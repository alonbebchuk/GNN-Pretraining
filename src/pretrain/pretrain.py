import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import wandb
import yaml
from torch.optim import AdamW

from src.data.data_setup import PRETRAIN_TUDATASETS
from src.data.pretrain_data_loaders import create_train_data_loader, create_val_data_loader
from src.models.pretrain_model import PretrainableGNN
from src.pretrain.losses import UncertaintyWeighter
from src.pretrain.metrics import compute_training_metrics, compute_validation_metrics
from src.pretrain.schedulers import CosineWithWarmup, GRLLambdaScheduler
from src.pretrain.tasks import (
    BasePretrainTask,
    DomainAdversarialTask,
    GraphContrastiveTask,
    GraphPropertyPredictionTask,
    LinkPredictionTask,
    NodeContrastiveTask,
    NodeFeatureMaskingTask,
)

BATCH_SIZE = 32
EPOCHS = 50
LR_MODEL = 3e-4
LR_UNCERTAINTY = 3e-3
PATIENCE_EPOCHS = 5
UNCERTAINTY_WEIGHT_DECAY = 0

PRETRAIN_DOMAINS = {
    'b2': PRETRAIN_TUDATASETS,
    'b3': PRETRAIN_TUDATASETS,
    'b4': ['ENZYMES'],
    's1': PRETRAIN_TUDATASETS,
    's2': PRETRAIN_TUDATASETS,
    's3': PRETRAIN_TUDATASETS,
    's4': PRETRAIN_TUDATASETS,
    's5': PRETRAIN_TUDATASETS,
}

ACTIVE_TASKS = {
    'b2': ['node_feat_mask'],
    'b3': ['node_contrast'],
    'b4': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast', 'graph_prop'],
    's1': ['node_feat_mask', 'link_pred'],
    's2': ['node_contrast', 'graph_contrast'],
    's3': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast'],
    's4': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast', 'graph_prop'],
    's5': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast', 'graph_prop', 'domain_adv'],
}

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "pretrain"
PROJECT_NAME = "gnn-pretraining-pretrain"


@dataclass
class PretrainConfig:
    exp_name: str
    seed: int
    pretrain_domains: List[str]
    active_tasks: List[str]

    def __post_init__(self):
        self.pretrain_domains = PRETRAIN_DOMAINS.get(self.exp_name, PRETRAIN_TUDATASETS)
        self.active_tasks = ACTIVE_TASKS[self.exp_name]


def build_config(args: argparse.Namespace) -> PretrainConfig:
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PretrainConfig(**data)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def instantiate_tasks(model: PretrainableGNN, active_tasks: List[str]) -> Dict[str, BasePretrainTask]:
    tasks = {}
    for name in active_tasks:
        if name == "node_feat_mask":
            tasks[name] = NodeFeatureMaskingTask(model)
        elif name == "link_pred":
            tasks[name] = LinkPredictionTask(model)
        elif name == "node_contrast":
            tasks[name] = NodeContrastiveTask(model)
        elif name == "graph_contrast":
            tasks[name] = GraphContrastiveTask(model)
        elif name == "graph_prop":
            tasks[name] = GraphPropertyPredictionTask(model)
        elif name == "domain_adv":
            tasks[name] = DomainAdversarialTask(model)
    return tasks


def run_training(
    model: PretrainableGNN,
    tasks: Dict[str, BasePretrainTask],
    weighter: UncertaintyWeighter,
    opt_model: AdamW,
    opt_uncertainty: AdamW,
    train_loader: torch.utils.data.DataLoader,
    generator: torch.Generator,
    grl_sched: GRLLambdaScheduler,
    lr_multiplier: CosineWithWarmup,
    device: torch.device,
    epoch: int,
    global_step_ref: List[int],
    cfg: PretrainConfig,
) -> None:
    model.train()

    for domain_batches in train_loader:
        step_start_time = time.time()
        global_step_ref[0] += 1

        for domain_name in domain_batches:
            domain_batches[domain_name] = domain_batches[domain_name].to(device)

        raw_losses = {}
        per_domain_per_task_raw_losses = {}

        for domain in cfg.pretrain_domains:
            per_domain_per_task_raw_losses[domain] = {}

        for task_name, task in tasks.items():
            if task_name == 'domain_adv':
                raw_loss, _ = task.compute_loss(domain_batches, generator, lambda_val=grl_sched())
            else:
                raw_loss, per_domain_loss = task.compute_loss(domain_batches, generator)
                for domain, domain_loss in per_domain_loss.items():
                    per_domain_per_task_raw_losses[domain][task_name] = float(domain_loss)
            raw_losses[task_name] = raw_loss

        total_weighted_loss = weighter(raw_losses, lambda_val=grl_sched())

        # Compute per-domain weighted losses for analysis
        per_domain_weighted_losses = {}
        for domain in per_domain_per_task_raw_losses.keys():
            # Create domain-specific raw losses dict
            domain_raw_losses = {}
            for task_name in per_domain_per_task_raw_losses[domain].keys():
                domain_raw_losses[task_name] = torch.tensor(
                    per_domain_per_task_raw_losses[domain][task_name], 
                    device=device
                )
            
            # Add domain_adv if present (shared across all domains)
            if 'domain_adv' in raw_losses:
                domain_raw_losses['domain_adv'] = raw_losses['domain_adv']
            
            # Compute weighted loss for this domain
            domain_weighted_loss = weighter(domain_raw_losses, lambda_val=grl_sched())
            per_domain_weighted_losses[domain] = float(domain_weighted_loss.detach().cpu())

        opt_model.zero_grad(set_to_none=True)
        opt_uncertainty.zero_grad(set_to_none=True)
        total_weighted_loss.backward()
        opt_model.step()
        opt_uncertainty.step()

        scale = lr_multiplier()
        for pg in opt_model.param_groups:
            pg["lr"] = LR_MODEL * scale
        for pg in opt_uncertainty.param_groups:
            pg["lr"] = LR_UNCERTAINTY * scale

        grl_sched.step()
        lr_multiplier.step()

        train_metrics = compute_training_metrics(
            per_domain_per_task_raw_losses,
            raw_losses,
            total_weighted_loss,
            weighter,
            grl_sched,
            opt_model,
            opt_uncertainty,
            model,
            epoch,
            global_step_ref[0],
            step_start_time,
            per_domain_weighted_losses
        )

        wandb.log(train_metrics, step=global_step_ref[0])


@torch.no_grad()
def run_evaluation(
    model: PretrainableGNN,
    tasks: Dict[str, BasePretrainTask],
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    generator: torch.Generator,
    weighter: UncertaintyWeighter,
    grl_sched: GRLLambdaScheduler,
    device: torch.device,
    epoch: int,
    best_total_weighted_loss: float,
    epochs_since_improvement: int,
    cfg: PretrainConfig,
    seed: int,
    global_step: List[int]
) -> Tuple[bool, Dict[str, float], float, int]:
    model.eval()

    val_tasks = tasks

    # Collect all validation batches first
    all_val_batches = {}
    for domain_name, val_loader in val_loaders.items():
        domain_batches = []
        for batch in val_loader:
            domain_batches.append(batch.to(device))
        all_val_batches[domain_name] = domain_batches

    # Compute raw losses and per-domain losses like in training
    raw_losses = {}
    per_domain_per_task_raw_losses = {}

    for domain_name in val_loaders.keys():
        per_domain_per_task_raw_losses[domain_name] = {}

    # Process each task like in run_training
    for task_name, task in val_tasks.items():
        if task_name == 'domain_adv':
            # Domain adversarial uses first batch from all domains
            domain_batches = {domain: batches[0] for domain, batches in all_val_batches.items()}
            raw_loss, _ = task.compute_loss(domain_batches, generator, lambda_val=grl_sched())
            raw_losses[task_name] = raw_loss
        else:
            # Regular tasks: compute per domain and aggregate
            task_losses = []
            for domain_name, domain_batches in all_val_batches.items():
                domain_task_losses = []
                for batch in domain_batches:
                    single_batch_dict = {domain_name: batch}
                    raw_loss, _ = task.compute_loss(single_batch_dict, generator)
                    domain_task_losses.append(raw_loss)
                
                # Average across batches for this domain-task
                domain_avg_loss = torch.stack(domain_task_losses).mean()
                per_domain_per_task_raw_losses[domain_name][task_name] = float(domain_avg_loss.detach().cpu())
                task_losses.append(domain_avg_loss)
            
            # Average across domains for this task  
            raw_losses[task_name] = torch.stack(task_losses).mean()

    # Compute total weighted loss like in training (not per-domain)
    # Average per-domain losses for each task
    task_domain_averages = {}
    for task_name in set().union(*[domain_tasks.keys() for domain_tasks in per_domain_per_task_raw_losses.values()]):
        task_losses = [per_domain_per_task_raw_losses[domain][task_name] 
                      for domain in per_domain_per_task_raw_losses.keys() 
                      if task_name in per_domain_per_task_raw_losses[domain]]
        task_domain_averages[task_name] = float(np.mean(task_losses))

    # Add domain_adv if present (computed once across all domains)
    if 'domain_adv' in raw_losses:
        task_domain_averages['domain_adv'] = float(raw_losses['domain_adv'].detach().cpu())

    # Convert to tensor dict for weighter (like training)
    domain_averaged_raw_losses = {
        task: torch.tensor(loss, device=device) 
        for task, loss in task_domain_averages.items()
    }
    
    # Compute total weighted loss once (like training)
    total_weighted_loss_tensor = weighter(domain_averaged_raw_losses, lambda_val=grl_sched())
    total_weighted_loss = float(total_weighted_loss_tensor.detach().cpu())

    # Compute per-domain weighted losses for validation analysis
    per_domain_weighted_losses_val = {}
    for domain in per_domain_per_task_raw_losses.keys():
        # Create domain-specific raw losses dict
        domain_raw_losses = {}
        for task_name in per_domain_per_task_raw_losses[domain].keys():
            domain_raw_losses[task_name] = torch.tensor(
                per_domain_per_task_raw_losses[domain][task_name], 
                device=device
            )
        
        # Add domain_adv if present (shared across all domains)
        if 'domain_adv' in raw_losses:
            domain_raw_losses['domain_adv'] = raw_losses['domain_adv']
        
        # Compute weighted loss for this domain
        domain_weighted_loss = weighter(domain_raw_losses, lambda_val=grl_sched())
        per_domain_weighted_losses_val[domain] = float(domain_weighted_loss.detach().cpu())

    val_metrics = compute_validation_metrics(
        per_domain_per_task_raw_losses,
        raw_losses,
        total_weighted_loss,
        weighter,
        grl_sched,
        epoch,
        global_step[0],
        per_domain_weighted_losses_val
    )

    current_total_weighted_loss = val_metrics['val/loss/total_weighted']
    is_best = current_total_weighted_loss < best_total_weighted_loss
    
    if is_best:
        new_best_loss = current_total_weighted_loss
        new_epochs_since_improvement = 0

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_metrics': val_metrics,
        }

        best_model_path = OUTPUT_DIR / f"best_model_{cfg.exp_name}_{seed}.pt"
        torch.save(checkpoint, best_model_path)

        artifact = wandb.Artifact(
            name=f"model_{cfg.exp_name}_{seed}",
            type="model",
        )
        artifact.add_file(str(best_model_path))
        wandb.log_artifact(artifact)
    else:
        new_best_loss = best_total_weighted_loss
        new_epochs_since_improvement = epochs_since_improvement + 1

    wandb.log(val_metrics, step=global_step[0])

    return is_best, val_metrics, new_best_loss, new_epochs_since_improvement


def pretrain(cfg: PretrainConfig, seed: int) -> None:
    set_global_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=PROJECT_NAME, name=f"{cfg.exp_name}_{seed}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    val_loaders = {}
    for domain in cfg.pretrain_domains:
        val_loaders[domain] = create_val_data_loader(domain, generator)

    model = PretrainableGNN(
        device=device,
        domain_names=cfg.pretrain_domains,
        task_names=cfg.active_tasks,
    )

    tasks = instantiate_tasks(model, cfg.active_tasks)

    weighter = UncertaintyWeighter(task_names=cfg.active_tasks).to(device)

    opt_model = AdamW(model.parameters(), lr=LR_MODEL)
    opt_uncertainty = AdamW(weighter.parameters(), lr=LR_UNCERTAINTY, weight_decay=UNCERTAINTY_WEIGHT_DECAY)

    train_loader = create_train_data_loader(cfg.pretrain_domains, generator)
    total_steps = len(train_loader) * EPOCHS

    grl_sched = GRLLambdaScheduler(total_steps=total_steps)
    lr_multiplier = CosineWithWarmup(total_steps=total_steps)

    best_total_weighted_loss = float("inf")
    epochs_since_improvement = 0

    global_step = [0]

    for epoch in range(1, EPOCHS + 1):
        run_training(
            model,
            tasks,
            weighter,
            opt_model,
            opt_uncertainty,
            train_loader,
            generator,
            grl_sched,
            lr_multiplier,
            device,
            epoch,
            global_step,
            cfg
        )

        is_best, val_metrics, best_total_weighted_loss, epochs_since_improvement = run_evaluation(
            model, tasks, val_loaders, generator, weighter, grl_sched, device, 
            epoch, best_total_weighted_loss, epochs_since_improvement, 
            cfg, seed, global_step
        )

        if epochs_since_improvement >= PATIENCE_EPOCHS:
            break

    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()
    cfg = build_config(args)

    pretrain(cfg, args.seed)


if __name__ == "__main__":
    main()