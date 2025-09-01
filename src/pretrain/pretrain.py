import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import wandb
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
EPOCHS = 5
LR_MODEL = 3e-4
LR_UNCERTAINTY = 3e-3
PATIENCE_FRACTION = 0.1
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

    pretrain_domains: List[str] = None
    active_tasks: List[str] = None

    def __post_init__(self):
        self.pretrain_domains = PRETRAIN_DOMAINS[self.exp_name]
        self.active_tasks = ACTIVE_TASKS[self.exp_name]


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
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

        per_task_raw_losses = {}
        per_domain_per_task_raw_losses = {domain: {} for domain in cfg.pretrain_domains}
        per_domain_weighted_losses = {}

        for task_name, task in tasks.items():
            task_raw_loss, task_per_domain_raw_loss = task.compute_loss(domain_batches, generator)
            per_task_raw_losses[task_name] = task_raw_loss
            for domain, per_domain_per_task_raw_loss in task_per_domain_raw_loss.items():
                per_domain_per_task_raw_losses[domain][task_name] = float(per_domain_per_task_raw_loss)

        for domain in per_domain_per_task_raw_losses.keys():
            domain_weighted_loss = weighter(per_domain_per_task_raw_losses[domain], lambda_val=grl_sched())
            per_domain_weighted_losses[domain] = float(domain_weighted_loss.detach().cpu())

        total_weighted_loss = weighter(per_task_raw_losses, lambda_val=grl_sched())

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
            per_task_raw_losses,
            per_domain_weighted_losses,
            total_weighted_loss,
            weighter,
            grl_sched,
            opt_model,
            opt_uncertainty,
            model,
            epoch,
            global_step_ref[0],
            step_start_time
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
    global_step: List[int]
) -> Tuple[float, int]:
    model.eval()

    per_task_raw_losses = {}
    per_domain_per_task_raw_losses = {domain: {} for domain in val_loaders.keys()}

    for task_name, task in tasks.items():
        domain_losses = []

        for domain_name, val_loader in val_loaders.items():
            batch_losses = []
            for batch in val_loader:
                batch = batch.to(device)
                loss, _ = task.compute_loss({domain_name: batch}, generator)
                batch_losses.append(loss)
            
            domain_task_raw_loss = torch.stack(batch_losses).mean()
            domain_losses.append(domain_task_raw_loss)
            per_domain_per_task_raw_losses[domain_name][task_name] = float(domain_task_raw_loss.detach().cpu())
        
        per_task_raw_losses[task_name] = torch.stack(domain_losses).mean()
    
    total_weighted_loss = weighter(per_task_raw_losses, lambda_val=grl_sched())

    per_domain_weighted_losses = {}
    for domain in per_domain_per_task_raw_losses.keys():
        domain_losses = {
            task: torch.tensor(loss, device=device)
            for task, loss in per_domain_per_task_raw_losses[domain].items()
        }
        domain_weighted_loss = weighter(domain_losses, lambda_val=grl_sched())
        per_domain_weighted_losses[domain] = float(domain_weighted_loss.detach().cpu())

    val_metrics = compute_validation_metrics(
        per_domain_per_task_raw_losses,
        per_task_raw_losses,
        per_domain_weighted_losses,
        total_weighted_loss,
        weighter,
        grl_sched,
        epoch
    )

    if total_weighted_loss < best_total_weighted_loss:
        best_total_weighted_loss = total_weighted_loss
        epochs_since_improvement = 0

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_metrics': val_metrics,
        }
        model_name = f"model_{cfg.exp_name}_{cfg.seed}"
        model_path = OUTPUT_DIR / f"{model_name}.pt"

        torch.save(checkpoint, model_path)

        artifact = wandb.Artifact(
            name=model_name,
            type="model",
        )
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
    else:
        epochs_since_improvement += 1

    wandb.log(val_metrics, step=global_step[0])

    return best_total_weighted_loss, epochs_since_improvement


def pretrain(cfg: PretrainConfig) -> None:
    set_global_seed(cfg.seed)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=PROJECT_NAME, name=f"{cfg.exp_name}_{cfg.seed}")

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

        best_total_weighted_loss, epochs_since_improvement = run_evaluation(
            model,
            tasks,
            val_loaders,
            generator,
            weighter,
            grl_sched,
            device,
            epoch,
            best_total_weighted_loss,
            epochs_since_improvement,
            cfg,
            global_step
        )

        if epochs_since_improvement >= int(EPOCHS * PATIENCE_FRACTION):
            break

    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    cfg = PretrainConfig(exp_name=args.exp_name, seed=args.seed)
    pretrain(cfg)


if __name__ == "__main__":
    main()
