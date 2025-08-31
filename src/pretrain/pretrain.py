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
WARMUP_FRACTION = 0.15

PRETRAIN_DOMAINS = {
    'b4': ['ENZYMES'],
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


def compute_task_synergy_metrics(
    per_domain_per_task_raw_losses: Dict[str, Dict[str, float]],
    total_balanced_loss: float,
    weighter: UncertaintyWeighter,
    active_tasks: List[str]
) -> Dict[str, float]:
    """Compute task synergy and contribution analysis metrics."""

    metrics = {}

    # Task contribution analysis
    for task_name in active_tasks:
        if task_name == 'domain_adv':
            continue

        # Compute average loss for this task across domains
        task_losses = []
        for domain_losses in per_domain_per_task_raw_losses.values():
            if task_name in domain_losses:
                task_losses.append(domain_losses[task_name])

        if task_losses:
            avg_task_loss = np.mean(task_losses)
            metrics[f'task_analysis/{task_name}_avg_loss'] = avg_task_loss
            metrics[f'task_analysis/{task_name}_loss_std'] = np.std(
                task_losses)

    # Task balance analysis using uncertainty weights
    if hasattr(weighter, 'log_sigma'):
        for i, task_name in enumerate(active_tasks):
            if task_name != 'domain_adv' and i < len(weighter.log_sigma):
                sigma = torch.exp(weighter.log_sigma[i]).item()
                metrics[f'task_balance/{task_name}_uncertainty_weight'] = 1.0 / (
                    sigma ** 2)
                metrics[f'task_balance/{task_name}_sigma'] = sigma

    # Overall task diversity
    task_losses_flat = []
    for domain_losses in per_domain_per_task_raw_losses.values():
        task_losses_flat.extend(domain_losses.values())

    if len(task_losses_flat) > 1:
        metrics['task_analysis/task_loss_diversity'] = np.std(task_losses_flat)
        metrics['task_analysis/task_loss_mean'] = np.mean(task_losses_flat)

    return metrics


def compute_domain_analysis_metrics(
    per_domain_per_task_raw_losses: Dict[str, Dict[str, float]],
    pretrain_domains: List[str]
) -> Dict[str, float]:
    """Compute cross-domain transfer and similarity metrics."""

    metrics = {}

    # Per-domain performance
    for domain in pretrain_domains:
        if domain in per_domain_per_task_raw_losses:
            domain_losses = list(
                per_domain_per_task_raw_losses[domain].values())
            if domain_losses:
                metrics[f'domain/{domain}_avg_loss'] = np.mean(domain_losses)
                metrics[f'domain/{domain}_loss_std'] = np.std(domain_losses)

    # Cross-domain consistency
    if len(pretrain_domains) > 1:
        domain_avg_losses = []
        for domain in pretrain_domains:
            if domain in per_domain_per_task_raw_losses:
                domain_losses = list(
                    per_domain_per_task_raw_losses[domain].values())
                if domain_losses:
                    domain_avg_losses.append(np.mean(domain_losses))

        if len(domain_avg_losses) > 1:
            metrics['domain/cross_domain_consistency'] = 1.0 / \
                (1.0 + np.std(domain_avg_losses))
            metrics['domain/domain_performance_gap'] = max(
                domain_avg_losses) - min(domain_avg_losses)

    return metrics


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
            domain_batches[domain_name] = domain_batches[domain_name].to(
                device)

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
                    per_domain_per_task_raw_losses[domain][task_name] = float(
                        domain_loss)
            raw_losses[task_name] = raw_loss

        total_weighted_loss, _ = weighter(raw_losses, lambda_val=grl_sched())

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
        )

        # Streamlined system metrics
        step_time_ms = (time.time() - step_start_time) * 1000
        train_metrics['system/avg_step_time_ms'] = step_time_ms

        if device.type == "cuda":
            train_metrics['system/gpu_memory_gb'] = torch.cuda.memory_allocated() / \
                1024**3

        wandb.log(train_metrics, step=global_step_ref[0])


@torch.no_grad()
def run_evaluation(
    model: PretrainableGNN,
    tasks: Dict[str, BasePretrainTask],
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    generator: torch.Generator,
    weighter: UncertaintyWeighter,
    device: torch.device,
    epoch: int,
    best_total_balanced_loss: float,
    epochs_since_improvement: int,
    cfg: PretrainConfig,
    seed: int,
    global_step: List[int]
) -> Tuple[bool, Dict[str, float], float, int]:
    model.eval()

    val_tasks = {k: v for k, v in tasks.items() if k != 'domain_adv'}

    per_domain_per_task_raw_losses = {}

    for domain_name, val_loader in val_loaders.items():
        per_domain_per_task_raw_losses[domain_name] = {}

        for task_name in val_tasks.keys():
            per_domain_per_task_raw_losses[domain_name][task_name] = []

        for batch in val_loader:
            batch = batch.to(device)
            domain_batches = {domain_name: batch}

            for task_name, task in val_tasks.items():
                raw_loss, _ = task.compute_loss(domain_batches, generator)
                raw_val = float(raw_loss.detach().cpu())
                per_domain_per_task_raw_losses[domain_name][task_name].append(
                    raw_val)

    for domain_name in val_loaders.keys():
        for task_name in val_tasks.keys():
            raw_losses = per_domain_per_task_raw_losses[domain_name][task_name]
            per_domain_per_task_raw_losses[domain_name][task_name] = float(
                np.mean(raw_losses))

    domain_weighted_means = []
    for domain in per_domain_per_task_raw_losses:
        tasks_list = list(per_domain_per_task_raw_losses[domain].keys())
        losses_list = list(per_domain_per_task_raw_losses[domain].values())
        losses_tensor = torch.tensor(
            losses_list, device=device, dtype=torch.float32)

        domain_raw_losses_tensor = dict(zip(tasks_list, losses_tensor))

        _, domain_weighted_components = weighter(domain_raw_losses_tensor)

        domain_weighted_losses = [loss.detach().cpu().item()
                                  for loss in domain_weighted_components.values()]
        domain_weighted_mean = float(np.mean(domain_weighted_losses))
        domain_weighted_means.append(domain_weighted_mean)

    total_balanced_loss = float(np.mean(domain_weighted_means))

    # Compute all validation metrics
    val_metrics = compute_validation_metrics(
        per_domain_per_task_raw_losses,
        total_balanced_loss,
        weighter,
    )
    
    # Add advanced analysis metrics
    task_synergy_metrics = compute_task_synergy_metrics(
        per_domain_per_task_raw_losses, total_balanced_loss, weighter, cfg.active_tasks
    )
    domain_analysis_metrics = compute_domain_analysis_metrics(
        per_domain_per_task_raw_losses, cfg.pretrain_domains
    )
    
    val_metrics.update(task_synergy_metrics)
    val_metrics.update(domain_analysis_metrics)

    current_total_balanced_loss = val_metrics['val/loss/total_balanced']
    is_best = current_total_balanced_loss < best_total_balanced_loss
    
    if is_best:
        new_best_loss = current_total_balanced_loss
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
        new_best_loss = best_total_balanced_loss
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
    warmup_steps = int(total_steps * WARMUP_FRACTION)

    grl_sched = GRLLambdaScheduler(total_steps=total_steps)
    lr_multiplier = CosineWithWarmup(total_steps=total_steps, warmup_steps=warmup_steps)

    best_total_balanced_loss = float("inf")
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

        is_best, val_metrics, best_total_balanced_loss, epochs_since_improvement = run_evaluation(
            model, tasks, val_loaders, generator, weighter, device, 
            epoch, best_total_balanced_loss, epochs_since_improvement, 
            cfg, seed, global_step
        )

        if epochs_since_improvement >= PATIENCE_EPOCHS:
            break

    final_metrics = {
        'convergence/total_epochs': epoch,
        'convergence/early_stopped': epochs_since_improvement >= PATIENCE_EPOCHS,
        'convergence/epochs_to_best': epoch - epochs_since_improvement,
        'convergence/training_efficiency': 1.0 / max(best_total_balanced_loss, 1e-6),
        'convergence/patience_utilization': epochs_since_improvement / PATIENCE_EPOCHS,
        'performance/best_total_balanced_loss': best_total_balanced_loss,
        'performance/final_loss': val_metrics.get('val/loss/total_balanced', best_total_balanced_loss),
        'performance/loss_improvement': abs(best_total_balanced_loss - val_metrics.get('val/loss/total_balanced', best_total_balanced_loss)),
        'experiment/scheme_name': cfg.exp_name,
        'experiment/num_domains': len(cfg.pretrain_domains),
        'experiment/num_active_tasks': len(cfg.active_tasks),
        'experiment/multitask_complexity': len(cfg.active_tasks) * len(cfg.pretrain_domains),
        'training/total_steps': global_step[0],
        'training/steps_per_epoch': global_step[0] / max(epoch, 1),
        'training/convergence_rate': (epoch - epochs_since_improvement) / max(epoch, 1),
    }

    wandb.log(final_metrics)

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
