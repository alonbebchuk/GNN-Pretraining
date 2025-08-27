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

from src.data.data_loaders import create_data_loader, create_pretrain_data_loader
from src.model.pretrain_model import PretrainableGNN
from src.pretraining.losses import UncertaintyWeighter
from src.pretraining.metrics import compute_training_metrics, compute_validation_metrics
from src.pretraining.schedulers import CosineWithWarmup, GRLLambdaScheduler
from src.pretraining.tasks import (
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
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "pretrain"
PATIENCE_EPOCHS = 5
UNCERTAINTY_WEIGHT_DECAY = 0
WARMUP_FRACTION = 0.15


@dataclass
class PretrainConfig:
    exp_name: str
    pretrain_domains: List[str]
    active_tasks: List[str]


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


def get_pretraining_train_loader(domains: List[str], seed: int, epoch: int) -> torch.utils.data.DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)

    return create_pretrain_data_loader(domains, generator)


def get_pretraining_val_loaders(domains: List[str], seed: int) -> Dict[str, torch.utils.data.DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    val_loaders = {}
    for domain in domains:
        val_loaders[domain] = create_data_loader(domain, "val", generator)

    return val_loaders


@torch.no_grad()
def run_validation(
    model: PretrainableGNN,
    tasks: Dict[str, BasePretrainTask],
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    weighter: UncertaintyWeighter,
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    model.eval()

    val_tasks = {k: v for k, v in tasks.items() if k != 'domain_adv'}

    per_domain_per_task_raw_losses = {}

    for domain_name, val_loader in val_loaders.items():
        per_domain_per_task_raw_losses[domain_name] = {}

        for task_name in val_tasks.keys():
            per_domain_per_task_raw_losses[domain_name][task_name] = []

        for batch in val_loader:
            batch = batch.to(device)
            batches_by_domain = {domain_name: batch}

            for task_name, task in val_tasks.items():
                raw_loss, _ = task.compute_loss(batches_by_domain)
                raw_val = float(raw_loss.detach().cpu())
                per_domain_per_task_raw_losses[domain_name][task_name].append(raw_val)

    for domain_name in val_loaders.keys():
        for task_name in val_tasks.keys():
            raw_losses = per_domain_per_task_raw_losses[domain_name][task_name]
            per_domain_per_task_raw_losses[domain_name][task_name] = float(np.mean(raw_losses))

    domain_weighted_means = []
    for domain in per_domain_per_task_raw_losses:
        tasks_list = list(per_domain_per_task_raw_losses[domain].keys())
        losses_list = list(per_domain_per_task_raw_losses[domain].values())
        losses_tensor = torch.tensor(losses_list, device=device, dtype=torch.float32)

        domain_raw_losses_tensor = dict(zip(tasks_list, losses_tensor))

        _, domain_weighted_components = weighter(domain_raw_losses_tensor)

        domain_weighted_losses = [loss.detach().cpu().item() for loss in domain_weighted_components.values()]
        domain_weighted_mean = float(np.mean(domain_weighted_losses))
        domain_weighted_means.append(domain_weighted_mean)

    total_balanced_loss = float(np.mean(domain_weighted_means))

    return per_domain_per_task_raw_losses, total_balanced_loss


def train(cfg: PretrainConfig, seed: int) -> None:
    set_global_seed(seed)
    device = torch.device("cuda")

    wandb.init(
        project="gnn-pretraining",
        name=f"{cfg.exp_name}_{seed}",
        tags=["pretraining", cfg.exp_name, f"seed_{seed}"],
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_loaders = get_pretraining_val_loaders(domains=cfg.pretrain_domains, seed=seed)
    model = PretrainableGNN(device=device, domain_names=cfg.pretrain_domains, task_names=cfg.active_tasks)
    tasks = instantiate_tasks(model, cfg.active_tasks)
    weighter = UncertaintyWeighter(task_names=cfg.active_tasks).to(device)

    opt_model = AdamW(model.parameters(), lr=LR_MODEL)
    opt_uncertainty = AdamW(weighter.parameters(), lr=LR_UNCERTAINTY, weight_decay=UNCERTAINTY_WEIGHT_DECAY)

    train_loader = get_pretraining_train_loader(cfg.pretrain_domains, seed, 0)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_FRACTION * total_steps)

    grl_sched = GRLLambdaScheduler(total_steps=total_steps)
    lr_multiplier = CosineWithWarmup(total_steps=total_steps, warmup_steps=warmup_steps)

    best_total_balanced_loss = float("inf")
    epochs_since_improvement = 0

    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loader = get_pretraining_train_loader(cfg.pretrain_domains, seed, epoch)

        for batches_by_domain in train_loader:
            step_start_time = time.time()
            global_step += 1

            for domain_name in batches_by_domain:
                batches_by_domain[domain_name] = batches_by_domain[domain_name].to(device)

            raw_losses = {}
            per_domain_per_task_raw_losses = {}

            for domain in cfg.pretrain_domains:
                per_domain_per_task_raw_losses[domain] = {}

            for task_name, task in tasks.items():
                raw_loss, per_domain_loss = task.compute_loss(batches_by_domain)
                raw_losses[task_name] = raw_loss

                if per_domain_loss is not None:
                    for domain, domain_loss in per_domain_loss.items():
                        per_domain_per_task_raw_losses[domain][task_name] = float(domain_loss)

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
                global_step,
            )

            step_time_ms = (time.time() - step_start_time) * 1000
            train_metrics['system/time_per_step_ms'] = step_time_ms

            if device.type == "cuda":
                train_metrics['system/gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
                train_metrics['system/gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3

            wandb.log(train_metrics, step=global_step)

        per_domain_per_task_raw_losses, total_balanced_loss = run_validation(model, tasks, val_loaders, weighter, device)

        val_metrics = compute_validation_metrics(
            per_domain_per_task_raw_losses,
            total_balanced_loss,
            weighter,
        )

        current_total_balanced_loss = val_metrics['val/loss/total_balanced']

        if current_total_balanced_loss < best_total_balanced_loss:
            best_total_balanced_loss = current_total_balanced_loss
            epochs_since_improvement = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }

            best_model_path = output_dir / f"best_model_{cfg.exp_name}_{seed}.pt"
            torch.save(checkpoint, best_model_path)

            artifact = wandb.Artifact(
                name=f"model_{cfg.exp_name}_{seed}",
                type="model",
            )
            artifact.add_file(str(best_model_path))
            wandb.log_artifact(artifact)
        else:
            epochs_since_improvement += 1

        wandb.log(val_metrics, step=global_step)

        if epochs_since_improvement >= PATIENCE_EPOCHS:
            break

    wandb.log({
        'final/total_epochs': epoch,
        'final/early_stopped': epochs_since_improvement >= PATIENCE_EPOCHS,
        'final/best_total_balanced_loss': best_total_balanced_loss,
    })

    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()
    cfg = build_config(args)

    train(cfg, args.seed)


if __name__ == "__main__":
    main()
