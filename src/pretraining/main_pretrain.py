import argparse
import os
import random
import json
import wandb
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, RandomSampler
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.common import (
    PRETRAIN_EPOCHS,
    PRETRAIN_BATCH_SIZE_PER_DOMAIN,
    PRETRAIN_LR_MODEL,
    PRETRAIN_LR_UNCERTAINTY,
    PRETRAIN_WEIGHT_DECAY,
    PRETRAIN_EVAL_EVERY_EPOCHS,
    PRETRAIN_OUTPUT_DIR,
    WANDB_PROJECT,
)
from src.model.pretrain_model import PretrainableGNN
from src.pretraining.losses import UncertaintyWeighter
from src.pretraining.schedulers import GRLLambdaScheduler
from src.pretraining.tasks import (
    DomainAdversarialTask,
    GraphContrastiveTask,
    GraphPropertyPredictionTask,
    LinkPredictionTask,
    NodeContrastiveTask,
    NodeFeatureMaskingTask,
)
from src.data.data_setup import PROCESSED_DIR


# -----------------------------
# Configuration
# -----------------------------


@dataclass
class TrainConfig:
    # Required
    exp_name: str
    pretrain_domains: List[str]
    active_tasks: List[str]

    # Training
    epochs: int = PRETRAIN_EPOCHS
    batch_size_per_domain: int = PRETRAIN_BATCH_SIZE_PER_DOMAIN
    lr_model: float = PRETRAIN_LR_MODEL
    lr_uncertainty: float = PRETRAIN_LR_UNCERTAINTY
    weight_decay: float = PRETRAIN_WEIGHT_DECAY
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every_steps: int = 10
    eval_every_epochs: int = PRETRAIN_EVAL_EVERY_EPOCHS

    # Checkpointing
    output_dir: str = PRETRAIN_OUTPUT_DIR


def build_config(args: argparse.Namespace) -> TrainConfig:
    base = TrainConfig()

    with open(args.config, "r", encoding="utf-8") as f:
        yaml_overrides = yaml.safe_load(f) or {}

    base.exp_name = yaml_overrides["exp_name"]
    base.pretrain_domains = list(yaml_overrides["pretrain_domains"])
    base.active_tasks = list(yaml_overrides["active_tasks"])

    return base


# -----------------------------
# Data utilities
# -----------------------------


class DomainSplitDataset(Dataset):
    def __init__(self, graphs: List, indices: Sequence[int]):
        self.graphs: List = graphs
        # Ensure list of ints (may come as tensor)
        self.indices: List[int] = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.graphs[self.indices[idx]]


def make_domain_loader(
    domain_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    num_steps: Optional[int] = None,
) -> Tuple[PyGDataLoader, int]:
    """
    Load processed graphs for a domain and return a DataLoader that yields
    batches as Sequence[(Data, domain_name)] to match task expectations.
    Returns the number of batches per epoch (len(loader)).
    """
    dom_dir = PROCESSED_DIR / f"{domain_name}_pretrain"
    data_path = dom_dir / "data.pt"
    splits_path = dom_dir / "splits.pt"

    if not data_path.exists() or not splits_path.exists():
        raise FileNotFoundError(
            f"Processed data for domain '{domain_name}' not found under {dom_dir}. "
            "Please run src/data/data_setup.py first."
        )

    graphs: List = torch.load(data_path)
    splits: Dict[str, torch.Tensor] = torch.load(splits_path)
    split_idx = splits[split]

    ds = DomainSplitDataset(graphs, split_idx)

    def collate_fn(batch_graphs: List) -> List[Tuple[object, str]]:
        # Keep as a simple list of (graph, domain_name) without PyG Batch collation
        return [(g, domain_name) for g in batch_graphs]

    if split == "train" and num_steps is not None:
        # Balanced with-replacement sampling for a fixed number of steps
        sampler = RandomSampler(ds, replacement=True, num_samples=num_steps * batch_size)
        loader = PyGDataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
        length = num_steps
    else:
        loader = PyGDataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
        # Compute length in batches (ceil)
        length = (len(ds) + batch_size - 1) // batch_size

    return loader, length


def build_multi_domain_loaders(
    domains: List[str],
    batch_size_per_domain: int,
    num_workers: int,
) -> Tuple[Dict[str, PyGDataLoader], Dict[str, PyGDataLoader], int]:
    # First pass to compute naive lengths per domain (without replacement)
    provisional_lengths: Dict[str, int] = {}
    for d in domains:
        _, tr_len = make_domain_loader(d, "train", batch_size_per_domain, num_workers)
        provisional_lengths[d] = tr_len

    # Balanced epoch length: minimum steps across domains
    steps_per_epoch = min(provisional_lengths.values()) if provisional_lengths else 0

    # Build loaders with with-replacement samplers to enforce equal steps per epoch
    train_loaders: Dict[str, PyGDataLoader] = {}
    val_loaders: Dict[str, PyGDataLoader] = {}
    for d in domains:
        tr_loader, _ = make_domain_loader(d, "train", batch_size_per_domain, num_workers, num_steps=steps_per_epoch)
        va_loader, _ = make_domain_loader(d, "val", batch_size_per_domain, num_workers)
        train_loaders[d] = tr_loader
        val_loaders[d] = va_loader

    return train_loaders, val_loaders, steps_per_epoch


# -----------------------------
# Task factory
# -----------------------------


def instantiate_tasks(model: PretrainableGNN, active_tasks: List[str]):
    mapping = {
        "node_feat_mask": NodeFeatureMaskingTask(model),
        "link_pred": LinkPredictionTask(model),
        "node_contrast": NodeContrastiveTask(model),
        "graph_contrast": GraphContrastiveTask(model),
        "graph_prop": GraphPropertyPredictionTask(model),
        "domain_adv": DomainAdversarialTask(model),
    }
    tasks: Dict[str, object] = {}
    for name in active_tasks:
        if name not in mapping:
            raise ValueError(f"Unknown task '{name}'.")
        tasks[name] = mapping[name]
    return tasks


# -----------------------------
# Training / validation
# -----------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def combine_next_batches(iters: Dict[str, Iterable]) -> List[Tuple[object, str]]:
    batch: List[Tuple[object, str]] = []
    for domain_name, it_ in iters.items():
        try:
            part = next(it_)
        except StopIteration:
            return []
        # part is a list[(Data, domain)] from our collate_fn
        batch.extend(part)
    return batch


@torch.no_grad()
def run_validation(
    model: PretrainableGNN,
    tasks: Dict[str, object],
    weighter: UncertaintyWeighter,
    val_loaders: Dict[str, PyGDataLoader],
    scheduler: GRLLambdaScheduler,
    device: torch.device,
    steps_cap: Optional[int] = None,
) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, float], float]:
    model.eval()

    # Build iterators per domain
    iters = {d: iter(l) for d, l in val_loaders.items()}
    steps_done = 0
    agg_raw: Dict[str, List[float]] = {}
    agg_weighted: Dict[str, List[float]] = {}
    total_losses: List[float] = []

    while True:
        if steps_cap is not None and steps_done >= steps_cap:
            break
        batch = combine_next_batches(iters)
        if not batch:
            break

        # Compute raw losses per task
        raw_losses: Dict[str, torch.Tensor] = {}
        for name, task in tasks.items():
            if name == "domain_adv":
                raw_losses[name] = task.compute_loss(batch, lambda_val=scheduler())
            else:
                raw_losses[name] = task.compute_loss(batch)

        # Weighted total for monitoring (uses current sigmas and lambda)
        total_loss, weighted = weighter(raw_losses, lambda_val=scheduler())
        total_losses.append(float(total_loss.detach().cpu()))

        for k, v in raw_losses.items():
            agg_raw.setdefault(k, []).append(float(v.detach().cpu()))
        for k, v in weighted.items():
            agg_weighted.setdefault(k, []).append(float(v.detach().cpu()))

        steps_done += 1

    # Averages across mixed-domain validation stream
    total = float(np.mean(total_losses)) if total_losses else 0.0
    raw_means = {k: float(np.mean(v)) for k, v in agg_raw.items()}
    weighted_means = {k: float(np.mean(v)) for k, v in agg_weighted.items()}

    # Compute domain-balanced validation: per-domain totals averaged equally
    per_domain_total: Dict[str, float] = {}
    for domain_name, loader in val_loaders.items():
        per_batch_totals: List[float] = []
        for batch in loader:
            # Ensure the batch is for a single domain
            # Our collate_fn yields [(g, domain_name)] with consistent domain
            raw_losses: Dict[str, torch.Tensor] = {}
            for name, task in tasks.items():
                if name == "domain_adv":
                    raw_losses[name] = task.compute_loss(batch, lambda_val=scheduler())
                else:
                    raw_losses[name] = task.compute_loss(batch)
            dom_total, _ = weighter(raw_losses, lambda_val=scheduler())
            per_batch_totals.append(float(dom_total.detach().cpu()))
        per_domain_total[domain_name] = float(np.mean(per_batch_totals)) if per_batch_totals else 0.0

    balanced_total = float(np.mean(list(per_domain_total.values()))) if per_domain_total else total

    return total, raw_means, weighted_means, per_domain_total, balanced_total


def train_single_seed(cfg: TrainConfig, seed: int) -> None:
    set_global_seed(seed)

    device = torch.device(cfg.device)

    # Data
    train_loaders, val_loaders, steps_per_epoch = build_multi_domain_loaders(
        domains=cfg.pretrain_domains,
        batch_size_per_domain=cfg.batch_size_per_domain,
        num_workers=cfg.num_workers,
    )

    if steps_per_epoch == 0:
        raise RuntimeError("No training steps available. Check your datasets and splits.")

    total_steps = cfg.epochs * steps_per_epoch

    # Model & tasks
    model = PretrainableGNN(device=device, domain_names=cfg.pretrain_domains, task_names=cfg.active_tasks)
    tasks = instantiate_tasks(model, cfg.active_tasks)

    # Loss weighter and optimizers
    weighter = UncertaintyWeighter(task_names=cfg.active_tasks).to(device)
    opt_model = Adam(model.parameters(), lr=cfg.lr_model, weight_decay=cfg.weight_decay)
    opt_uncertainty = Adam(weighter.parameters(), lr=cfg.lr_uncertainty, weight_decay=0.0)

    # GRL lambda scheduler
    grl_sched = GRLLambdaScheduler(total_steps=total_steps)

    # W&B
    # Compose informative tags for easy filtering in W&B UI
    wb_tags = [
        "phase:pretrain",
        *(f"domain:{d}" for d in cfg.pretrain_domains),
        *(f"task:{t}" for t in cfg.active_tasks),
    ]

    run = wandb.init(
        project=WANDB_PROJECT,
        group=cfg.exp_name,
        name=f"{cfg.exp_name}-seed{seed}",
        tags=wb_tags,
        config={**asdict(cfg), "seed": seed},
        reinit=True,
    )
    # Add derived schedule info to config
    wandb.config.update({
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "domains": cfg.pretrain_domains,
        "tasks": cfg.active_tasks,
    }, allow_val_change=True)
    # Link out to run URL for downstream manifests
    run_url = getattr(wandb.run, "url", None)

    os.makedirs(cfg.output_dir, exist_ok=True)
    best_val_total = float("inf")
    best_ckpt_path = Path(cfg.output_dir) / f"best_{cfg.exp_name}_seed{seed}.pt"
    manifest_path = Path(cfg.output_dir) / f"manifest_{cfg.exp_name}_seed{seed}.json"
    best_epoch = -1

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()

        # Domain iterators for balanced multi-domain batches
        iterators = {d: iter(l) for d, l in train_loaders.items()}

        for step_in_epoch in range(steps_per_epoch):
            batch = combine_next_batches(iterators)
            if not batch:
                break

            # Raw task losses
            raw_losses: Dict[str, torch.Tensor] = {}
            for name, task in tasks.items():
                if name == "domain_adv":
                    raw_losses[name] = task.compute_loss(batch, lambda_val=grl_sched())
                else:
                    raw_losses[name] = task.compute_loss(batch)

            # Weighted total
            total_loss, weighted_components = weighter(raw_losses, lambda_val=grl_sched())

            opt_model.zero_grad(set_to_none=True)
            opt_uncertainty.zero_grad(set_to_none=True)
            total_loss.backward()
            opt_model.step()
            opt_uncertainty.step()

            # Scheduler step
            grl_sched.step(1)

            # Logging
            if global_step % cfg.log_every_steps == 0:
                log_dict: Dict[str, float] = {
                    "train/total_loss": float(total_loss.detach().cpu()),
                    "train/grl_lambda": grl_sched(),
                    "epoch": float(epoch),
                    "global_step": float(global_step),
                }
                # Raw and weighted
                for k, v in raw_losses.items():
                    log_dict[f"train/raw/{k}"] = float(v.detach().cpu())
                for k, v in weighted_components.items():
                    log_dict[f"train/weighted/{k}"] = float(v.detach().cpu())
                # Learned sigmas
                for t, sigma in weighter.get_task_sigmas().items():
                    log_dict[f"train/sigma/{t}"] = float(sigma)

                wandb.log(log_dict, step=global_step)

            global_step += 1

        # Validation
        if (epoch % cfg.eval_every_epochs) == 0:
            val_total, val_raw, val_weighted, val_domain_totals, val_balanced = run_validation(
                model=model,
                tasks=tasks,
                weighter=weighter,
                val_loaders=val_loaders,
                scheduler=grl_sched,
                device=device,
            )

            log_dict = {
                "val/total_loss": val_total,
                "val/grl_lambda": grl_sched(),
                "val/balanced_total": val_balanced,
                "epoch": float(epoch),
                "global_step": float(global_step),
            }
            for k, v in val_raw.items():
                log_dict[f"val/raw/{k}"] = v
            for k, v in val_weighted.items():
                log_dict[f"val/weighted/{k}"] = v
            # Per-domain totals (balanced metric components)
            for d, v in val_domain_totals.items():
                log_dict[f"val/domain_total/{d}"] = v

            wandb.log(log_dict, step=global_step)

            # Checkpoint best
            if val_total < best_val_total:
                best_val_total = val_total
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "weighter_state_dict": weighter.state_dict(),
                        "cfg": asdict(cfg),
                        "domain_names": cfg.pretrain_domains,
                        "task_names": cfg.active_tasks,
                        "seed": seed,
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_total": best_val_total,
                    },
                    best_ckpt_path,
                )
                # Track with W&B
                wandb.save(str(best_ckpt_path), policy="now")

                # Write/update a manifest for finetuning discovery
                manifest = {
                    "checkpoint_path": str(best_ckpt_path),
                    "exp_name": cfg.exp_name,
                    "seed": seed,
                    "pretrain_domains": cfg.pretrain_domains,
                    "active_tasks": cfg.active_tasks,
                    "best_val_total": float(best_val_total),
                    "best_epoch": int(best_epoch),
                    "steps_per_epoch": int(steps_per_epoch),
                    "total_steps": int(total_steps),
                    "wandb_run_id": getattr(wandb.run, "id", None),
                    "wandb_project": WANDB_PROJECT,
                    "wandb_run_url": run_url,
                }
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                wandb.save(str(manifest_path), policy="now")

    # Summaries for analysis
    wandb.run.summary["best_val_total"] = float(best_val_total)
    wandb.run.summary["best_epoch"] = int(best_epoch)
    wandb.run.summary["best_ckpt_path"] = str(best_ckpt_path)

    run.finish()


def main():
    parser = argparse.ArgumentParser(description="Multi-domain GNN pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config defining the scheme (exp_name, pretrain_domains, active_tasks)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for this single run")

    args = parser.parse_args()
    cfg = build_config(args)

    # Single-run execution with provided seed
    train_single_seed(cfg, args.seed)


if __name__ == "__main__":
    main()

