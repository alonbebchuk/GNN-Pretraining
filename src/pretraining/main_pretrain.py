import argparse
import os
import random
import json
import wandb
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
from numpy.typing import NDArray

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, RandomSampler
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.common import (
    PRETRAIN_EPOCHS,
    PRETRAIN_BATCH_SIZE,
    PRETRAIN_LR_MODEL,
    PRETRAIN_LR_UNCERTAINTY,
    PRETRAIN_MODEL_WEIGHT_DECAY,
    PRETRAIN_EVAL_EVERY_EPOCHS,
    PRETRAIN_LOG_EVERY_STEPS,
    PRETRAIN_NUM_WORKERS,
    PRETRAIN_OUTPUT_DIR,
    WANDB_PROJECT,
    PRETRAIN_PIN_MEMORY,
    PRETRAIN_DROP_LAST,
    PRETRAIN_LR_WARMUP_FRACTION,
    PRETRAIN_LR_MIN_FACTOR,
    PRETRAIN_ADAM_BETAS,
    PRETRAIN_ADAM_EPS,
    PRETRAIN_UNCERTAINTY_WEIGHT_DECAY,
)
from src.model.pretrain_model import PretrainableGNN
from src.pretraining.losses import UncertaintyWeighter
from src.pretraining.schedulers import GRLLambdaScheduler, CosineWithWarmup
from src.pretraining.tasks import (
    DomainAdversarialTask,
    GraphContrastiveTask,
    GraphPropertyPredictionTask,
    LinkPredictionTask,
    NodeContrastiveTask,
    NodeFeatureMaskingTask,
)
from src.data.data_setup import PROCESSED_DIR
from src.common import OVERLAP_TUDATASETS
from src.common import OVERLAP_TUDATASETS


# -----------------------------
# Configuration
# -----------------------------


@dataclass
class TrainConfig:
    exp_name: str
    pretrain_domains: List[str]
    active_tasks: List[str]
    batch_size_per_domain: int

    epochs: int = PRETRAIN_EPOCHS
    batch_size: int = PRETRAIN_BATCH_SIZE
    num_workers: int = PRETRAIN_NUM_WORKERS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every_steps: int = PRETRAIN_LOG_EVERY_STEPS
    eval_every_epochs: int = PRETRAIN_EVAL_EVERY_EPOCHS
    output_dir: str = PRETRAIN_OUTPUT_DIR
    lr_model: float = PRETRAIN_LR_MODEL
    lr_uncertainty: float = PRETRAIN_LR_UNCERTAINTY
    adam_betas: Tuple[float, float] = PRETRAIN_ADAM_BETAS
    adam_eps: float = PRETRAIN_ADAM_EPS
    model_weight_decay: float = PRETRAIN_MODEL_WEIGHT_DECAY
    uncertainty_weight_decay: float = PRETRAIN_UNCERTAINTY_WEIGHT_DECAY


def build_config(args: argparse.Namespace) -> TrainConfig:
    """
    Build TrainConfig by FIRST loading the YAML specified by --config,
    then constructing TrainConfig(**yaml_dict). This avoids instantiating
    TrainConfig() with missing required fields.
    """
    cfg_path = Path(getattr(args, "config", None) or "")
    if not cfg_path.exists():
        raise FileNotFoundError(f"--config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}

    # If the YAML nests config under a single top-level key, unwrap it:
    if isinstance(data, dict) and len(data) == 1 and isinstance(next(iter(data.values())), dict):
        data = next(iter(data.values()))

    return TrainConfig(**data)




class DomainSplitDataset(Dataset):
    def __init__(self, graphs: List[Any], indices: NDArray[np.int_]) -> None:
        self.graphs: List[Any] = graphs
        self.indices: List[int] = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Any:
        return self.graphs[self.indices[idx]]


def make_domain_loader(
    domain_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    num_steps: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[PyGDataLoader, int]:
    """
    Load processed graphs for a domain and return a DataLoader that yields
    batches as List[(Data, domain_name)] to match task expectations.
    Returns the number of batches per epoch (len(loader)).
    """
    dom_dir = PROCESSED_DIR / f"{domain_name}_pretrain"
    data_path = dom_dir / "data.pt"
    splits_path = dom_dir / "splits.pt"

    graphs = torch.load(data_path)
    splits = torch.load(splits_path)
    split_idx = splits[split]

    ds = DomainSplitDataset(graphs, split_idx)

    if split == "train" and num_steps is not None:
        sampler = RandomSampler(ds, replacement=True, num_samples=num_steps * batch_size)
        loader = PyGDataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=PRETRAIN_PIN_MEMORY,
            drop_last=PRETRAIN_DROP_LAST,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
        length = num_steps
    else:
        loader = PyGDataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=PRETRAIN_PIN_MEMORY,
            drop_last=PRETRAIN_DROP_LAST,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
        length = (len(ds) + batch_size - 1) // batch_size

    return loader, length


def build_multi_domain_loaders(
    domains: List[str],
    batch_size_per_domain: int,
    num_workers: int,
    seed: int,
) -> Tuple[Dict[str, PyGDataLoader], Dict[str, PyGDataLoader], int]:
    provisional_lengths: Dict[str, int] = {}
    proto_gen = torch.Generator()
    proto_gen.manual_seed(seed)
    for d in domains:
        _, tr_len = make_domain_loader(d, "train", batch_size_per_domain, num_workers, generator=proto_gen)
        provisional_lengths[d] = tr_len

    steps_per_epoch = max(provisional_lengths.values())

    train_loaders: Dict[str, PyGDataLoader] = {}
    val_loaders: Dict[str, PyGDataLoader] = {}
    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    val_gen = torch.Generator()
    val_gen.manual_seed(seed + 1)
    for d in domains:
        tr_loader, _ = make_domain_loader(d, "train", batch_size_per_domain, num_workers, num_steps=steps_per_epoch, generator=train_gen)
        va_loader, _ = make_domain_loader(d, "val", batch_size_per_domain, num_workers, generator=val_gen)
        train_loaders[d] = tr_loader
        val_loaders[d] = va_loader

    return train_loaders, val_loaders, steps_per_epoch


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------
# Task factory
# -----------------------------


def instantiate_tasks(model: PretrainableGNN, active_tasks: List[str]) -> Dict[str, Any]:
    tasks: Dict[str, Any] = {}
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
        else:
            raise ValueError(f"Unknown task '{name}'.")
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


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return batch.to(device)


def combine_next_batches(iters: Dict[str, Iterable]) -> Dict[str, Batch]:
    batches: Dict[str, Batch] = {}
    for domain_name, it_ in iters.items():
        try:
            part_batch = next(it_)
        except StopIteration:
            return {}
        batches[domain_name] = part_batch
    return batches


def move_batches_to_device(batches_by_domain: Dict[str, Batch], device: torch.device) -> Dict[str, Batch]:
    return {domain: move_batch_to_device(batch, device) for domain, batch in batches_by_domain.items()}


@torch.no_grad()
def run_validation(
    model: PretrainableGNN,
    tasks: Dict[str, object],
    weighter: UncertaintyWeighter,
    val_loaders: Dict[str, PyGDataLoader],
    scheduler: GRLLambdaScheduler,
    device: torch.device,
) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, float], float, Dict[str, Dict[str, float]]]:
    model.eval()

    iters = {d: iter(l) for d, l in val_loaders.items()}
    agg_raw: Dict[str, List[float]] = {}
    agg_weighted: Dict[str, List[float]] = {}
    agg_per_task_per_domain: Dict[str, Dict[str, List[float]]] = {}
    agg_per_domain_totals: Dict[str, List[float]] = {}
    total_losses: List[float] = []

    while True:
        batches_by_domain = combine_next_batches(iters)
        if not batches_by_domain:
            break

        batches_by_domain = move_batches_to_device(batches_by_domain, device)

        raw_losses: Dict[str, torch.Tensor] = {}
        per_task_per_domain_losses: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, task in tasks.items():
            if name == "domain_adv":
                total_loss, _ = task.compute_loss(batches_by_domain, lambda_val=scheduler())
                raw_losses[name] = total_loss
            else:
                total_loss, domain_losses = task.compute_loss(batches_by_domain)
                raw_losses[name] = total_loss
                per_task_per_domain_losses[name] = domain_losses

        raw_losses_no_adv: Dict[str, torch.Tensor] = {k: v for k, v in raw_losses.items() if k != "domain_adv"}

        total_loss, weighted = weighter(raw_losses_no_adv, lambda_val=scheduler())
        total_losses.append(float(total_loss.detach().cpu()))

        for k, v in raw_losses.items():
            agg_raw.setdefault(k, []).append(float(v.detach().cpu()))
        for k, v in weighted.items():
            agg_weighted.setdefault(k, []).append(float(v.detach().cpu()))

        for task_name, domain_losses in per_task_per_domain_losses.items():
            if task_name not in agg_per_task_per_domain:
                agg_per_task_per_domain[task_name] = {}
            for domain_name, loss in domain_losses.items():
                agg_per_task_per_domain[task_name].setdefault(domain_name, []).append(float(loss.detach().cpu()))

        for domain_name in batches_by_domain.keys():
            domain_raw_losses = {task: per_task_per_domain_losses[task][domain_name] for task in per_task_per_domain_losses.keys()}
            domain_total, _ = weighter(domain_raw_losses, lambda_val=scheduler())
            agg_per_domain_totals.setdefault(domain_name, []).append(float(domain_total.detach().cpu()))

    total = float(np.mean(total_losses))
    raw_means = {k: float(np.mean(v)) for k, v in agg_raw.items()}
    weighted_means = {k: float(np.mean(v)) for k, v in agg_weighted.items()}

    per_task_per_domain_means: Dict[str, Dict[str, float]] = {}
    for task_name, domain_dict in agg_per_task_per_domain.items():
        per_task_per_domain_means[task_name] = {}
        for domain_name, loss_list in domain_dict.items():
            per_task_per_domain_means[task_name][domain_name] = float(np.mean(loss_list))

    per_domain_total: Dict[str, float] = {}
    domain_totals_for_balanced: List[float] = []
    
    for domain_name, totals_list in agg_per_domain_totals.items():
        domain_avg = float(np.mean(totals_list))
        per_domain_total[domain_name] = domain_avg
        domain_totals_for_balanced.append(domain_avg)

    balanced_total = float(np.mean(domain_totals_for_balanced))

    return total, raw_means, weighted_means, per_domain_total, balanced_total, per_task_per_domain_means


def train_single_seed(cfg: TrainConfig, seed: int) -> None:
    set_global_seed(seed)

    device = torch.device(cfg.device)

    train_loaders, val_loaders, steps_per_epoch = build_multi_domain_loaders(
        domains=cfg.pretrain_domains,
        batch_size_per_domain=cfg.batch_size_per_domain,
        num_workers=cfg.num_workers,
        seed=seed,
    )

    total_steps = cfg.epochs * steps_per_epoch

    model = PretrainableGNN(device=device, domain_names=cfg.pretrain_domains, task_names=cfg.active_tasks)
    tasks = instantiate_tasks(model, cfg.active_tasks)

    weighter = UncertaintyWeighter(task_names=cfg.active_tasks).to(device)
    opt_model = AdamW(
        model.parameters(),
        lr=cfg.lr_model,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
        weight_decay=cfg.model_weight_decay,
    )
    opt_uncertainty = AdamW(
        weighter.parameters(),
        lr=cfg.lr_uncertainty,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
        weight_decay=cfg.uncertainty_weight_decay,
    )

    grl_sched = GRLLambdaScheduler(total_steps=total_steps)
    lr_multiplier = CosineWithWarmup(
        total_steps=total_steps,
        warmup_fraction=PRETRAIN_LR_WARMUP_FRACTION,
        lr_min_factor=PRETRAIN_LR_MIN_FACTOR,
    )

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
    wandb.config.update({
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "domains": cfg.pretrain_domains,
        "tasks": cfg.active_tasks,
    }, allow_val_change=True)
    run_url = getattr(wandb.run, "url", None)

    os.makedirs(cfg.output_dir, exist_ok=True)
    best_val_total = float("inf")
    best_ckpt_path = Path(cfg.output_dir) / f"best_{cfg.exp_name}_seed{seed}.pt"
    manifest_path = Path(cfg.output_dir) / f"manifest_{cfg.exp_name}_seed{seed}.json"
    best_epoch = -1

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()

        iterators = {d: iter(l) for d, l in train_loaders.items()}

        for _ in range(steps_per_epoch):
            batches_by_domain = combine_next_batches(iterators)
            if not batches_by_domain:
                break

            batches_by_domain = move_batches_to_device(batches_by_domain, device)

            raw_losses: Dict[str, torch.Tensor] = {}
            per_task_per_domain_losses: Dict[str, Dict[str, torch.Tensor]] = {}
            for name, task in tasks.items():
                if name == "domain_adv":
                    total_loss, _ = task.compute_loss(batches_by_domain, lambda_val=grl_sched())
                    raw_losses[name] = total_loss
                else:
                    total_loss, domain_losses = task.compute_loss(batches_by_domain)
                    raw_losses[name] = total_loss
                    per_task_per_domain_losses[name] = domain_losses

            total_loss, weighted_components = weighter(raw_losses, lambda_val=grl_sched())

            opt_model.zero_grad(set_to_none=True)
            opt_uncertainty.zero_grad(set_to_none=True)
            total_loss.backward()
            opt_model.step()
            opt_uncertainty.step()

            scale = lr_multiplier()
            for pg in opt_model.param_groups:
                pg["lr"] = cfg.lr_model * scale
            for pg in opt_uncertainty.param_groups:
                pg["lr"] = cfg.lr_uncertainty * scale

            grl_sched.step(1)
            lr_multiplier.step(1)

            if global_step % cfg.log_every_steps == 0:
                log_dict: Dict[str, float] = {
                    "train/total_loss": float(total_loss.detach().cpu()),
                    "train/grl_lambda": grl_sched(),
                    "epoch": float(epoch),
                    "global_step": float(global_step),
                }

                for k, v in raw_losses.items():
                    log_dict[f"train/raw/{k}"] = float(v.detach().cpu())
                for k, v in weighted_components.items():
                    log_dict[f"train/weighted/{k}"] = float(v.detach().cpu())

                for task_name, domain_losses in per_task_per_domain_losses.items():
                    for domain_name, loss in domain_losses.items():
                        log_dict[f"train/domain/{task_name}/{domain_name}"] = float(loss.detach().cpu())

                for t, sigma in weighter.get_task_sigmas().items():
                    log_dict[f"train/sigma/{t}"] = float(sigma)

                log_dict["train/lr_scale"] = float(scale)
                log_dict["train/lr_model"] = float(cfg.lr_model * scale)
                log_dict["train/lr_uncertainty"] = float(cfg.lr_uncertainty * scale)

                wandb.log(log_dict, step=global_step)

            global_step += 1

        if (epoch % cfg.eval_every_epochs) == 0:
            val_total, val_raw, val_weighted, val_domain_totals, val_balanced, val_per_task_per_domain = run_validation(
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

            for d, v in val_domain_totals.items():
                log_dict[f"val/domain_total/{d}"] = v

            for task_name, domain_losses in val_per_task_per_domain.items():
                for domain_name, loss in domain_losses.items():
                    log_dict[f"val/domain/{task_name}/{domain_name}"] = loss

            ood_domains = [d for d in val_domain_totals.keys() if d not in OVERLAP_TUDATASETS]
            if len(ood_domains) > 0:
                selection_total = float(np.mean([val_domain_totals[d] for d in ood_domains]))
            else:
                selection_total = float(val_balanced)

            log_dict["val/selection_total"] = selection_total
            wandb.log(log_dict, step=global_step)

            if epoch == 1 or selection_total < best_val_total:
                best_val_total = selection_total
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
                wandb.save(str(best_ckpt_path), policy="now")

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

    wandb.run.summary["best_val_total"] = float(best_val_total)
    wandb.run.summary["best_epoch"] = int(best_epoch)
    wandb.run.summary["best_ckpt_path"] = str(best_ckpt_path)

    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-domain GNN pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config defining the scheme (exp_name, pretrain_domains, active_tasks)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for this single run")

    args = parser.parse_args()
    cfg = build_config(args)

    train_single_seed(cfg, args.seed)


if __name__ == "__main__":
    main()
