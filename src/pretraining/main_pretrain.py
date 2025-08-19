import argparse
import logging
import os
import random
import json
import time
import wandb
import yaml
from dataclasses import asdict, dataclass
from datetime import datetime
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
    PATIENCE,
    # Model architecture constants
    GNN_HIDDEN_DIM,
    GNN_NUM_LAYERS,
    DROPOUT_RATE,
    CONTRASTIVE_PROJ_DIM,
    GRAPH_PROP_HEAD_HIDDEN_DIM,
    DOMAIN_ADV_HEAD_HIDDEN_DIM,
    DOMAIN_ADV_HEAD_OUT_DIM,
    MASK_TOKEN_INIT_MEAN,
    MASK_TOKEN_INIT_STD,
    # Task constants
    NODE_FEATURE_MASKING_MASK_RATE,
    NODE_CONTRASTIVE_TEMPERATURE,
    GRAPH_PROPERTY_DIM,
    # Augmentation constants
    AUGMENTATION_ATTR_MASK_PROB,
    AUGMENTATION_ATTR_MASK_RATE,
    AUGMENTATION_EDGE_DROP_PROB,
    AUGMENTATION_EDGE_DROP_RATE,
    AUGMENTATION_NODE_DROP_PROB,
    AUGMENTATION_NODE_DROP_RATE,
    AUGMENTATION_MIN_NODES_PER_GRAPH,
    # Loss constants
    UNCERTAINTY_LOSS_COEF,
    LOGSIGMA_TO_SIGMA_SCALE,
    GRL_GAMMA,
    GRL_LAMBDA_MIN,
    GRL_LAMBDA_MAX,
    # Domain information
    DOMAIN_DIMENSIONS,
    FEATURE_TYPES,
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
from src.pretraining.model_registry import register_model_completion


# -----------------------------
# Configuration
# -----------------------------


@dataclass
class TrainConfig:
    exp_name: str
    pretrain_domains: List[str]
    active_tasks: List[str]
    batch_size_per_domain: Optional[int] = None

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

    def __post_init__(self):
        """Automatically calculate batch_size_per_domain if not provided."""
        if self.batch_size_per_domain is None:
            num_domains = len(self.pretrain_domains)
            self.batch_size_per_domain = self.batch_size // num_domains


def build_config(args: argparse.Namespace) -> TrainConfig:
    """
    Build TrainConfig by loading the YAML configuration file.
    batch_size_per_domain will be automatically calculated if not provided.
    """
    cfg_path = Path(getattr(args, "config", None) or "")
    if not cfg_path.exists():
        raise FileNotFoundError(f"--config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

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
    
    # Load graph properties separately if they exist (workaround for PyG bug)
    props_path = dom_dir / "graph_properties.pt"
    if props_path.exists():
        graph_properties_tensor = torch.load(props_path)
        # Attach properties to each graph at runtime
        for i, graph in enumerate(graphs):
            graph.graph_properties = graph_properties_tensor[i]
        logging.info(f"Loaded and attached graph_properties to {len(graphs)} graphs for domain {domain_name}")

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
# Experiment Classification Helpers
# -----------------------------

def classify_scheme(exp_name: str, active_tasks: List[str]) -> str:
    """Classify experiment scheme type for analysis."""
    if exp_name.startswith("B"):
        return "baseline"
    elif len(active_tasks) == 1:
        return "single_task"
    else:
        return "multi_task"

def get_paradigm(active_tasks: List[str]) -> str:
    """Determine the paradigm based on active tasks."""
    generative_tasks = {"node_feat_mask", "link_pred"}
    contrastive_tasks = {"node_contrast", "graph_contrast"}
    
    has_generative = any(task in generative_tasks for task in active_tasks)
    has_contrastive = any(task in contrastive_tasks for task in active_tasks)
    
    if has_generative and has_contrastive:
        return "mixed"
    elif has_generative:
        return "generative"
    elif has_contrastive:
        return "contrastive"
    else:
        return "other"

def has_auxiliary_tasks(active_tasks: List[str]) -> bool:
    """Check if auxiliary supervised tasks are present."""
    return "graph_prop" in active_tasks

def has_adversarial_tasks(active_tasks: List[str]) -> bool:
    """Check if adversarial tasks are present."""
    return "domain_adv" in active_tasks

def get_comparison_groups(exp_name: str) -> List[str]:
    """Get comparison groups this experiment belongs to for analysis."""
    groups = []
    
    # Baseline vs Pre-trained comparison
    if exp_name == "B1":
        groups.append("from_scratch_baseline")
    else:
        groups.append("pretrained")
    
    # Scheme type groups
    if exp_name.startswith("B"):
        groups.append("baseline_schemes")
        if exp_name in ["B2", "B3"]:
            groups.append("single_task_baselines")
        elif exp_name == "B4":
            groups.append("single_domain_baseline")
    elif exp_name.startswith("S"):
        groups.append("multi_task_schemes")
        
        # Specific scheme groups for key comparisons
        if exp_name in ["S1", "S2"]:
            groups.append("paradigm_comparison")  # S1 vs S2
        if exp_name in ["S3", "S4", "S5"]:
            groups.append("complexity_progression")  # S3 -> S4 -> S5
        if exp_name in ["S4", "B4"]:
            groups.append("cross_domain_comparison")  # Multi vs single domain
    
    return groups



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

    # Classify experiment for analysis
    scheme_type = classify_scheme(cfg.exp_name, cfg.active_tasks)
    paradigm = get_paradigm(cfg.active_tasks)
    comparison_groups = get_comparison_groups(cfg.exp_name)
    
    wb_tags = [
        "phase:pretrain",
        *(f"domain:{d}" for d in cfg.pretrain_domains),
        *(f"task:{t}" for t in cfg.active_tasks),
        # Analysis classification tags
        f"scheme_type:{scheme_type}",
        f"paradigm:{paradigm}",
        f"has_auxiliary:{has_auxiliary_tasks(cfg.active_tasks)}",
        f"has_adversarial:{has_adversarial_tasks(cfg.active_tasks)}",
        *(f"group:{group}" for group in comparison_groups),
    ]

    # Comprehensive config for WandB - include ALL hyperparameters for reproducibility
    full_config = {
        **asdict(cfg),
        "seed": seed,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "domains": cfg.pretrain_domains,
        "tasks": cfg.active_tasks,
        
        # Model architecture hyperparameters
        "model": {
            "gnn_hidden_dim": GNN_HIDDEN_DIM,
            "gnn_num_layers": GNN_NUM_LAYERS,
            "dropout_rate": DROPOUT_RATE,
            "contrastive_proj_dim": CONTRASTIVE_PROJ_DIM,
            "graph_prop_head_hidden_dim": GRAPH_PROP_HEAD_HIDDEN_DIM,
            "domain_adv_head_hidden_dim": DOMAIN_ADV_HEAD_HIDDEN_DIM,
            "domain_adv_head_out_dim": DOMAIN_ADV_HEAD_OUT_DIM,
            "mask_token_init_mean": MASK_TOKEN_INIT_MEAN,
            "mask_token_init_std": MASK_TOKEN_INIT_STD,
        },
        
        # Task-specific hyperparameters
        "tasks_config": {
            "node_feature_masking_mask_rate": NODE_FEATURE_MASKING_MASK_RATE,
            "node_contrastive_temperature": NODE_CONTRASTIVE_TEMPERATURE,
            "graph_property_dim": GRAPH_PROPERTY_DIM,
        },
        
        # Augmentation hyperparameters
        "augmentations": {
            "attr_mask_prob": AUGMENTATION_ATTR_MASK_PROB,
            "attr_mask_rate": AUGMENTATION_ATTR_MASK_RATE,
            "edge_drop_prob": AUGMENTATION_EDGE_DROP_PROB,
            "edge_drop_rate": AUGMENTATION_EDGE_DROP_RATE,
            "node_drop_prob": AUGMENTATION_NODE_DROP_PROB,
            "node_drop_rate": AUGMENTATION_NODE_DROP_RATE,
            "min_nodes_per_graph": AUGMENTATION_MIN_NODES_PER_GRAPH,
        },
        
        # Loss weighting and scheduling
        "loss_config": {
            "uncertainty_loss_coef": UNCERTAINTY_LOSS_COEF,
            "logsigma_to_sigma_scale": LOGSIGMA_TO_SIGMA_SCALE,
            "grl_gamma": GRL_GAMMA,
            "grl_lambda_min": GRL_LAMBDA_MIN,
            "grl_lambda_max": GRL_LAMBDA_MAX,
            "lr_warmup_fraction": PRETRAIN_LR_WARMUP_FRACTION,
            "lr_min_factor": PRETRAIN_LR_MIN_FACTOR,
        },
        
        # Domain information
        "domain_info": {
            "domain_dimensions": {d: DOMAIN_DIMENSIONS[d] for d in cfg.pretrain_domains},
            "feature_types": {d: FEATURE_TYPES[d] for d in cfg.pretrain_domains},
            "overlap_tudatasets": OVERLAP_TUDATASETS,
        },
        
        # Experiment classification for analysis
        "experiment_classification": {
            "scheme_id": cfg.exp_name,  # B1, B2, S1, etc.
            "scheme_type": scheme_type,
            "paradigm": paradigm,
            "task_count": len(cfg.active_tasks),
            "domain_count": len(cfg.pretrain_domains),
            "is_baseline": cfg.exp_name.startswith("B"),
            "is_single_task": len(cfg.active_tasks) == 1,
            "is_multi_domain": len(cfg.pretrain_domains) > 1,
            "has_auxiliary": has_auxiliary_tasks(cfg.active_tasks),
            "has_adversarial": has_adversarial_tasks(cfg.active_tasks),
            "comparison_groups": comparison_groups,
        },
        
        # System information
        "system": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        }
    }
    
    run = wandb.init(
        project=WANDB_PROJECT,
        group=cfg.exp_name,
        name=f"{cfg.exp_name}-seed{seed}",
        tags=wb_tags,
        config=full_config,
        reinit=True,
    )
    run_url = getattr(wandb.run, "url", None)

    os.makedirs(cfg.output_dir, exist_ok=True)
    best_val_total = float("inf")
    best_ckpt_path = Path(cfg.output_dir) / f"best_{cfg.exp_name}_seed{seed}.pt"
    manifest_path = Path(cfg.output_dir) / f"manifest_{cfg.exp_name}_seed{seed}.json"
    best_epoch = -1
    
    # Early stopping configuration
    patience = PATIENCE  # Stop if no improvement for PATIENCE epochs
    epochs_without_improvement = 0
    
    # Timing tracking
    training_start_time = time.time()
    epoch_start_times = {}
    step_times = []

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        epoch_start_time = time.time()
        epoch_start_times[epoch] = epoch_start_time
        model.train()

        iterators = {d: iter(l) for d, l in train_loaders.items()}

        for _ in range(steps_per_epoch):
            step_start_time = time.time()
            
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
            
            # Track step timing
            step_duration = time.time() - step_start_time
            step_times.append(step_duration)

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
                
                # Add domain loss balance monitoring
                domain_losses_list = []
                for task_name, domain_losses in per_task_per_domain_losses.items():
                    for domain_name, loss in domain_losses.items():
                        domain_losses_list.append(float(loss.detach().cpu()))
                
                if domain_losses_list:
                    log_dict["train/domain_balance/mean"] = float(np.mean(domain_losses_list))
                    log_dict["train/domain_balance/std"] = float(np.std(domain_losses_list))
                    log_dict["train/domain_balance/cv"] = float(np.std(domain_losses_list) / (np.mean(domain_losses_list) + 1e-8))
                
                # Add timing metrics
                current_time = time.time()
                log_dict["timing/cumulative_training_time_hours"] = (current_time - training_start_time) / 3600
                if len(step_times) >= 10:  # Only log after some steps for stability
                    recent_step_times = step_times[-10:]
                    log_dict["timing/avg_step_time_seconds"] = float(np.mean(recent_step_times))
                    log_dict["timing/steps_per_second"] = 1.0 / float(np.mean(recent_step_times))

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
                
                # Add scheduler states for debugging
                "schedulers/grl_lambda": grl_sched(),
                "schedulers/lr_scale": lr_multiplier(),
                "schedulers/current_lr_model": cfg.lr_model * lr_multiplier(),
                "schedulers/current_lr_uncertainty": cfg.lr_uncertainty * lr_multiplier(),
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
                    
            # Log uncertainty weights for each task during validation
            for t, sigma in weighter.get_task_sigmas().items():
                log_dict[f"val/sigma/{t}"] = float(sigma)
            
            # Add epoch timing metrics
            if epoch > 1:
                epoch_duration = time.time() - epoch_start_times[epoch]
                log_dict["timing/epoch_duration_seconds"] = epoch_duration
                log_dict["timing/epoch_duration_minutes"] = epoch_duration / 60
                
                # Efficiency metrics
                cumulative_time = time.time() - training_start_time
                log_dict["timing/cumulative_training_time_seconds"] = cumulative_time
                log_dict["timing/average_epoch_time_seconds"] = cumulative_time / epoch
                
                # Convergence efficiency
                if best_epoch > 0:
                    log_dict["efficiency/epochs_to_best"] = best_epoch
                    log_dict["efficiency/time_to_best_hours"] = (epoch_start_times.get(best_epoch, training_start_time) - training_start_time) / 3600

            ood_domains = [d for d in val_domain_totals.keys() if d not in OVERLAP_TUDATASETS]
            if len(ood_domains) > 0:
                selection_total = float(np.mean([val_domain_totals[d] for d in ood_domains]))
                selection_method = "ood_average"
            else:
                selection_total = float(val_balanced)
                selection_method = "balanced_average"

            log_dict["val/selection_total"] = selection_total
            
            wandb.log(log_dict, step=global_step)

            if epoch == 1 or selection_total < best_val_total:
                best_val_total = selection_total
                best_epoch = epoch
                epochs_without_improvement = 0  # Reset counter
                
                # Log improvement for monitoring
                wandb.log({
                    "checkpoints/new_best_epoch": epoch,
                    "checkpoints/new_best_val_total": selection_total,
                    "checkpoints/improvement": (wandb.run.summary.get("best_val_total", float("inf")) - selection_total) if epoch > 1 else 0,
                }, step=global_step)
                
                # Save comprehensive checkpoint with all states needed for resuming/fine-tuning
                checkpoint_dict = {
                    # Model states
                    "model_state_dict": model.state_dict(),
                    "weighter_state_dict": weighter.state_dict(),
                    
                    # Optimizer states (crucial for resuming training)
                    "opt_model_state_dict": opt_model.state_dict(),
                    "opt_uncertainty_state_dict": opt_uncertainty.state_dict(),
                    
                    # Scheduler states
                    "grl_scheduler_state": grl_sched.state_dict(),
                    "lr_scheduler_state": lr_multiplier.state_dict(),
                    
                    # Training state
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_total": best_val_total,
                    "best_epoch": best_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                    
                    # Configuration and metadata
                    "cfg": asdict(cfg),
                    "domain_names": cfg.pretrain_domains,
                    "task_names": cfg.active_tasks,
                    "seed": seed,
                    
                    # Model architecture info (for loading compatibility)
                    "model_config": {
                        "gnn_hidden_dim": GNN_HIDDEN_DIM,
                        "gnn_num_layers": GNN_NUM_LAYERS,
                        "dropout_rate": DROPOUT_RATE,
                        "domain_dimensions": {d: DOMAIN_DIMENSIONS[d] for d in cfg.pretrain_domains},
                        "contrastive_proj_dim": CONTRASTIVE_PROJ_DIM,
                        "graph_prop_head_hidden_dim": GRAPH_PROP_HEAD_HIDDEN_DIM,
                        "domain_adv_head_hidden_dim": DOMAIN_ADV_HEAD_HIDDEN_DIM,
                    },
                    
                    # System info for debugging
                    "torch_version": torch.__version__,
                    "device": str(device),
                    "timestamp": datetime.now().isoformat(),
                }
                
                torch.save(checkpoint_dict, best_ckpt_path)

                wandb.save(str(best_ckpt_path), policy="now")

                # Create comprehensive model artifact with all necessary metadata
                artifact = wandb.Artifact(
                    name=f"model-{cfg.exp_name}-seed{seed}",
                    type="model",
                    description=f"Best model checkpoint for {cfg.exp_name} seed {seed} at epoch {epoch} (val_loss: {best_val_total:.4f})",
                    metadata={
                        # Training state
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_total": float(best_val_total),
                        "best_epoch": int(best_epoch),
                        
                        # Experiment configuration
                        "exp_name": cfg.exp_name,
                        "seed": seed,
                        "domains": cfg.pretrain_domains,
                        "tasks": cfg.active_tasks,
                        
                        # Model architecture
                        "model_config": {
                            "gnn_hidden_dim": GNN_HIDDEN_DIM,
                            "gnn_num_layers": GNN_NUM_LAYERS,
                            "dropout_rate": DROPOUT_RATE,
                            "domain_dimensions": {d: DOMAIN_DIMENSIONS[d] for d in cfg.pretrain_domains},
                        },
                        
                        # Training hyperparameters (crucial for fine-tuning)
                        "training_config": {
                            "lr_model": cfg.lr_model,
                            "lr_uncertainty": cfg.lr_uncertainty,
                            "batch_size": cfg.batch_size,
                            "batch_size_per_domain": cfg.batch_size_per_domain,
                            "model_weight_decay": cfg.model_weight_decay,
                            "uncertainty_weight_decay": cfg.uncertainty_weight_decay,
                            "adam_betas": cfg.adam_betas,
                            "adam_eps": cfg.adam_eps,
                        },
                        
                        # For reproducibility
                        "system_info": {
                            "torch_version": torch.__version__,
                            "device": str(device),
                            "cuda_available": torch.cuda.is_available(),
                        },
                        
                        # Analysis metadata
                        "analysis_metadata": {
                            "scheme_classification": full_config["experiment_classification"],
                            "comparison_groups": get_comparison_groups(cfg.exp_name),
                            "expected_downstream_tasks": ["graph_classification", "node_classification", "link_prediction"],
                            "validation_selection_method": selection_method,
                            "validation_ood_domains": ood_domains,
                            "validation_overlap_domains": [d for d in val_domain_totals.keys() if d in OVERLAP_TUDATASETS],
                        },
                        
                        # File information
                        "checkpoint_filename": best_ckpt_path.name,
                        "total_steps": total_steps,
                        "steps_per_epoch": steps_per_epoch,
                    }
                )
                
                # Create manifest first before adding to artifacts
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

                # Add both checkpoint and manifest to artifact
                artifact.add_file(str(best_ckpt_path))
                artifact.add_file(str(manifest_path))  # Include manifest in artifact
                wandb.log_artifact(artifact)
                
                # Also log as model artifact for Model Registry
                model_artifact = wandb.Artifact(
                    name=f"trained-model-{cfg.exp_name}",
                    type="model",
                    description=f"Production-ready model from {cfg.exp_name} experiment"
                )
                model_artifact.add_file(str(best_ckpt_path))
                model_artifact.add_file(str(manifest_path))
                wandb.log_artifact(model_artifact, aliases=["latest", f"seed-{seed}", f"epoch-{epoch}"])
                
                # Register model in central registry
                register_model_completion(
                    exp_name=cfg.exp_name,
                    seed=seed,
                    checkpoint_path=str(best_ckpt_path),
                    manifest_path=str(manifest_path),
                    best_val_total=float(best_val_total),
                    best_epoch=int(best_epoch),
                    wandb_run_id=getattr(wandb.run, "id", None),
                    wandb_artifact_name=f"model-{cfg.exp_name}-seed{seed}",
                    domains=cfg.pretrain_domains,
                    tasks=cfg.active_tasks
                )
            else:
                epochs_without_improvement += 1
                wandb.log({
                    "early_stopping/epochs_without_improvement": epochs_without_improvement,
                    "early_stopping/patience_remaining": patience - epochs_without_improvement,
                }, step=global_step)
                
                # Early stopping check
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    wandb.log({
                        "early_stopping/triggered": True,
                        "early_stopping/stopped_at_epoch": epoch,
                        "early_stopping/final_epochs_without_improvement": epochs_without_improvement,
                    }, step=global_step)
                    break

    # Calculate final timing metrics
    total_training_time = time.time() - training_start_time
    
    # Comprehensive run summary for easy filtering and analysis
    wandb.run.summary.update({
        # Best performance metrics
        "best_val_total": float(best_val_total),
        "best_epoch": int(best_epoch),
        "best_ckpt_path": str(best_ckpt_path),
        
        # Training completion status
        "training_completed": True,
        "total_epochs_trained": epoch,  # Actual epochs trained (may be less due to early stopping)
        "planned_epochs": cfg.epochs,
        "early_stopped": epochs_without_improvement >= patience,
        "epochs_without_improvement": epochs_without_improvement,
        "total_steps_trained": global_step,
        
        # Experiment metadata
        "exp_name": cfg.exp_name,
        "seed": seed,
        "domains_count": len(cfg.pretrain_domains),
        "tasks_count": len(cfg.active_tasks),
        "domains_list": ",".join(cfg.pretrain_domains),
        "tasks_list": ",".join(cfg.active_tasks),
        
        # Model size information
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        
        # Timing and efficiency metrics
        "timing/total_training_time_seconds": total_training_time,
        "timing/total_training_time_hours": total_training_time / 3600,
        "timing/average_epoch_time_seconds": total_training_time / epoch,
        "timing/convergence_epoch": best_epoch,
        "timing/time_to_convergence_hours": (epoch_start_times.get(best_epoch, training_start_time) - training_start_time) / 3600 if best_epoch > 0 else 0,
        "efficiency/training_efficiency": best_epoch / epoch if epoch > 0 else 0,  # Fraction of epochs needed
        "efficiency/steps_per_second_avg": len(step_times) / total_training_time if step_times else 0,
        
        # System information
        "device_used": str(device),
        "torch_version": torch.__version__,
    })

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
