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
from torch_geometric.utils import negative_sampling

from src.data.finetune_data_loaders import create_finetune_data_loader
from src.finetune.artifacts import get_pretrained_model_path
from src.finetune.metrics import aggregate_metrics, compute_evaluation_metrics, compute_training_metrics
from src.models.finetune_model import FinetuneGNN, create_finetune_model
from src.pretrain.schedulers import CosineWithWarmup

from src.data.data_setup import NUM_CLASSES, TASK_TYPES
import torch.nn.functional as F

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "finetune"
WARMUP_FRACTION = 0.15

LR_BACKBONES = {
    'full_finetune': 1e-4,
    'linear_probe': 0.0,
}
LR_HEAD = 1e-3
BATCH_SIZES = {
    'ENZYMES': 32,
    'PTC_MR': 32,
    'Cora_NC': 512,
    'CiteSeer_NC': 512,
    'Cora_LP': 256,
    'CiteSeer_LP': 256,
}
EPOCHS = {
    'ENZYMES': 100,
    'PTC_MR': 100,
    'Cora_NC': 200,
    'CiteSeer_NC': 200,
    'Cora_LP': 300,
    'CiteSeer_LP': 300,
}
PATIENCE_FRACTION = 0.1

@dataclass
class FinetuneConfig:
    domain_name: str
    pretrained_scheme: str
    finetune_strategy: str
    seed: int

    exp_name: str
    task_type: str
    batch_size: int
    epochs: int
    lr_backbone: float
    lr_head: float
    patience: int

    def __post_init__(self):
        self.exp_name = f"{self.domain_name}_{self.finetune_strategy}_{self.pretrained_scheme}"
        self.task_type = TASK_TYPES[self.domain_name]
        self.lr_backbone = LR_BACKBONES[self.finetune_strategy]
        self.lr_head = LR_HEAD
        self.batch_size = BATCH_SIZES[self.domain_name]
        self.epochs = EPOCHS[self.domain_name]
        self.patience = int(self.epochs * PATIENCE_FRACTION)


def build_config(args: argparse.Namespace) -> FinetuneConfig:
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return FinetuneConfig(**data)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_finetune_train_loader(domain_name: str, seed: int, batch_size: int, epoch: int) -> torch.utils.data.DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return create_finetune_data_loader(domain_name, 'train', batch_size, generator)


def get_finetune_evaluation_loaders(domain_name: str, seed: int, batch_size: int) -> Dict[str, torch.utils.data.DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    evaluation_loaders = {
        'val': create_finetune_data_loader(domain_name, 'val', batch_size, generator),
        'test': create_finetune_data_loader(domain_name, 'test', batch_size, generator)
    }

    return evaluation_loaders


def compute_loss_and_metrics(model: torch.nn.Module, batch, device: torch.device, task_type: str, domain_name: str) -> Dict[str, float]:
    model.eval()

    with torch.no_grad():
        if task_type == 'graph_classification':
            batch = batch.to(device)
            logits = model(batch)
            targets = batch.y

            loss = F.cross_entropy(logits, targets)

            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        elif task_type == 'node_classification':
            data, node_indices, targets = batch
            data, node_indices, targets = data.to(device), node_indices.to(device), targets.to(device)

            full_logits = model(data)
            logits = full_logits[node_indices]

            loss = F.cross_entropy(logits, targets)

            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        elif task_type == 'link_prediction':
            data, edges, targets = batch
            data, edges, targets = data.to(device), edges.to(device), targets.to(device)

            edge_probs = model(data, edge_index=edges)

            loss = F.binary_cross_entropy(edge_probs, targets)

            predictions = (edge_probs > 0.5).float()

            probabilities = torch.stack([1 - edge_probs, edge_probs], dim=1)
            targets = targets.long()
            predictions = predictions.long()

        num_classes = NUM_CLASSES[domain_name]
        task_classification_type = 'binary' if num_classes == 2 else 'multiclass'
        metrics = compute_evaluation_metrics(targets, predictions, probabilities, task_classification_type)

        metrics['loss'] = float(loss.item())
        metrics['num_samples'] = len(targets)

    return metrics


@torch.no_grad()
def run_evaluation(
    model: FinetuneGNN,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    best_selection_metric: float,
    epochs_since_improvement: int,
    cfg: FinetuneConfig,
    seed: int,
    global_step: List[int]
) -> Tuple[bool, Dict[str, float], float, int]:
    model.eval()

    # Compute validation metrics
    batch_metrics = []
    for batch_or_data in val_loader:
        metrics = compute_loss_and_metrics(
            model, batch_or_data, device, cfg.task_type, cfg.domain_name)
        batch_metrics.append(metrics)

    val_metrics = aggregate_metrics(batch_metrics, 'val')

    # Determine selection metric
    selection_metric_name = 'val/auc' if cfg.task_type == 'link_prediction' else 'val/accuracy'
    current_val_metric = val_metrics[selection_metric_name]
    is_best = current_val_metric > best_selection_metric

    if is_best:
        new_best_metric = current_val_metric
        new_epochs_since_improvement = 0

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_metrics': val_metrics,
        }
        best_checkpoint_path = OUTPUT_DIR / \
            f"best_model_{cfg.exp_name}_{seed}.pt"
        torch.save(checkpoint, best_checkpoint_path)

        artifact = wandb.Artifact(f"model_{cfg.exp_name}_{seed}", type="model")
        artifact.add_file(str(best_checkpoint_path))
        wandb.log_artifact(artifact)
    else:
        new_best_metric = best_selection_metric
        new_epochs_since_improvement = epochs_since_improvement + 1

    # Log epoch metrics
    epoch_metrics = {
        **val_metrics,
        'train/epochs_since_improvement': new_epochs_since_improvement,
        selection_metric_name: current_val_metric,
    }
    wandb.log(epoch_metrics, step=global_step[0])

    return is_best, val_metrics, new_best_metric, new_epochs_since_improvement


def run_training(
    model: FinetuneGNN,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    lr_multiplier: CosineWithWarmup,
    device: torch.device,
    epoch: int,
    global_step_ref: List[int],
    cfg: FinetuneConfig
) -> None:
    model.train()

    for batch_or_data in train_loader:
        step_start_time = time.time()
        global_step_ref[0] += 1

        if cfg.task_type == 'graph_classification':
            batch = batch_or_data.to(device)
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(logits, batch.y)
        elif cfg.task_type == 'node_classification':
            data, node_indices, targets = batch_or_data
            data, node_indices, targets = data.to(
                device), node_indices.to(device), targets.to(device)
            full_logits = model(data)
            loss = torch.nn.functional.cross_entropy(
                full_logits[node_indices], targets)
        elif cfg.task_type == 'link_prediction':
            data, pos_edges, _ = batch_or_data
            data, pos_edges = data.to(device), pos_edges.to(device)
            neg_edges = negative_sampling(
                data.edge_index, data.num_nodes, pos_edges.size(1)).to(device)
            all_edges = torch.cat([pos_edges, neg_edges], dim=1)
            edge_labels = torch.cat([torch.ones(pos_edges.size(
                1), device=device), torch.zeros(neg_edges.size(1), device=device)])
            edge_probs = model(data, edge_index=all_edges)
            loss = torch.nn.functional.binary_cross_entropy(
                edge_probs, edge_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scale = lr_multiplier()
        for pg in optimizer.param_groups:
            if pg['name'] == 'backbone':
                pg['lr'] = cfg.lr_backbone * scale
            else:
                pg['lr'] = cfg.lr_head * scale
        lr_multiplier.step()

        # Log core training metrics every iteration
        current_lr_backbone = optimizer.param_groups[0]['lr'] if len(
            optimizer.param_groups) > 1 else 0.0
        current_lr_head = optimizer.param_groups[-1]['lr']
        train_metrics = compute_training_metrics(
            epoch, global_step_ref[0], loss.item(), current_lr_backbone, current_lr_head, model)

        # Add system metrics less frequently to reduce overhead
        if global_step_ref[0] % 10 == 0:
            step_time_ms = (time.time() - step_start_time) * 1000
            train_metrics['system/avg_step_time_ms'] = step_time_ms

            if device.type == "cuda":
                train_metrics['system/gpu_memory_gb'] = torch.cuda.memory_allocated() / \
                    1024**3

        wandb.log(train_metrics, step=global_step_ref[0])


def finetune(cfg: FinetuneConfig, seed: int) -> None:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="gnn-pretraining-finetune",
               name=f"{cfg.exp_name}_{seed}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    evaluation_loaders = get_finetune_evaluation_loaders(cfg.domain_name, seed, cfg.batch_size)
    val_loader = evaluation_loaders['val']
    test_loader = evaluation_loaders['test']

    model = create_finetune_model(
        device=device,
        domain_name=cfg.domain_name,
        finetune_strategy=cfg.finetune_strategy,
        task_type=cfg.task_type,
        pretrained_path=get_pretrained_model_path(cfg.pretrained_scheme, seed) if cfg.pretrained_scheme != 'b1' else None,
    )

    param_groups = model.get_optimizer_param_groups(
        cfg.lr_backbone, cfg.lr_head)
    optimizer = AdamW(param_groups)

    train_loader = get_finetune_train_loader(
        cfg.domain_name, seed, cfg.batch_size, 0)
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * WARMUP_FRACTION)

    lr_multiplier = CosineWithWarmup(
        total_steps=total_steps, warmup_steps=warmup_steps)

    best_selection_metric = -float('inf')
    epochs_since_improvement = 0

    global_step = [0]

    for epoch in range(1, cfg.epochs + 1):
        train_loader = get_finetune_train_loader(
            cfg.domain_name, seed, cfg.batch_size, epoch)

        run_training(
            model,
            optimizer,
            train_loader,
            lr_multiplier,
            device,
            epoch,
            global_step,
            cfg
        )

        is_best, formatted_val_metrics, best_selection_metric, epochs_since_improvement = run_evaluation(
            model, val_loader, device, epoch, best_selection_metric,
            epochs_since_improvement, cfg, seed, global_step
        )

        if epochs_since_improvement >= cfg.patience:
            break

    best_checkpoint_path = OUTPUT_DIR / f"best_model_{cfg.exp_name}_{seed}.pt"
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # Test evaluation using simpler function
    batch_metrics = []
    for batch_or_data in test_loader:
        metrics = compute_loss_and_metrics(
            model, batch_or_data, device, cfg.task_type, cfg.domain_name)
        batch_metrics.append(metrics)

    test_metrics = aggregate_metrics(batch_metrics, 'test')

    # Convergence analysis
    convergence_metrics = {
        'convergence/total_epochs': epoch,
        'convergence/early_stopped': epochs_since_improvement >= cfg.patience,
        'convergence/epochs_to_best': epoch - epochs_since_improvement,
        'convergence/training_efficiency': best_selection_metric / max(epoch, 1),
        'convergence/patience_utilization': epochs_since_improvement / cfg.patience
    }

    # Model analysis
    param_info = count_parameters(model)
    model_metrics = {
        'model/param_count_total': param_info['total'],
        'model/param_count_trainable': param_info['trainable'],
        'model/param_efficiency': best_selection_metric / max(param_info['trainable'], 1),
        'model/freeze_ratio': 1.0 - (param_info['trainable'] / max(param_info['total'], 1))
    }

    # Strategy analysis
    strategy_metrics = {
        'strategy/finetune_method': cfg.finetune_strategy,
        'strategy/pretrain_method': cfg.pretrained_scheme,
        'strategy/is_from_scratch': cfg.pretrained_scheme == 'b1',
        'strategy/selection_metric_used': 'val/auc' if cfg.task_type == 'link_prediction' else 'val/accuracy',
        'strategy/final_selection_score': best_selection_metric
    }

    final_metrics = {
        **test_metrics,
        **convergence_metrics,
        **model_metrics,
        **strategy_metrics
    }

    wandb.log(final_metrics, step=global_step[0])
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()
    cfg = build_config(args)

    finetune(cfg, args.seed)


if __name__ == "__main__":
    main()
