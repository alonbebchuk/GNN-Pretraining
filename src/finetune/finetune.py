import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import wandb
import yaml
from torch.optim import AdamW
from torch_geometric.utils import negative_sampling

from src.data.finetune_data_loaders import create_finetune_data_loader
from src.finetune.metrics import compute_batch_metrics, compute_training_metrics, compute_validation_metrics, compute_test_metrics
from src.models.finetune_model import FinetuneGNN, create_finetune_model, LR_BACKBONE, LR_FINETUNE
from src.pretrain.schedulers import CosineWithWarmup

import torch.nn.functional as F

from src.data.data_setup import TASK_TYPES

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "finetune"

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
    patience: int

    def __post_init__(self):
        self.exp_name = f"{self.domain_name}_{self.finetune_strategy}_{self.pretrained_scheme}"
        self.task_type = TASK_TYPES[self.domain_name]
        self.batch_size = BATCH_SIZES[self.domain_name]
        self.epochs = EPOCHS[self.domain_name]
        self.patience = int(self.epochs * PATIENCE_FRACTION)


def build_config(args: argparse.Namespace) -> FinetuneConfig:
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return FinetuneConfig(**data)


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_batch_by_task_type(
    model: torch.nn.Module, 
    batch, 
    device: torch.device, 
    task_type: str,
    generator: torch.Generator = None
) -> tuple:
    
    if task_type == 'graph_classification':
        batch = batch.to(device)
        logits = model(batch)
        targets = batch.y
        
        loss = F.cross_entropy(logits, targets)
        
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        return logits, loss, targets, predictions, probabilities, None, None
        
    elif task_type == 'node_classification':
        data, node_indices, targets = batch
        data, node_indices, targets = data.to(device), node_indices.to(device), targets.to(device)
        
        full_logits = model(data)
        logits = full_logits[node_indices]
        
        loss = F.cross_entropy(logits, targets)
        
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        return logits, loss, targets, predictions, probabilities, None, None
        
    elif task_type == 'link_prediction':
        # For training, generate negative samples; for evaluation, edges are pre-provided
        if model.training and generator is not None:
            data, pos_edges, _ = batch
            data, pos_edges = data.to(device), pos_edges.to(device)
            
            neg_edges = negative_sampling(data.edge_index, data.num_nodes, pos_edges.size(1), generator=generator).to(device)
            all_edges = torch.cat([pos_edges, neg_edges], dim=1)
            edge_labels = torch.cat([torch.ones(pos_edges.size(1), device=device), torch.zeros(neg_edges.size(1), device=device)])
            edge_probs = model(data, edge_index=all_edges)
        else:
            # Evaluation case - edges and targets provided in batch
            data, all_edges, edge_labels = batch
            data, all_edges, edge_labels = data.to(device), all_edges.to(device), edge_labels.to(device)
            edge_probs = model(data, edge_index=all_edges)
        
        loss = F.binary_cross_entropy(edge_probs, edge_labels)
        
        predictions = (edge_probs > 0.5).float()
        probabilities = torch.stack([1 - edge_probs, edge_probs], dim=1)
        targets = edge_labels.long()
        predictions = predictions.long()
        
        return None, loss, targets, predictions, probabilities, edge_probs, edge_labels
    
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


def compute_loss_and_metrics(model: torch.nn.Module, batch, device: torch.device, task_type: str, domain_name: str) -> Dict[str, float]:
    model.eval()
    
    with torch.no_grad():
        logits, loss, targets, predictions, probabilities, edge_probs, edge_labels = process_batch_by_task_type(
            model, batch, device, task_type
        )
        metrics = compute_batch_metrics(domain_name, targets, predictions, probabilities, loss)
    
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

    val_metrics = compute_validation_metrics(batch_metrics, epoch)

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
            f"model_{cfg.exp_name}_{seed}.pt"
        torch.save(checkpoint, best_checkpoint_path)

        artifact = wandb.Artifact(f"model_{cfg.exp_name}_{seed}", type="model")
        artifact.add_file(str(best_checkpoint_path))
        wandb.log_artifact(artifact)
    else:
        new_best_metric = best_selection_metric
        new_epochs_since_improvement = epochs_since_improvement + 1

    # Log validation metrics
    wandb.log(val_metrics, step=global_step[0])

    return is_best, val_metrics, new_best_metric, new_epochs_since_improvement


def run_training(
    model: FinetuneGNN,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    lr_multiplier: CosineWithWarmup,
    device: torch.device,
    epoch: int,
    global_step_ref: List[int],
    cfg: FinetuneConfig,
    generator: torch.Generator
) -> None:
    model.train()

    for batch in train_loader:
        step_start_time = time.time()
        global_step_ref[0] += 1

        logits, loss, targets, predictions, probabilities, edge_probs, edge_labels = process_batch_by_task_type(
            model, batch, device, cfg.task_type, generator
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scale = lr_multiplier()
        for pg in optimizer.param_groups:
            if pg['name'] == 'backbone':
                pg['lr'] = LR_BACKBONE[cfg.finetune_strategy] * scale
            else:
                pg['lr'] = LR_FINETUNE * scale
        lr_multiplier.step()

        train_metrics = compute_training_metrics(
            epoch=epoch,
            step=global_step_ref[0],
            loss=loss,
            optimizer=optimizer,
            domain_name=cfg.domain_name,
            targets=targets,
            predictions=predictions,
            probabilities=probabilities,
            step_start_time=step_start_time,
            model=model
        )
        
        wandb.log(train_metrics, step=global_step_ref[0])


def finetune(cfg: FinetuneConfig, seed: int) -> None:
    training_start_time = time.time()
    
    set_global_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="gnn-pretraining-finetune", name=f"{cfg.exp_name}_{seed}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    val_loader = create_finetune_data_loader(cfg.domain_name, 'val', cfg.batch_size, generator)
    test_loader = create_finetune_data_loader(cfg.domain_name, 'test', cfg.batch_size, generator)

    model = create_finetune_model(
        device=device,
        domain_name=cfg.domain_name,
        finetune_strategy=cfg.finetune_strategy,
        pretrained_scheme=cfg.pretrained_scheme,
        seed=seed,
    )

    optimizer = AdamW(model.param_groups)

    train_loader = create_finetune_data_loader(cfg.domain_name, 'train', cfg.batch_size, generator)
    total_steps = len(train_loader) * cfg.epochs

    lr_multiplier = CosineWithWarmup(total_steps=total_steps)

    best_selection_metric = -float('inf')
    epochs_since_improvement = 0

    global_step = [0]

    for epoch in range(1, cfg.epochs + 1):
        run_training(
            model,
            optimizer,
            train_loader,
            lr_multiplier,
            device,
            epoch,
            global_step,
            cfg,
            generator
        )

        is_best, formatted_val_metrics, best_selection_metric, epochs_since_improvement = run_evaluation(
            model, val_loader, device, epoch, best_selection_metric,
            epochs_since_improvement, cfg, seed, global_step
        )

        if epochs_since_improvement >= cfg.patience:
            break

    best_checkpoint_path = OUTPUT_DIR / f"model_{cfg.exp_name}_{seed}.pt"
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    batch_metrics = []
    for batch_or_data in test_loader:
        metrics = compute_loss_and_metrics(
            model, batch_or_data, device, cfg.task_type, cfg.domain_name)
        batch_metrics.append(metrics)

    final_metrics = compute_test_metrics(
        batch_metrics=batch_metrics,
        epoch=epoch,
        epochs_since_improvement=epochs_since_improvement,
        training_start_time=training_start_time,
        model=model
    )

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
