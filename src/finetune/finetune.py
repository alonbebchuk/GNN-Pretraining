import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
import yaml
from torch.optim import AdamW
from torch_geometric.utils import negative_sampling

from src.data.finetune_data_loaders import create_finetune_data_loader, get_dataset_info
from src.data.data_setup import NUM_CLASSES, TASK_TYPES
from src.model.finetune_model import FinetuneGNN, create_finetune_model
from src.finetune.metrics import (
    aggregate_metrics,
    compute_loss_and_metrics, 
    compute_training_metrics,
    compute_validation_metrics,
    print_metrics_summary
)
from src.finetune.artifacts import (
    count_parameters,
    get_experiment_output_dir,
    get_pretrained_model_path,
    print_model_info,
    save_finetune_checkpoint,
    validate_config_compatibility
)

# Training hyperparameters
EPOCHS = 100
LR_BACKBONE = 1e-4  # Lower LR for pretrained backbone
LR_HEAD = 1e-3      # Higher LR for new head
PATIENCE_EPOCHS = 10
BATCH_SIZE = 32

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "finetune"


@dataclass
class FinetuneConfig:
    exp_name: str                    # Experiment name
    domain_name: str                 # Target domain ('ENZYMES', 'PTC_MR', 'Cora', 'CiteSeer')
    pretrained_scheme: str           # Pretraining scheme ('b1_from_scratch', 's4_all_objectives', etc.)
    finetune_strategy: str           # 'full_finetune' or 'linear_probe'
    
    # Optional overrides
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    lr_backbone: float = LR_BACKBONE
    lr_head: float = LR_HEAD
    patience: int = PATIENCE_EPOCHS


def build_config(args: argparse.Namespace) -> FinetuneConfig:
    """Build configuration from YAML file and command line args."""
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    # Override with command line args if provided
    if args.domain:
        data['domain_name'] = args.domain
    if args.scheme:
        data['pretrained_scheme'] = args.scheme
    if args.strategy:
        data['finetune_strategy'] = args.strategy
    
    return FinetuneConfig(**data)


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(domain_name: str, batch_size: int, seed: int) -> Dict[str, torch.utils.data.DataLoader]:
    """Create data loaders for train/val/test splits."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    loaders = {}
    for split in ['train', 'val', 'test']:
        loaders[split] = create_finetune_data_loader(
            domain_name=domain_name,
            split=split,
            generator=generator,
            batch_size=batch_size,
            shuffle=(split == 'train')
        )
    
    return loaders


def evaluate_model(
    model: FinetuneGNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    task_type: str,
    domain_name: str
) -> Dict[str, float]:
    """Evaluate model on given data loader."""
    model.eval()
    
    batch_metrics = []
    
    for batch_or_data in data_loader:
        metrics = compute_loss_and_metrics(
            model=model,
            batch_or_data=batch_or_data,
            device=device,
            task_type=task_type,
            domain_name=domain_name
        )
        batch_metrics.append(metrics)
    
    # Aggregate metrics across batches
    aggregated = aggregate_metrics(batch_metrics)
    
    return aggregated


def train_epoch(
    model: FinetuneGNN,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    task_type: str,
    domain_name: str
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    epoch_losses = []
    step = 0
    
    for batch_or_data in train_loader:
        step_start_time = time.time()
        
        # Forward pass and loss computation
        if task_type == 'graph_classification':
            batch = batch_or_data.to(device)
            logits = model(batch)
            targets = batch.y
            loss = torch.nn.functional.cross_entropy(logits, targets)
            
        elif task_type == 'node_classification':
            data, node_indices, targets = batch_or_data
            data = data.to(device)
            node_indices = node_indices.to(device)
            targets = targets.to(device)
            
            full_logits = model(data)
            logits = full_logits[node_indices]
            loss = torch.nn.functional.cross_entropy(logits, targets)
            
        elif task_type == 'link_prediction':
            data, pos_edges, _ = batch_or_data  # Third element is None for training
            data = data.to(device)
            pos_edges = pos_edges.to(device)
            
            # Generate negative edges on-the-fly
            neg_edges = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=pos_edges.size(1)
            ).to(device)
            
            # Combine positive and negative edges
            all_edges = torch.cat([pos_edges, neg_edges], dim=1)
            edge_labels = torch.cat([
                torch.ones(pos_edges.size(1), device=device),
                torch.zeros(neg_edges.size(1), device=device)
            ])
            
            # Forward pass
            edge_probs = model(data, edge_index_for_prediction=all_edges)
            
            # Binary cross entropy loss
            loss = torch.nn.functional.binary_cross_entropy(edge_probs, edge_labels)
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        epoch_losses.append(loss.item())
        
        # Log training metrics occasionally
        if step % 50 == 0:
            # Get current learning rates
            lr_backbone = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 1 else 0.0
            lr_head = optimizer.param_groups[-1]['lr']
            
            train_metrics = compute_training_metrics(
                epoch=epoch,
                step=step,
                train_loss=loss.item(),
                lr_backbone=lr_backbone,
                lr_head=lr_head,
                model=model
            )
            
            # Add timing
            step_time_ms = (time.time() - step_start_time) * 1000
            train_metrics['system/time_per_step_ms'] = step_time_ms
            
            # Add GPU memory if available
            if device.type == "cuda":
                train_metrics['system/gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
                train_metrics['system/gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
            
            wandb.log(train_metrics)
        
        step += 1
    
    return {'train_loss_avg': np.mean(epoch_losses)}


def run_finetuning(cfg: FinetuneConfig, seed: int) -> Dict[str, float]:
    """Run complete finetuning experiment."""
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb
    wandb.init(
        project="gnn-finetuning",
        name=f"{cfg.exp_name}_{seed}",
        tags=["finetuning", cfg.domain_name, cfg.pretrained_scheme, cfg.finetune_strategy, f"seed_{seed}"],
        config=cfg.__dict__
    )
    
    # Create output directory
    output_dir = get_experiment_output_dir(
        base_dir=OUTPUT_DIR,
        domain_name=cfg.domain_name,
        scheme_name=cfg.pretrained_scheme,
        strategy=cfg.finetune_strategy,
        seed=seed
    )
    
    # Get dataset info
    dataset_info = get_dataset_info(cfg.domain_name)
    print(f"Dataset info: {dataset_info}")
    
    # Create data loaders
    data_loaders = get_data_loaders(cfg.domain_name, cfg.batch_size, seed)
    
    # Get pretrained model path (if not from scratch)
    pretrained_path = None
    if cfg.pretrained_scheme != 'b1_from_scratch':
        pretrained_path = get_pretrained_model_path(
            scheme_name=cfg.pretrained_scheme,
            seed=seed,
            download_dir=OUTPUT_DIR / "pretrained_models"
        )
    
    # Create model
    freeze_backbone = (cfg.finetune_strategy == 'linear_probe')
    model = create_finetune_model(
        domain_name=cfg.domain_name,
        pretrained_checkpoint_path=pretrained_path,
        freeze_backbone=freeze_backbone,
        device=device
    )
    
    # Print model info
    print_model_info(model)
    
    # Validate config compatibility
    validate_config_compatibility(cfg.__dict__, model)
    
    # Setup optimizer with discriminative learning rates
    param_groups = model.get_optimizer_param_groups(cfg.lr_backbone, cfg.lr_head)
    optimizer = AdamW(param_groups)
    
    print(f"Optimizer parameter groups: {[g['name'] for g in param_groups]}")
    
    # Training loop
    best_val_metric = -float('inf')  # We'll use accuracy as primary metric
    epochs_since_improvement = 0
    task_type = TASK_TYPES[cfg.domain_name]
    
    for epoch in range(1, cfg.epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=data_loaders['train'],
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            task_type=task_type,
            domain_name=cfg.domain_name
        )
        
        # Validate
        val_metrics = evaluate_model(
            model=model,
            data_loader=data_loaders['val'],
            device=device,
            task_type=task_type,
            domain_name=cfg.domain_name
        )
        
        # Format for logging
        formatted_val_metrics = compute_validation_metrics(val_metrics, 'val')
        
        # Determine if this is the best model
        current_val_metric = formatted_val_metrics['val/accuracy']  # Use accuracy as primary
        is_best = current_val_metric > best_val_metric
        
        if is_best:
            best_val_metric = current_val_metric
            epochs_since_improvement = 0
            
            # Save best checkpoint
            best_checkpoint_path = output_dir / "best_model.pt"
            save_finetune_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=formatted_val_metrics,
                checkpoint_path=best_checkpoint_path,
                is_best=True
            )
        else:
            epochs_since_improvement += 1
        
        # Log epoch metrics
        epoch_time = time.time() - epoch_start_time
        epoch_metrics = {
            **formatted_val_metrics,
            'val/is_best': is_best,
            'train/epoch_time_min': epoch_time / 60.0,
            'train/epochs_since_improvement': epochs_since_improvement,
        }
        wandb.log(epoch_metrics)
        
        # Print progress
        if epoch % 10 == 0 or is_best:
            print(f"Epoch {epoch:3d} | "
                  f"Val Acc: {current_val_metric:.4f} | "
                  f"Best: {best_val_metric:.4f} | "
                  f"Patience: {epochs_since_improvement}/{cfg.patience}")
        
        # Early stopping
        if epochs_since_improvement >= cfg.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model for final test evaluation
    print("\nLoading best model for test evaluation...")
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final test evaluation
    test_metrics = evaluate_model(
        model=model,
        data_loader=data_loaders['test'],
        device=device,
        task_type=task_type,
        domain_name=cfg.domain_name
    )
    
    formatted_test_metrics = compute_validation_metrics(test_metrics, 'test')
    
    # Print final results
    print_metrics_summary(formatted_test_metrics, 'test', cfg.domain_name)
    
    # Log final metrics
    final_metrics = {
        **formatted_test_metrics,
        'final/total_epochs': epoch,
        'final/early_stopped': epochs_since_improvement >= cfg.patience,
        'final/best_val_accuracy': best_val_metric,
        'final/param_count': count_parameters(model)['total'],
        'final/param_count_trainable': count_parameters(model)['trainable'],
    }
    
    wandb.log(final_metrics)
    wandb.finish()
    
    return formatted_test_metrics


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fine-tune pretrained GNN models")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to YAML config file")
    parser.add_argument("--seed", type=int, required=True,
                       help="Random seed")
    
    # Optional overrides
    parser.add_argument("--domain", type=str, help="Target domain name")
    parser.add_argument("--scheme", type=str, help="Pretraining scheme") 
    parser.add_argument("--strategy", type=str, help="Finetuning strategy")
    
    args = parser.parse_args()
    
    # Build configuration
    cfg = build_config(args)
    
    print(f"\nStarting finetuning experiment:")
    print(f"  Domain: {cfg.domain_name}")
    print(f"  Pretraining: {cfg.pretrained_scheme}")
    print(f"  Strategy: {cfg.finetune_strategy}")
    print(f"  Seed: {args.seed}")
    print(f"  Config: {args.config}")
    
    # Run experiment
    final_metrics = run_finetuning(cfg, args.seed)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"Test Accuracy: {final_metrics.get('test/accuracy', 0.0):.4f}")
    print(f"Test F1 Macro: {final_metrics.get('test/f1_macro', 0.0):.4f}")
    if 'test/auc_macro' in final_metrics:
        print(f"Test AUC Macro: {final_metrics['test/auc_macro']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
