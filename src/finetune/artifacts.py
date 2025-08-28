import os
from pathlib import Path
from typing import Dict, Optional

import torch
import wandb
from torch import nn

from src.models.finetune_model import FinetuneGNN


def download_wandb_artifact(
    project_name: str,
    artifact_name: str,
    download_dir: Path,
    version: str = "latest"
) -> str:
    """
    Download a WandB artifact (pretrained model) to local directory.
    
    Args:
        project_name: WandB project name
        artifact_name: Artifact name (e.g., "model_s4_all_objectives_42")
        download_dir: Local directory to download to
        version: Artifact version ("latest" or specific version)
    
    Returns:
        Path to the downloaded model file
    """
    # Initialize WandB API
    api = wandb.Api()
    
    # Construct full artifact name
    full_artifact_name = f"{project_name}/{artifact_name}:{version}"
    
    print(f"Downloading artifact: {full_artifact_name}")
    
    # Download artifact
    artifact = api.artifact(full_artifact_name)
    artifact_dir = artifact.download(root=str(download_dir))
    
    # Find the model file (should be .pt file)
    model_files = list(Path(artifact_dir).glob("*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No .pt files found in artifact {artifact_name}")
    
    model_path = str(model_files[0])
    print(f"Downloaded model to: {model_path}")
    
    return model_path


def get_pretrained_model_path(
    scheme_name: str,
    seed: int,
    download_dir: Path,
    project_name: str = "gnn-pretraining"
) -> str:
    """
    Get path to pretrained model, downloading from WandB if necessary.
    
    Args:
        scheme_name: Pretraining scheme (e.g., "s4_all_objectives", "b1_from_scratch")
        seed: Random seed used in pretraining
        download_dir: Directory to download models to
        project_name: WandB project name
    
    Returns:
        Path to the pretrained model checkpoint
    """
    # Check if model already exists locally
    local_model_name = f"best_model_{scheme_name}_{seed}.pt"
    local_model_path = download_dir / local_model_name
    
    if local_model_path.exists():
        print(f"Using cached model: {local_model_path}")
        return str(local_model_path)
    
    # Download from WandB
    artifact_name = f"model_{scheme_name}_{seed}"
    
    try:
        return download_wandb_artifact(
            project_name=project_name,
            artifact_name=artifact_name,
            download_dir=download_dir,
            version="latest"
        )
    except Exception as e:
        print(f"Failed to download artifact {artifact_name}: {e}")
        raise


def save_finetune_checkpoint(
    model: FinetuneGNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: Path,
    is_best: bool = False
) -> None:
    """
    Save finetuning checkpoint.
    
    Args:
        model: Finetuning model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'model_config': {
            'domain_name': model.domain_name,
            'task_type': model.task_type,
            'num_classes': model.num_classes,
            'is_in_domain': model.is_in_domain,
            'freeze_backbone': model.freeze_backbone,
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        print(f"Saved best checkpoint to: {checkpoint_path}")
    else:
        print(f"Saved checkpoint to: {checkpoint_path}")


def load_finetune_checkpoint(
    checkpoint_path: str,
    model: FinetuneGNN,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cuda')
) -> Dict:
    """
    Load finetuning checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to map tensors to
    
    Returns:
        Checkpoint metadata (epoch, metrics, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {}),
        'model_config': checkpoint.get('model_config', {})
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in model by component.
    
    Args:
        model: Model to analyze
    
    Returns:
        Parameter counts by component
    """
    counts = {
        'total': sum(p.numel() for p in model.parameters()),
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'frozen': sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    
    # Component-specific counts for FinetuneGNN
    if isinstance(model, FinetuneGNN):
        counts['backbone'] = sum(p.numel() for p in model.gnn_backbone.parameters())
        counts['encoder'] = sum(p.numel() for p in model.input_encoder.parameters())
        counts['head'] = sum(p.numel() for p in model.classification_head.parameters())
        
        counts['backbone_trainable'] = sum(
            p.numel() for p in model.gnn_backbone.parameters() if p.requires_grad
        )
        counts['encoder_trainable'] = sum(
            p.numel() for p in model.input_encoder.parameters() if p.requires_grad
        )
        counts['head_trainable'] = sum(
            p.numel() for p in model.classification_head.parameters() if p.requires_grad
        )
    
    return counts


def print_model_info(model: FinetuneGNN) -> None:
    """Print comprehensive model information."""
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    
    # Basic info
    print(f"Domain: {model.domain_name}")
    print(f"Task Type: {model.task_type}")
    print(f"Num Classes: {model.num_classes}")
    print(f"In Domain: {model.is_in_domain}")
    print(f"Backbone Frozen: {model.freeze_backbone}")
    print(f"Device: {model.device}")
    
    # Parameter counts
    param_counts = count_parameters(model)
    print(f"\nParameter Counts:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Frozen: {param_counts['frozen']:,}")
    
    if 'backbone' in param_counts:
        print(f"\nBy Component:")
        print(f"  Backbone: {param_counts['backbone']:,} ({param_counts['backbone_trainable']:,} trainable)")
        print(f"  Encoder:  {param_counts['encoder']:,} ({param_counts['encoder_trainable']:,} trainable)")
        print(f"  Head:     {param_counts['head']:,} ({param_counts['head_trainable']:,} trainable)")
    
    print("="*50)


def validate_config_compatibility(
    config: Dict,
    model: FinetuneGNN
) -> None:
    """
    Validate that config is compatible with model.
    
    Args:
        config: Experiment configuration
        model: Finetuning model
    
    Raises:
        ValueError: If config incompatible with model
    """
    # Check domain match
    if config['domain_name'] != model.domain_name:
        raise ValueError(
            f"Config domain ({config['domain_name']}) doesn't match "
            f"model domain ({model.domain_name})"
        )
    
    # Check task type match
    if config['task_type'] != model.task_type:
        raise ValueError(
            f"Config task type ({config['task_type']}) doesn't match "
            f"model task type ({model.task_type})"
        )
    
    # Check freezing strategy
    if config.get('freeze_backbone', False) != model.freeze_backbone:
        print(
            f"Warning: Config freeze_backbone ({config.get('freeze_backbone', False)}) "
            f"doesn't match model freeze_backbone ({model.freeze_backbone})"
        )


def cleanup_downloaded_models(download_dir: Path, keep_latest: int = 5) -> None:
    """
    Clean up old downloaded model files to save space.
    
    Args:
        download_dir: Directory containing downloaded models
        keep_latest: Number of latest models to keep per scheme
    """
    model_files = list(download_dir.glob("best_model_*.pt"))
    
    if len(model_files) <= keep_latest:
        return
    
    # Sort by modification time
    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Delete older files
    for old_file in model_files[keep_latest:]:
        try:
            old_file.unlink()
            print(f"Deleted old model: {old_file.name}")
        except OSError as e:
            print(f"Failed to delete {old_file.name}: {e}")


def get_experiment_output_dir(
    base_dir: Path,
    domain_name: str,
    scheme_name: str,
    strategy: str,
    seed: int
) -> Path:
    """
    Get standardized output directory for experiment.
    
    Args:
        base_dir: Base output directory
        domain_name: Target domain
        scheme_name: Pretraining scheme
        strategy: Finetuning strategy ('full_finetune' or 'linear_probe')
        seed: Random seed
    
    Returns:
        Path to experiment output directory
    """
    exp_dir = base_dir / domain_name / scheme_name / strategy / f"seed_{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir
