"""
Comprehensive checkpointing system for GNN training.

This module provides robust checkpointing capabilities including:
- Model state saving and loading
- Training state resumption
- Best model tracking
- WandB artifact integration
- Automatic cleanup of old checkpoints
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
import json
import shutil
import glob
from datetime import datetime
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("WandB not available for checkpoint artifacts")


class CheckpointManager:
    """
    Comprehensive checkpoint manager for training state and model persistence.
    
    This class handles:
    - Saving/loading complete training state
    - Best model tracking based on metrics
    - Automatic cleanup of old checkpoints
    - Integration with experiment tracking
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path],
                 keep_n_checkpoints: int = 3,
                 save_best: bool = True,
                 best_metric: str = 'val/loss_total',
                 best_mode: str = 'min',
                 config=None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_n_checkpoints: Number of regular checkpoints to keep
            save_best: Whether to track and save best model
            best_metric: Metric to use for best model selection
            best_mode: 'min' or 'max' for best metric
            config: Configuration object to save with checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_n_checkpoints = keep_n_checkpoints
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.config = config
        
        # Initialize WandB reference (will be set by experiment tracker)
        self.wandb_run = None
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best metric value
        self.best_metric_value = float('inf') if best_mode == 'min' else float('-inf')
        
        # Paths for special checkpoints
        self.latest_checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        self.best_checkpoint_path = self.checkpoint_dir / 'best_checkpoint.pt'
        
        logging.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
        logging.info(f"Best metric: {self.best_metric} ({self.best_mode})")
    
    def set_wandb_run(self, wandb_run):
        """Set WandB run reference for artifact logging."""
        self.wandb_run = wandb_run
    
    def save_checkpoint(self, epoch: int, model: nn.Module, optimizer, 
                       scheduler_manager, metrics: Dict[str, float], 
                       is_best: bool = False, is_final: bool = False) -> Optional[str]:
        """
        Save model checkpoint with comprehensive state information.
        
        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer state
            scheduler_manager: Scheduler states
            metrics: Current metrics
            is_best: Whether this is the best model so far
            is_final: Whether this is the final model
            
        Returns:
            Path to saved checkpoint file
        """
        try:
            # Create checkpoint directory
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine checkpoint filename
            if is_final:
                checkpoint_path = self.checkpoint_dir / 'final_model.pt'
            elif is_best:
                checkpoint_path = self.checkpoint_dir / 'best_model.pt'
            else:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
            
            # Prepare checkpoint data
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
            }
            
            # Add scheduler states
            if scheduler_manager:
                checkpoint_data['scheduler_states'] = {
                    'lr_scheduler': scheduler_manager.lr_scheduler.state_dict() if scheduler_manager.lr_scheduler else None,
                    'da_scheduler': scheduler_manager.da_scheduler.state_dict() if hasattr(scheduler_manager.da_scheduler, 'state_dict') else None
                }
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Verify file was created and has reasonable size
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file was not created: {checkpoint_path}")
            
            file_size = checkpoint_path.stat().st_size
            if file_size < 1000:  # Less than 1KB is suspicious
                raise ValueError(f"Checkpoint file is too small ({file_size} bytes): {checkpoint_path}")
            
            logging.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Save to WandB artifacts if available
            if self.wandb_run and WANDB_AVAILABLE:
                try:
                    # Ensure we use the absolute path and it exists
                    abs_checkpoint_path = checkpoint_path.resolve()
                    if abs_checkpoint_path.exists():
                        artifact_name = f"model-checkpoint-epoch-{epoch}"
                        if is_best:
                            artifact_name = "best-model"
                        elif is_final:
                            artifact_name = "final-model"
                        
                        artifact = wandb.Artifact(
                            name=artifact_name,
                            type="model",
                            description=f"Model checkpoint at epoch {epoch}"
                        )
                        
                        # Add file with explicit name
                        artifact.add_file(str(abs_checkpoint_path), name=checkpoint_path.name)
                        
                        # Log artifact with appropriate alias
                        aliases = ["latest"]
                        if is_best:
                            aliases.append("best")
                        if is_final:
                            aliases.append("final")
                        
                        self.wandb_run.log_artifact(artifact, aliases=aliases)
                        logging.info(f"Checkpoint artifact logged: {artifact_name}")
                    else:
                        logging.error(f"Checkpoint file does not exist for artifact: {abs_checkpoint_path}")
                except Exception as e:
                    logging.error(f"Failed to save checkpoint artifact: {str(e)}")
            elif self.wandb_run and not WANDB_AVAILABLE:
                logging.warning("WandB run available but WandB not installed - skipping artifact upload")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {str(e)}")
            return None
    
    def load_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None,
                       load_best: bool = False,
                       load_latest: bool = False) -> Dict[str, Any]:
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint path to load
            load_best: Load the best checkpoint
            load_latest: Load the latest checkpoint
            
        Returns:
            Loaded checkpoint dictionary
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        # Determine which checkpoint to load
        if load_best:
            path = self.best_checkpoint_path
        elif load_latest:
            path = self.latest_checkpoint_path
        elif checkpoint_path:
            path = Path(checkpoint_path)
        else:
            raise ValueError("Must specify checkpoint_path, load_best=True, or load_latest=True")
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        try:
            checkpoint = torch.load(path, map_location='cpu')
            logging.info(f"Checkpoint loaded: {path}")
            
            # Log checkpoint info
            if 'epoch' in checkpoint:
                logging.info(f"  Epoch: {checkpoint['epoch']}")
            if 'timestamp' in checkpoint:
                logging.info(f"  Timestamp: {checkpoint['timestamp']}")
            if 'metrics' in checkpoint and checkpoint['metrics']:
                logging.info(f"  Metrics: {checkpoint['metrics']}")
            
            return checkpoint
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {str(e)}")
            raise
    
    def resume_training(self, model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler=None,
                       loss_computer=None,
                       checkpoint_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Resume training from a checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into (optional)
            loss_computer: Loss computer to load state into (optional)
            checkpoint_path: Specific checkpoint to resume from (otherwise loads latest)
            
        Returns:
            Dictionary with resume information (epoch, metrics, etc.)
        """
        # Load checkpoint
        if checkpoint_path:
            checkpoint = self.load_checkpoint(checkpoint_path)
        else:
            try:
                checkpoint = self.load_checkpoint(load_latest=True)
            except FileNotFoundError:
                logging.warning("No latest checkpoint found. Starting from scratch.")
                return {'epoch': 0, 'metrics': {}}
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info("Model state loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load model state: {str(e)}")
                raise
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logging.info("Optimizer state loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load optimizer state: {str(e)}")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logging.info("Scheduler state loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load scheduler state: {str(e)}")
        
        # Load loss computer state
        if loss_computer is not None and 'loss_computer_state_dict' in checkpoint:
            try:
                # Assuming loss computer has uncertainty weighting parameters
                if hasattr(loss_computer, 'uncertainty_weighting') and loss_computer.uncertainty_weighting:
                    loss_computer.uncertainty_weighting.load_state_dict(
                        checkpoint['loss_computer_state_dict']
                    )
                logging.info("Loss computer state loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load loss computer state: {str(e)}")
        
        # Update best metric value if available
        if 'best_metric_value' in checkpoint:
            self.best_metric_value = checkpoint['best_metric_value']
        
        resume_info = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', ''),
            'resumed': True
        }
        
        logging.info(f"Training resumed from epoch {resume_info['epoch']}")
        return resume_info
    
    def create_state_dict(self, model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler=None,
                         loss_computer=None,
                         **kwargs) -> Dict[str, Any]:
        """
        Create a complete state dictionary for checkpointing.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Scheduler to save (optional)
            loss_computer: Loss computer to save (optional)
            **kwargs: Additional items to save
            
        Returns:
            Complete state dictionary
        """
        state_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric_value': self.best_metric_value,
        }
        
        # Add scheduler state if available
        if scheduler is not None:
            if hasattr(scheduler, 'state_dict'):
                state_dict['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add loss computer state if available
        if loss_computer is not None:
            if hasattr(loss_computer, 'uncertainty_weighting') and loss_computer.uncertainty_weighting:
                state_dict['loss_computer_state_dict'] = loss_computer.uncertainty_weighting.state_dict()
        
        # Add any additional items
        state_dict.update(kwargs)
        
        return state_dict
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """
        Get information about available checkpoints.
        
        Returns:
            Dictionary with checkpoint information
        """
        info = {
            'checkpoint_dir': str(self.checkpoint_dir),
            'best_metric': self.best_metric,
            'best_metric_value': self.best_metric_value,
            'has_latest': self.latest_checkpoint_path.exists(),
            'has_best': self.best_checkpoint_path.exists(),
            'regular_checkpoints': []
        }
        
        # Find all regular checkpoints
        checkpoint_pattern = self.checkpoint_dir / 'checkpoint_epoch_*.pt'
        checkpoint_files = sorted(glob.glob(str(checkpoint_pattern)))
        
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                info['regular_checkpoints'].append({
                    'path': checkpoint_file,
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'timestamp': checkpoint.get('timestamp', 'unknown'),
                    'metrics': checkpoint.get('metrics', {})
                })
            except Exception as e:
                logging.warning(f"Failed to read checkpoint info from {checkpoint_file}: {str(e)}")
        
        return info
    
    def _is_better_metric(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.best_mode == 'min':
            return current < best
        else:
            return current > best
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint files, keeping only the most recent ones."""
        if self.keep_n_checkpoints <= 0:
            return
        
        # Find all regular checkpoint files
        checkpoint_pattern = self.checkpoint_dir / 'checkpoint_epoch_*.pt'
        checkpoint_files = sorted(glob.glob(str(checkpoint_pattern)))
        
        # Remove oldest files if we have too many
        if len(checkpoint_files) > self.keep_n_checkpoints:
            files_to_remove = checkpoint_files[:-self.keep_n_checkpoints]
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    logging.debug(f"Removed old checkpoint: {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove old checkpoint {file_path}: {str(e)}")
    
    def cleanup_all(self):
        """Remove all checkpoint files."""
        try:
            shutil.rmtree(self.checkpoint_dir)
            logging.info(f"All checkpoints removed: {self.checkpoint_dir}")
        except Exception as e:
            logging.error(f"Failed to cleanup checkpoints: {str(e)}")


class ModelSaver:
    """
    Utility class for saving trained models in various formats.
    
    This class provides methods for saving models for different use cases:
    - State dict only (for resuming training)
    - Complete model (for inference)
    - ONNX export (for deployment)
    - TorchScript (for production)
    """
    
    def __init__(self, save_dir: Union[str, Path]):
        """
        Initialize model saver.
        
        Args:
            save_dir: Directory to save models
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save_state_dict(self, model: nn.Module, filename: str,
                       metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save model state dictionary.
        
        Args:
            model: Model to save
            filename: Filename for saved model
            metadata: Additional metadata to save
            
        Returns:
            Path to saved file
        """
        save_path = self.save_dir / filename
        
        save_data = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(save_data, save_path)
        logging.info(f"Model state dict saved: {save_path}")
        
        return save_path
    
    def save_complete_model(self, model: nn.Module, filename: str) -> Path:
        """
        Save complete model (architecture + weights).
        
        Args:
            model: Model to save
            filename: Filename for saved model
            
        Returns:
            Path to saved file
        """
        save_path = self.save_dir / filename
        
        torch.save(model, save_path)
        logging.info(f"Complete model saved: {save_path}")
        
        return save_path
    
    def save_onnx(self, model: nn.Module, dummy_input: torch.Tensor,
                  filename: str, **kwargs) -> Path:
        """
        Export model to ONNX format.
        
        Args:
            model: Model to export
            dummy_input: Example input for tracing
            filename: Filename for ONNX model
            **kwargs: Additional arguments for torch.onnx.export
            
        Returns:
            Path to saved ONNX file
        """
        save_path = self.save_dir / filename
        
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            **kwargs
        )
        
        logging.info(f"ONNX model saved: {save_path}")
        return save_path
    
    def save_torchscript(self, model: nn.Module, filename: str,
                        example_input: Optional[torch.Tensor] = None) -> Path:
        """
        Save model as TorchScript.
        
        Args:
            model: Model to save
            filename: Filename for TorchScript model
            example_input: Example input for tracing (optional)
            
        Returns:
            Path to saved TorchScript file
        """
        save_path = self.save_dir / filename
        
        model.eval()
        
        if example_input is not None:
            # Use tracing
            traced_model = torch.jit.trace(model, example_input)
        else:
            # Use scripting
            traced_model = torch.jit.script(model)
        
        traced_model.save(str(save_path))
        logging.info(f"TorchScript model saved: {save_path}")
        
        return save_path


def create_checkpoint_manager(config) -> CheckpointManager:
    """
    Create checkpoint manager from configuration.
    
    Args:
        config: Training configuration object
        
    Returns:
        Initialized checkpoint manager
    """
    return CheckpointManager(
        checkpoint_dir=config.training.checkpoint_dir,
        keep_n_checkpoints=config.training.keep_n_checkpoints,
        save_best=True,
        best_metric=config.training.validation_metric,
        best_mode=config.training.metric_mode,
        config=config # Pass the config object
    )


if __name__ == '__main__':
    # Test the checkpointing system
    import tempfile
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / 'checkpoints'
        
        # Initialize checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            keep_n_checkpoints=3,
            best_metric='val/loss',
            best_mode='min'
        )
        
        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test saving checkpoints
        for epoch in range(5):
            # Create dummy metrics
            metrics = {
                'train/loss': 1.0 - epoch * 0.1,
                'val/loss': 0.8 - epoch * 0.05,
                'val/accuracy': 0.6 + epoch * 0.08
            }
            
            # Create state dict
            state_dict = manager.create_state_dict(model, optimizer)
            
            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(
                state_dict=state_dict,
                epoch=epoch,
                metrics=metrics
            )
            
            print(f"Saved checkpoint for epoch {epoch}: {checkpoint_path}")
        
        # Test loading best checkpoint
        try:
            best_checkpoint = manager.load_checkpoint(load_best=True)
            print(f"Best checkpoint: epoch {best_checkpoint['epoch']}, metrics: {best_checkpoint['metrics']}")
        except FileNotFoundError:
            print("No best checkpoint found")
        
        # Test checkpoint info
        info = manager.get_checkpoint_info()
        print(f"Checkpoint info: {len(info['regular_checkpoints'])} regular checkpoints")
        
        # Test model saver
        model_saver = ModelSaver(checkpoint_dir / 'models')
        model_path = model_saver.save_state_dict(
            model=model,
            filename='test_model.pt',
            metadata={'test': True, 'epoch': 4}
        )
        print(f"Model saved: {model_path}")
        
        print("Checkpointing system test completed successfully!") 