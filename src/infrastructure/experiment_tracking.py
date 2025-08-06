"""
Experiment tracking and logging using Weights & Biases.

This module provides comprehensive experiment tracking including:
- WandB initialization and configuration
- Metric logging and visualization
- Model versioning with artifacts
- Hyperparameter tracking
- System monitoring
"""

import wandb
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
import json
import time
from datetime import datetime
import os
import shutil


class WandBTracker:
    """
    Weights & Biases experiment tracker.
    
    This class handles all interactions with WandB including initialization,
    metric logging, model versioning, and artifact management.
    """
    
    def __init__(self, config, model: Optional[nn.Module] = None, 
                 watch_model: bool = True, offline: bool = False):
        """
        Initialize WandB tracker.
        
        Args:
            config: Configuration object containing run details
            model: Model to track (optional)
            watch_model: Whether to watch model parameters and gradients
            offline: Whether to run in offline mode
        """
        self.config = config
        self.model = model
        self.offline = offline
        self.run = None
        self.step_count = 0
        self.epoch_count = 0
        
        # Initialize WandB
        self._initialize_wandb()
        
        # Watch model if provided
        if model is not None and watch_model and not offline:
            self._watch_model()
        
        logging.info(f"WandB tracker initialized. Run: {self.run.name if self.run else 'offline'}")
    
    def _initialize_wandb(self):
        """Initialize WandB run."""
        try:
            # Set offline mode if requested
            if self.offline:
                os.environ["WANDB_MODE"] = "offline"
            
            # Initialize run
            self.run = wandb.init(
                project=self.config.run.project_name,
                entity=self.config.run.entity,
                name=self.config.run.run_name,
                tags=self.config.run.tags,
                notes=self.config.run.notes,
                config=self._config_to_dict(),
                reinit=True
            )
            
            logging.info(f"WandB initialized successfully. Run ID: {self.run.id}")
            
        except Exception as e:
            logging.error(f"Failed to initialize WandB: {str(e)}")
            self.run = None
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration object to dictionary for WandB."""
        config_dict = {}
        
        # Convert dataclass config to dict
        if hasattr(self.config, '__dict__'):
            for key, value in self.config.__dict__.items():
                if hasattr(value, '__dict__'):
                    # Nested dataclass
                    config_dict[key] = {k: v for k, v in value.__dict__.items() 
                                      if not k.startswith('_')}
                elif isinstance(value, dict):
                    # Dictionary (like tasks)
                    config_dict[key] = {}
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, '__dict__'):
                            config_dict[key][sub_key] = {k: v for k, v in sub_value.__dict__.items() 
                                                       if not k.startswith('_')}
                        else:
                            config_dict[key][sub_key] = sub_value
                elif not key.startswith('_'):
                    config_dict[key] = value
        
        # Add system information
        config_dict['system'] = {
            'device': str(self.config.device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            config_dict['system']['cuda_device_name'] = torch.cuda.get_device_name(0)
        
        return config_dict
    
    def _watch_model(self):
        """Watch model parameters and gradients."""
        if self.run is not None and self.model is not None:
            try:
                wandb.watch(
                    self.model,
                    log='all',  # Log parameters and gradients
                    log_freq=100,  # Log every 100 steps
                    log_graph=True  # Log model graph
                )
                logging.info("Model watching enabled in WandB")
            except Exception as e:
                logging.warning(f"Failed to watch model: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int, torch.Tensor]], 
                   step: Optional[int] = None, epoch: Optional[int] = None,
                   commit: bool = True):
        """
        Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (auto-incremented if None)
            epoch: Epoch number (auto-incremented if None)
            commit: Whether to commit the log (send to server)
        """
        if self.run is None:
            return
        
        # Update counters
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
        
        if epoch is not None:
            self.epoch_count = epoch
        
        # Process metrics
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    processed_metrics[key] = value.item()
                else:
                    # Log tensor statistics
                    processed_metrics[f"{key}_mean"] = value.mean().item()
                    processed_metrics[f"{key}_std"] = value.std().item()
                    processed_metrics[f"{key}_min"] = value.min().item()
                    processed_metrics[f"{key}_max"] = value.max().item()
            else:
                processed_metrics[key] = value
        
        # Add step and epoch info
        processed_metrics['step'] = self.step_count
        processed_metrics['epoch'] = self.epoch_count
        
        try:
            wandb.log(processed_metrics, step=self.step_count, commit=commit)
        except Exception as e:
            logging.warning(f"Failed to log metrics: {str(e)}")
    
    def log_learning_curves(self, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float], epoch: int):
        """
        Log training and validation metrics for learning curves.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            epoch: Current epoch
        """
        if self.run is None:
            return
        
        # Combine metrics with prefixes
        all_metrics = {}
        
        for key, value in train_metrics.items():
            all_metrics[f"train/{key}"] = value
        
        for key, value in val_metrics.items():
            all_metrics[f"val/{key}"] = value
        
        self.log_metrics(all_metrics, epoch=epoch)
    
    def log_model_info(self, model: nn.Module):
        """
        Log model architecture information.
        
        Args:
            model: PyTorch model
        """
        if self.run is None:
            return
        
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                'model/total_parameters': total_params,
                'model/trainable_parameters': trainable_params,
                'model/non_trainable_parameters': total_params - trainable_params,
                'model/parameter_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            }
            
            # Get model info if available
            if hasattr(model, 'get_model_info'):
                additional_info = model.get_model_info()
                for key, value in additional_info.items():
                    if isinstance(value, (int, float, bool)):
                        model_info[f'model/{key}'] = value
            
            self.log_metrics(model_info, commit=False)
            
            # Log model architecture as text
            model_str = str(model)
            wandb.log({"model_architecture": wandb.Html(f"<pre>{model_str}</pre>")})
            
        except Exception as e:
            logging.warning(f"Failed to log model info: {str(e)}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters to WandB.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        if self.run is None:
            return
        
        try:
            wandb.config.update(hyperparams)
        except Exception as e:
            logging.warning(f"Failed to log hyperparameters: {str(e)}")
    
    def save_checkpoint_artifact(self, checkpoint_path: Union[str, Path], 
                                alias: str = "latest", 
                                metadata: Optional[Dict[str, Any]] = None):
        """
        Save model checkpoint as WandB artifact.
        
        Args:
            checkpoint_path: Path to checkpoint file
            alias: Alias for the artifact (e.g., "best", "latest", "epoch_10")
            metadata: Additional metadata to store with artifact
        """
        if self.run is None:
            return
        
        try:
            checkpoint_path = Path(checkpoint_path)
            
            # Create artifact
            artifact_name = f"{self.run.name}-checkpoint"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model_checkpoint",
                description=f"Model checkpoint for {self.run.name}",
                metadata=metadata or {}
            )
            
            # Add checkpoint file
            artifact.add_file(str(checkpoint_path))
            
            # Log artifact
            self.run.log_artifact(artifact, aliases=[alias])
            
            logging.info(f"Checkpoint artifact saved with alias '{alias}'")
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint artifact: {str(e)}")
    
    def save_model_artifact(self, model: nn.Module, model_path: Union[str, Path],
                           alias: str = "latest",
                           metadata: Optional[Dict[str, Any]] = None):
        """
        Save trained model as WandB artifact.
        
        Args:
            model: Trained model
            model_path: Path where model is saved
            alias: Alias for the artifact
            metadata: Additional metadata
        """
        if self.run is None:
            return
        
        try:
            model_path = Path(model_path)
            
            # Ensure model is saved
            if not model_path.exists():
                torch.save(model.state_dict(), model_path)
            
            # Create artifact
            artifact_name = f"{self.run.name}-model"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="trained_model",
                description=f"Trained model for {self.run.name}",
                metadata=metadata or {}
            )
            
            # Add model file
            artifact.add_file(str(model_path))
            
            # Add model info if available
            if hasattr(model, 'get_model_info'):
                model_info = model.get_model_info()
                artifact.metadata.update(model_info)
            
            # Log artifact
            self.run.log_artifact(artifact, aliases=[alias])
            
            logging.info(f"Model artifact saved with alias '{alias}'")
            
        except Exception as e:
            logging.error(f"Failed to save model artifact: {str(e)}")
    
    def log_confusion_matrix(self, y_true, y_pred, class_names: Optional[List[str]] = None):
        """
        Log confusion matrix to WandB.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
        """
        if self.run is None:
            return
        
        try:
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names
                )
            })
        except Exception as e:
            logging.warning(f"Failed to log confusion matrix: {str(e)}")
    
    def log_histogram(self, data: torch.Tensor, name: str):
        """
        Log histogram to WandB.
        
        Args:
            data: Data tensor
            name: Name for the histogram
        """
        if self.run is None:
            return
        
        try:
            wandb.log({name: wandb.Histogram(data.detach().cpu().numpy())})
        except Exception as e:
            logging.warning(f"Failed to log histogram: {str(e)}")
    
    def log_gradients(self, model: nn.Module):
        """
        Log gradient statistics.
        
        Args:
            model: PyTorch model
        """
        if self.run is None:
            return
        
        try:
            grad_stats = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_stats[f"gradients/{name}_norm"] = grad_norm
                    grad_stats[f"gradients/{name}_mean"] = param.grad.mean().item()
                    grad_stats[f"gradients/{name}_std"] = param.grad.std().item()
            
            if grad_stats:
                self.log_metrics(grad_stats, commit=False)
                
        except Exception as e:
            logging.warning(f"Failed to log gradients: {str(e)}")
    
    def log_system_metrics(self):
        """Log system metrics (GPU memory, etc.)."""
        if self.run is None:
            return
        
        try:
            system_metrics = {}
            
            if torch.cuda.is_available():
                # GPU memory
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)  # GB
                
                system_metrics['system/gpu_memory_allocated_gb'] = gpu_memory
                system_metrics['system/gpu_memory_cached_gb'] = gpu_memory_cached
            
            if system_metrics:
                self.log_metrics(system_metrics, commit=False)
                
        except Exception as e:
            logging.warning(f"Failed to log system metrics: {str(e)}")
    
    def finish(self):
        """Finish the WandB run."""
        if self.run is not None:
            try:
                wandb.finish()
                logging.info("WandB run finished")
            except Exception as e:
                logging.warning(f"Error finishing WandB run: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()

    def connect_checkpoint_manager(self, checkpoint_manager):
        """Connect checkpoint manager for artifact logging."""
        if hasattr(self, 'run') and self.run is not None:
            checkpoint_manager.set_wandb_run(self.run)
        else:
            logging.warning("WandB run not initialized - checkpoint artifacts will not be logged")


class MetricsLogger:
    """
    Utility class for collecting and logging metrics during training.
    
    This class provides a convenient interface for accumulating metrics
    over multiple batches and computing averages.
    """
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, Union[float, torch.Tensor]], count: int = 1):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric values
            count: Number of samples (for averaging)
        """
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value * count
            self.counts[key] += count
    
    def compute_averages(self) -> Dict[str, float]:
        """Compute average values for all metrics."""
        averages = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                averages[key] = self.metrics[key] / self.counts[key]
            else:
                averages[key] = 0.0
        return averages
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()
    
    def __len__(self):
        """Return number of different metrics."""
        return len(self.metrics)


def create_experiment_tracker(config, model: Optional[nn.Module] = None,
                            offline: bool = False) -> WandBTracker:
    """
    Create and initialize experiment tracker.
    
    Args:
        config: Configuration object
        model: Model to track (optional)
        offline: Whether to run in offline mode
        
    Returns:
        Initialized WandB tracker
    """
    return WandBTracker(config=config, model=model, offline=offline)


if __name__ == '__main__':
    # Test the experiment tracking system
    from dataclasses import dataclass
    
    @dataclass
    class TestRunConfig:
        project_name: str = 'test-project'
        run_name: str = 'test-run'
        entity: str = None
        tags: List[str] = None
        notes: str = 'Test run'
    
    @dataclass
    class TestConfig:
        run: TestRunConfig
        device: torch.device = torch.device('cpu')
    
    # Create test configuration
    config = TestConfig(run=TestRunConfig())
    
    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    
    # Test tracker (offline mode)
    with create_experiment_tracker(config, model, offline=True) as tracker:
        # Log model info
        tracker.log_model_info(model)
        
        # Log some metrics
        for epoch in range(5):
            train_metrics = {
                'loss': 0.5 - epoch * 0.1,
                'accuracy': 0.7 + epoch * 0.05
            }
            val_metrics = {
                'loss': 0.6 - epoch * 0.08,
                'accuracy': 0.65 + epoch * 0.04
            }
            
            tracker.log_learning_curves(train_metrics, val_metrics, epoch)
        
        # Test metrics logger
        metrics_logger = MetricsLogger()
        for i in range(10):
            batch_metrics = {
                'batch_loss': 0.5 + i * 0.01,
                'batch_acc': 0.8 - i * 0.005
            }
            metrics_logger.update(batch_metrics, count=32)  # Batch size 32
        
        avg_metrics = metrics_logger.compute_averages()
        tracker.log_metrics(avg_metrics)
        
        print("Experiment tracking test completed successfully!")
        print(f"Logged metrics: {avg_metrics}") 