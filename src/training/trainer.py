"""
Core training loop for GNN pre-training.

This module implements the main training loop that orchestrates:
- Model forward/backward passes
- Multi-task loss computation
- Validation and metric tracking
- Checkpointing and model saving
- Experiment logging
- Early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
import logging
from tqdm import tqdm
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch.nn.functional as F
import json

# Mixed precision training
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    logging.warning("Mixed precision training not available (PyTorch < 1.6)")

try:
    # Try relative imports first (for package usage)
    from ..infrastructure.config import Config
    from ..core.models.pretrain_model import PretrainableGNN
    from .losses import MultiTaskLossComputer, LossOutput
    from ..infrastructure.scheduler import SchedulerManager
    from ..infrastructure.experiment_tracking import WandBTracker, MetricsLogger
    from ..infrastructure.checkpointing import CheckpointManager
    from ..data.data_loading import negative_sampling, compute_graph_properties
except ImportError:
    # Fallback to absolute imports (for script usage)
    from infrastructure.config import Config
    from core.models.pretrain_model import PretrainableGNN
    from training.losses import MultiTaskLossComputer, LossOutput
    from infrastructure.scheduler import SchedulerManager
    from infrastructure.experiment_tracking import WandBTracker, MetricsLogger
    from infrastructure.checkpointing import CheckpointManager
    from data.data_loading import negative_sampling, compute_graph_properties


class MemoryMonitor:
    """
    Monitor GPU memory usage and provide recommendations for batch size adjustment.
    """
    
    def __init__(self, target_usage: float = 0.8):
        """
        Initialize memory monitor.
        
        Args:
            target_usage: Target GPU memory usage (0.0-1.0)
        """
        self.target_usage = target_usage
        self.memory_history = []
        self.oom_count = 0
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage statistics."""
        if not torch.cuda.is_available():
            return {'used': 0.0, 'total': 0.0, 'usage_ratio': 0.0}
        
        memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        usage_ratio = memory_used / memory_total if memory_total > 0 else 0.0
        
        return {
            'used': memory_used,
            'total': memory_total,
            'usage_ratio': usage_ratio
        }
    
    def should_reduce_batch_size(self) -> bool:
        """Check if batch size should be reduced based on memory usage."""
        memory_stats = self.get_memory_usage()
        return memory_stats['usage_ratio'] > self.target_usage
    
    def record_oom(self):
        """Record an out-of-memory event."""
        self.oom_count += 1
        logging.warning(f"OOM event recorded. Total OOM count: {self.oom_count}")
    
    def get_recommended_batch_size(self, current_batch_size: int, min_batch_size: int = 4) -> int:
        """
        Get recommended batch size based on memory usage.
        
        Args:
            current_batch_size: Current batch size
            min_batch_size: Minimum allowed batch size
            
        Returns:
            Recommended batch size
        """
        memory_stats = self.get_memory_usage()
        usage_ratio = memory_stats['usage_ratio']
        
        if usage_ratio > self.target_usage * 1.1:  # 10% buffer
            # Reduce batch size
            new_batch_size = max(min_batch_size, int(current_batch_size * 0.8))
            logging.info(f"High memory usage ({usage_ratio:.2%}), reducing batch size: {current_batch_size} -> {new_batch_size}")
            return new_batch_size
        elif usage_ratio < self.target_usage * 0.7:  # Room to increase
            # Increase batch size (but conservatively)
            new_batch_size = int(current_batch_size * 1.1)
            logging.info(f"Low memory usage ({usage_ratio:.2%}), increasing batch size: {current_batch_size} -> {new_batch_size}")
            return new_batch_size
        
        return current_batch_size


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 mode: str = 'min', restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for the monitored metric
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
        
        self.compare_fn = self._get_compare_function()
    
    def _get_compare_function(self):
        """Get comparison function based on mode."""
        if self.mode == 'min':
            return self._compare_min
        else:
            return self._compare_max
    
    def _compare_min(self, current, best):
        """Compare function for minimization."""
        return current < best - self.min_delta
    
    def _compare_max(self, current, best):
        """Compare function for maximization."""
        return current > best + self.min_delta
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop early.
        
        Args:
            score: Current validation score
            epoch: Current epoch
            
        Returns:
            True if training should stop early
        """
        if self.compare_fn(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            logging.info(f"Early stopping triggered after {self.patience} epochs without improvement")
            logging.info(f"Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
        
        return self.early_stop


class PretrainTrainer:
    """
    Main trainer class for GNN pre-training.
    
    This class orchestrates the entire training process including:
    - Model training and validation
    - Multi-task loss computation
    - Metric tracking and logging
    - Checkpointing and model saving
    - Early stopping
    """
    
    def __init__(self, 
                 config: Config,
                 model: PretrainableGNN,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 optimizer: torch.optim.Optimizer,
                 loss_computer: MultiTaskLossComputer,
                 scheduler_manager: SchedulerManager,
                 checkpoint_manager: CheckpointManager,
                 experiment_tracker: Optional[WandBTracker] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            loss_computer: Multi-task loss computer
            scheduler_manager: Learning rate and lambda scheduler manager
            checkpoint_manager: Checkpoint manager
            experiment_tracker: Experiment tracker (optional)
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_computer = loss_computer
        self.scheduler_manager = scheduler_manager
        self.checkpoint_manager = checkpoint_manager
        self.experiment_tracker = experiment_tracker
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode=config.training.metric_mode,
            restore_best_weights=True
        )
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_score = float('inf') if config.training.metric_mode == 'min' else float('-inf')
        
        # Mixed precision training
        self.use_amp = config.training.use_amp and AMP_AVAILABLE and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler(init_scale=config.training.amp_init_scale)
            logging.info("Mixed precision training enabled")
        else:
            self.scaler = None
            if config.training.use_amp and not AMP_AVAILABLE:
                logging.warning("Mixed precision requested but not available")
        
        # Calculate total steps
        if config.training.max_steps > 0:
            self.total_steps = config.training.max_steps
            self.total_epochs = config.training.max_steps // max(len(train_loader), 1)  # Avoid division by zero
        else:
            self.total_epochs = config.training.max_epochs
            self.total_steps = config.training.max_epochs * max(len(train_loader), 1)  # Avoid division by zero
        
        logging.info(f"Trainer initialized: {self.total_epochs} epochs, {self.total_steps} steps")
        
        # Memory monitoring for dynamic batch sizing
        self.memory_monitor = MemoryMonitor() if config.training.dynamic_batch_sizing else None
        
        # Performance profiler for optimization insights
        self.performance_profiler = self.PerformanceProfiler() if logging.getLogger().isEnabledFor(logging.DEBUG) else None
        
        # Validate trainer setup
        self._validate_trainer_setup()
    
    def _validate_trainer_setup(self):
        """Validate that all trainer components are properly configured."""
        # Check model is on correct device
        model_device = next(self.model.parameters()).device
        if model_device != self.config.device:
            logging.warning(f"Model is on {model_device}, but config specifies {self.config.device}")
        
        # Check data loader compatibility
        if len(self.train_loader) == 0:
            logging.warning("Training data loader is empty - this is expected if no processed data exists")
        
        # Check that we can iterate through at least one batch (if data exists)
        try:
            if len(self.train_loader) > 0:
                test_batch = next(iter(self.train_loader))
                logging.info("Data loader iteration test passed")
        except Exception as e:
            logging.warning(f"Data loader iteration test failed: {e}")
        
        # Check task configuration consistency
        enabled_tasks = [name for name, config in self.config.tasks.items() if config.enabled]
        if not enabled_tasks:
            raise ValueError("No tasks are enabled for training")
        
        # Check mixed precision compatibility
        if self.use_amp and not torch.cuda.is_available():
            logging.warning("Mixed precision enabled but CUDA not available - falling back to CPU")
        
        # Validate checkpoint directory
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        if not checkpoint_dir.parent.exists():
            logging.warning(f"Checkpoint parent directory doesn't exist: {checkpoint_dir.parent}")
        
        logging.info("Trainer validation completed successfully")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check of the entire training pipeline.
        
        Returns:
            Dictionary with health check results
        """
        health_results = {
            'overall_status': 'healthy',
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # 1. Model health check
            model_validation = self.model.validate_model_architecture()
            health_results['checks']['model_architecture'] = {
                'status': 'pass' if model_validation['architecture_valid'] else 'fail',
                'total_params': model_validation['total_parameters'],
                'issues': model_validation['issues']
            }
            
            if not model_validation['architecture_valid']:
                health_results['errors'].extend(model_validation['issues'])
            
            # 2. Data loader health check
            try:
                # Check if data loader has data
                if len(self.train_loader) == 0:
                    health_results['checks']['data_loading'] = {
                        'status': 'warning',
                        'batch_size': 0,
                        'data_type': 'empty',
                        'message': 'No training data available - expected if data not processed yet'
                    }
                    health_results['warnings'].append("Training data loader is empty")
                else:
                    # Test data loading
                    test_batch = next(iter(self.train_loader))
                    batch_data, domain_labels = test_batch
                    
                    health_results['checks']['data_loading'] = {
                        'status': 'pass',
                        'batch_size': len(domain_labels),
                        'data_type': 'list' if isinstance(batch_data, list) else 'batch'
                    }
                    
                    # Test forward pass
                    if isinstance(batch_data, list):
                        batch_data = [graph.to(self.config.device) for graph in batch_data]
                    else:
                        batch_data = batch_data.to(self.config.device)
                    domain_labels = domain_labels.to(self.config.device)
                    
                    # Quick forward pass test
                    with torch.no_grad():
                        domain_name = self.train_loader.dataset.get_domain_name(domain_labels[0].item())
                        if isinstance(batch_data, list):
                            test_output = self.model(batch_data[0], domain_name)
                        else:
                            # This shouldn't happen with current setup, but handle gracefully
                            test_output = self.model(batch_data.get_example(0), domain_name)
                    
                    health_results['checks']['forward_pass'] = {
                        'status': 'pass',
                        'output_keys': list(test_output.keys())
                    }
                
            except Exception as e:
                health_results['checks']['data_loading'] = {'status': 'fail', 'error': str(e)}
                health_results['errors'].append(f"Data loading test failed: {str(e)}")
            
            # 3. Loss computation health check
            try:
                # Test loss computation with dummy data
                dummy_outputs = {
                    'node_feat_mask': torch.randn(10, self.config.model.hidden_dim, device=self.config.device),
                    'link_pred': torch.randn(20, device=self.config.device),
                    'domain_adv': torch.randn(2, len(self.model.get_domain_list()), device=self.config.device)
                }
                dummy_targets = {
                    'node_feat_mask': torch.randn(10, self.config.model.hidden_dim, device=self.config.device),
                    'link_pred': torch.randint(0, 2, (20,), device=self.config.device),
                    'domain_adv': torch.randint(0, len(self.model.get_domain_list()), (2,), device=self.config.device)
                }
                
                loss_output = self.loss_computer.compute_losses(dummy_outputs, dummy_targets)
                
                health_results['checks']['loss_computation'] = {
                    'status': 'pass',
                    'total_loss': loss_output.total_loss.item(),
                    'individual_losses': len(loss_output.individual_losses),
                    'uncertainty_weighting': self.loss_computer.use_uncertainty_weighting
                }
                
            except Exception as e:
                health_results['checks']['loss_computation'] = {'status': 'fail', 'error': str(e)}
                health_results['errors'].append(f"Loss computation test failed: {str(e)}")
            
            # 4. Optimizer health check
            try:
                # Check optimizer state
                param_groups = len(self.optimizer.param_groups)
                total_params = sum(len(group['params']) for group in self.optimizer.param_groups)
                
                health_results['checks']['optimizer'] = {
                    'status': 'pass',
                    'param_groups': param_groups,
                    'total_params': total_params,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                
            except Exception as e:
                health_results['checks']['optimizer'] = {'status': 'fail', 'error': str(e)}
                health_results['errors'].append(f"Optimizer check failed: {str(e)}")
            
            # 5. Scheduler health check
            try:
                current_lr = self.scheduler_manager.get_current_lr()
                current_lambda = self.scheduler_manager.get_current_lambda()
                
                health_results['checks']['scheduler'] = {
                    'status': 'pass',
                    'current_lr': current_lr[0] if current_lr else 0,
                    'current_lambda': current_lambda
                }
                
            except Exception as e:
                health_results['checks']['scheduler'] = {'status': 'fail', 'error': str(e)}
                health_results['errors'].append(f"Scheduler check failed: {str(e)}")
            
            # 6. Memory and device checks
            try:
                device_info = {
                    'model_device': str(next(self.model.parameters()).device),
                    'config_device': str(self.config.device),
                    'cuda_available': torch.cuda.is_available(),
                    'amp_enabled': self.use_amp
                }
                
                if torch.cuda.is_available():
                    device_info['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
                    device_info['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
                
                health_results['checks']['device_memory'] = {
                    'status': 'pass',
                    **device_info
                }
                
                # Check device consistency
                if device_info['model_device'] != device_info['config_device']:
                    health_results['warnings'].append(f"Device mismatch: model on {device_info['model_device']}, config specifies {device_info['config_device']}")
                
            except Exception as e:
                health_results['checks']['device_memory'] = {'status': 'fail', 'error': str(e)}
                health_results['warnings'].append(f"Device/memory check failed: {str(e)}")
            
            # 7. Checkpointing health check
            try:
                checkpoint_dir = Path(self.config.training.checkpoint_dir)
                checkpoint_info = {
                    'dir_exists': checkpoint_dir.exists(),
                    'dir_writable': checkpoint_dir.parent.exists(),
                    'wandb_connected': self.checkpoint_manager.wandb_run is not None
                }
                
                health_results['checks']['checkpointing'] = {
                    'status': 'pass',
                    **checkpoint_info
                }
                
                if not checkpoint_info['dir_writable']:
                    health_results['warnings'].append(f"Checkpoint directory may not be writable: {checkpoint_dir}")
                
            except Exception as e:
                health_results['checks']['checkpointing'] = {'status': 'fail', 'error': str(e)}
                health_results['warnings'].append(f"Checkpointing check failed: {str(e)}")
            
            # Determine overall status
            failed_checks = [name for name, check in health_results['checks'].items() if check.get('status') == 'fail']
            if failed_checks:
                health_results['overall_status'] = 'unhealthy'
                health_results['errors'].append(f"Failed checks: {', '.join(failed_checks)}")
            elif health_results['warnings']:
                health_results['overall_status'] = 'warning'
            
        except Exception as e:
            health_results['overall_status'] = 'error'
            health_results['errors'].append(f"Health check failed: {str(e)}")
        
        return health_results
    
    def print_health_report(self):
        """Print a comprehensive health report."""
        health_results = self.health_check()
        
        print("\n" + "="*60)
        print("           TRAINING PIPELINE HEALTH CHECK")
        print("="*60)
        
        status_emoji = {
            'healthy': '🟢',
            'warning': '🟡', 
            'unhealthy': '🔴',
            'error': '💥'
        }
        
        print(f"Overall Status: {status_emoji.get(health_results['overall_status'], '❓')} {health_results['overall_status'].upper()}")
        
        print(f"\nComponent Checks:")
        for check_name, check_result in health_results['checks'].items():
            status = check_result.get('status', 'unknown')
            status_symbol = '✅' if status == 'pass' else '❌' if status == 'fail' else '❓'
            print(f"  {status_symbol} {check_name}: {status}")
            
            # Print additional info
            for key, value in check_result.items():
                if key != 'status' and key != 'error':
                    print(f"    {key}: {value}")
            
            if 'error' in check_result:
                print(f"    Error: {check_result['error']}")
        
        if health_results['warnings']:
            print(f"\n⚠️  Warnings ({len(health_results['warnings'])}):")
            for warning in health_results['warnings']:
                print(f"    - {warning}")
        
        if health_results['errors']:
            print(f"\n❌ Errors ({len(health_results['errors'])}):")
            for error in health_results['errors']:
                print(f"    - {error}")
        
        print("="*60)
        
        return health_results['overall_status'] in ['healthy', 'warning']

    def _save_training_progress(self, epoch: int, metrics: Dict[str, float]):
        """Save training progress to a JSON file for monitoring."""
        try:
            progress_file = Path(self.config.training.checkpoint_dir) / 'training_progress.json'
            
            progress_data = {
                'current_epoch': epoch,
                'current_step': self.current_step,
                'total_epochs': self.total_epochs,
                'total_steps': self.total_steps,
                'best_val_score': self.best_val_score,
                'early_stopping_counter': self.early_stopping.counter,
                'metrics_history': getattr(self, '_metrics_history', []),
                'timestamp': time.time()
            }
            
            # Add current metrics
            progress_data['current_metrics'] = metrics
            
            # Maintain metrics history (last 10 epochs)
            if not hasattr(self, '_metrics_history'):
                self._metrics_history = []
            
            self._metrics_history.append({
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': time.time()
            })
            
            # Keep only last 10 entries
            if len(self._metrics_history) > 10:
                self._metrics_history = self._metrics_history[-10:]
            
            progress_data['metrics_history'] = self._metrics_history
            
            # Save to file
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2, default=str)
                
        except Exception as e:
            logging.warning(f"Failed to save training progress: {e}")
    
    class PerformanceProfiler:
        """
        Profile training performance to identify bottlenecks.
        """
        
        def __init__(self):
            self.timings = {}
            self.step_count = 0
            
        def start_timer(self, name: str):
            """Start timing an operation."""
            self.timings[name] = time.time()
            
        def end_timer(self, name: str) -> float:
            """End timing an operation and return duration."""
            if name in self.timings:
                duration = time.time() - self.timings[name]
                del self.timings[name]
                return duration
            return 0.0
            
        def log_performance_summary(self, step: int):
            """Log performance summary every N steps."""
            if step % 100 == 0 and self.step_count > 0:
                logging.debug("=== Performance Summary ===")
                # Add performance metrics logging here
                logging.debug(f"Steps processed: {self.step_count}")
    
    def _validate_resumption_compatibility(self, resume_info: Dict[str, Any]) -> bool:
        """
        Validate that the current configuration is compatible with resumed training.
        
        Args:
            resume_info: Information from resumed checkpoint
            
        Returns:
            True if compatible, False otherwise
        """
        if not resume_info.get('resumed', False):
            return True
            
        # Check if model architecture matches
        if 'config' in resume_info:
            saved_config = resume_info['config']
            current_model_config = {
                'hidden_dim': self.config.model.hidden_dim,
                'num_layers': self.config.model.num_layers,
                'dropout_rate': self.config.model.dropout_rate
            }
            
            if isinstance(saved_config, dict) and 'model' in saved_config:
                saved_model_config = saved_config['model']
                for key, value in current_model_config.items():
                    if key in saved_model_config and saved_model_config[key] != value:
                        logging.warning(f"Model config mismatch: {key} was {saved_model_config[key]}, now {value}")
                        return False
        
        return True
    
    def train(self) -> Dict[str, Any]:
        """
        Run the complete training process.
        
        Returns:
            Dictionary with training results and statistics
        """
        logging.info("Starting training...")
        
        # Log model info
        if self.experiment_tracker:
            self.experiment_tracker.log_model_info(self.model)
        
        # Training loop
        training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.total_epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self._train_epoch()
                
                # Validation phase
                if self.val_loader is not None and epoch % self.config.training.validation_freq == 0:
                    val_metrics = self._validate_epoch()
                else:
                    val_metrics = {}
                
                # Log learning curves
                if self.experiment_tracker and val_metrics:
                    self.experiment_tracker.log_learning_curves(train_metrics, val_metrics, epoch)
                
                # Update schedulers
                self.scheduler_manager.step(epoch=epoch, step=self.current_step)
                
                # Log scheduler states
                if self.experiment_tracker:
                    scheduler_metrics = {
                        'scheduler/lr': self.scheduler_manager.get_current_lr()[0],
                        'scheduler/lambda_da': self.scheduler_manager.get_current_lambda()
                    }
                    self.experiment_tracker.log_metrics(scheduler_metrics, epoch=epoch)
                
                # Check early stopping
                if val_metrics and self.config.training.validation_metric in val_metrics:
                    val_score = val_metrics[self.config.training.validation_metric]
                    should_stop = self.early_stopping(val_score, epoch)
                    
                    if should_stop:
                        logging.info("Early stopping triggered")
                        break
                
                # Save checkpoint
                if epoch % self.config.training.save_checkpoint_freq == 0:
                    self._save_checkpoint(epoch, {**train_metrics, **val_metrics})
                
                # Save training progress
                combined_metrics = {**train_metrics, **val_metrics}
                self._save_training_progress(epoch, combined_metrics)
                
                # Check if we've reached max steps
                if self.config.training.max_steps > 0 and self.current_step >= self.total_steps:
                    logging.info(f"Reached maximum steps: {self.total_steps}")
                    break
        
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except Exception as e:
            logging.error(f"Training failed with error: {str(e)}")
            raise
        
        # Save final checkpoint
        final_metrics = {**train_metrics, **val_metrics} if 'val_metrics' in locals() else train_metrics
        self._save_checkpoint(self.current_epoch, final_metrics, is_final=True)
        
        # Save final progress
        if 'final_metrics' in locals():
            self._save_training_progress(self.current_epoch, final_metrics)
        
        # Training summary
        training_time = time.time() - training_start_time
        
        training_results = {
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.current_step,
            'training_time': training_time,
            'best_val_score': self.best_val_score,
            'best_epoch': self.early_stopping.best_epoch,
            'early_stopped': self.early_stopping.early_stop
        }
        
        logging.info(f"Training completed in {training_time:.2f} seconds")
        logging.info(f"Best validation score: {self.best_val_score:.6f} at epoch {self.early_stopping.best_epoch}")
        
        return training_results
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        metrics_logger = MetricsLogger()
        
        epoch_start_time = time.time()
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}', leave=False)
        
        for batch_idx, (batch_data, domain_labels) in enumerate(pbar):
            batch_start_time = time.time()
            
            # Move data to device
            if isinstance(batch_data, list):
                # Handle list of individual graphs
                batch_data = [graph.to(self.config.device) for graph in batch_data]
            else:
                # Handle single batched graph (fallback)
                batch_data = batch_data.to(self.config.device)
            domain_labels = domain_labels.to(self.config.device)
            
            # Forward pass and compute losses
            if self.use_amp:
                with autocast():
                    loss_output = self._compute_batch_losses(batch_data, domain_labels)
            else:
                loss_output = self._compute_batch_losses(batch_data, domain_labels)
            
            # Scale loss for gradient accumulation
            if self.config.training.gradient_accumulation_steps > 1:
                loss_output.total_loss = loss_output.total_loss / self.config.training.gradient_accumulation_steps
            
            # Check for valid loss
            if torch.isnan(loss_output.total_loss) or torch.isinf(loss_output.total_loss):
                logging.warning(f"Invalid loss detected: {loss_output.total_loss.item()}")
                continue
            
            # Check for loss spikes (potential training instability)
            if hasattr(self, '_last_loss') and self._last_loss is not None:
                loss_ratio = loss_output.total_loss.item() / self._last_loss
                if loss_ratio > 5.0:  # Loss spike detection
                    logging.warning(f"Large loss spike detected: {self._last_loss:.4f} -> {loss_output.total_loss.item():.4f}")
                    if self.use_amp and self.scaler.get_scale() > 1024:
                        # Reduce AMP scale to prevent numerical instability
                        self.scaler.update(new_scale=self.scaler.get_scale() * 0.5)
                        logging.info(f"Reduced AMP scale to {self.scaler.get_scale()}")
            
            self._last_loss = loss_output.total_loss.item()
            
            # Backward pass (accumulate gradients)
            if self.use_amp:
                # Scaled backward pass
                self.scaler.scale(loss_output.total_loss).backward()
            else:
                # Standard backward pass
                loss_output.total_loss.backward()
            
            # Perform optimizer step with gradient accumulation
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping with scaled gradients
                    if self.config.training.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
                    
                    # Optimizer step with scaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    if self.config.training.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                
                # Zero gradients after accumulation
                self.optimizer.zero_grad()
                
                # Step schedulers (only after actual optimizer step)
                self.scheduler_manager.step(epoch=self.current_epoch, step=self.current_step)
            
            # Update metrics
            batch_metrics = self._extract_metrics(loss_output, prefix='train')
            batch_metrics['train/batch_time'] = time.time() - batch_start_time
            
            # Handle both list and batched data formats
            if isinstance(batch_data, list):
                num_graphs = len(batch_data)
            else:
                num_graphs = batch_data.num_graphs
            metrics_logger.update(batch_metrics, count=num_graphs)
            
            # Log to experiment tracker
            if (self.experiment_tracker and 
                self.current_step % self.config.training.log_freq == 0):
                
                step_metrics = batch_metrics.copy()
                step_metrics.update({
                    'train/lr': self.scheduler_manager.get_current_lr()[0],
                    'train/lambda_da': self.scheduler_manager.get_current_lambda()
                })
                
                self.experiment_tracker.log_metrics(step_metrics, step=self.current_step)
                
                # Log system metrics occasionally
                if self.current_step % (self.config.training.log_freq * 10) == 0:
                    self.experiment_tracker.log_system_metrics()
                    
                    # Log memory usage if monitoring is enabled
                    if self.memory_monitor:
                        memory_stats = self.memory_monitor.get_memory_usage()
                        memory_metrics = {
                            'system/gpu_memory_used_gb': memory_stats['used'],
                            'system/gpu_memory_total_gb': memory_stats['total'],
                            'system/gpu_memory_usage_ratio': memory_stats['usage_ratio']
                        }
                        self.experiment_tracker.log_metrics(memory_metrics, step=self.current_step)
                    
                    # Clear cache to prevent memory buildup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Update progress bar
            current_metrics = metrics_logger.compute_averages()
            pbar.set_postfix({
                'loss': f"{current_metrics.get('train/loss_total', 0):.4f}",
                'lr': f"{self.scheduler_manager.get_current_lr()[0]:.2e}"
            })
            
            self.current_step += 1
            
            # Early exit if max steps reached
            if self.config.training.max_steps > 0 and self.current_step >= self.total_steps:
                break
        
        # Compute epoch averages
        epoch_metrics = metrics_logger.compute_averages()
        epoch_metrics['train/epoch_time'] = time.time() - epoch_start_time
        
        return epoch_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """
        Run one validation epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        metrics_logger = MetricsLogger()
        
        # Collect all outputs for comprehensive metrics
        all_model_outputs = {task: [] for task in ['link_pred', 'domain_adv', 'graph_contrast']}
        all_targets = {task: [] for task in ['link_pred', 'domain_adv', 'graph_contrast']}
        
        with torch.no_grad():
            for batch_data, domain_labels in tqdm(self.val_loader, desc='Validation', leave=False):
                # Move data to device
                if isinstance(batch_data, list):
                    # Handle list of individual graphs
                    batch_data = [graph.to(self.config.device) for graph in batch_data]
                else:
                    # Handle single batched graph (fallback)
                    batch_data = batch_data.to(self.config.device)
                domain_labels = domain_labels.to(self.config.device)
                
                # Forward pass and compute losses
                if self.use_amp:
                    with autocast():
                        loss_output = self._compute_batch_losses(batch_data, domain_labels)
                        # Get model outputs for metrics
                        model_outputs, targets = self._get_model_outputs_and_targets(batch_data, domain_labels)
                else:
                    loss_output = self._compute_batch_losses(batch_data, domain_labels)
                    # Get model outputs for metrics
                    model_outputs, targets = self._get_model_outputs_and_targets(batch_data, domain_labels)
                
                # Collect outputs for comprehensive metrics
                for task in all_model_outputs:
                    if task in model_outputs:
                        all_model_outputs[task].append(model_outputs[task])
                        all_targets[task].append(targets[task])
                
                # Update metrics
                batch_metrics = self._extract_metrics(loss_output, prefix='val')
                # Handle both list and batched data formats
                if isinstance(batch_data, list):
                    num_graphs = len(batch_data)
                else:
                    num_graphs = batch_data.num_graphs
                metrics_logger.update(batch_metrics, count=num_graphs)
        
        # Compute epoch averages
        val_metrics = metrics_logger.compute_averages()
        
        # Compute comprehensive validation metrics
        final_model_outputs = {}
        final_targets = {}
        for task in all_model_outputs:
            if all_model_outputs[task]:
                if task == 'domain_adv':
                    final_model_outputs[task] = torch.cat(all_model_outputs[task], dim=0)
                    final_targets[task] = torch.cat(all_targets[task], dim=0)
                else:
                    final_model_outputs[task] = torch.cat(all_model_outputs[task], dim=0)
                    final_targets[task] = torch.cat(all_targets[task], dim=0)
        
        comprehensive_metrics = self._compute_validation_metrics(final_model_outputs, final_targets)
        val_metrics.update(comprehensive_metrics)
        
        # Update best validation score
        if self.config.training.validation_metric in val_metrics:
            val_score = val_metrics[self.config.training.validation_metric]
            is_better = (
                (self.config.training.metric_mode == 'min' and val_score < self.best_val_score) or
                (self.config.training.metric_mode == 'max' and val_score > self.best_val_score)
            )
            
            if is_better:
                self.best_val_score = val_score
        
        return val_metrics
    
    def _compute_batch_losses(self, batch_data, domain_labels) -> LossOutput:
        """
        Compute all losses for a batch efficiently with minimal forward passes.
        
        Args:
            batch_data: List of individual graph data objects
            domain_labels: Domain labels for graphs in batch
            
        Returns:
            LossOutput containing all computed losses
        """
        model_outputs = {}
        targets = {}
        
        # Single forward pass per graph to get all embeddings
        graph_outputs = []
        for graph, domain_label in zip(batch_data, domain_labels):
            domain_name = self.train_loader.dataset.get_domain_name(domain_label.item())
            outputs = self.model(graph, domain_name)
            graph_outputs.append((outputs, domain_name, graph))
        
        # Task 1: Node Feature Masking
        if self.config.tasks['node_feat_mask'].enabled:
            mask_outputs = []
            mask_targets = []
            
            for i, (graph, domain_label) in enumerate(zip(batch_data, domain_labels)):
                domain_name = self.train_loader.dataset.get_domain_name(domain_label.item())
                
                masked_data, mask_indices, target_h0 = self.model.apply_node_masking(
                    graph, domain_name, 
                    mask_rate=self.config.tasks['node_feat_mask'].mask_rate
                )
                
                if len(mask_indices) > 0:
                    # Forward pass on masked data
                    masked_outputs = self.model(masked_data, domain_name)
                    
                    # Get predictions for masked nodes
                    mask_head = self.model.get_head('node_feat_mask')
                    predictions = mask_head(masked_outputs['node_embeddings'][mask_indices])
                    
                    mask_outputs.append(predictions)
                    mask_targets.append(target_h0)
            
            if mask_outputs:
                model_outputs['node_feat_mask'] = torch.cat(mask_outputs, dim=0)
                targets['node_feat_mask'] = torch.cat(mask_targets, dim=0)
        
        # Task 2: Link Prediction
        if self.config.tasks['link_pred'].enabled:
            link_outputs = []
            link_targets = []
            
            for (outputs, domain_name, graph), domain_label in zip(graph_outputs, domain_labels):
                node_embeddings = outputs['node_embeddings']
                
                # Generate positive and negative edges
                pos_edges = graph.edge_index
                neg_edges = negative_sampling(
                    pos_edges, 
                    graph.num_nodes,
                    num_neg_samples=int(pos_edges.shape[1] * self.config.tasks['link_pred'].negative_sampling_ratio)
                )
                
                # Combine edges and create labels
                all_edges = torch.cat([pos_edges, neg_edges], dim=1)
                edge_labels = torch.cat([
                    torch.ones(pos_edges.shape[1], device=self.config.device),
                    torch.zeros(neg_edges.shape[1], device=self.config.device)
                ])
                
                # Get edge predictions
                link_head = self.model.get_head('link_pred')
                edge_logits = link_head(node_embeddings, all_edges)
                
                link_outputs.append(edge_logits)
                link_targets.append(edge_labels)
            
            if link_outputs:
                model_outputs['link_pred'] = torch.cat(link_outputs, dim=0)
                targets['link_pred'] = torch.cat(link_targets, dim=0)
        
        # Task 3: Node Contrastive Learning
        if self.config.tasks['node_contrast'].enabled:
            contrast_outputs = []
            
            for (_, domain_name, graph), domain_label in zip(graph_outputs, domain_labels):
                # Create two augmented views
                aug1, aug2 = self.model.create_augmented_views(graph, num_views=2)
                
                # Forward pass on augmented views
                out1 = self.model(aug1, domain_name)
                out2 = self.model(aug2, domain_name)
                
                # Project to contrastive space
                contrast_head = self.model.get_head('node_contrast')
                z1 = contrast_head(out1['node_embeddings'])
                z2 = contrast_head(out2['node_embeddings'])
                
                contrast_outputs.append((z1, z2))
            
            if contrast_outputs:
                all_z1 = torch.cat([z1 for z1, z2 in contrast_outputs], dim=0)
                all_z2 = torch.cat([z2 for z1, z2 in contrast_outputs], dim=0)
                model_outputs['node_contrast'] = (all_z1, all_z2)
        
        # Task 4: Graph Contrastive Learning (Optimized)
        if self.config.tasks['graph_contrast'].enabled:
            pos_scores = []
            neg_scores = []
            
            graph_contrast_head = self.model.get_head('graph_contrast')
            
            # Pre-compute all graph embeddings
            all_graph_embeddings = [outputs['graph_embedding'] for outputs, _, _ in graph_outputs]
            
            for i, (outputs, domain_name, graph) in enumerate(graph_outputs):
                node_embeddings = outputs['node_embeddings']
                graph_embedding = outputs['graph_embedding']
                
                # Positive pairs: nodes from this graph with this graph's summary
                graph_emb_expanded = graph_embedding.unsqueeze(0).expand(node_embeddings.shape[0], -1)
                pos_scores_i = graph_contrast_head(node_embeddings, graph_emb_expanded)
                pos_scores.append(pos_scores_i)
                
                # Negative pairs: nodes from this graph with other graphs' summaries
                for j, other_graph_embedding in enumerate(all_graph_embeddings):
                    if i != j:
                        other_graph_emb_expanded = other_graph_embedding.unsqueeze(0).expand(node_embeddings.shape[0], -1)
                        neg_scores_i = graph_contrast_head(node_embeddings, other_graph_emb_expanded)
                        neg_scores.append(neg_scores_i)
            
            if pos_scores and neg_scores:
                all_pos_scores = torch.cat(pos_scores, dim=0)
                all_neg_scores = torch.cat(neg_scores, dim=0)
                
                all_scores = torch.cat([all_pos_scores, all_neg_scores], dim=0)
                all_labels = torch.cat([
                    torch.ones(all_pos_scores.shape[0], device=self.config.device),
                    torch.zeros(all_neg_scores.shape[0], device=self.config.device)
                ])
                
                model_outputs['graph_contrast'] = all_scores
                targets['graph_contrast'] = all_labels
        
        # Task 5: Graph Property Prediction
        if self.config.tasks['graph_prop'].enabled:
            graph_predictions = []
            graph_props = []
            
            for (outputs, domain_name, graph), domain_label in zip(graph_outputs, domain_labels):
                graph_embedding = outputs['graph_embedding']
                
                # Get predictions
                prop_head = self.model.get_head('graph_prop')
                pred = prop_head(graph_embedding)
                graph_predictions.append(pred)
                
                # Compute true properties
                props = compute_graph_properties(graph)
                graph_props.append(props)
            
            if graph_predictions:
                model_outputs['graph_prop'] = torch.stack(graph_predictions, dim=0)
                targets['graph_prop'] = torch.stack(graph_props, dim=0)
        
        # Task 6: Domain Adversarial Training
        if self.config.tasks['domain_adv'].enabled:
            domain_predictions = []
            lambda_da = self.scheduler_manager.get_current_lambda()
            
            for (outputs, domain_name, graph), domain_label in zip(graph_outputs, domain_labels):
                graph_embedding = outputs['graph_embedding']
                
                # Apply gradient reversal
                reversed_embedding = self.model.apply_gradient_reversal(graph_embedding, lambda_da)
                
                # Predict domain
                domain_head = self.model.get_head('domain_adv')
                domain_logits = domain_head(reversed_embedding)
                domain_predictions.append(domain_logits)
            
            if domain_predictions:
                model_outputs['domain_adv'] = torch.stack(domain_predictions, dim=0)
                targets['domain_adv'] = domain_labels
        
        # Compute losses with domain adversarial lambda
        lambda_da = self.scheduler_manager.get_current_lambda()
        loss_output = self.loss_computer.compute_losses(model_outputs, targets, lambda_da=lambda_da)
        return loss_output
    
    def _get_model_outputs_and_targets(self, batch_data, domain_labels) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get model outputs and targets for validation metrics computation.
        This is a simplified version that focuses on classification tasks.
        
        Args:
            batch_data: List of individual graph data objects
            domain_labels: Domain labels for graphs in batch
            
        Returns:
            Tuple of (model_outputs, targets) for validation metrics
        """
        model_outputs = {}
        targets = {}
        
        # Single forward pass per graph to get embeddings
        graph_outputs = []
        for graph, domain_label in zip(batch_data, domain_labels):
            domain_name = self.train_loader.dataset.get_domain_name(domain_label.item())
            outputs = self.model(graph, domain_name)
            graph_outputs.append((outputs, domain_name, graph))
        
        # Link Prediction outputs
        if self.config.tasks['link_pred'].enabled:
            link_outputs = []
            link_targets = []
            
            for (outputs, domain_name, graph), domain_label in zip(graph_outputs, domain_labels):
                node_embeddings = outputs['node_embeddings']
                
                # Generate edges (simplified for validation)
                pos_edges = graph.edge_index
                if pos_edges.shape[1] > 0:
                    neg_edges = negative_sampling(pos_edges, graph.num_nodes, 
                                                num_neg_samples=min(pos_edges.shape[1], 50))  # Limit for efficiency
                    
                    all_edges = torch.cat([pos_edges, neg_edges], dim=1)
                    edge_labels = torch.cat([
                        torch.ones(pos_edges.shape[1], device=self.config.device),
                        torch.zeros(neg_edges.shape[1], device=self.config.device)
                    ])
                    
                    link_head = self.model.get_head('link_pred')
                    edge_logits = link_head(node_embeddings, all_edges)
                    
                    link_outputs.append(edge_logits)
                    link_targets.append(edge_labels)
            
            if link_outputs:
                model_outputs['link_pred'] = torch.cat(link_outputs, dim=0)
                targets['link_pred'] = torch.cat(link_targets, dim=0)
        
        # Domain Adversarial outputs
        if self.config.tasks['domain_adv'].enabled:
            domain_predictions = []
            lambda_da = self.scheduler_manager.get_current_lambda()
            
            for (outputs, domain_name, graph), domain_label in zip(graph_outputs, domain_labels):
                graph_embedding = outputs['graph_embedding']
                reversed_embedding = self.model.apply_gradient_reversal(graph_embedding, lambda_da)
                
                domain_head = self.model.get_head('domain_adv')
                domain_logits = domain_head(reversed_embedding)
                domain_predictions.append(domain_logits)
            
            if domain_predictions:
                model_outputs['domain_adv'] = torch.stack(domain_predictions, dim=0)
                targets['domain_adv'] = domain_labels
        
        # Graph Contrastive outputs (simplified)
        if self.config.tasks['graph_contrast'].enabled:
            pos_scores = []
            neg_scores = []
            
            if len(graph_outputs) > 1:  # Need at least 2 graphs for contrastive
                graph_contrast_head = self.model.get_head('graph_contrast')
                all_graph_embeddings = [outputs['graph_embedding'] for outputs, _, _ in graph_outputs]
                
                for i, (outputs, domain_name, graph) in enumerate(graph_outputs[:2]):  # Limit for efficiency
                    node_embeddings = outputs['node_embeddings'][:10]  # Limit nodes for efficiency
                    graph_embedding = outputs['graph_embedding']
                    
                    # Positive pairs
                    graph_emb_expanded = graph_embedding.unsqueeze(0).expand(node_embeddings.shape[0], -1)
                    pos_scores_i = graph_contrast_head(node_embeddings, graph_emb_expanded)
                    pos_scores.append(pos_scores_i)
                    
                    # Negative pairs (with next graph)
                    j = (i + 1) % len(all_graph_embeddings)
                    other_graph_embedding = all_graph_embeddings[j]
                    other_graph_emb_expanded = other_graph_embedding.unsqueeze(0).expand(node_embeddings.shape[0], -1)
                    neg_scores_i = graph_contrast_head(node_embeddings, other_graph_emb_expanded)
                    neg_scores.append(neg_scores_i)
                
                if pos_scores and neg_scores:
                    all_pos_scores = torch.cat(pos_scores, dim=0)
                    all_neg_scores = torch.cat(neg_scores, dim=0)
                    
                    all_scores = torch.cat([all_pos_scores, all_neg_scores], dim=0)
                    all_labels = torch.cat([
                        torch.ones(all_pos_scores.shape[0], device=self.config.device),
                        torch.zeros(all_neg_scores.shape[0], device=self.config.device)
                    ])
                    
                    model_outputs['graph_contrast'] = all_scores
                    targets['graph_contrast'] = all_labels
        
        return model_outputs, targets
    
    def _extract_metrics(self, loss_output: LossOutput, prefix: str = '') -> Dict[str, float]:
        """
        Extract metrics from loss output.
        
        Args:
            loss_output: Loss computation results
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Total loss
        metrics[f'{prefix}/loss_total'] = loss_output.total_loss.item()
        
        # Individual losses
        for task_name, loss_value in loss_output.individual_losses.items():
            metrics[f'{prefix}/loss_{task_name}'] = loss_value.item()
        
        # Weighted losses
        for task_name, loss_value in loss_output.weighted_losses.items():
            metrics[f'{prefix}/weighted_loss_{task_name}'] = loss_value.item()
        
        # Uncertainty parameters
        if loss_output.uncertainty_params:
            for task_name, sigma in loss_output.uncertainty_params.items():
                metrics[f'{prefix}/uncertainty_{task_name}'] = sigma
        
        return metrics
    
    def _compute_validation_metrics(self, model_outputs: Dict[str, Any], 
                                   targets: Dict[str, Any], prefix: str = 'val') -> Dict[str, float]:
        """
        Compute comprehensive validation metrics including accuracy, F1, and AUC.
        
        Args:
            model_outputs: Model predictions
            targets: Ground truth targets
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of validation metrics
        """
        metrics = {}
        
        # Link prediction metrics
        if 'link_pred' in model_outputs and 'link_pred' in targets:
            logits = model_outputs['link_pred']
            labels = targets['link_pred']
            
            if len(logits) > 0:
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                labels_np = labels.detach().cpu().numpy()
                
                try:
                    metrics[f'{prefix}/link_pred_accuracy'] = accuracy_score(labels_np, preds)
                    metrics[f'{prefix}/link_pred_f1'] = f1_score(labels_np, preds, average='binary')
                    if len(np.unique(labels_np)) > 1:  # AUC requires both classes
                        metrics[f'{prefix}/link_pred_auc'] = roc_auc_score(labels_np, probs)
                except Exception as e:
                    logging.warning(f"Error computing link prediction metrics: {e}")
        
        # Domain adversarial metrics
        if 'domain_adv' in model_outputs and 'domain_adv' in targets:
            logits = model_outputs['domain_adv']
            labels = targets['domain_adv']
            
            if len(logits) > 0:
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                preds = np.argmax(probs, axis=1)
                labels_np = labels.detach().cpu().numpy()
                
                try:
                    metrics[f'{prefix}/domain_adv_accuracy'] = accuracy_score(labels_np, preds)
                    metrics[f'{prefix}/domain_adv_f1'] = f1_score(labels_np, preds, average='macro')
                except Exception as e:
                    logging.warning(f"Error computing domain adversarial metrics: {e}")
        
        # Graph contrastive metrics
        if 'graph_contrast' in model_outputs and 'graph_contrast' in targets:
            logits = model_outputs['graph_contrast']
            labels = targets['graph_contrast']
            
            if len(logits) > 0:
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                labels_np = labels.detach().cpu().numpy()
                
                try:
                    metrics[f'{prefix}/graph_contrast_accuracy'] = accuracy_score(labels_np, preds)
                    metrics[f'{prefix}/graph_contrast_f1'] = f1_score(labels_np, preds, average='binary')
                except Exception as e:
                    logging.warning(f"Error computing graph contrastive metrics: {e}")
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_final: bool = False):
        """
        Save checkpoint using the checkpoint manager.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_final: Whether this is the final checkpoint
        """
        try:
            # Check if this is the best model
            is_best = False
            if self.config.training.validation_metric in metrics:
                current_score = metrics[self.config.training.validation_metric]
                if self.config.training.metric_mode == 'min':
                    is_best = current_score < self.best_val_score
                else:
                    is_best = current_score > self.best_val_score
                    
                if is_best:
                    self.best_val_score = current_score
                    logging.info(f"New best model saved! {self.config.training.validation_metric}: {current_score}")
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler_manager=self.scheduler_manager,
                metrics=metrics,
                is_best=is_best,
                is_final=is_final
            )
            
            if checkpoint_path and self.experiment_tracker:
                # Log checkpoint info to experiment tracker
                self.experiment_tracker.log_metrics({
                    'checkpoint/epoch': epoch,
                    'checkpoint/is_best': is_best,
                    'checkpoint/is_final': is_final
                }, step=self.current_step)
                
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file (loads latest if None)
        """
        resume_info = self.checkpoint_manager.resume_training(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler_manager,
            loss_computer=self.loss_computer,
            checkpoint_path=checkpoint_path
        )
        
        # Update training state
        self.current_epoch = resume_info.get('epoch', 0)
        self.current_step = resume_info.get('current_step', 0)
        self.best_val_score = resume_info.get('best_val_score', self.best_val_score)
        
        # Restore early stopping state if available
        if 'early_stopping_state' in resume_info:
            es_state = resume_info['early_stopping_state']
            for key, value in es_state.items():
                if hasattr(self.early_stopping, key):
                    setattr(self.early_stopping, key, value)
        
        logging.info(f"Training resumed from epoch {self.current_epoch}, step {self.current_step}")


if __name__ == '__main__':
    # This would be used by the main training script
    logging.info("Trainer module loaded successfully!")
    print("Use this trainer with the main training script.") 