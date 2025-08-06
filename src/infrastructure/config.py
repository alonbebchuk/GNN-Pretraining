"""
Configuration management system for GNN training pipeline.

This module provides comprehensive configuration parsing, validation, and
type-safe access to all training parameters from YAML files.
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import warnings


@dataclass
class RunConfig:
    """Configuration for run details and experiment tracking."""
    project_name: str = 'Graph-Multitask-Learning'
    run_name: str = 'default-run'
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    pretrain_datasets: List[str] = field(default_factory=lambda: ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES'])
    batch_size: int = 32
    domain_balanced_sampling: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = True


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    hidden_dim: int = 256
    num_layers: int = 5
    dropout_rate: float = 0.2
    enable_augmentations: bool = True


@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings."""
    name: str = 'AdamW'
    lr: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduling."""
    type: str = 'cosine'  # 'cosine', 'linear', 'step', 'none'
    warmup_epochs: int = 10
    warmup_fraction: float = 0.1  # Fraction of total steps for warmup
    min_lr_ratio: float = 0.01  # Minimum LR as fraction of initial LR
    step_size: int = 50  # For step scheduler
    gamma: float = 0.1  # For step scheduler


@dataclass
class TaskConfig:
    """Configuration for individual training tasks."""
    enabled: bool = True
    weight: float = 1.0
    loss_type: str = 'auto'  # 'auto', 'BCE', 'CE', 'MSE', 'NT-Xent'
    
    # Task-specific parameters
    mask_rate: float = 0.15  # For node feature masking
    negative_sampling_ratio: float = 1.0  # For link prediction
    temperature: float = 0.1  # For contrastive learning
    projection_dim: int = 128  # For contrastive learning


@dataclass
class DomainAdversarialConfig:
    """Configuration for domain-adversarial training."""
    enabled: bool = True
    schedule_type: str = 'dann'  # 'dann', 'linear', 'constant'
    initial_lambda: float = 0.0
    final_lambda: float = 1.0
    gamma: float = 10.0  # For DANN schedule
    warmup_epochs: int = 0


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    max_epochs: int = 100
    max_steps: int = 100000  # Will be used if > 0, otherwise use max_epochs
    validation_freq: int = 1  # Validate every N epochs
    log_freq: int = 100  # Log metrics every N steps
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    validation_metric: str = 'val/loss_total'
    metric_mode: str = 'min'  # 'min' or 'max'
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_checkpoint_freq: int = 10
    keep_n_checkpoints: int = 3
    
    # Uncertainty weighting
    use_uncertainty_weighting: bool = True
    uncertainty_init: float = 0.0  # Initial log(sigma^2)
    
    # Gradient clipping
    grad_clip_norm: float = 0.0  # 0 means no clipping
    
    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision
    amp_init_scale: float = 65536.0  # Initial loss scaling for AMP
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps
    
    # Memory optimization
    dynamic_batch_sizing: bool = False  # Automatically adjust batch size based on GPU memory
    max_memory_usage: float = 0.8  # Maximum GPU memory usage (0.0-1.0)
    min_batch_size: int = 4  # Minimum batch size for dynamic sizing


@dataclass
class Config:
    """Main configuration container."""
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    domain_adversarial: DomainAdversarialConfig = field(default_factory=DomainAdversarialConfig)
    
    # Task configurations
    tasks: Dict[str, TaskConfig] = field(default_factory=lambda: {
        'node_feat_mask': TaskConfig(enabled=True, weight=1.0, mask_rate=0.15),
        'link_pred': TaskConfig(enabled=True, weight=1.0, negative_sampling_ratio=1.0),
        'node_contrast': TaskConfig(enabled=True, weight=1.0, temperature=0.1),
        'graph_contrast': TaskConfig(enabled=True, weight=1.0),
        'graph_prop': TaskConfig(enabled=True, weight=1.0),
        'domain_adv': TaskConfig(enabled=True, weight=1.0)
    })
    
    # Computed properties
    device: torch.device = field(init=False)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        # Validate batch size
        if self.data.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.data.batch_size}")
        
        # Validate model parameters
        if self.model.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.model.hidden_dim}")
        
        if self.model.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.model.num_layers}")
        
        if not 0 <= self.model.dropout_rate <= 1:
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.model.dropout_rate}")
        
        # Validate optimizer
        if self.optimizer.lr <= 0:
            raise ValueError(f"learning rate must be positive, got {self.optimizer.lr}")
        
        # Validate training parameters
        if self.training.max_epochs <= 0 and self.training.max_steps <= 0:
            raise ValueError("Either max_epochs or max_steps must be positive")
        
        if self.training.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.training.patience}")
        
        # Validate metric mode
        if self.training.metric_mode not in ['min', 'max']:
            raise ValueError(f"metric_mode must be 'min' or 'max', got {self.training.metric_mode}")
        
        # Validate task configurations
        enabled_tasks = [name for name, config in self.tasks.items() if config.enabled]
        if not enabled_tasks:
            warnings.warn("No tasks are enabled in the configuration")
        
        # Validate domain adversarial config
        if self.domain_adversarial.enabled:
            if self.domain_adversarial.schedule_type not in ['dann', 'linear', 'constant']:
                raise ValueError(f"Invalid domain adversarial schedule type: {self.domain_adversarial.schedule_type}")
        
        # Validate gradient accumulation
        if self.training.gradient_accumulation_steps < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {self.training.gradient_accumulation_steps}")
        
        # Validate memory optimization settings
        if self.training.dynamic_batch_sizing:
            if not 0.1 <= self.training.max_memory_usage <= 1.0:
                raise ValueError(f"max_memory_usage must be between 0.1 and 1.0, got {self.training.max_memory_usage}")
            if self.training.min_batch_size < 1:
                raise ValueError(f"min_batch_size must be >= 1, got {self.training.min_batch_size}")
        
        # Validate AMP settings
        if self.training.use_amp and self.training.amp_init_scale <= 0:
            raise ValueError(f"amp_init_scale must be positive, got {self.training.amp_init_scale}")
        
        # Warn about potentially problematic combinations
        if self.training.use_amp and self.training.gradient_accumulation_steps > 8:
            warnings.warn("Large gradient accumulation steps with AMP may cause numerical instability")
        
        if self.data.batch_size * self.training.gradient_accumulation_steps > 128:
            warnings.warn("Very large effective batch size may hurt convergence for contrastive learning tasks")


def load_config(config_path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML configuration file
        overrides: Optional dictionary of parameter overrides
        
    Returns:
        Parsed and validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    if yaml_config is None:
        yaml_config = {}
    
    # Apply overrides
    if overrides:
        yaml_config = _deep_update(yaml_config, overrides)
    
    # Parse into structured config
    config = _parse_config(yaml_config)
    
    return config


def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Recursively update base_dict with update_dict."""
    result = base_dict.copy()
    
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    
    return result


def _parse_config(yaml_config: Dict[str, Any]) -> Config:
    """Parse YAML configuration into structured Config object."""
    # Initialize with defaults
    config = Config()
    
    # Parse run configuration
    if 'run' in yaml_config:
        run_dict = yaml_config['run']
        config.run = RunConfig(
            project_name=run_dict.get('project_name', config.run.project_name),
            run_name=run_dict.get('run_name', config.run.run_name),
            entity=run_dict.get('entity', config.run.entity),
            tags=run_dict.get('tags', config.run.tags),
            notes=run_dict.get('notes', config.run.notes)
        )
    
    # Parse data configuration
    if 'data' in yaml_config:
        data_dict = yaml_config['data']
        config.data = DataConfig(
            pretrain_datasets=data_dict.get('pretrain_datasets', config.data.pretrain_datasets),
            batch_size=data_dict.get('batch_size', config.data.batch_size),
            domain_balanced_sampling=data_dict.get('domain_balanced_sampling', config.data.domain_balanced_sampling),
            num_workers=data_dict.get('num_workers', config.data.num_workers),
            pin_memory=data_dict.get('pin_memory', config.data.pin_memory),
            shuffle=data_dict.get('shuffle', config.data.shuffle),
            drop_last=data_dict.get('drop_last', config.data.drop_last)
        )
    
    # Parse model configuration
    if 'model' in yaml_config:
        model_dict = yaml_config['model']
        config.model = ModelConfig(
            hidden_dim=model_dict.get('hidden_dim', config.model.hidden_dim),
            num_layers=model_dict.get('num_layers', config.model.num_layers),
            dropout_rate=model_dict.get('dropout_rate', config.model.dropout_rate),
            enable_augmentations=model_dict.get('enable_augmentations', config.model.enable_augmentations)
        )
    
    # Parse optimizer configuration
    if 'optimizer' in yaml_config:
        opt_dict = yaml_config['optimizer']
        config.optimizer = OptimizerConfig(
            name=opt_dict.get('name', config.optimizer.name),
            lr=opt_dict.get('lr', config.optimizer.lr),
            beta1=opt_dict.get('beta1', config.optimizer.beta1),
            beta2=opt_dict.get('beta2', config.optimizer.beta2),
            weight_decay=opt_dict.get('weight_decay', config.optimizer.weight_decay),
            eps=opt_dict.get('eps', config.optimizer.eps)
        )
    
    # Parse scheduler configuration
    if 'scheduler' in yaml_config:
        sched_dict = yaml_config['scheduler']
        config.scheduler = SchedulerConfig(
            type=sched_dict.get('type', config.scheduler.type),
            warmup_epochs=sched_dict.get('warmup_epochs', config.scheduler.warmup_epochs),
            warmup_fraction=sched_dict.get('warmup_fraction', config.scheduler.warmup_fraction),
            min_lr_ratio=sched_dict.get('min_lr_ratio', config.scheduler.min_lr_ratio),
            step_size=sched_dict.get('step_size', config.scheduler.step_size),
            gamma=sched_dict.get('gamma', config.scheduler.gamma)
        )
    
    # Parse training configuration
    if 'training' in yaml_config:
        train_dict = yaml_config['training']
        config.training = TrainingConfig(
            max_epochs=train_dict.get('max_epochs', config.training.max_epochs),
            max_steps=train_dict.get('max_steps', config.training.max_steps),
            validation_freq=train_dict.get('validation_freq', config.training.validation_freq),
            log_freq=train_dict.get('log_freq', config.training.log_freq),
            patience=train_dict.get('patience', config.training.patience),
            min_delta=train_dict.get('min_delta', config.training.min_delta),
            validation_metric=train_dict.get('validation_metric', config.training.validation_metric),
            metric_mode=train_dict.get('metric_mode', config.training.metric_mode),
            checkpoint_dir=train_dict.get('checkpoint_dir', config.training.checkpoint_dir),
            save_checkpoint_freq=train_dict.get('save_checkpoint_freq', config.training.save_checkpoint_freq),
            keep_n_checkpoints=train_dict.get('keep_n_checkpoints', config.training.keep_n_checkpoints),
            use_uncertainty_weighting=train_dict.get('use_uncertainty_weighting', config.training.use_uncertainty_weighting),
            uncertainty_init=train_dict.get('uncertainty_init', config.training.uncertainty_init),
            grad_clip_norm=train_dict.get('grad_clip_norm', config.training.grad_clip_norm),
            use_amp=train_dict.get('use_amp', config.training.use_amp),
            amp_init_scale=train_dict.get('amp_init_scale', config.training.amp_init_scale),
            gradient_accumulation_steps=train_dict.get('gradient_accumulation_steps', config.training.gradient_accumulation_steps),
            dynamic_batch_sizing=train_dict.get('dynamic_batch_sizing', config.training.dynamic_batch_sizing),
            max_memory_usage=train_dict.get('max_memory_usage', config.training.max_memory_usage),
            min_batch_size=train_dict.get('min_batch_size', config.training.min_batch_size)
        )
    
    # Parse domain adversarial configuration
    if 'domain_adversarial' in yaml_config:
        da_dict = yaml_config['domain_adversarial']
        config.domain_adversarial = DomainAdversarialConfig(
            enabled=da_dict.get('enabled', config.domain_adversarial.enabled),
            schedule_type=da_dict.get('schedule_type', config.domain_adversarial.schedule_type),
            initial_lambda=da_dict.get('initial_lambda', config.domain_adversarial.initial_lambda),
            final_lambda=da_dict.get('final_lambda', config.domain_adversarial.final_lambda),
            gamma=da_dict.get('gamma', config.domain_adversarial.gamma),
            warmup_epochs=da_dict.get('warmup_epochs', config.domain_adversarial.warmup_epochs)
        )
    
    # Parse task configurations
    if 'tasks' in yaml_config:
        for task_name, task_dict in yaml_config['tasks'].items():
            if task_name in config.tasks:
                # Update existing task config
                current_config = config.tasks[task_name]
                config.tasks[task_name] = TaskConfig(
                    enabled=task_dict.get('enabled', current_config.enabled),
                    weight=task_dict.get('weight', current_config.weight),
                    loss_type=task_dict.get('loss_type', current_config.loss_type),
                    mask_rate=task_dict.get('mask_rate', current_config.mask_rate),
                    negative_sampling_ratio=task_dict.get('negative_sampling_ratio', current_config.negative_sampling_ratio),
                    temperature=task_dict.get('temperature', current_config.temperature),
                    projection_dim=task_dict.get('projection_dim', current_config.projection_dim)
                )
            else:
                # Add new task config
                config.tasks[task_name] = TaskConfig(
                    enabled=task_dict.get('enabled', True),
                    weight=task_dict.get('weight', 1.0),
                    loss_type=task_dict.get('loss_type', 'auto'),
                    mask_rate=task_dict.get('mask_rate', 0.15),
                    negative_sampling_ratio=task_dict.get('negative_sampling_ratio', 1.0),
                    temperature=task_dict.get('temperature', 0.1),
                    projection_dim=task_dict.get('projection_dim', 128)
                )
    
    return config


def create_default_config_file(output_path: Union[str, Path]) -> None:
    """
    Create a default configuration YAML file.
    
    Args:
        output_path: Path where to save the default config file
    """
    default_config = {
        'run': {
            'project_name': 'Graph-Multitask-Learning',
            'run_name': 'default-experiment',
            'entity': None,
            'tags': [],
            'notes': 'Default configuration for GNN pre-training'
        },
        'data': {
            'pretrain_datasets': ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES'],
            'batch_size': 32,
            'domain_balanced_sampling': True,
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': True,
            'drop_last': True
        },
        'model': {
            'hidden_dim': 256,
            'num_layers': 5,
            'dropout_rate': 0.2,
            'enable_augmentations': True
        },
        'optimizer': {
            'name': 'AdamW',
            'lr': 5.0e-4,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.01,
            'eps': 1.0e-8
        },
        'scheduler': {
            'type': 'cosine',
            'warmup_epochs': 10,
            'warmup_fraction': 0.1,
            'min_lr_ratio': 0.01
        },
        'training': {
            'max_epochs': 100,
            'max_steps': 100000,
            'validation_freq': 1,
            'log_freq': 100,
            'patience': 10,
            'min_delta': 1.0e-4,
            'validation_metric': 'val/loss_total',
            'metric_mode': 'min',
            'checkpoint_dir': './checkpoints',
            'save_checkpoint_freq': 10,
            'keep_n_checkpoints': 3,
            'use_uncertainty_weighting': True,
            'uncertainty_init': 0.0,
            'grad_clip_norm': 0.0,
            'use_amp': True,
            'amp_init_scale': 65536.0,
            'gradient_accumulation_steps': 1,
            'dynamic_batch_sizing': False,
            'max_memory_usage': 0.8,
            'min_batch_size': 4
        },
        'domain_adversarial': {
            'enabled': True,
            'schedule_type': 'dann',
            'initial_lambda': 0.0,
            'final_lambda': 1.0,
            'gamma': 10.0,
            'warmup_epochs': 0
        },
        'tasks': {
            'node_feat_mask': {
                'enabled': True,
                'weight': 1.0,
                'mask_rate': 0.15
            },
            'link_pred': {
                'enabled': True,
                'weight': 1.0,
                'negative_sampling_ratio': 1.0
            },
            'node_contrast': {
                'enabled': True,
                'weight': 1.0,
                'temperature': 0.1
            },
            'graph_contrast': {
                'enabled': True,
                'weight': 1.0
            },
            'graph_prop': {
                'enabled': True,
                'weight': 1.0
            },
            'domain_adv': {
                'enabled': True,
                'weight': 1.0
            }
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration saved to: {output_path}")


if __name__ == '__main__':
    # Create a default config file for reference
    create_default_config_file('configs/default.yaml')
    
    # Test loading the config
    config = load_config('configs/default.yaml')
    print("Configuration loaded successfully!")
    print(f"Device: {config.device}")
    print(f"Enabled tasks: {[name for name, task in config.tasks.items() if task.enabled]}") 