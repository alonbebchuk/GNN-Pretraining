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
    type: str = 'GIN'
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
    augmentations: Optional[List[Dict[str, Any]]
                            ] = None  # For contrastive learning


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
    # Automatically adjust batch size based on GPU memory
    dynamic_batch_sizing: bool = False
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
    domain_adversarial: DomainAdversarialConfig = field(
        default_factory=DomainAdversarialConfig)

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
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')


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

    # Aliases/compatibility with proposal YAMLs
    # Support top-level `pretrain_datasets` alias for data.pretrain_datasets
    if 'pretrain_datasets' in yaml_config:
        yaml_config.setdefault('data', {})
        yaml_config['data'].setdefault(
            'pretrain_datasets', yaml_config['pretrain_datasets'])

    # Support `lr_scheduler` alias for `scheduler`
    if 'lr_scheduler' in yaml_config and 'scheduler' not in yaml_config:
        yaml_config['scheduler'] = yaml_config['lr_scheduler']

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
            pretrain_datasets=data_dict.get(
                'pretrain_datasets', config.data.pretrain_datasets),
            batch_size=data_dict.get('batch_size', config.data.batch_size),
            domain_balanced_sampling=data_dict.get(
                'domain_balanced_sampling', config.data.domain_balanced_sampling),
            num_workers=data_dict.get('num_workers', config.data.num_workers),
            pin_memory=data_dict.get('pin_memory', config.data.pin_memory),
            shuffle=data_dict.get('shuffle', config.data.shuffle),
            drop_last=data_dict.get('drop_last', config.data.drop_last)
        )

    # Parse model configuration
    if 'model' in yaml_config:
        model_dict = yaml_config['model']
        # Allow alias `dropout` for `dropout_rate`
        parsed_dropout = model_dict.get(
            'dropout_rate', model_dict.get('dropout', config.model.dropout_rate))
        config.model = ModelConfig(
            type=model_dict.get('type', config.model.type),
            hidden_dim=model_dict.get('hidden_dim', config.model.hidden_dim),
            num_layers=model_dict.get('num_layers', config.model.num_layers),
            dropout_rate=parsed_dropout,
            enable_augmentations=model_dict.get(
                'enable_augmentations', config.model.enable_augmentations)
        )

    # Parse optimizer configuration
    if 'optimizer' in yaml_config:
        opt_dict = yaml_config['optimizer']
        # Accept `betas: [beta1, beta2]` as an alternative specification
        betas_list = opt_dict.get('betas', None)
        beta1 = opt_dict.get('beta1', config.optimizer.beta1)
        beta2 = opt_dict.get('beta2', config.optimizer.beta2)
        if isinstance(betas_list, (list, tuple)) and len(betas_list) >= 2:
            beta1, beta2 = betas_list[0], betas_list[1]
        config.optimizer = OptimizerConfig(
            name=opt_dict.get('name', config.optimizer.name),
            lr=opt_dict.get('lr', config.optimizer.lr),
            beta1=beta1,
            beta2=beta2,
            weight_decay=opt_dict.get(
                'weight_decay', config.optimizer.weight_decay),
            eps=opt_dict.get('eps', config.optimizer.eps)
        )

    # Parse scheduler configuration
    if 'scheduler' in yaml_config:
        sched_dict = yaml_config['scheduler']
        config.scheduler = SchedulerConfig(
            type=sched_dict.get('type', config.scheduler.type),
            warmup_epochs=sched_dict.get(
                'warmup_epochs', config.scheduler.warmup_epochs),
            warmup_fraction=sched_dict.get(
                'warmup_fraction', config.scheduler.warmup_fraction),
            min_lr_ratio=sched_dict.get(
                'min_lr_ratio', config.scheduler.min_lr_ratio),
            step_size=sched_dict.get('step_size', config.scheduler.step_size),
            gamma=sched_dict.get('gamma', config.scheduler.gamma)
        )

    # Parse training configuration
    if 'training' in yaml_config:
        train_dict = yaml_config['training']
        # Accept alias `num_steps` for `max_steps`
        max_steps = train_dict.get('max_steps', train_dict.get(
            'num_steps', config.training.max_steps))
        config.training = TrainingConfig(
            max_epochs=train_dict.get(
                'max_epochs', config.training.max_epochs),
            max_steps=max_steps,
            validation_freq=train_dict.get(
                'validation_freq', config.training.validation_freq),
            log_freq=train_dict.get('log_freq', config.training.log_freq),
            patience=train_dict.get('patience', config.training.patience),
            min_delta=train_dict.get('min_delta', config.training.min_delta),
            validation_metric=train_dict.get(
                'validation_metric', config.training.validation_metric),
            metric_mode=train_dict.get(
                'metric_mode', config.training.metric_mode),
            checkpoint_dir=train_dict.get(
                'checkpoint_dir', config.training.checkpoint_dir),
            save_checkpoint_freq=train_dict.get(
                'save_checkpoint_freq', config.training.save_checkpoint_freq),
            keep_n_checkpoints=train_dict.get(
                'keep_n_checkpoints', config.training.keep_n_checkpoints),
            use_uncertainty_weighting=train_dict.get(
                'use_uncertainty_weighting', config.training.use_uncertainty_weighting),
            uncertainty_init=train_dict.get(
                'uncertainty_init', config.training.uncertainty_init),
            grad_clip_norm=train_dict.get(
                'grad_clip_norm', config.training.grad_clip_norm),
            use_amp=train_dict.get('use_amp', config.training.use_amp),
            amp_init_scale=train_dict.get(
                'amp_init_scale', config.training.amp_init_scale),
            gradient_accumulation_steps=train_dict.get(
                'gradient_accumulation_steps', config.training.gradient_accumulation_steps),
            dynamic_batch_sizing=train_dict.get(
                'dynamic_batch_sizing', config.training.dynamic_batch_sizing),
            max_memory_usage=train_dict.get(
                'max_memory_usage', config.training.max_memory_usage),
            min_batch_size=train_dict.get(
                'min_batch_size', config.training.min_batch_size)
        )

    # Parse domain adversarial configuration
    if 'domain_adversarial' in yaml_config:
        da_dict = yaml_config['domain_adversarial']
        config.domain_adversarial = DomainAdversarialConfig(
            enabled=da_dict.get('enabled', config.domain_adversarial.enabled),
            schedule_type=da_dict.get(
                'schedule_type', config.domain_adversarial.schedule_type),
            initial_lambda=da_dict.get(
                'initial_lambda', config.domain_adversarial.initial_lambda),
            final_lambda=da_dict.get(
                'final_lambda', config.domain_adversarial.final_lambda),
            gamma=da_dict.get('gamma', config.domain_adversarial.gamma),
            warmup_epochs=da_dict.get(
                'warmup_epochs', config.domain_adversarial.warmup_epochs)
        )

    # Parse task configurations (supports dict- and list-style as in proposal)
    if 'tasks' in yaml_config:
        tasks_yaml = yaml_config['tasks']
        # Map proposal names to internal keys
        name_map = {
            'NodeFeatureMasking': 'node_feat_mask',
            'LinkPrediction': 'link_pred',
            'NodeContrastive': 'node_contrast',
            'GraphContrastive': 'graph_contrast',
            'GraphPropertyPrediction': 'graph_prop',
            'DomainAdversarial': 'domain_adv'
        }

        def _merge_task(internal_name: str, item: Dict[str, Any]) -> None:
            # Accept both flat and nested params
            params = item.get('params', {}) if isinstance(item, dict) else {}
            # Handle synonyms for negative sampling ratio
            neg_ratio = params.get(
                'negative_sampling_ratio', params.get('neg_sample_ratio', None))
            current = config.tasks.get(internal_name, TaskConfig())
            new_cfg = TaskConfig(
                enabled=item.get('enabled', current.enabled),
                weight=item.get('weight', current.weight),
                loss_type=item.get('loss_type', current.loss_type),
                mask_rate=params.get('mask_rate', current.mask_rate),
                negative_sampling_ratio=neg_ratio if neg_ratio is not None else current.negative_sampling_ratio,
                temperature=params.get('temperature', current.temperature),
                projection_dim=params.get(
                    'projection_dim', current.projection_dim),
                augmentations=params.get(
                    'augmentations', current.augmentations)
            )
            config.tasks[internal_name] = new_cfg
            # Special routing for domain adversarial knobs
            if internal_name == 'domain_adv':
                if 'weight' in item and isinstance(item['weight'], (int, float)):
                    config.domain_adversarial.final_lambda = float(
                        item['weight'])
                if 'gamma' in params and isinstance(params['gamma'], (int, float)):
                    config.domain_adversarial.gamma = float(params['gamma'])

        if isinstance(tasks_yaml, list):
            for item in tasks_yaml:
                if not isinstance(item, dict) or 'name' not in item:
                    continue
                internal = name_map.get(item['name'], item['name'])
                _merge_task(internal, item)
        elif isinstance(tasks_yaml, dict):
            for task_name, task_dict in tasks_yaml.items():
                internal = name_map.get(task_name, task_name)
                # Flatten optional nested params for compatibility
                params = task_dict.get('params', {}) if isinstance(
                    task_dict, dict) else {}
                merged = task_dict.copy() if isinstance(task_dict, dict) else {}
                if isinstance(params, dict):
                    # Only add missing keys from params to preserve explicit top-level overrides
                    for k, v in params.items():
                        merged.setdefault(k, v)
                # Normalize neg_sample_ratio key
                if 'neg_sample_ratio' in merged and 'negative_sampling_ratio' not in merged:
                    merged['negative_sampling_ratio'] = merged['neg_sample_ratio']
                # Update or insert
                current = config.tasks.get(internal, TaskConfig())
                cfg = TaskConfig(
                    enabled=merged.get('enabled', current.enabled),
                    weight=merged.get('weight', current.weight),
                    loss_type=merged.get('loss_type', current.loss_type),
                    mask_rate=merged.get('mask_rate', current.mask_rate),
                    negative_sampling_ratio=merged.get(
                        'negative_sampling_ratio', current.negative_sampling_ratio),
                    temperature=merged.get('temperature', current.temperature),
                    projection_dim=merged.get(
                        'projection_dim', current.projection_dim),
                    augmentations=merged.get(
                        'augmentations', current.augmentations)
                )
                config.tasks[internal] = cfg
                if internal == 'domain_adv':
                    if 'weight' in merged and isinstance(merged['weight'], (int, float)):
                        config.domain_adversarial.final_lambda = float(
                            merged['weight'])
                    if 'gamma' in merged and isinstance(merged['gamma'], (int, float)):
                        config.domain_adversarial.gamma = float(
                            merged['gamma'])

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
            'type': 'GIN',
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
            'seed': 42,
            'seeds': [42, 84, 126],
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
