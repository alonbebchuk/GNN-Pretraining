"""
Infrastructure and supporting systems for GNN research pipeline.

This module contains:
- Configuration management
- Checkpointing and model persistence
- Experiment tracking (WandB integration)
- Learning rate and lambda scheduling
- Input validation
"""

from .config import Config, load_config, create_default_config_file
from .config_validator import validate_config, ConfigValidator
from .checkpointing import CheckpointManager
from .experiment_tracking import WandBTracker, MetricsLogger
from .scheduler import (
    CosineAnnealingWithWarmup,
    LinearWarmupScheduler,
    DomainAdversarialScheduler,
    SchedulerManager
)

__all__ = [
    'Config',
    'load_config',
    'create_default_config_file',
    'validate_config',
    'ConfigValidator',
    'CheckpointManager',
    'WandBTracker',
    'MetricsLogger',
    'CosineAnnealingWithWarmup',
    'LinearWarmupScheduler',
    'DomainAdversarialScheduler',
    'SchedulerManager'
]
