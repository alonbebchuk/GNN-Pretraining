"""
Pre-training pipeline for GNN multi-task learning.

This module contains:
- Main pre-training script
- Training loop implementation
- Multi-task loss functions
- Graph augmentations
"""

# Core training components
from .trainer import PretrainTrainer, MemoryMonitor, EarlyStopping
from .losses import MultiTaskLossComputer, UncertaintyWeighting
from .augmentations import GraphAugmentor, AttributeMasking, EdgeDropping, SubgraphSampling

__all__ = [
    'PretrainTrainer',
    'MemoryMonitor',
    'EarlyStopping',
    'MultiTaskLossComputer',
    'UncertaintyWeighting',
    'GraphAugmentor',
    'AttributeMasking',
    'EdgeDropping',
    'SubgraphSampling'
]
