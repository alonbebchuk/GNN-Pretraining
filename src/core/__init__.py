"""
Core model components for GNN multi-task pre-training.

This module contains the fundamental building blocks:
- Custom layers (GRL)
- Model architectures (GNN, heads, meta-model)
"""

from . import models
from .layers import GradientReversalLayer, GradientReversalFn

__all__ = [
    'models',
    'GradientReversalLayer',
    'GradientReversalFn'
]
