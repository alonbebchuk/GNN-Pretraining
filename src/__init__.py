"""
GNN Multi-Task Pre-training Research Package.

This package provides a complete implementation of multi-task, cross-domain
pre-training for Graph Neural Networks as described in the research plan.
"""

# Core model components
from . import core

# Data processing pipeline
from . import data

# Training pipeline
from . import training

# Evaluation pipeline
from . import evaluation

# Infrastructure and utilities
from . import infrastructure

__version__ = "1.0.0"
__author__ = "GNN Research Team"

__all__ = [
    'core',
    'data', 
    'training',
    'evaluation',
    'infrastructure'
]