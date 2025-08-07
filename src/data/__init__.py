"""
Data processing pipeline for GNN multi-task pre-training.

This module handles:
- Data setup and preprocessing
- Pre-training data loading
- Downstream data loading
"""

from .data_setup import main as setup_data
from .data_loading import create_data_loaders, GraphDataset, DomainBalancedSampler
from .downstream_data_loading import load_downstream_data, get_dataset_info

__all__ = [
    'setup_data',
    'create_data_loaders',
    'GraphDataset', 
    'DomainBalancedSampler',
    'load_downstream_data',
    'get_dataset_info'
]
