"""
Neural network models and components for GNN pre-training.

This package contains:
- gnn.py: Core GNN components (InputEncoder, GINLayer, GIN_Backbone)
- heads.py: Prediction heads (MLPHead, DotProductDecoder, BilinearDiscriminator)  
- pretrain_model.py: Meta-model for multi-domain pre-training (PretrainableGNN)
"""

from .gnn import InputEncoder, GINLayer, GIN_Backbone
from .heads import MLPHead, DotProductDecoder, BilinearDiscriminator
from .pretrain_model import PretrainableGNN, create_full_pretrain_model

__all__ = [
    'InputEncoder',
    'GINLayer', 
    'GIN_Backbone',
    'MLPHead',
    'DotProductDecoder',
    'BilinearDiscriminator',
    'PretrainableGNN',
    'create_full_pretrain_model'
] 