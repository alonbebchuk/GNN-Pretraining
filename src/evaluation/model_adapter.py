#!/usr/bin/env python3
"""
Model Adapter for Fine-tuning Pipeline.

This module handles the adaptation of pre-trained models for downstream tasks,
including proper handling of domain-specific encoders and task heads.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)


class ModelAdapter:
    """
    Handles adaptation of pre-trained models for downstream fine-tuning.
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize model adapter.
        
        Args:
            device: Device to load models on
        """
        self.device = device
    
    def load_pretrained_model(self, artifact_path: str) -> Optional[nn.Module]:
        """
        Load pre-trained model from artifact path with robust error handling.
        
        Args:
            artifact_path: Path to model artifact
            
        Returns:
            Loaded pre-trained model or None if loading fails
        """
        try:
            from src.models.pretrain_model import create_full_pretrain_model
        except ImportError:
            try:
                from models.pretrain_model import create_full_pretrain_model
            except ImportError:
                logger.error("Cannot import pretrain model creation function")
                return None
        
        # Handle local fallback paths
        if artifact_path in ["local-fallback", "local-model-checkpoint"] or artifact_path.startswith('local-'):
            logger.info("Creating fresh model for testing (no pre-trained weights)")
            return create_full_pretrain_model(device=self.device)
        
        # Look for checkpoint file
        artifact_path_obj = Path(artifact_path)
        if not artifact_path_obj.exists():
            logger.warning(f"Artifact path doesn't exist: {artifact_path}")
            logger.info("Creating fresh model for testing")
            return create_full_pretrain_model(device=self.device)
        
        checkpoint_files = list(artifact_path_obj.glob("*.pt")) + list(artifact_path_obj.glob("*.pth"))
        
        if not checkpoint_files:
            logger.warning(f"No checkpoint files found in {artifact_path}")
            logger.info("Creating fresh model for testing")
            return create_full_pretrain_model(device=self.device)
        
        # Use the first checkpoint file (or look for specific names)
        checkpoint_path = checkpoint_files[0]
        for cf in checkpoint_files:
            if 'best' in cf.name.lower():
                checkpoint_path = cf
                break
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Create encoder model
            pretrained_model = create_full_pretrain_model(device=self.device)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load the state dict with error handling
            missing_keys, unexpected_keys = pretrained_model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading model: {len(missing_keys)} keys")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading model: {len(unexpected_keys)} keys")
            
            logger.info("Pre-trained model loaded successfully")
            return pretrained_model
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Creating fresh model for testing")
            return create_full_pretrain_model(device=self.device)
    
    def extract_encoder_components(self, pretrained_model: nn.Module, dataset_name: str) -> Dict[str, Any]:
        """
        Extract relevant encoder components from pre-trained model.
        
        Args:
            pretrained_model: Pre-trained model
            dataset_name: Target dataset name
            
        Returns:
            Dictionary with encoder components
        """
        try:
            from src.downstream_data_loading import get_dataset_info
        except ImportError:
            from downstream_data_loading import get_dataset_info
        
        dataset_info = get_dataset_info(dataset_name)
        is_in_domain = dataset_info.get('in_domain', False)
        domain_name = dataset_info.get('domain_name')
        
        components = {
            'gnn_backbone': None,
            'input_encoder': None,
            'is_in_domain': is_in_domain,
            'domain_name': domain_name,
            'available_domains': []
        }
        
        # Extract GNN backbone
        if hasattr(pretrained_model, 'gnn_backbone'):
            components['gnn_backbone'] = pretrained_model.gnn_backbone
        else:
            logger.error("Pre-trained model missing gnn_backbone")
            return components
        
        # Extract available domains
        if hasattr(pretrained_model, 'get_domain_list'):
            components['available_domains'] = pretrained_model.get_domain_list()
        
        # Extract input encoder
        if is_in_domain and domain_name:
            # Try to get domain-specific encoder
            if (hasattr(pretrained_model, 'input_encoders') and 
                domain_name in pretrained_model.input_encoders):
                components['input_encoder'] = pretrained_model.input_encoders[domain_name]
                logger.info(f"Using in-domain encoder for {domain_name}")
            else:
                logger.warning(f"Domain {domain_name} not found in pre-trained encoders")
                components['input_encoder'] = self._create_new_input_encoder(dataset_info)
        else:
            # Create new encoder for out-of-domain tasks
            components['input_encoder'] = self._create_new_input_encoder(dataset_info)
            logger.info(f"Created new encoder for out-of-domain dataset {dataset_name}")
        
        return components
    
    def _create_new_input_encoder(self, dataset_info: Dict[str, Any]) -> nn.Module:
        """Create a new input encoder for the dataset."""
        try:
            from src.models.gnn import InputEncoder
        except ImportError:
            from models.gnn import InputEncoder
        
        return InputEncoder(
            dim_in=dataset_info['input_dim'],
            hidden_dim=256,  # Standard hidden dimension
            dropout_rate=0.2
        )
    
    def create_task_head(self, task_type: str, num_classes: int, hidden_dim: int = 256) -> nn.Module:
        """
        Create appropriate task head for the downstream task.
        
        Args:
            task_type: Type of downstream task
            num_classes: Number of output classes
            hidden_dim: Hidden dimension from encoder
            
        Returns:
            Task-specific head module
        """
        try:
            from src.models.heads import MLPHead
        except ImportError:
            try:
                from models.heads import MLPHead
            except ImportError:
                # Fallback to simple implementation
                if task_type == 'graph_classification':
                    return nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim // 2, num_classes)
                    )
                elif task_type == 'node_classification':
                    return nn.Linear(hidden_dim, num_classes)
                else:
                    return nn.Linear(hidden_dim, num_classes)
        
        # Use MLPHead for all tasks
        if task_type in ['graph_classification', 'node_classification']:
            return MLPHead(
                dim_in=hidden_dim,
                dim_hidden=hidden_dim // 2,
                dim_out=num_classes,
                dropout_rate=0.2
            )
        elif task_type == 'link_prediction':
            # Binary classification for link prediction
            return MLPHead(
                dim_in=hidden_dim * 2,  # Concatenated node embeddings
                dim_hidden=hidden_dim,
                dim_out=2,
                dropout_rate=0.2
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def create_downstream_model(self, 
                              pretrained_model: nn.Module,
                              dataset_name: str,
                              task_type: str,
                              freeze_encoder: bool = True) -> nn.Module:
        """
        Create complete downstream model by combining pre-trained components with new task head.
        
        Args:
            pretrained_model: Pre-trained model
            dataset_name: Target dataset name
            task_type: Type of downstream task
            freeze_encoder: Whether to freeze encoder weights
            
        Returns:
            Complete downstream model
        """
        try:
            from src.downstream_data_loading import get_dataset_info
        except ImportError:
            from downstream_data_loading import get_dataset_info
        
        # Get dataset information
        dataset_info = get_dataset_info(dataset_name)
        
        # Validate task compatibility
        expected_task_type = dataset_info['task_type']
        if task_type != expected_task_type and task_type != 'link_prediction':
            logger.warning(f"Task type mismatch: {dataset_name} expects {expected_task_type}, got {task_type}")
        
        # Extract encoder components
        components = self.extract_encoder_components(pretrained_model, dataset_name)
        
        if not components['gnn_backbone'] or not components['input_encoder']:
            raise ValueError("Failed to extract required model components")
        
        # Create task head
        task_head = self.create_task_head(
            task_type=task_type,
            num_classes=dataset_info['num_classes'],
            hidden_dim=256  # Standard hidden dimension
        )
        
        # Create downstream model
        downstream_model = DownstreamModelWrapper(
            input_encoder=components['input_encoder'],
            gnn_backbone=components['gnn_backbone'],
            task_head=task_head,
            task_type=task_type,
            dataset_info=dataset_info,
            freeze_encoder=freeze_encoder
        )
        
        return downstream_model.to(self.device)


class DownstreamModelWrapper(nn.Module):
    """
    Wrapper for downstream fine-tuning that combines pre-trained components with task head.
    """
    
    def __init__(self,
                 input_encoder: nn.Module,
                 gnn_backbone: nn.Module,
                 task_head: nn.Module,
                 task_type: str,
                 dataset_info: Dict[str, Any],
                 freeze_encoder: bool = True):
        """
        Initialize downstream model wrapper.
        
        Args:
            input_encoder: Input encoder (pre-trained or new)
            gnn_backbone: GNN backbone (pre-trained)
            task_head: Task-specific head (new)
            task_type: Type of downstream task
            dataset_info: Information about the dataset
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()
        
        self.input_encoder = input_encoder
        self.gnn_backbone = gnn_backbone
        self.task_head = task_head
        self.task_type = task_type
        self.dataset_info = dataset_info
        
        # Freeze encoder components if requested
        if freeze_encoder:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.input_encoder.parameters():
            param.requires_grad = False
        for param in self.gnn_backbone.parameters():
            param.requires_grad = False
        logger.info("Encoder parameters frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.input_encoder.parameters():
            param.requires_grad = True
        for param in self.gnn_backbone.parameters():
            param.requires_grad = True
        logger.info("Encoder parameters unfrozen")
    
    def forward(self, data):
        """
        Forward pass through the model.
        
        Args:
            data: Input graph data
            
        Returns:
            Task-specific predictions
        """
        # Encode input features
        h = self.input_encoder(data.x)
        
        # Pass through GNN backbone
        node_embeddings = self.gnn_backbone(h, data.edge_index)
        
        # Task-specific prediction
        if self.task_type == 'graph_classification':
            # Global pooling for graph-level prediction
            try:
                from torch_geometric.nn import global_mean_pool
                batch = getattr(data, 'batch', torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device))
                graph_embedding = global_mean_pool(node_embeddings, batch)
                return self.task_head(graph_embedding)
            except ImportError:
                # Fallback: simple mean pooling
                graph_embedding = node_embeddings.mean(dim=0, keepdim=True)
                return self.task_head(graph_embedding)
        
        elif self.task_type == 'node_classification':
            # Direct node-level prediction
            return self.task_head(node_embeddings)
        
        elif self.task_type == 'link_prediction':
            # Edge-level prediction (simplified)
            # This would need more sophisticated implementation for production
            edge_index = data.edge_index
            src_embeddings = node_embeddings[edge_index[0]]
            dst_embeddings = node_embeddings[edge_index[1]]
            edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
            return self.task_head(edge_embeddings)
        
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def get_trainable_parameters(self):
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'task_type': self.task_type,
            'dataset_info': self.dataset_info,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'encoder_frozen': not any(p.requires_grad for p in self.input_encoder.parameters())
        }


def create_model_adapter(device: torch.device) -> ModelAdapter:
    """
    Convenience function to create a model adapter.
    
    Args:
        device: Device to run on
        
    Returns:
        ModelAdapter instance
    """
    return ModelAdapter(device) 