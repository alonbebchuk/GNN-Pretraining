import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any
import warnings

# Core components
from .gnn import InputEncoder, GIN_Backbone
from .heads import MLPHead, DotProductDecoder, BilinearDiscriminator
from ..layers import GradientReversalLayer


class PretrainableGNN(nn.Module):
    """
    Meta-model for multi-domain GNN pre-training.

    This model assembles all components needed for pre-training:
    - Domain-specific input encoders
    - Shared GNN backbone
    - Task-specific prediction heads
    - Gradient reversal layer for domain-adversarial training
    - Learnable [MASK] tokens for node feature masking
    - Graph augmentation capabilities for contrastive learning
    """

    def __init__(self, domain_dimensions: Dict[str, int], hidden_dim: int = 256, num_layers: int = 5, dropout_rate: float = 0.2, device: Optional[Union[str, torch.device]] = None, enable_augmentations: bool = True):
        """
        Initialize the pretrainable GNN model.

        Args:
            domain_dimensions: Dict mapping domain names to their input dimensions
            hidden_dim: Hidden dimension for the shared representation
            num_layers: Number of GIN layers in the backbone
            dropout_rate: Dropout rate for regularization
            device: Device to place the model on ('cpu', 'cuda', or torch.device)
            enable_augmentations: Whether to enable graph augmentations
        """
        super(PretrainableGNN, self).__init__()

        # Device handling
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.hidden_dim = hidden_dim
        self.num_domains = len(domain_dimensions)
        self.domain_dimensions = domain_dimensions
        self.dropout_rate = dropout_rate

        # --- Domain-specific Input Encoders ---
        self.input_encoders = nn.ModuleDict()
        for domain_name, dim_in in domain_dimensions.items():
            self.input_encoders[domain_name] = InputEncoder(
                dim_in=dim_in,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate
            )

        # --- Domain-specific [MASK] tokens for node feature masking ---
        self.mask_tokens = nn.ParameterDict()
        for domain_name, dim_in in domain_dimensions.items():
            # Each domain needs its own [MASK] token with the correct input dimension
            # Use normal initialization (0 mean, small std)
            mask_token = torch.zeros(dim_in)
            nn.init.normal_(mask_token, mean=0.0, std=0.02)  # BERT-style initialization
            self.mask_tokens[domain_name] = nn.Parameter(mask_token)

        # --- Shared GNN Backbone ---
        self.gnn_backbone = GIN_Backbone(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

        # --- Task-specific Prediction Heads ---
        self.heads = nn.ModuleDict({
            # Node feature masking head (reconstruction)
            'node_feat_mask': MLPHead(
                dim_in=hidden_dim,
                dim_hidden=hidden_dim,
                dim_out=hidden_dim,
                dropout_rate=dropout_rate
            ),

            # Link prediction head (non-parametric)
            'link_pred': DotProductDecoder(),

            # Node contrastive learning projection head
            'node_contrast': MLPHead(
                dim_in=hidden_dim,
                dim_hidden=hidden_dim,
                dim_out=hidden_dim // 2,
                dropout_rate=dropout_rate
            ),

            # Graph contrastive learning discriminator
            'graph_contrast': BilinearDiscriminator(
                dim1=hidden_dim,  # Node embeddings
                dim2=hidden_dim   # Graph embeddings
            ),

            # Graph property prediction head
            'graph_prop': MLPHead(
                dim_in=hidden_dim,
                dim_hidden=hidden_dim * 2,
                dim_out=15,  # Predict 15 comprehensive graph properties
                dropout_rate=dropout_rate
            ),

            # Domain adversarial classifier
            'domain_adv': MLPHead(
                dim_in=hidden_dim,
                dim_hidden=hidden_dim // 2,
                dim_out=self.num_domains,
                dropout_rate=0.0  # No dropout for strongest classifier
            )
        })

        # --- Gradient Reversal Layer ---
        self.grl = GradientReversalLayer()

        # --- Graph Augmentation ---
        if enable_augmentations:
            # Lazy import to avoid circular dependency
            self.augmentor = self._create_default_augmentor()
        else:
            self.augmentor = None

        # Move to device
        self.to(self.device)

    def _create_default_augmentor(self):
        """Create default augmentor with lazy import to avoid circular dependency."""
        try:
            from ...training.augmentations import GraphAugmentor
        except ImportError:
            # Fallback for different import contexts
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            from training.augmentations import GraphAugmentor

        return GraphAugmentor(
            attr_mask_prob=0.5,
            attr_mask_rate=0.15,
            edge_drop_prob=0.5,
            edge_drop_rate=0.15,
            subgraph_prob=0.5,
            walk_length=10
        )

    def apply_node_masking(self, data, domain_name: str, mask_rate: float = 0.15):
        """
        Apply node feature masking for the node feature masking pre-training task.

        Args:
            data: PyTorch Geometric Data object
            domain_name: Name of the domain
            mask_rate: Fraction of nodes to mask

        Returns:
            Tuple of (masked_data, mask_indices, target_h0):
                - masked_data: Data object with masked node features
                - mask_indices: Indices of masked nodes
                - target_h0: Original h_0 embeddings for masked nodes (None if compute_targets=False)
        """
        # Move data to the correct device
        data = data.to(self.device)

        # Compute original h_0 embeddings
        encoder = self.input_encoders[domain_name]
        with torch.no_grad():
            original_h0 = encoder(data.x)

        # Clone the data to avoid modifying the original
        masked_data = data.clone()
        num_nodes = data.x.shape[0]
        num_mask = int(num_nodes * mask_rate)

        # Randomly select nodes to mask
        mask_indices = torch.randperm(num_nodes, device=self.device)[:num_mask]

        # Store original h_0 embeddings for reconstruction targets
        target_h0 = original_h0[mask_indices].clone()

        # Replace selected node features with the learnable [MASK] token
        mask_token = self.mask_tokens[domain_name]
        masked_data.x[mask_indices] = mask_token.unsqueeze(0).expand(num_mask, -1)

        return masked_data, mask_indices, target_h0

    def create_augmented_views(self, data, num_views: int = 2):
        """
        Create multiple augmented views of the graph for contrastive learning.

        Args:
            data: PyTorch Geometric Data object
            num_views: Number of augmented views to create

        Returns:
            List of augmented data objects
        """
        return [self.augmentor(data) for _ in range(num_views)]

    def forward(self, data, domain_name: str):
        """
        Forward pass through the model.

        Args:
            data: PyTorch Geometric Data object
            domain_name: Name of the domain

        Returns:
            Dictionary containing:
                - 'node_embeddings': Final node embeddings (num_nodes, hidden_dim)
                - 'graph_embedding': Graph-level embedding (hidden_dim,)
                - 'h_0': Initial embeddings for masking reconstruction
        """
        # Move data to the correct device
        data = data.to(self.device)

        # 1. Select domain-specific encoder
        encoder = self.input_encoders[domain_name]

        # 2. Encode domain-specific features to shared representation
        h_0 = encoder(data.x)

        # 3. Process with shared GNN backbone
        final_node_embeddings = self.gnn_backbone(h_0, data.edge_index)

        # 4. Compute graph-level summary embedding (mean pooling)
        graph_summary_embedding = final_node_embeddings.mean(dim=0)

        return {
            'node_embeddings': final_node_embeddings,
            'graph_embedding': graph_summary_embedding,
            'h_0': h_0  # Include initial embeddings for masking reconstruction
        }

    def get_head(self, head_name: str):
        """
        Get a specific prediction head.

        Args:
            head_name: Name of the head to retrieve

        Returns:
            The requested prediction head module
        """
        return self.heads[head_name]

    def apply_gradient_reversal(self, embeddings, lambda_val):
        """
        Apply gradient reversal to embeddings for domain-adversarial training.

        Args:
            embeddings: Input embeddings tensor
            lambda_val: Scaling factor for gradient reversal

        Returns:
            Embeddings with gradient reversal applied
        """
        return self.grl(embeddings, lambda_val)

    def get_domain_list(self):
        """
        Get list of available domains.

        Returns:
            List of domain names
        """
        return list(self.input_encoders.keys())

    def get_mask_token(self, domain_name: str):
        """
        Get the learnable [MASK] token for a specific domain.

        Args:
            domain_name: Name of the domain

        Returns:
            The [MASK] token parameter for the domain
        """
        return self.mask_tokens[domain_name]


# Convenience function to create model with complete domain configuration
def create_full_pretrain_model(device: Optional[Union[str, torch.device]] = None, **kwargs) -> PretrainableGNN:
    """
    Create a PretrainableGNN model with the complete domain configuration.

    Args:
        device: Device to place the model on
        **kwargs: Additional arguments to pass to PretrainableGNN

    Returns:
        PretrainableGNN model with all domains configured
    """
    # Complete domain configuration
    complete_domain_configs = {
        # Pre-training domains
        'MUTAG': 7,
        'PROTEINS': 4,
        'NCI1': 37,
        'ENZYMES': 21,

        # Additional TUDatasets
        'FRANKENSTEIN': 780,
        'PTC_MR': 18,

        # Planetoid datasets
        'Cora': 1433,
        'CiteSeer': 3703
    }

    return PretrainableGNN(
        domain_dimensions=complete_domain_configs,
        device=device,
        **kwargs
    )
