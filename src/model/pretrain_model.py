from typing import Optional
import torch
import torch.nn as nn

from src.common import (
    DOMAIN_DIMENSIONS,
    GNN_HIDDEN_DIM,
    NODE_FEATURE_MASKING_MASK_RATE,
    MASK_TOKEN_INIT_MEAN,
    MASK_TOKEN_INIT_STD,
    GRAPH_PROPERTY_DIM,
    CONTRASTIVE_PROJ_DIM_FACTOR,
    GRAPH_PROP_HEAD_HIDDEN_FACTOR,
    DOMAIN_ADV_HEAD_HIDDEN_FACTOR,
    NODE_FEATURE_MASKING_MIN_NODES,
)
from src.model.layers import GradientReversalLayer
from src.model.gnn import InputEncoder, GIN_Backbone
from src.model.heads import MLPHead, DotProductDecoder, BilinearDiscriminator


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

    def __init__(self, device: torch.device, domain_names: list[str], task_names: list[str]):
        """
        Initialize the pretrainable GNN model.

        Args:
            domain_dimensions: Dict mapping domain names to their input dimensions
            device: Device to place the model on ('cpu', 'cuda', or torch.device)
        """
        super(PretrainableGNN, self).__init__()

        self.device = device
        self.num_domains = len(domain_names)

        # --- Domain-specific Input Encoders ---
        self.input_encoders = nn.ModuleDict()
        for domain_name in domain_names:
            dim_in = DOMAIN_DIMENSIONS[domain_name]
            self.input_encoders[domain_name] = InputEncoder(dim_in=dim_in)

        # --- Shared [MASK] token in hidden space for node feature masking ---
        self.mask_token = nn.Parameter(torch.zeros(GNN_HIDDEN_DIM))
        nn.init.normal_(self.mask_token, mean=MASK_TOKEN_INIT_MEAN, std=MASK_TOKEN_INIT_STD)

        # --- Shared GNN Backbone ---
        self.gnn_backbone = GIN_Backbone()

        # --- Task-specific Prediction Heads ---
        self.heads = nn.ModuleDict()
        for task_name in task_names:
            if task_name == 'node_feat_mask':
                # Domain-specific reconstruction heads
                dom_heads = nn.ModuleDict()
                for domain_name in domain_names:
                    dom_heads[domain_name] = MLPHead()
                self.heads[task_name] = dom_heads
            elif task_name == 'link_pred':
                # Link prediction head (non-parametric, shared)
                self.heads[task_name] = DotProductDecoder()
            elif task_name == 'node_contrast':
                # Domain-specific projection heads
                proj_dim = int(GNN_HIDDEN_DIM * CONTRASTIVE_PROJ_DIM_FACTOR)
                dom_heads = nn.ModuleDict()
                for domain_name in domain_names:
                    dom_heads[domain_name] = MLPHead(dim_out=proj_dim)
                self.heads[task_name] = dom_heads
            elif task_name == 'graph_contrast':
                # Domain-specific discriminators
                dom_heads = nn.ModuleDict()
                for domain_name in domain_names:
                    dom_heads[domain_name] = BilinearDiscriminator()
                self.heads[task_name] = dom_heads
            elif task_name == 'graph_prop':
                # Domain-specific graph property heads
                hidden_dim = int(GNN_HIDDEN_DIM * GRAPH_PROP_HEAD_HIDDEN_FACTOR)
                dom_heads = nn.ModuleDict()
                for domain_name in domain_names:
                    dom_heads[domain_name] = MLPHead(dim_hidden=hidden_dim, dim_out=GRAPH_PROPERTY_DIM)
                self.heads[task_name] = dom_heads
            elif task_name == 'domain_adv':
                # Shared domain adversarial classifier across all domains (predicts domain)
                hidden_dim = int(GNN_HIDDEN_DIM * DOMAIN_ADV_HEAD_HIDDEN_FACTOR)
                # Use no-dropout head to align with paper (strongest possible classifier)
                self.heads[task_name] = MLPHead(dim_hidden=hidden_dim, dim_out=self.num_domains, apply_dropout=False)

        # --- Gradient Reversal Layer ---
        self.grl = GradientReversalLayer()

        # Move to device
        self.to(self.device)

    def apply_node_masking(self, data, domain_name: str):
        """
        Apply node masking in hidden space for the Node Feature Masking task.

        Args:
            data: PyTorch Geometric Data object
            domain_name: Name of the domain

        Returns:
            Tuple of (masked_h0, mask_indices, target_h0):
                - masked_h0: h_0 embeddings with masked nodes replaced by shared [MASK] token
                - mask_indices: Indices of masked nodes
                - target_h0: Original h_0 embeddings for masked nodes
        """
        # Move data to the correct device
        data = data.to(self.device)

        # Compute original h_0 embeddings
        encoder = self.input_encoders[domain_name]
        with torch.no_grad():
            original_h0 = encoder(data.x)

        num_nodes = data.x.shape[0]

        # Ensure at least a minimum number of nodes are masked
        num_mask = max(NODE_FEATURE_MASKING_MIN_NODES, int(num_nodes * NODE_FEATURE_MASKING_MASK_RATE))

        # Randomly select nodes to mask
        mask_indices = torch.randperm(num_nodes, device=self.device)[:num_mask]

        # Store original h_0 embeddings for reconstruction targets
        target_h0 = original_h0[mask_indices].clone()

        # Create masked h_0 by replacing selected node embeddings with the shared [MASK] token
        masked_h0 = original_h0.clone()
        masked_h0[mask_indices] = self.mask_token.unsqueeze(0).expand(num_mask, -1)

        return masked_h0, mask_indices, target_h0

    def forward(self, data, domain_name: str):
        """
        Forward pass through the model.

        Args:
            data: PyTorch Geometric Data object
            domain_name: Name of the domain

        Returns:
            Final node embeddings (num_nodes, hidden_dim)
        """
        # Move data to the correct device
        data = data.to(self.device)

        # 1. Select domain-specific encoder
        encoder = self.input_encoders[domain_name]

        # 2. Encode domain-specific features to shared representation
        h_0 = encoder(data.x)

        # 3. Process with shared GNN backbone
        final_node_embeddings = self.gnn_backbone(h_0, data.edge_index)

        return final_node_embeddings

    def forward_with_h0(self, h_0: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass starting from precomputed h_0 embeddings (used by masking task).
        """
        return self.gnn_backbone(h_0, edge_index)

    def get_head(self, head_name: str, domain_name: Optional[str] = None):
        """
        Get a specific prediction head.

        Args:
            head_name: Name of the head to retrieve
            domain_name: Optional domain when the head is domain-specific

        Returns:
            The requested prediction head module
        """
        head = self.heads[head_name]
        if domain_name is not None:
            return head[domain_name]
        return head

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
