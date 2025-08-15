from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from src.common import (
    DOMAIN_DIMENSIONS,
    GNN_HIDDEN_DIM,
    NODE_FEATURE_MASKING_MASK_RATE,
    MASK_TOKEN_INIT_MEAN,
    MASK_TOKEN_INIT_STD,
    GRAPH_PROPERTY_DIM,
    CONTRASTIVE_PROJ_DIM,
    GRAPH_PROP_HEAD_HIDDEN_DIM,
)
from src.model.gnn import InputEncoder, GIN_Backbone
from src.model.heads import MLPHead, DotProductDecoder, BilinearDiscriminator, DomainClassifierHead


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

    def __init__(self, device: torch.device, domain_names: list[str], task_names: list[str]) -> None:
        """
        Initialize the pretrainable GNN model.

        Args:
            domain_dimensions: Dict mapping domain names to their input dimensions
            device: Device to place the model on ('cpu', 'cuda', or torch.device)
        """
        super(PretrainableGNN, self).__init__()

        self.device = device

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
                dom_heads = nn.ModuleDict()
                for domain_name in domain_names:
                    dom_heads[domain_name] = MLPHead(dim_out=CONTRASTIVE_PROJ_DIM)
                self.heads[task_name] = dom_heads
            elif task_name == 'graph_contrast':
                # Domain-specific discriminators
                dom_heads = nn.ModuleDict()
                for domain_name in domain_names:
                    dom_heads[domain_name] = BilinearDiscriminator()
                self.heads[task_name] = dom_heads
            elif task_name == 'graph_prop':
                # Domain-specific graph property heads
                dom_heads = nn.ModuleDict()
                for domain_name in domain_names:
                    dom_heads[domain_name] = MLPHead(dim_hidden=GRAPH_PROP_HEAD_HIDDEN_DIM, dim_out=GRAPH_PROPERTY_DIM)
                self.heads[task_name] = dom_heads
            elif task_name == 'domain_adv':
                # Shared domain adversarial classifier across all domains (predicts domain)
                self.heads[task_name] = DomainClassifierHead()

        # Move to device
        self.to(self.device)

    def apply_node_masking(self, batch: Batch, domain_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply node masking in hidden space for the Node Feature Masking task.

        Args:
            batch: PyTorch Geometric Batch object
            domain_name: Name of the domain

        Returns:
            Tuple of (masked_h0, mask_indices, target_h0):
                - masked_h0: h_0 embeddings with masked nodes replaced by shared [MASK] token
                - mask_indices: Indices of masked nodes
                - target_h0: Original h_0 embeddings for masked nodes
        """
        # Move batch to the correct device
        batch = batch.to(self.device)

        # Compute original h_0 embeddings
        encoder = self.input_encoders[domain_name]
        with torch.no_grad():
            original_h0 = encoder(batch.x)

        num_nodes = batch.x.shape[0]

        mask_candidates = (torch.rand(num_nodes, device=self.device) < NODE_FEATURE_MASKING_MASK_RATE)
        
        # Ensure at least one node is NOT masked per graph (to prevent empty graphs)
        num_graphs = int(batch.batch.max().item()) + 1
        mask_counts = torch.bincount(batch.batch, weights=mask_candidates.to(torch.long), minlength=num_graphs)
        node_counts = torch.bincount(batch.batch, minlength=num_graphs)
        all_masked_graphs = (mask_counts == node_counts) & (node_counts > 0)        
        if all_masked_graphs.any():
            ptr = torch.zeros(num_graphs + 1, device=self.device, dtype=torch.long)
            ptr[1:] = torch.cumsum(node_counts, dim=0)
            starts = ptr[:-1][all_masked_graphs]
            mask_candidates[starts] = False
        
        mask_indices = mask_candidates.nonzero(as_tuple=True)[0]

        target_h0 = original_h0[mask_indices].clone()

        masked_h0 = original_h0.clone()
        masked_h0[mask_indices] = self.mask_token.unsqueeze(0).expand(mask_indices.size(0), -1)

        return masked_h0, mask_indices, target_h0

    def forward(self, batch: Batch, domain_name: str) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            batch: PyTorch Geometric Batch object
            domain_name: Name of the domain

        Returns:
            Final node embeddings (num_nodes, hidden_dim)
        """
        # Move batch to the correct device
        batch = batch.to(self.device)

        # Select domain-specific encoder
        encoder = self.input_encoders[domain_name]

        # Encode domain-specific features to shared representation
        h_0 = encoder(batch.x)

        # Process with shared GNN backbone
        final_node_embeddings = self.gnn_backbone(h_0, batch.edge_index)

        return final_node_embeddings

    def forward_with_h0(self, h_0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass starting from precomputed h_0 embeddings (used by masking task).
        
        Args:
            h_0: Initial node embeddings tensor of shape [total_nodes_in_batch, hidden_dim]
                 Contains concatenated embeddings from all nodes across all graphs in the batch
            edge_index: Edge indices tensor of shape [2, total_edges_in_batch]
                       Contains concatenated edges from all graphs in the batch
        
        Returns:
            Final node embeddings tensor of shape [total_nodes_in_batch, hidden_dim]
        """
        return self.gnn_backbone(h_0, edge_index)

    def get_head(self, head_name: str, domain_name: Optional[str] = None) -> nn.Module:
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
