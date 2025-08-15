from typing import Optional
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
from src.model.layers import GradientReversalLayer
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

        # --- Gradient Reversal Layer ---
        self.grl = GradientReversalLayer()

        # Move to device
        self.to(self.device)

    def apply_node_masking(self, batch: Batch, domain_name: str):
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

        # Handle batched data with per-graph safeguards
        if hasattr(batch, 'batch') and batch.batch is not None:
            # Batched processing with per-graph safeguards
            mask_prob = NODE_FEATURE_MASKING_MASK_RATE
            mask_candidates = (torch.rand(num_nodes, device=self.device) < mask_prob)
            
            # Ensure at least one node is NOT masked per graph (to prevent empty graphs)
            num_graphs = int(batch.batch.max().item()) + 1 if batch.batch.numel() > 0 else 0
            if num_graphs > 0:
                mask_counts = torch.bincount(batch.batch, weights=mask_candidates.to(torch.long), minlength=num_graphs)
                node_counts = torch.bincount(batch.batch, minlength=num_graphs)
                
                # Find graphs where ALL nodes would be masked (dangerous!)
                all_masked_graphs = (mask_counts == node_counts) & (node_counts > 0)
                
                if all_masked_graphs.any():
                    # For each fully-masked graph, unmask its first node
                    ptr = torch.zeros(num_graphs + 1, device=self.device, dtype=torch.long)
                    ptr[1:] = torch.cumsum(node_counts, dim=0)
                    starts = ptr[:-1][all_masked_graphs]  # First node of each fully-masked graph
                    mask_candidates[starts] = False  # Unmask at least one node per graph
            
            mask_indices = mask_candidates.nonzero(as_tuple=True)[0]
        else:
            # Single graph processing
            num_mask = max(1, min(num_nodes - 1, int(num_nodes * NODE_FEATURE_MASKING_MASK_RATE)))
            mask_indices = torch.randperm(num_nodes, device=self.device)[:num_mask]

        # Store original h_0 embeddings for reconstruction targets
        target_h0 = original_h0[mask_indices].clone() if mask_indices.numel() > 0 else torch.empty(0, original_h0.size(1), device=self.device)

        # Create masked h_0 by replacing selected node embeddings with the shared [MASK] token
        masked_h0 = original_h0.clone()
        if mask_indices.numel() > 0:
            num_mask = mask_indices.size(0)
            masked_h0[mask_indices] = self.mask_token.unsqueeze(0).expand(num_mask, -1)

        return masked_h0, mask_indices, target_h0

    def forward(self, batch: Batch, domain_name: str):
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

        # 1. Select domain-specific encoder
        encoder = self.input_encoders[domain_name]

        # 2. Encode domain-specific features to shared representation
        h_0 = encoder(batch.x)

        # 3. Process with shared GNN backbone
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

    def apply_gradient_reversal(self, embeddings: torch.Tensor, lambda_val: float) -> torch.Tensor:
        """
        Apply gradient reversal to embeddings for domain-adversarial training.

        Args:
            embeddings: Input embeddings tensor of shape [batch_size, embedding_dim]
            lambda_val: Scaling factor for gradient reversal (float from GRL scheduler)

        Returns:
            Embeddings with gradient reversal applied, same shape as input
        """
        return self.grl(embeddings, lambda_val)
