"""
Graph augmentations for contrastive learning tasks.

This module implements the augmentations required for node-level contrastive learning.
Each augmentation is applied with p=0.5 probability as part of a sequential composition:

- Node dropping (% of nodes)
- Edge dropping (% of edges)
- Feature masking (% of feature dimensions)

The implementation follows a GraphCL-style contrastive learning approach where
two different augmented versions (G', G'') are created for each graph to form
positive pairs for contrastive learning.
"""

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
from torch_geometric.transforms import BaseTransform
import random

from src.common import (
    AUGMENTATION_ATTR_MASK_PROB,
    AUGMENTATION_ATTR_MASK_RATE,
    AUGMENTATION_EDGE_DROP_PROB,
    AUGMENTATION_EDGE_DROP_RATE,
    AUGMENTATION_NODE_DROP_PROB,
    AUGMENTATION_NODE_DROP_RATE,
)


class NodeDropping(BaseTransform):
    """Randomly drop a percentage of nodes from the graph and relabel nodes."""

    def __call__(self, batch: Batch) -> Batch:
        """
        Apply node dropping to the batch.

        Args:
            batch: PyTorch Geometric Batch object

        Returns:
            Batch object with a node-induced subgraph
        """
        batch = batch.clone()
        device = batch.x.device
        
        # Batched processing with per-graph safeguards
        keep_prob = 1.0 - AUGMENTATION_NODE_DROP_RATE
        keep_mask = (torch.rand(batch.x.size(0), device=device) < keep_prob)
        
        # Ensure at least one node kept per graph
        num_graphs = int(batch.batch.max().item()) + 1
        keep_counts = torch.bincount(batch.batch, weights=keep_mask.to(torch.long), minlength=num_graphs)
        zero_graphs_mask = (keep_counts == 0)
        if zero_graphs_mask.any():
            node_counts = torch.bincount(batch.batch, minlength=num_graphs)
            ptr = torch.zeros(num_graphs + 1, device=device, dtype=torch.long)
            ptr[1:] = torch.cumsum(node_counts, dim=0)
            starts = ptr[:-1][zero_graphs_mask]
            keep_mask[starts] = True
        
        keep_nodes = keep_mask.nonzero(as_tuple=True)[0]
        edge_index, _ = subgraph(keep_nodes, batch.edge_index)

        batch.x = batch.x[keep_nodes]
        batch.node_indices = batch.node_indices[keep_nodes]
        batch.batch = batch.batch[keep_nodes]
        batch.edge_index = edge_index
        return batch


class EdgeDropping(BaseTransform):
    """Randomly drop a percentage of edges from the graph."""

    def __call__(self, batch: Batch) -> Batch:
        """
        Apply edge dropping to the batch with per-graph safeguards.

        Args:
            batch: PyTorch Geometric Batch object

        Returns:
            Batch object with dropped edges (ensuring each graph retains at least one edge if it had any)
        """
        batch = batch.clone()
        
        if batch.edge_index.numel() > 0:
            device = batch.edge_index.device
            num_edges = batch.edge_index.shape[1]
            
            # Handle batched data with per-graph safeguards
            if hasattr(batch, 'batch') and batch.batch is not None:
                # Batched processing with per-graph safeguards
                keep_prob = 1.0 - AUGMENTATION_EDGE_DROP_RATE
                keep_mask = (torch.rand(num_edges, device=device) < keep_prob)
                
                # Map each edge to its source graph
                edge_to_graph = batch.batch[batch.edge_index[0]]  # Use source node's graph ID
                num_graphs = int(batch.batch.max().item()) + 1
                
                # Count kept edges per graph
                keep_counts = torch.bincount(edge_to_graph, weights=keep_mask.to(torch.long), minlength=num_graphs)
                edge_counts = torch.bincount(edge_to_graph, minlength=num_graphs)
                
                # Find graphs that would lose ALL edges (but originally had edges)
                zero_edge_graphs = (keep_counts == 0) & (edge_counts > 0)
                
                if zero_edge_graphs.any():
                    # For each graph losing all edges, keep its first edge
                    for graph_id in zero_edge_graphs.nonzero(as_tuple=True)[0]:
                        # Find first edge belonging to this graph
                        first_edge_mask = (edge_to_graph == graph_id)
                        first_edge_idx = first_edge_mask.nonzero(as_tuple=True)[0][0]
                        keep_mask[first_edge_idx] = True
                
                keep_indices = keep_mask.nonzero(as_tuple=True)[0]
            else:
                # Single graph processing
                num_keep = max(1, int(num_edges * (1.0 - AUGMENTATION_EDGE_DROP_RATE)))
                keep_indices = torch.randperm(num_edges, device=device)[:num_keep]
            
            if keep_indices.numel() > 0:
                batch.edge_index = batch.edge_index[:, keep_indices]

        return batch


class AttributeMasking(BaseTransform):
    """Randomly mask a percentage of node feature dimensions."""

    def __call__(self, batch: Batch) -> Batch:
        """
        Apply attribute masking to the batch.

        Args:
            batch: PyTorch Geometric Batch object

        Returns:
            Batch object with masked node features
        """
        batch = batch.clone()
        
        if batch.x.numel() > 0:
            device = batch.x.device
            num_features = batch.x.shape[1]
            num_mask = max(1, int(num_features * AUGMENTATION_ATTR_MASK_RATE))

            # Randomly select feature dimensions to mask
            mask_dims = torch.randperm(num_features, device=device)[:num_mask]
            batch.x[:, mask_dims] = 0.0

        return batch


class GraphAugmentor:
    """Composite augmentor that applies multiple transformations with specified probabilities."""

    def __init__(self) -> None:
        self.transforms = [
            (NodeDropping(), AUGMENTATION_NODE_DROP_PROB),
            (EdgeDropping(), AUGMENTATION_EDGE_DROP_PROB),
            (AttributeMasking(), AUGMENTATION_ATTR_MASK_PROB),
        ]

    def __call__(self, batch: Batch) -> Batch:
        """Apply the sequential augmentations to an entire batch using individual transform classes."""
        # Initialize node_indices for tracking original node mappings
        device = batch.x.device
        batch.node_indices = torch.arange(batch.x.size(0), device=device, dtype=torch.long)
        
        # Apply each transformation with its specified probability
        for transform, prob in self.transforms:
            if random.random() < prob:
                batch = transform(batch)
        
        return batch
