"""
Graph augmentations for contrastive learning tasks.

This module provides static utility classes for graph augmentations used in node-level 
contrastive learning. The GraphAugmentor creates two views with:

- Shared node dropping (same nodes in both views)
- Independent edge dropping (different edge patterns)
- Independent feature masking (different feature noise)

This approach ensures proper positive pair alignment while learning invariance
to edge connectivity and feature perturbations.

All augmentation classes are static utilities with no instance state.
"""

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
import random
from typing import Tuple

from src.common import (
    AUGMENTATION_ATTR_MASK_PROB,
    AUGMENTATION_ATTR_MASK_RATE,
    AUGMENTATION_EDGE_DROP_PROB,
    AUGMENTATION_EDGE_DROP_RATE,
    AUGMENTATION_NODE_DROP_PROB,
    AUGMENTATION_NODE_DROP_RATE,
)


class NodeDropping:
    """Static utility for node dropping augmentation."""

    @staticmethod
    def apply(batch: Batch) -> Batch:
        """
        Apply node dropping to the batch.

        Args:
            batch: PyTorch Geometric Batch object

        Returns:
            Batch object with a node-induced subgraph
        """
        batch = batch.clone()
        device = batch.x.device

        keep_mask = (torch.rand(batch.x.size(0), device=device) < (1.0 - AUGMENTATION_NODE_DROP_RATE))

        num_graphs = int(batch.batch.max().item()) + 1
        keep_counts = torch.bincount(batch.batch, weights=keep_mask.to(torch.long), minlength=num_graphs)
        zero_node_graphs = (keep_counts == 0)
        if zero_node_graphs.any():
            node_counts = torch.bincount(batch.batch, minlength=num_graphs)
            ptr = torch.zeros(num_graphs + 1, device=device, dtype=torch.long)
            ptr[1:] = torch.cumsum(node_counts, dim=0)
            starts = ptr[:-1][zero_node_graphs]
            keep_mask[starts] = True

        keep_nodes = keep_mask.nonzero(as_tuple=True)[0]
        edge_index, _ = subgraph(keep_nodes, batch.edge_index)

        batch.x = batch.x[keep_nodes]
        batch.batch = batch.batch[keep_nodes]
        batch.edge_index = edge_index
        return batch


class EdgeDropping:
    """Static utility for edge dropping augmentation."""

    @staticmethod
    def apply(batch: Batch) -> Batch:
        """
        Apply edge dropping to the batch with per-graph safeguards.

        Args:
            batch: PyTorch Geometric Batch object

        Returns:
            Batch object with dropped edges (ensuring each graph retains at least one edge if it had any)
        """
        batch = batch.clone()
        device = batch.edge_index.device
        num_edges = batch.edge_index.shape[1]
        edge_to_graph = batch.batch[batch.edge_index[0]]
        
        keep_mask = (torch.rand(num_edges, device=device) < (1.0 - AUGMENTATION_EDGE_DROP_RATE))
        
        num_graphs = int(batch.batch.max().item()) + 1
        keep_counts = torch.bincount(edge_to_graph, weights=keep_mask.to(torch.long), minlength=num_graphs)
        zero_edge_graphs = (keep_counts == 0)
        if zero_edge_graphs.any():
            edge_counts = torch.bincount(edge_to_graph, minlength=num_graphs)
            ptr = torch.zeros(num_graphs + 1, device=device, dtype=torch.long)
            ptr[1:] = torch.cumsum(edge_counts, dim=0)
            starts = ptr[:-1][zero_edge_graphs]
            keep_mask[starts] = True
        
        keep_edges = keep_mask.nonzero(as_tuple=True)[0]

        batch.edge_index = batch.edge_index[:, keep_edges]

        return batch


class AttributeMasking:
    """Static utility for attribute masking augmentation."""

    @staticmethod
    def apply(batch: Batch) -> Batch:
        """
        Apply attribute masking to the batch.

        Args:
            batch: PyTorch Geometric Batch object

        Returns:
            Batch object with masked node features
        """
        batch = batch.clone()
        device = batch.x.device
        num_features = batch.x.shape[1]

        num_mask = max(1, int(num_features * AUGMENTATION_ATTR_MASK_RATE))

        mask_dims = torch.randperm(num_features, device=device)[:num_mask]

        batch.x[:, mask_dims] = 0.0

        return batch


class GraphAugmentor:
    """Static class for creating augmented graph views for contrastive learning."""

    @staticmethod
    def create_two_views(batch: Batch) -> Tuple[Batch, Batch]:
        """
        Create two augmented views with shared node dropping but independent edge/attribute augmentations.
        
        This is more efficient and conceptually correct for contrastive learning than independent
        augmentations followed by intersection computation.
        
        Args:
            batch: Original batch
            
        Returns:
            Tuple of (view1, view2) with same nodes but different edge/attribute augmentations
        """
        batch_v1 = batch.clone()
        if random.random() < AUGMENTATION_NODE_DROP_PROB:
            batch_v1 = NodeDropping.apply(batch_v1)
        
        batch_v2 = batch_v1.clone()
        
        if random.random() < AUGMENTATION_EDGE_DROP_PROB:
            batch_v1 = EdgeDropping.apply(batch_v1)
        if random.random() < AUGMENTATION_EDGE_DROP_PROB:
            batch_v2 = EdgeDropping.apply(batch_v2)
            
        if random.random() < AUGMENTATION_ATTR_MASK_PROB:
            batch_v1 = AttributeMasking.apply(batch_v1)
        if random.random() < AUGMENTATION_ATTR_MASK_PROB:
            batch_v2 = AttributeMasking.apply(batch_v2)
        
        return batch_v1, batch_v2
