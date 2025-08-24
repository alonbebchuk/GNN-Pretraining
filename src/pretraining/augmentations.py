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
    AUGMENTATION_MIN_NODES_PER_GRAPH,
    AUGMENTATION_MIN_EDGES_PER_GRAPH,
    SMALL_GRAPH_EDGE_THRESHOLD,
    MIN_EDGE_RETENTION_RATE,
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
            Batch object with node-induced subgraph
        """
        device = batch.x.device
        
        # Calculate nodes to keep per graph
        num_graphs = int(batch.batch.max().item()) + 1
        node_counts = torch.bincount(batch.batch, minlength=num_graphs)
        
        # Build keep_mask by selecting nodes to KEEP
        keep_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=device)
        
        # Process each graph: determine how many nodes to keep
        ptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_counts, dim=0)])
        
        for graph_idx in range(num_graphs):
            start_idx, end_idx = ptr[graph_idx], ptr[graph_idx + 1]
            graph_size = end_idx - start_idx
            
            # Keep (1 - drop_rate) fraction, but at least MIN_NODES_PER_GRAPH
            nodes_to_keep = max(
                int(graph_size * (1 - AUGMENTATION_NODE_DROP_RATE)),
                AUGMENTATION_MIN_NODES_PER_GRAPH
            )
            
            # Randomly select nodes_to_keep nodes
            graph_indices = torch.arange(start_idx, end_idx, device=device)
            selected_indices = graph_indices[torch.randperm(graph_size, device=device)[:nodes_to_keep]]
            keep_mask[selected_indices] = True

        # Use subgraph to handle edge remapping correctly
        keep_nodes = keep_mask.nonzero(as_tuple=True)[0]
        edge_index, _ = subgraph(keep_nodes, batch.edge_index, relabel_nodes=True, num_nodes=batch.x.size(0))
        
        # Create new batch with selected nodes and remapped edges
        new_batch = Batch()
        new_batch.x = batch.x[keep_nodes]
        new_batch.batch = batch.batch[keep_nodes] 
        new_batch.edge_index = edge_index
        
        return new_batch


class EdgeDropping:
    """Static utility for edge dropping augmentation."""

    @staticmethod
    def apply(batch: Batch) -> Batch:
        """
        Apply structure-aware edge dropping to preserve graph connectivity.
        
        Uses conservative dropping for small graphs and ensures minimum edge retention
        to maintain meaningful graph structure. Creates new batch to avoid modification.

        Args:
            batch: PyTorch Geometric Batch object (NEVER modified)

        Returns:
            New Batch object with structure-aware edge dropping applied
        """
        device = batch.edge_index.device
        num_edges = batch.edge_index.size(1)
        
        # Get edge-to-graph mapping for per-graph processing
        edge_to_graph = batch.batch[batch.edge_index[0]]
        num_graphs = int(batch.batch.max().item()) + 1
        
        # Initialize keep mask
        keep_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        
        # Process each graph individually with structure-aware dropping
        for graph_idx in range(num_graphs):
            graph_edges = (edge_to_graph == graph_idx).nonzero(as_tuple=True)[0]
            graph_edge_count = len(graph_edges)
            
            if graph_edge_count == 0:
                continue  # No edges in this graph
            
            # Determine how many edges to keep based on graph size
            if graph_edge_count <= SMALL_GRAPH_EDGE_THRESHOLD:
                # For small graphs, use conservative dropping
                min_edges_to_keep = max(
                    int(graph_edge_count * MIN_EDGE_RETENTION_RATE),
                    AUGMENTATION_MIN_EDGES_PER_GRAPH
                )
                edges_to_keep = max(
                    int(graph_edge_count * (1 - AUGMENTATION_EDGE_DROP_RATE)),
                    min_edges_to_keep
                )
            else:
                # For larger graphs, apply normal drop rate but ensure minimum
                edges_to_keep = max(
                    int(graph_edge_count * (1 - AUGMENTATION_EDGE_DROP_RATE)),
                    AUGMENTATION_MIN_EDGES_PER_GRAPH
                )
            
            # Randomly select edges_to_keep edges
            selected_edges = graph_edges[torch.randperm(graph_edge_count, device=device)[:edges_to_keep]]
            keep_mask[selected_edges] = True
        
        # Create new batch with selected edges
        new_batch = Batch()
        new_batch.x = batch.x
        new_batch.batch = batch.batch
        
        new_batch.edge_index = batch.edge_index[:, keep_mask]
        
        return new_batch


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
        # Deep clone with contiguous memory layout to prevent corruption
        batch = batch.clone()
        batch.x = batch.x.contiguous()
        device = batch.x.device
        num_features = batch.x.shape[1]

        num_mask = max(1, int(num_features * AUGMENTATION_ATTR_MASK_RATE))

        mask_dims = torch.randperm(num_features, device=device)[:num_mask]

        batch.x[:, mask_dims] = 0.0
        return batch


class GraphAugmentor:
    """Static class for creating augmented graph views for contrastive learning."""

    def create_two_views(batch: Batch) -> Tuple[Batch, Batch]:
        """
        Create two augmented views with shared node dropping but independent edge/attribute augmentations.

        Args:
            batch: Original batch

        Returns:
            Tuple of (view1, view2) - two independent batch objects with different augmentations
        """
        # Apply shared node dropping first
        if random.random() < AUGMENTATION_NODE_DROP_PROB:
            batch = NodeDropping.apply(batch)
        
        batch_v1 = batch.clone()
        batch_v2 = batch.clone()
        
        # Apply independent edge dropping to each view
        if random.random() < AUGMENTATION_EDGE_DROP_PROB:
            batch_v1 = EdgeDropping.apply(batch_v1)
        if random.random() < AUGMENTATION_EDGE_DROP_PROB:
            batch_v2 = EdgeDropping.apply(batch_v2)
        
        # Apply independent attribute masking to each view
        if random.random() < AUGMENTATION_ATTR_MASK_PROB:
            batch_v1 = AttributeMasking.apply(batch_v1)
        if random.random() < AUGMENTATION_ATTR_MASK_PROB:
            batch_v2 = AttributeMasking.apply(batch_v2)

        return batch_v1, batch_v2
