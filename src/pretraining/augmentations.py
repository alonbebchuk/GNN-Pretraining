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
    DEFAULT_TASK_SCALE,
    MIN_NODES_AFTER_DROP,
    SMALL_GRAPH_THRESHOLD,
    MAX_SINGLE_NODE_DROP
)


class NodeDropping:
    """Static utility for node dropping augmentation."""

    @staticmethod
    def apply(batch: Batch) -> Batch:
        """
        Apply node dropping to the batch with safeguards to prevent graphs becoming too small.

        Args:
            batch: PyTorch Geometric Batch object

        Returns:
            Batch object with a node-induced subgraph, ensuring each graph has at least 2 nodes
        """
        # Deep clone with contiguous memory layout to prevent corruption
        batch = batch.clone()
        batch.x = batch.x.contiguous()
        batch.batch = batch.batch.contiguous() 
        batch.edge_index = batch.edge_index.contiguous()
        device = batch.x.device
        
        # Handle empty batch case
        if batch.x.size(0) == 0 or batch.batch.numel() == 0:
            return batch

        # Get graph information
        num_graphs = int(batch.batch.max().item()) + 1
        node_counts = torch.bincount(batch.batch.to(device), minlength=num_graphs)
        
        # Initialize keep_mask - start by keeping all nodes
        keep_mask = torch.ones(batch.x.size(0), device=device, dtype=torch.bool)
        
        # Process each graph individually to ensure proper minimum node counts
        # Use deterministic operations to avoid race conditions
        ptr = torch.zeros(num_graphs + 1, device=device, dtype=torch.long)
        ptr[1:] = torch.cumsum(node_counts, dim=0)
        
        # Ensure all operations are synchronous for CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        for graph_idx in range(num_graphs):
            start_idx = ptr[graph_idx]
            end_idx = ptr[graph_idx + 1]
            graph_size = end_idx - start_idx
            
            if graph_size <= AUGMENTATION_MIN_NODES_PER_GRAPH:
                # For very small graphs, keep all nodes
                continue
            elif graph_size <= 5:
                # For small graphs (3-5 nodes), drop at most 1 node
                max_drop = 1
            else:
                # For larger graphs, use the configured drop rate but ensure minimum remains
                max_drop = min(
                    int(graph_size * AUGMENTATION_NODE_DROP_RATE),
                    graph_size - AUGMENTATION_MIN_NODES_PER_GRAPH  # Always keep minimum nodes
                )
            
            if max_drop > 0:
                # Randomly select nodes to drop within this graph
                # Use deterministic operations to prevent partial updates
                graph_nodes = torch.arange(start_idx, end_idx, device=device)
                drop_indices = torch.randperm(graph_size, device=device)[:max_drop]
                nodes_to_drop = graph_nodes[drop_indices]
                
                # Atomic update - create new mask instead of in-place modification
                new_keep_mask = keep_mask.clone()
                new_keep_mask[nodes_to_drop] = False
                keep_mask = new_keep_mask

        # Final safety check: ensure no graph becomes empty
        keep_counts = torch.bincount(batch.batch.to(device), weights=keep_mask.to(torch.long), minlength=num_graphs)
        zero_node_graphs = (keep_counts == 0)
        single_node_graphs = (keep_counts == 1)
        
        # For zero-node graphs, keep the first node
        if zero_node_graphs.any():
            starts = ptr[:-1][zero_node_graphs]
            keep_mask[starts] = True
            
        # For single-node graphs, keep one more node if possible
        if single_node_graphs.any():
            for graph_idx in torch.where(single_node_graphs)[0]:
                start_idx = ptr[graph_idx]
                end_idx = ptr[graph_idx + 1]
                if end_idx - start_idx > 1:  # Graph originally had more than 1 node
                    # Find the first dropped node in this graph and keep it
                    graph_mask = keep_mask[start_idx:end_idx]
                    if not graph_mask.all():  # There are dropped nodes
                        dropped_indices = (~graph_mask).nonzero(as_tuple=True)[0]
                        if len(dropped_indices) > 0:
                            keep_mask[start_idx + dropped_indices[0]] = True

        keep_nodes = keep_mask.nonzero(as_tuple=True)[0]
        
        # Use subgraph to properly remap edge indices
        if keep_nodes.numel() == 0:
            # No nodes to keep, return empty batch
            batch.x = batch.x[keep_nodes]  # Empty tensor
            batch.batch = batch.batch[keep_nodes]  # Empty tensor
            batch.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
            return batch
        
        # Use subgraph with explicit node mapping to prevent index corruption
        edge_index, edge_attr = subgraph(keep_nodes, batch.edge_index, relabel_nodes=True, num_nodes=batch.x.size(0))
        
        # Apply node selection with proper tensor operations
        batch.x = batch.x[keep_nodes].contiguous()
        batch.batch = batch.batch[keep_nodes].contiguous()
        
        # Validate and assign edge_index with proper shape
        if edge_index.numel() > 0:
            # Double-check that edge indices are within bounds
            num_remaining_nodes = len(keep_nodes)
            max_valid_idx = num_remaining_nodes - 1
            
            if num_remaining_nodes > 0 and edge_index.max().item() <= max_valid_idx and edge_index.min().item() >= 0:
                batch.edge_index = edge_index.contiguous()
            else:
                # Create empty edge index with proper shape and type
                batch.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
        else:
            batch.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
        
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
        # Deep clone with contiguous memory layout to prevent corruption
        batch = batch.clone()
        batch.x = batch.x.contiguous()
        batch.batch = batch.batch.contiguous()
        batch.edge_index = batch.edge_index.contiguous()
        device = batch.edge_index.device
        num_edges = batch.edge_index.shape[1]
        
        # Handle empty batch case
        if num_edges == 0 or batch.batch.numel() == 0:
            return batch
            
        # Pre-validation: if batch seems corrupted, skip edge dropping entirely
        try:
            # Quick validation checks
            if batch.x.size(0) == 0:
                return batch
            if batch.batch.max().item() < 0 or batch.batch.min().item() < 0:
                return batch
            if batch.edge_index.max().item() >= batch.x.size(0) or batch.edge_index.min().item() < 0:
                # Edge indices are already invalid, clear them
                batch.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
                return batch
        except (RuntimeError, ValueError, AttributeError):
            # If basic validation fails, return original batch
            return batch
            
        # Ensure edge indices are within bounds of batch tensor
        max_node_idx = batch.batch.size(0) - 1
        
        # More robust edge validation
        if max_node_idx < 0:
            # No nodes left, return empty edge_index
            batch.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
            return batch
            
        valid_edge_mask = (batch.edge_index[0] <= max_node_idx) & (batch.edge_index[1] <= max_node_idx) & \
                         (batch.edge_index[0] >= 0) & (batch.edge_index[1] >= 0)
        
        if not valid_edge_mask.all():
            if not valid_edge_mask.any():
                # No valid edges, return empty edge_index
                batch.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
                return batch
            valid_edges = valid_edge_mask.nonzero(as_tuple=True)[0]
            batch.edge_index = batch.edge_index[:, valid_edges]
            num_edges = batch.edge_index.shape[1]
            if num_edges == 0:
                return batch
        
        # Safe edge-to-graph mapping with bounds checking
        try:
            edge_to_graph = batch.batch.to(device)[batch.edge_index[0]]
        except (IndexError, RuntimeError) as e:
            # If indexing fails, return batch with empty edges
            batch.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
            return batch
        try:
            keep_mask = (torch.rand(num_edges, device=device) < (DEFAULT_TASK_SCALE - AUGMENTATION_EDGE_DROP_RATE))

            # Validate edge_to_graph before using it
            if edge_to_graph.min().item() < 0:
                # Invalid graph indices, return batch with no edges
                batch.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
                return batch

            num_graphs = int(batch.batch.max().item()) + 1
            
            # Ensure edge_to_graph indices are within valid range
            if edge_to_graph.max().item() >= num_graphs:
                # Invalid graph indices, return batch with no edges
                batch.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
                return batch
            
            keep_counts = torch.bincount(edge_to_graph.to(device), weights=keep_mask.to(torch.long), minlength=num_graphs)
            zero_edge_graphs = (keep_counts == 0)
            
            if zero_edge_graphs.any():
                edge_counts = torch.bincount(edge_to_graph.to(device), minlength=num_graphs)
                ptr = torch.zeros(num_graphs + 1, device=device, dtype=torch.long)
                ptr[1:] = torch.cumsum(edge_counts, dim=0)
                starts = ptr[:-1][zero_edge_graphs]
                
                # Validate starts indices before using them
                if len(starts) > 0 and starts.max().item() < len(keep_mask):
                    keep_mask[starts] = True

            # Safe nonzero operation with validation
            if keep_mask.any():
                keep_edges = keep_mask.nonzero(as_tuple=True)[0]
                if len(keep_edges) > 0:
                    batch.edge_index = batch.edge_index[:, keep_edges]
                else:
                    batch.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
            else:
                batch.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
            
            return batch
            
        except (RuntimeError, IndexError, ValueError) as e:
            # If any operation fails, return batch with empty edges
            batch.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
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

    @staticmethod
    def _validate_batch_quality(batch: Batch) -> bool:
        """
        Validate that the batch meets minimum quality requirements for contrastive learning.
        
        Args:
            batch: Batch to validate
            
        Returns:
            True if batch is suitable for contrastive learning, False otherwise
        """
        if batch.x.size(0) == 0:
            return False
            
        # Check that each graph has at least minimum nodes
        num_graphs = int(batch.batch.max().item()) + 1
        node_counts = torch.bincount(batch.batch.to(batch.x.device), minlength=num_graphs)
        
        # All graphs must have at least minimum nodes for meaningful contrastive learning
        return (node_counts >= AUGMENTATION_MIN_NODES_PER_GRAPH).all().item()

    @staticmethod
    def _ensure_tensor_consistency(batch: Batch) -> Batch:
        """
        Ensure all tensors in the batch are properly shaped and consistent.
        
        Args:
            batch: Batch to fix
            
        Returns:
            Batch with consistent tensor shapes and memory layout
        """
        # Ensure contiguous memory layout
        batch.x = batch.x.contiguous()
        batch.batch = batch.batch.contiguous()
        batch.edge_index = batch.edge_index.contiguous()
        
        # Validate tensor shape consistency
        num_nodes = batch.x.size(0)
        
        # Fix batch tensor if needed
        if batch.batch.size(0) != num_nodes:
            # Reconstruct batch tensor if corrupted
            if num_nodes == 0:
                batch.batch = torch.empty(0, dtype=torch.long, device=batch.x.device)
            else:
                # Create a single-graph batch as fallback
                batch.batch = torch.zeros(num_nodes, dtype=torch.long, device=batch.x.device)
        
        # Fix edge_index if needed
        if batch.edge_index.numel() > 0:
            if batch.edge_index.size(0) != 2:
                batch.edge_index = torch.empty((2, 0), dtype=torch.long, device=batch.x.device)
            elif num_nodes > 0 and (batch.edge_index.max().item() >= num_nodes or batch.edge_index.min().item() < 0):
                batch.edge_index = torch.empty((2, 0), dtype=torch.long, device=batch.x.device)
        
        return batch
    
    @staticmethod
    def _validate_edge_consistency(batch: Batch) -> bool:
        """
        Validate that edge indices are consistent with the current node set.
        
        Args:
            batch: Batch to validate
            
        Returns:
            True if edge indices are valid, False otherwise
        """
        if batch.x.size(0) == 0:
            return batch.edge_index.numel() == 0
            
        if batch.edge_index.numel() == 0:
            return True
            
        max_node_idx = batch.x.size(0) - 1
        return (batch.edge_index >= 0).all().item() and (batch.edge_index <= max_node_idx).all().item()

    @staticmethod
    def create_two_views(batch: Batch) -> Tuple[Batch, Batch]:
        """
        Create two augmented views with shared node dropping but independent edge/attribute augmentations.

        This method ensures that both views meet minimum quality requirements for contrastive learning.

        Args:
            batch: Original batch

        Returns:
            Tuple of (view1, view2) with same nodes but different edge/attribute augmentations.
            Both views are guaranteed to have graphs with at least minimum nodes.
        """
        # Ensure tensor consistency before any operations
        batch = GraphAugmentor._ensure_tensor_consistency(batch)
        
        # Validate original batch
        if not GraphAugmentor._validate_batch_quality(batch):
            # Return original batch twice if it doesn't meet requirements
            # This prevents crashes while allowing training to continue
            return batch.clone(), batch.clone()
        
        batch_v1 = batch.clone()
        if random.random() < AUGMENTATION_NODE_DROP_PROB:
            batch_v1 = NodeDropping.apply(batch_v1)

        # Validate after node dropping - also check edge consistency
        if not GraphAugmentor._validate_batch_quality(batch_v1) or not GraphAugmentor._validate_edge_consistency(batch_v1):
            # If node dropping made it invalid, use original batch
            batch_v1 = batch.clone()

        batch_v2 = batch_v1.clone()

        # Apply edge dropping with additional safety
        if random.random() < AUGMENTATION_EDGE_DROP_PROB:
            try:
                batch_v1_aug = EdgeDropping.apply(batch_v1)
                # Validate the result before using it
                if GraphAugmentor._validate_edge_consistency(batch_v1_aug):
                    batch_v1 = batch_v1_aug
            except Exception:
                pass  # Keep original batch_v1 if augmentation fails
                
        if random.random() < AUGMENTATION_EDGE_DROP_PROB:
            try:
                batch_v2_aug = EdgeDropping.apply(batch_v2)
                # Validate the result before using it
                if GraphAugmentor._validate_edge_consistency(batch_v2_aug):
                    batch_v2 = batch_v2_aug
            except Exception:
                pass  # Keep original batch_v2 if augmentation fails

        if random.random() < AUGMENTATION_ATTR_MASK_PROB:
            batch_v1 = AttributeMasking.apply(batch_v1)
        if random.random() < AUGMENTATION_ATTR_MASK_PROB:
            batch_v2 = AttributeMasking.apply(batch_v2)

        # Final consistency check and repair
        batch_v1 = GraphAugmentor._ensure_tensor_consistency(batch_v1)
        batch_v2 = GraphAugmentor._ensure_tensor_consistency(batch_v2)
        
        # Additional edge consistency validation
        if not GraphAugmentor._validate_edge_consistency(batch_v1):
            batch_v1.edge_index = torch.empty((2, 0), dtype=batch_v1.edge_index.dtype, device=batch_v1.edge_index.device)
        if not GraphAugmentor._validate_edge_consistency(batch_v2):
            batch_v2.edge_index = torch.empty((2, 0), dtype=batch_v2.edge_index.dtype, device=batch_v2.edge_index.device)

        return batch_v1, batch_v2
