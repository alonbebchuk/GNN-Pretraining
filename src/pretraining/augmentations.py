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
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.transforms import BaseTransform
import random
from typing import List, Tuple

from src.common import (
    AUGMENTATION_ATTR_MASK_PROB,
    AUGMENTATION_ATTR_MASK_RATE,
    AUGMENTATION_EDGE_DROP_PROB,
    AUGMENTATION_EDGE_DROP_RATE,
    AUGMENTATION_NODE_DROP_PROB,
    AUGMENTATION_NODE_DROP_RATE,
    AUGMENTATION_MIN_ATTR_MASK_DIM,
    AUGMENTATION_MIN_EDGE_NUM_KEEP,
    AUGMENTATION_MIN_NODE_NUM_KEEP,
)


class NodeDropping(BaseTransform):
    """
    Randomly drop a percentage of nodes from the graph and relabel nodes.

    This augmentation induces a node-induced subgraph by keeping a subset of nodes.
    """

    def __call__(self, data: Data) -> Data:
        """
        Apply node dropping to the data.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Data object with a node-induced subgraph
        """
        data = data.clone()
        num_nodes = data.x.shape[0]
        num_keep = max(AUGMENTATION_MIN_NODE_NUM_KEEP, int(num_nodes * (1.0 - AUGMENTATION_NODE_DROP_RATE)))

        # Randomly select nodes to keep
        keep_nodes = torch.randperm(num_nodes)[:num_keep]
        edge_index, _ = subgraph(keep_nodes, data.edge_index)
        data.x = data.x[keep_nodes]
        data.node_indices = data.node_indices[keep_nodes]
        data.edge_index = edge_index

        return data


class EdgeDropping(BaseTransform):
    """
    Randomly drop a percentage of edges from the graph.

    This augmentation removes edges to create structural variations
    while preserving the overall connectivity.
    """

    def __call__(self, data: Data) -> Data:
        """
        Apply edge dropping to the data.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Data object with dropped edges
        """
        data = data.clone()
        num_edges = data.edge_index.shape[1]
        num_keep = max(AUGMENTATION_MIN_EDGE_NUM_KEEP, int(num_edges * (1.0 - AUGMENTATION_EDGE_DROP_RATE)))

        # Randomly select edges to keep
        keep_indices = torch.randperm(num_edges)[:num_keep]
        data.edge_index = data.edge_index[:, keep_indices]

        return data


class AttributeMasking(BaseTransform):
    """
    Randomly mask a percentage of node feature dimensions.

    This augmentation sets selected feature dimensions to zero,
    forcing the model to be robust to missing features.
    """

    def __call__(self, data: Data) -> Data:
        """
        Apply attribute masking to the data.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Data object with masked node features
        """
        data = data.clone()
        num_features = data.x.shape[1]
        num_mask = max(AUGMENTATION_MIN_ATTR_MASK_DIM, int(num_features * AUGMENTATION_ATTR_MASK_RATE))

        # Randomly select feature dimensions to mask
        mask_dims = torch.randperm(num_features)[:num_mask]
        data.x[:, mask_dims] = 0.0

        return data


class GraphAugmentor:
    """
    Composite augmentor that applies multiple transformations with specified probabilities.

    This class orchestrates the application of different augmentations and handles
    node index tracking for contrastive learning. It implements the sequential composition 
    where each transformation is applied with p=0.5 probability as required for 
    GraphCL-style node-level contrastive learning.

    The augmentor automatically initializes node_indices to track the mapping between
    augmented graph nodes and their original indices. Only transformations that change
    the node set need to update this mapping.
    """

    def __init__(self):
        """
        Initialize the graph augmentor.
        """
        self.transforms = [
            (NodeDropping(), AUGMENTATION_NODE_DROP_PROB),
            (EdgeDropping(), AUGMENTATION_EDGE_DROP_PROB),
            (AttributeMasking(), AUGMENTATION_ATTR_MASK_PROB),
        ]

    def __call__(self, data: Data) -> Data:
        """
        Apply augmentations to the data.

        Each transformation is applied with its specified probability.
        The transformations are applied sequentially.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Augmented data object
        """
        augmented_data = data.clone()

        # Initialize node indices for contrastive learning
        # This maps each node in the augmented graph to its original index
        augmented_data.node_indices = torch.arange(augmented_data.x.shape[0], dtype=torch.long)

        for transform, prob in self.transforms:
            if random.random() < prob:
                augmented_data = transform(augmented_data)

        return augmented_data

    def create_augmented_pair(self, data: Data) -> Tuple[Data, Data]:
        """
        Create two different augmented versions of the same graph.

        This is used for contrastive learning where we need two
        different views of the same graph.

        Args:
            data: Original PyTorch Geometric Data object

        Returns:
            Tuple of two augmented data objects (G', G'')
        """
        aug1 = self(data)
        aug2 = self(data)
        return aug1, aug2

    @staticmethod
    def get_overlapping_nodes(view1_data: Data, view2_data: Data) -> torch.Tensor:
        """
        Get nodes that appear in both augmented views for contrastive learning.

        This implements the intersection-based approach from GraphCL, where only
        nodes that appear in both views are used for computing contrastive loss.
        Non-overlapping nodes still contribute to representation learning through
        message passing but don't directly participate in the contrastive objective.

        Args:
            view1_data: First augmented view
            view2_data: Second augmented view

        Returns:
            torch.Tensor: Node indices that appear in both views
        """
        # Get node indices from both views
        view1_nodes = view1_data.node_indices
        view2_nodes = view2_data.node_indices

        # Find intersection of node sets
        view1_set = set(view1_nodes.tolist())
        view2_set = set(view2_nodes.tolist())
        overlap = view1_set & view2_set

        return torch.tensor(list(overlap), dtype=torch.long)

    @staticmethod
    def get_contrastive_pairs(view1_data: Data, view2_data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get positive pairs for contrastive learning from two augmented views.

        Args:
            view1_data: First augmented view
            view2_data: Second augmented view

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (view1_indices, view2_indices)
                where view1_indices[i] and view2_indices[i] form a positive pair
        """
        overlapping_nodes = GraphAugmentor.get_overlapping_nodes(view1_data, view2_data)

        # Get node indices from both views (these map local indices to original indices)
        view1_nodes = view1_data.node_indices
        view2_nodes = view2_data.node_indices

        # Find positions of overlapping nodes in each view using broadcasting
        # view1_mask[i,j] = True if overlapping_nodes[i] == view1_nodes[j]
        view1_mask = view1_nodes.unsqueeze(0) == overlapping_nodes.unsqueeze(1)  # [len(overlap), len(view1)]
        view2_mask = view2_nodes.unsqueeze(0) == overlapping_nodes.unsqueeze(1)  # [len(overlap), len(view2)]

        # Get the local indices (positions) in each view for overlapping nodes
        view1_indices = view1_mask.nonzero(as_tuple=True)[1]  # Local positions in view1
        view2_indices = view2_mask.nonzero(as_tuple=True)[1]  # Local positions in view2

        return view1_indices, view2_indices
