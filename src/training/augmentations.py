"""
Graph augmentations for contrastive learning tasks.

This module implements the augmentations required for node-level contrastive learning.
Each augmentation is applied with p=0.5 probability as part of a sequential composition:

- Attribute masking (% of feature dimensions)
- Edge dropping (% of edges) 
- Subgraph sampling (via random walks of length k)

The implementation follows the GraphCL-style contrastive learning approach where
two different augmented versions (G', G'') are created for each graph to form
positive pairs for contrastive learning.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.transforms import BaseTransform
import random
from typing import List, Tuple

from src.common import AUGMENTATION_ATTR_MASK_PROB, AUGMENTATION_ATTR_MASK_RATE, AUGMENTATION_EDGE_DROP_PROB, AUGMENTATION_EDGE_DROP_RATE, AUGMENTATION_SUBGRAPH_PROB, AUGMENTATION_WALK_LENGTH, AUGMENTATION_MIN_NODES_RATIO


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

        # Ensure at least 1 feature is masked
        num_mask = max(1, int(num_features * AUGMENTATION_ATTR_MASK_RATE))

        # Randomly select feature dimensions to mask
        mask_dims = torch.randperm(num_features)[:num_mask]
        # Set selected dimensions to zero for all nodes
        data.x[:, mask_dims] = 0.0

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
        num_keep = int(num_edges * (1 - AUGMENTATION_EDGE_DROP_RATE))

        # Randomly select edges to keep
        keep_indices = torch.randperm(num_edges)[:num_keep]
        data.edge_index = data.edge_index[:, keep_indices]

        return data


class SubgraphSampling(BaseTransform):
    """
    Sample a subgraph using random walks.

    This augmentation creates a subgraph by performing random walks
    from randomly selected starting nodes.
    """

    def _random_walk(self, edge_index: torch.Tensor, start_node: int, num_nodes: int) -> List[int]:
        """
        Perform a single random walk starting from a given node.

        Args:
            edge_index: Edge connectivity
            start_node: Starting node for the walk
            num_nodes: Total number of nodes in the graph

        Returns:
            List of nodes visited during the walk
        """
        # Create adjacency list for efficient neighbor lookup
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(dst)

        walk = [start_node]
        current_node = start_node

        for _ in range(AUGMENTATION_WALK_LENGTH - 1):
            neighbors = adj_list[current_node]
            if len(neighbors) == 0:
                break

            next_node = random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node

        return walk

    def __call__(self, data: Data) -> Data:
        """
        Apply subgraph sampling to the data.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Data object representing the sampled subgraph
        """
        num_nodes = data.x.shape[0]

        # Collect nodes from random walks
        sampled_nodes = set()

        # Start random walks from random nodes
        num_start_nodes = max(1, int(num_nodes * AUGMENTATION_MIN_NODES_RATIO))
        start_nodes = torch.randperm(num_nodes)[:num_start_nodes]

        for start_node in start_nodes:
            walk = self._random_walk(data.edge_index, start_node.item(), num_nodes)
            sampled_nodes.update(walk)

        sampled_nodes = torch.tensor(list(sampled_nodes), dtype=torch.long)

        # Extract subgraph with node relabeling for proper indexing
        edge_index, _ = subgraph(sampled_nodes, data.edge_index)

        # Create new data object with subgraph
        new_data = Data(x=data.x[sampled_nodes], edge_index=edge_index)

        # Store original node indices for contrastive learning
        # This maps the new node indices to original indices
        new_data.node_indices = sampled_nodes

        return new_data


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
            (AttributeMasking(), AUGMENTATION_ATTR_MASK_PROB),
            (EdgeDropping(), AUGMENTATION_EDGE_DROP_PROB),
            (SubgraphSampling(), AUGMENTATION_SUBGRAPH_PROB)
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
