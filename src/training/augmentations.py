"""
Graph augmentations for contrastive learning tasks.

This module implements the augmentations required for node-level contrastive learning:
- Attribute masking (15% of feature dimensions)
- Edge dropping (15% of edges)
- Subgraph sampling (via random walks of length 10)
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.transforms import BaseTransform
import random
from typing import Optional, List, Tuple


class AttributeMasking(BaseTransform):
    """
    Randomly mask a percentage of node feature dimensions.
    
    This augmentation sets selected feature dimensions to zero,
    forcing the model to be robust to missing features.
    """
    
    def __init__(self, mask_rate: float = 0.15):
        """
        Initialize attribute masking transform.
        
        Args:
            mask_rate: Fraction of feature dimensions to mask (default: 15%)
        """
        self.mask_rate = mask_rate
    
    def __call__(self, data: Data) -> Data:
        """
        Apply attribute masking to the data.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Data object with masked node features
        """
        if data.x is None:
            return data
        
        data = data.clone()
        num_features = data.x.shape[1]
        num_mask = int(num_features * self.mask_rate)
        
        if num_mask > 0:
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
    
    def __init__(self, drop_rate: float = 0.15):
        """
        Initialize edge dropping transform.
        
        Args:
            drop_rate: Fraction of edges to drop (default: 15%)
        """
        self.drop_rate = drop_rate
    
    def __call__(self, data: Data) -> Data:
        """
        Apply edge dropping to the data.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Data object with dropped edges
        """
        if data.edge_index is None or data.edge_index.shape[1] == 0:
            return data
        
        data = data.clone()
        num_edges = data.edge_index.shape[1]
        num_keep = int(num_edges * (1 - self.drop_rate))
        
        if num_keep < num_edges and num_keep > 0:
            # Randomly select edges to keep
            keep_indices = torch.randperm(num_edges)[:num_keep]
            data.edge_index = data.edge_index[:, keep_indices]
            
            # Also drop corresponding edge attributes if they exist
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr[keep_indices]
        
        return data


class SubgraphSampling(BaseTransform):
    """
    Sample a subgraph using random walks.
    
    This augmentation creates a subgraph by performing random walks
    from randomly selected starting nodes.
    """
    
    def __init__(self, walk_length: int = 10, num_walks_per_node: int = 1):
        """
        Initialize subgraph sampling transform.
        
        Args:
            walk_length: Length of each random walk (default: 10)
            num_walks_per_node: Number of walks per starting node (default: 1)
        """
        self.walk_length = walk_length
        self.num_walks_per_node = num_walks_per_node
    
    def _random_walk(self, edge_index: torch.Tensor, start_node: int, 
                     walk_length: int, num_nodes: int) -> List[int]:
        """
        Perform a single random walk starting from a given node.
        
        Args:
            edge_index: Edge connectivity
            start_node: Starting node for the walk
            walk_length: Length of the walk
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
        
        for _ in range(walk_length - 1):
            neighbors = adj_list[current_node]
            if len(neighbors) == 0:
                break  # No neighbors, end walk
            
            # Randomly select next node
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
        if data.edge_index is None or data.x is None:
            return data
        
        num_nodes = data.x.shape[0]
        if num_nodes <= 1:
            return data
        
        # Collect nodes from random walks
        sampled_nodes = set()
        
        # Start random walks from random nodes
        num_start_nodes = max(1, num_nodes // 4)  # Start from ~25% of nodes
        start_nodes = torch.randperm(num_nodes)[:num_start_nodes]
        
        for start_node in start_nodes:
            for _ in range(self.num_walks_per_node):
                walk = self._random_walk(
                    data.edge_index, start_node.item(), 
                    self.walk_length, num_nodes
                )
                sampled_nodes.update(walk)
        
        # Ensure we have at least some nodes
        if len(sampled_nodes) == 0:
            sampled_nodes = {0}  # Fallback to first node
        
        sampled_nodes = torch.tensor(list(sampled_nodes), dtype=torch.long)
        
        # Extract subgraph
        edge_index, edge_attr = subgraph(
            sampled_nodes, data.edge_index, data.edge_attr,
            relabel_nodes=True, num_nodes=num_nodes
        )
        
        # Create new data object with subgraph
        new_data = Data(
            x=data.x[sampled_nodes],
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        # Copy other attributes if they exist
        for key, value in data.__dict__.items():
            if key not in ['x', 'edge_index', 'edge_attr'] and not key.startswith('_'):
                setattr(new_data, key, value)
        
        return new_data


class GraphAugmentor:
    """
    Composite augmentor that applies multiple transformations with specified probabilities.
    
    This class orchestrates the application of different augmentations
    as specified in the research plan for contrastive learning.
    """
    
    def __init__(self, 
                 attr_mask_prob: float = 0.5,
                 attr_mask_rate: float = 0.15,
                 edge_drop_prob: float = 0.5,
                 edge_drop_rate: float = 0.15,
                 subgraph_prob: float = 0.5,
                 walk_length: int = 10):
        """
        Initialize the graph augmentor.
        
        Args:
            attr_mask_prob: Probability of applying attribute masking
            attr_mask_rate: Rate of attribute masking when applied
            edge_drop_prob: Probability of applying edge dropping
            edge_drop_rate: Rate of edge dropping when applied
            subgraph_prob: Probability of applying subgraph sampling
            walk_length: Length of random walks for subgraph sampling
        """
        self.transforms = [
            (AttributeMasking(attr_mask_rate), attr_mask_prob),
            (EdgeDropping(edge_drop_rate), edge_drop_prob),
            (SubgraphSampling(walk_length), subgraph_prob)
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


# Convenience function for easy usage
def create_default_augmentor() -> GraphAugmentor:
    """
    Create a default graph augmentor with settings from the research plan.
    
    Returns:
        GraphAugmentor configured with default parameters
    """
    return GraphAugmentor(
        attr_mask_prob=0.5,
        attr_mask_rate=0.15,
        edge_drop_prob=0.5,
        edge_drop_rate=0.15,
        subgraph_prob=0.5,
        walk_length=10
    ) 