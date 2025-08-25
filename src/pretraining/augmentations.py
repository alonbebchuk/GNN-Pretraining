import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import random
from typing import Tuple, Dict, Optional

# Standard GraphCL augmentation parameters
ATTR_MASK_PROB = 0.5     # Probability of applying attribute masking
ATTR_MASK_RATE = 0.3     # Fraction of features to mask
EDGE_DROP_PROB = 0.5     # Probability of applying edge dropping  
EDGE_DROP_RATE = 0.2     # Fraction of edges to drop
NODE_DROP_PROB = 0.5     # Probability of applying node dropping
NODE_DROP_RATE = 0.1     # Fraction of nodes to drop


def attribute_mask(data: Data, mask_rate: float = ATTR_MASK_RATE) -> Tuple[Data, torch.Tensor]:
    """
    Randomly mask node features by setting them to zero.
    Standard GraphCL attribute masking augmentation.
    
    Returns:
        Tuple[Data, torch.Tensor]: (augmented_data, kept_nodes_indices)
            kept_nodes_indices contains all original node indices (no nodes dropped)
    """
    # Handle edge cases
    if data.x is None or data.x.size(0) == 0:
        return data, torch.empty(0, dtype=torch.long, device=data.x.device if data.x is not None else 'cpu')
    
    # Clone to avoid modifying original data
    x = data.x.clone()
    num_nodes, num_features = x.shape
    
    # Handle edge case of no features
    if num_features == 0:
        return data, torch.arange(num_nodes, device=x.device)
    
    # Create random mask - mask_rate fraction of features will be set to zero
    mask = torch.rand(num_nodes, num_features, device=x.device) < mask_rate
    x[mask] = 0.0
    
    # All nodes are kept (no nodes dropped in attribute masking)
    kept_nodes_indices = torch.arange(num_nodes, device=x.device)
    
    augmented_data = Data(x=x, edge_index=data.edge_index, batch=getattr(data, 'batch', None))
    return augmented_data, kept_nodes_indices


def edge_drop(data: Data, drop_rate: float = EDGE_DROP_RATE) -> Tuple[Data, torch.Tensor]:
    """
    Randomly drop edges from the graph.
    Standard GraphCL edge perturbation augmentation.
    
    Returns:
        Tuple[Data, torch.Tensor]: (augmented_data, kept_nodes_indices)
            kept_nodes_indices contains all original node indices (no nodes dropped)
    """
    # Handle edge cases
    if data.x is None or data.x.size(0) == 0:
        device = data.edge_index.device if hasattr(data, 'edge_index') else 'cpu'
        return data, torch.empty(0, dtype=torch.long, device=device)
    
    num_nodes = data.x.size(0)
    kept_nodes_indices = torch.arange(num_nodes, device=data.x.device)
    
    # Handle case of no edges
    if not hasattr(data, 'edge_index') or data.edge_index.size(1) == 0:
        return data, kept_nodes_indices
    
    edge_index = data.edge_index.clone()
    num_edges = edge_index.size(1)
    
    # Handle case where drop_rate would remove all edges
    num_edges_to_keep = max(1, int(num_edges * (1 - drop_rate))) if num_edges > 0 else 0
    
    if num_edges_to_keep >= num_edges:
        # No edges to drop
        return data, kept_nodes_indices
    
    keep_indices = torch.randperm(num_edges, device=edge_index.device)[:num_edges_to_keep]
    edge_index = edge_index[:, keep_indices]
    
    augmented_data = Data(x=data.x, edge_index=edge_index, batch=getattr(data, 'batch', None))
    return augmented_data, kept_nodes_indices


def node_drop(data: Data, drop_rate: float = NODE_DROP_RATE) -> Tuple[Data, torch.Tensor]:
    """
    Randomly drop nodes and their connections.
    Standard GraphCL node dropping augmentation using subgraph sampling.
    
    Returns:
        Tuple[Data, torch.Tensor]: (augmented_data, kept_nodes_indices)
            kept_nodes_indices contains the indices of nodes that were KEPT from the original graph
    """
    # Handle edge cases
    if data.x is None or data.x.size(0) == 0:
        return data, torch.empty(0, dtype=torch.long, device=data.x.device if data.x is not None else 'cpu')
    
    num_nodes = data.num_nodes
    
    # Don't drop if too few nodes or drop_rate is 0
    if num_nodes <= 2 or drop_rate == 0:
        kept_nodes_indices = torch.arange(num_nodes, device=data.x.device)
        return data, kept_nodes_indices
    
    num_nodes_to_keep = max(2, int(num_nodes * (1 - drop_rate)))
    
    # If we're keeping all nodes, return original data
    if num_nodes_to_keep >= num_nodes:
        kept_nodes_indices = torch.arange(num_nodes, device=data.x.device)
        return data, kept_nodes_indices
    
    # Randomly select nodes to keep - these are the indices from the original graph
    kept_nodes_indices = torch.randperm(num_nodes, device=data.x.device)[:num_nodes_to_keep]
    kept_nodes_indices = kept_nodes_indices.sort()[0]  # Sort for consistent ordering
    
    # Create subgraph with kept nodes
    edge_index, _ = subgraph(kept_nodes_indices, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
    x = data.x[kept_nodes_indices]
    
    # Handle batch info if present
    batch = None
    if hasattr(data, 'batch') and data.batch is not None:
        batch = data.batch[kept_nodes_indices]
    
    augmented_data = Data(x=x, edge_index=edge_index, batch=batch)
    return augmented_data, kept_nodes_indices


def create_augmented_view(data: Data) -> Tuple[Data, torch.Tensor]:
    """
    Create one augmented view by randomly applying standard GraphCL augmentations.
    Each augmentation has a 50% chance of being applied.
    
    Returns:
        Tuple[Data, torch.Tensor]: (augmented_data, kept_nodes_indices)
            kept_nodes_indices contains the indices of nodes from the original graph that survived augmentation
    """
    aug_data = data.clone()
    kept_nodes_indices = torch.arange(data.num_nodes, device=data.x.device)
    
    # Randomly apply augmentations (standard GraphCL approach)
    # Node dropping must come first since it changes the number of nodes
    if random.random() < NODE_DROP_PROB:
        aug_data, kept_nodes_indices = node_drop(aug_data)
    
    if random.random() < EDGE_DROP_PROB:
        aug_data, _ = edge_drop(aug_data)  # Edge drop doesn't change kept nodes
        
    if random.random() < ATTR_MASK_PROB:
        aug_data, _ = attribute_mask(aug_data)  # Attr mask doesn't change kept nodes
        
    return aug_data, kept_nodes_indices


class GraphAugmentor:
    """
    Simplified GraphCL-style augmentor using intersection-based node correspondence.
    """
    
    @staticmethod
    def create_two_views(data: Data) -> Tuple[Data, Data]:
        """
        Create two independently augmented views of the input data.
        For backward compatibility - returns only the augmented data.
        """
        view_1, _ = create_augmented_view(data)
        view_2, _ = create_augmented_view(data)
        return view_1, view_2
    
    @staticmethod
    def create_two_views_with_mappings(data: Data) -> Tuple[Data, Data, torch.Tensor, torch.Tensor]:
        """
        Create two independently augmented views with their node mappings.
        
        Returns:
            Tuple[Data, Data, torch.Tensor, torch.Tensor]: 
                (view_1, view_2, kept_nodes_1, kept_nodes_2)
        
        Use find_common_nodes_for_contrastive_loss() to handle the intersection logic.
        """
        view_1, kept_nodes_1 = create_augmented_view(data)
        view_2, kept_nodes_2 = create_augmented_view(data)
        return view_1, view_2, kept_nodes_1, kept_nodes_2


def find_common_nodes_for_contrastive_loss(kept_nodes_1: torch.Tensor, 
                                          kept_nodes_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find nodes common to both views and their indices for contrastive learning.
    This implements the simple intersection-based approach.
    
    Args:
        kept_nodes_1: Original node indices kept in view 1
        kept_nodes_2: Original node indices kept in view 2
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - common_original_nodes: Original node indices present in both views
            - mask_1: Boolean mask to select corresponding embeddings from view 1
            - mask_2: Boolean mask to select corresponding embeddings from view 2
            
    Usage in training loop:
        h1 = encoder(view_1)  # Shape: [num_nodes_in_view1, hidden_dim]
        h2 = encoder(view_2)  # Shape: [num_nodes_in_view2, hidden_dim]
        
        common_nodes, mask_1, mask_2 = find_common_nodes_for_contrastive_loss(kept_nodes_1, kept_nodes_2)
        
        if len(common_nodes) == 0:
            continue  # Skip this step if no nodes overlap
        
        h1_common = h1[mask_1]  # Shape: [num_common_nodes, hidden_dim]
        h2_common = h2[mask_2]  # Shape: [num_common_nodes, hidden_dim]
        
        # Now h1_common[i] and h2_common[i] are embeddings of the same original node
        loss = contrastive_loss(h1_common, h2_common)
    """
    # Find intersection of nodes present in both views
    combined = torch.cat((kept_nodes_1, kept_nodes_2))
    uniques, counts = combined.unique(return_counts=True)
    common_original_nodes = uniques[counts > 1]
    
    if len(common_original_nodes) == 0:
        # No nodes in common - return empty tensors
        device = kept_nodes_1.device
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.bool, device=device),
                torch.empty(0, dtype=torch.bool, device=device))
    
    # Create masks to select the right embeddings from each view
    mask_1 = torch.isin(kept_nodes_1, common_original_nodes)
    mask_2 = torch.isin(kept_nodes_2, common_original_nodes)
    
    return common_original_nodes, mask_1, mask_2


def generate_view(data: Data) -> Tuple[Data, torch.Tensor]:
    """
    Convenience function that generates one view by randomly choosing one augmentation.
    This follows the approach suggested in the GraphCL paper.
    
    Returns:
        Tuple[Data, torch.Tensor]: (augmented_data, kept_nodes_indices)
    """
    augmentations = [
        lambda x: node_drop(x, NODE_DROP_RATE),
        lambda x: edge_drop(x, EDGE_DROP_RATE), 
        lambda x: attribute_mask(x, ATTR_MASK_RATE)
    ]
    aug_fn = random.choice(augmentations)
    return aug_fn(data)


# Example usage in training loop:
"""
Standard GraphCL training loop with intersection-based node correspondence:

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from src.pretraining.augmentations import GraphAugmentor, find_common_nodes_for_contrastive_loss

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class ProjectionHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        return self.linear(x)

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    similarity_matrix = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)  # Your graph data
encoder = GCNEncoder(data.num_features, 128, 64).to(device)
projection_head = ProjectionHead(64, 32).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=0.001)

for epoch in range(200):
    encoder.train()
    projection_head.train()
    optimizer.zero_grad()
    
    # 1. Generate two views with their node mappings
    view_1, view_2, kept_nodes_1, kept_nodes_2 = GraphAugmentor.create_two_views_with_mappings(data)
    
    # 2. Encode the augmented views
    h1 = encoder(view_1.x, view_1.edge_index)
    h2 = encoder(view_2.x, view_2.edge_index)
    
    z1 = projection_head(h1)
    z2 = projection_head(h2)
    
    # 3. Find common nodes and create masks for contrastive loss
    common_nodes, mask_1, mask_2 = find_common_nodes_for_contrastive_loss(kept_nodes_1, kept_nodes_2)
    
    # 4. Skip if no nodes overlap
    if len(common_nodes) == 0:
        continue
    
    # 5. Extract aligned embeddings for positive pairs
    z1_common = z1[mask_1]  # Shape: [num_common_nodes, projection_dim]
    z2_common = z2[mask_2]  # Shape: [num_common_nodes, projection_dim]
    
    # 6. Compute contrastive loss on aligned embeddings
    loss = contrastive_loss(z1_common, z2_common)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Common nodes: {len(common_nodes)}")
"""
