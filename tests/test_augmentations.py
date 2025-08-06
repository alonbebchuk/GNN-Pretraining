import pytest
import torch
from torch_geometric.data import Data

from src.augmentations import (
    AttributeMasking, EdgeDropping, SubgraphSampling,
    GraphAugmentor, create_default_augmentor
)


class TestAttributeMasking:
    """Test attribute masking transformation."""
    
    def test_attribute_masking_basic(self):
        """Test basic attribute masking functionality."""
        transform = AttributeMasking(mask_rate=0.5)
        
        # Create test data
        data = Data(
            x=torch.ones(10, 4),  # 10 nodes, 4 features all set to 1
            edge_index=torch.randint(0, 10, (2, 15))
        )
        
        # Apply transformation
        augmented = transform(data)
        
        # Check that some features are masked (set to 0)
        assert not torch.allclose(augmented.x, data.x)
        assert augmented.x.shape == data.x.shape
        
        # Check that exactly 50% of feature dimensions are masked
        masked_dims = (augmented.x == 0).all(dim=0).sum().item()
        expected_masked = int(4 * 0.5)  # 50% of 4 features
        assert masked_dims == expected_masked
    
    def test_attribute_masking_zero_rate(self):
        """Test attribute masking with zero mask rate."""
        transform = AttributeMasking(mask_rate=0.0)
        
        data = Data(
            x=torch.randn(5, 3),
            edge_index=torch.randint(0, 5, (2, 8))
        )
        
        augmented = transform(data)
        
        # Should be unchanged
        assert torch.allclose(augmented.x, data.x)
    
    def test_attribute_masking_no_features(self):
        """Test attribute masking with no node features."""
        transform = AttributeMasking(mask_rate=0.3)
        
        data = Data(
            x=None,
            edge_index=torch.randint(0, 5, (2, 8))
        )
        
        augmented = transform(data)
        
        # Should return unchanged data
        assert augmented.x is None


class TestEdgeDropping:
    """Test edge dropping transformation."""
    
    def test_edge_dropping_basic(self):
        """Test basic edge dropping functionality."""
        transform = EdgeDropping(drop_rate=0.3)
        
        # Create test data
        data = Data(
            x=torch.randn(8, 5),
            edge_index=torch.randint(0, 8, (2, 20))  # 20 edges
        )
        
        # Apply transformation
        augmented = transform(data)
        
        # Check that edges are dropped
        original_edges = data.edge_index.shape[1]
        remaining_edges = augmented.edge_index.shape[1]
        expected_remaining = int(original_edges * 0.7)  # 70% should remain
        
        assert remaining_edges == expected_remaining
        assert augmented.x.shape == data.x.shape  # Node features unchanged
    
    def test_edge_dropping_with_edge_attr(self):
        """Test edge dropping with edge attributes."""
        transform = EdgeDropping(drop_rate=0.4)
        
        data = Data(
            x=torch.randn(6, 3),
            edge_index=torch.randint(0, 6, (2, 15)),
            edge_attr=torch.randn(15, 2)  # Edge attributes
        )
        
        augmented = transform(data)
        
        # Check that edge attributes are also dropped
        assert augmented.edge_attr.shape[0] == augmented.edge_index.shape[1]
        assert augmented.edge_attr.shape[1] == data.edge_attr.shape[1]
    
    def test_edge_dropping_no_edges(self):
        """Test edge dropping with no edges."""
        transform = EdgeDropping(drop_rate=0.5)
        
        data = Data(
            x=torch.randn(5, 4),
            edge_index=torch.empty((2, 0), dtype=torch.long)
        )
        
        augmented = transform(data)
        
        # Should return unchanged data
        assert augmented.edge_index.shape[1] == 0


class TestSubgraphSampling:
    """Test subgraph sampling transformation."""
    
    def test_subgraph_sampling_basic(self):
        """Test basic subgraph sampling functionality."""
        transform = SubgraphSampling(walk_length=5, num_walks_per_node=1)
        
        # Create a connected graph
        data = Data(
            x=torch.randn(12, 6),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11, 10]])
        )
        
        # Apply transformation
        augmented = transform(data)
        
        # Check that we get a subgraph
        assert augmented.x.shape[0] <= data.x.shape[0]  # Fewer or equal nodes
        assert augmented.x.shape[1] == data.x.shape[1]  # Same feature dimension
        assert augmented.edge_index.shape[0] == 2  # Still edge format
    
    def test_subgraph_sampling_single_node(self):
        """Test subgraph sampling with single node."""
        transform = SubgraphSampling(walk_length=3)
        
        data = Data(
            x=torch.randn(1, 4),
            edge_index=torch.empty((2, 0), dtype=torch.long)
        )
        
        augmented = transform(data)
        
        # Should return the same single node
        assert augmented.x.shape[0] == 1
    
    def test_subgraph_sampling_no_features(self):
        """Test subgraph sampling with no node features."""
        transform = SubgraphSampling(walk_length=4)
        
        data = Data(
            x=None,
            edge_index=torch.randint(0, 8, (2, 12))
        )
        
        augmented = transform(data)
        
        # Should return unchanged data
        assert augmented.x is None


class TestGraphAugmentor:
    """Test the composite graph augmentor."""
    
    def test_graph_augmentor_initialization(self):
        """Test graph augmentor initialization."""
        augmentor = GraphAugmentor(
            attr_mask_prob=0.6,
            attr_mask_rate=0.2,
            edge_drop_prob=0.7,
            edge_drop_rate=0.25,
            subgraph_prob=0.4,
            walk_length=8
        )
        
        assert len(augmentor.transforms) == 3  # Three transforms
        assert all(len(transform_prob) == 2 for transform_prob in augmentor.transforms)
    
    def test_graph_augmentor_application(self):
        """Test graph augmentor application."""
        # Use high probabilities to ensure transformations are applied
        augmentor = GraphAugmentor(
            attr_mask_prob=1.0,  # Always apply
            attr_mask_rate=0.3,
            edge_drop_prob=1.0,  # Always apply
            edge_drop_rate=0.2,
            subgraph_prob=0.0,   # Never apply (to keep predictable)
            walk_length=5
        )
        
        data = Data(
            x=torch.ones(10, 6),  # Easy to detect masking
            edge_index=torch.randint(0, 10, (2, 25))
        )
        
        augmented = augmentor(data)
        
        # Should have applied attribute masking and edge dropping
        assert not torch.allclose(augmented.x, data.x)  # Features changed
        assert augmented.edge_index.shape[1] < data.edge_index.shape[1]  # Edges dropped
    
    def test_create_augmented_pair(self):
        """Test creating augmented pairs for contrastive learning."""
        augmentor = GraphAugmentor()
        
        data = Data(
            x=torch.randn(8, 4),
            edge_index=torch.randint(0, 8, (2, 16))
        )
        
        aug1, aug2 = augmentor.create_augmented_pair(data)
        
        # Should get two valid augmented versions
        # Note: shapes might be different due to subgraph sampling
        assert aug1.x.shape[1] == aug2.x.shape[1] == data.x.shape[1]  # Feature dim preserved
        assert aug1.edge_index.shape[0] == aug2.edge_index.shape[0] == 2  # Edge format preserved
        
        # Both should be valid Data objects
        assert isinstance(aug1, Data)
        assert isinstance(aug2, Data)
        
        # Both should have at least some nodes
        assert aug1.x.shape[0] > 0
        assert aug2.x.shape[0] > 0


class TestDefaultAugmentor:
    """Test the default augmentor creation."""
    
    def test_create_default_augmentor(self):
        """Test creating default augmentor."""
        augmentor = create_default_augmentor()
        
        assert isinstance(augmentor, GraphAugmentor)
        assert len(augmentor.transforms) == 3
        
        # Test that it works on sample data
        data = Data(
            x=torch.randn(6, 5),
            edge_index=torch.randint(0, 6, (2, 12))
        )
        
        augmented = augmentor(data)
        
        # Should return valid Data object
        assert isinstance(augmented, Data)
        assert augmented.x.shape[1] == data.x.shape[1]  # Feature dim preserved


class TestIntegration:
    """Integration tests for augmentations."""
    
    def test_augmentation_pipeline(self):
        """Test complete augmentation pipeline."""
        # Create a realistic graph
        data = Data(
            x=torch.randn(15, 8),
            edge_index=torch.randint(0, 15, (2, 30)),
            edge_attr=torch.randn(30, 3)
        )
        
        # Apply each transformation individually
        attr_mask = AttributeMasking(0.25)
        edge_drop = EdgeDropping(0.3)
        subgraph = SubgraphSampling(walk_length=6)
        
        # Sequential application
        step1 = attr_mask(data)
        step2 = edge_drop(step1)
        step3 = subgraph(step2)
        
        # Should produce valid result
        assert isinstance(step3, Data)
        assert step3.x is not None
        assert step3.edge_index is not None
        
        # Compare with composite augmentor
        augmentor = GraphAugmentor(
            attr_mask_prob=1.0, attr_mask_rate=0.25,
            edge_drop_prob=1.0, edge_drop_rate=0.3,
            subgraph_prob=1.0, walk_length=6
        )
        
        composite_result = augmentor(data)
        
        # Should produce valid result (though different due to randomness)
        assert isinstance(composite_result, Data)
        assert composite_result.x is not None
        assert composite_result.edge_index is not None 