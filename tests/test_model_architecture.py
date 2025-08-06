import pytest
import torch
from torch_geometric.data import Data

from src.layers import GradientReversalLayer
from src.models import (
    InputEncoder, GINLayer, GIN_Backbone,
    MLPHead, DotProductDecoder, BilinearDiscriminator,
    PretrainableGNN, create_full_pretrain_model
)


class TestCustomLayers:
    """Test custom layers implementation."""
    
    def test_gradient_reversal_layer_forward(self):
        """Test that GRL acts as identity in forward pass."""
        grl = GradientReversalLayer()
        x = torch.randn(10, 5)
        lambda_val = 0.5
        
        output = grl(x, lambda_val)
        
        # Should be identity in forward pass
        assert torch.allclose(output, x)
    
    def test_gradient_reversal_layer_backward(self):
        """Test that GRL reverses gradients in backward pass."""
        grl = GradientReversalLayer()
        x = torch.randn(10, 5, requires_grad=True)
        lambda_val = 0.5
        
        output = grl(x, lambda_val)
        loss = output.sum()
        loss.backward()
        
        # Gradient should be reversed and scaled
        expected_grad = torch.ones_like(x) * (-lambda_val)
        assert torch.allclose(x.grad, expected_grad)


class TestPredictionHeads:
    """Test prediction heads implementation."""
    
    def test_mlp_head_shapes(self):
        """Test MLPHead with various input/output dimensions."""
        head = MLPHead(dim_in=256, dim_hidden=128, dim_out=10, dropout_rate=0.1)
        x = torch.randn(32, 256)
        
        output = head(x)
        
        assert output.shape == (32, 10)
    
    def test_dot_product_decoder(self):
        """Test DotProductDecoder for link prediction."""
        decoder = DotProductDecoder()
        h = torch.randn(100, 64)  # 100 nodes, 64-dim embeddings
        edge_index = torch.randint(0, 100, (2, 50))  # 50 edges
        
        scores = decoder(h, edge_index)
        
        assert scores.shape == (50,)
        assert torch.all(scores >= 0) and torch.all(scores <= 1)  # Sigmoid output
    
    def test_bilinear_discriminator(self):
        """Test BilinearDiscriminator for contrastive learning."""
        discriminator = BilinearDiscriminator(dim1=256, dim2=256)
        x = torch.randn(32, 256)
        y = torch.randn(32, 256)
        
        scores = discriminator(x, y)
        
        assert scores.shape == (32,)
        assert torch.all(scores >= 0) and torch.all(scores <= 1)  # Sigmoid output


class TestGNNComponents:
    """Test core GNN components."""
    
    def test_input_encoder(self):
        """Test InputEncoder with different input dimensions."""
        encoder = InputEncoder(dim_in=7, hidden_dim=256, dropout_rate=0.2)
        x = torch.randn(50, 7)
        
        output = encoder(x)
        
        assert output.shape == (50, 256)
    
    def test_gin_layer(self):
        """Test single GIN layer."""
        layer = GINLayer(hidden_dim=256, dropout_rate=0.2)
        h = torch.randn(20, 256)
        edge_index = torch.randint(0, 20, (2, 30))
        
        output = layer(h, edge_index)
        
        assert output.shape == (20, 256)
    
    def test_gin_backbone(self):
        """Test stacked GIN backbone."""
        backbone = GIN_Backbone(num_layers=3, hidden_dim=256, dropout_rate=0.2)
        h = torch.randn(20, 256)
        edge_index = torch.randint(0, 20, (2, 30))
        
        output = backbone(h, edge_index)
        
        assert output.shape == (20, 256)


class TestPretrainableGNN:
    """Test the complete meta-model."""
    
    @pytest.fixture
    def domain_configs(self):
        """Sample domain configurations for testing."""
        return {
            'MUTAG': {'dim_in': 7},
            'PROTEINS': {'dim_in': 1},
            'NCI1': {'dim_in': 37},
            'ENZYMES': {'dim_in': 18}
        }
    
    @pytest.fixture
    def sample_data(self):
        """Sample PyTorch Geometric data for testing."""
        return Data(
            x=torch.randn(20, 7),  # 20 nodes, 7 features (MUTAG)
            edge_index=torch.randint(0, 20, (2, 30))  # 30 edges
        )
    
    def test_model_initialization(self, domain_configs):
        """Test that model initializes correctly."""
        model = PretrainableGNN(
            pretrain_domain_configs=domain_configs,
            hidden_dim=256,
            num_layers=5,
            dropout_rate=0.2
        )
        
        # Check that all components are created
        assert len(model.input_encoders) == 4
        assert 'MUTAG' in model.input_encoders
        assert 'PROTEINS' in model.input_encoders
        assert len(model.heads) == 6  # All prediction heads
        assert model.num_domains == 4
        
        # Check that mask tokens are created with proper initialization
        assert len(model.mask_tokens) == 4
        assert 'MUTAG' in model.mask_tokens
        assert model.mask_tokens['MUTAG'].shape == (7,)  # MUTAG has 7 input features
        assert model.mask_tokens['PROTEINS'].shape == (1,)  # PROTEINS has 1 input feature
        
        # Check that augmentor is created
        assert model.augmentor is not None
    
    def test_input_validation(self):
        """Test input validation in model initialization."""
        # Test empty domain configs
        with pytest.raises(ValueError, match="pretrain_domain_configs cannot be empty"):
            PretrainableGNN({})
        
        # Test negative hidden_dim
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            PretrainableGNN({'MUTAG': {'dim_in': 7}}, hidden_dim=-1)
        
        # Test invalid dropout rate
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            PretrainableGNN({'MUTAG': {'dim_in': 7}}, dropout_rate=1.5)
        
        # Test missing dim_in in config
        with pytest.raises(ValueError, match="missing 'dim_in' in config"):
            PretrainableGNN({'MUTAG': {}})
    
    def test_device_handling(self, domain_configs):
        """Test device handling functionality."""
        # Test CPU device
        model_cpu = PretrainableGNN(domain_configs, device='cpu')
        assert model_cpu.device == torch.device('cpu')
        
        # Test auto device selection
        model_auto = PretrainableGNN(domain_configs, device=None)
        expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert model_auto.device == expected_device
        
        # Test torch.device object
        device_obj = torch.device('cpu')
        model_obj = PretrainableGNN(domain_configs, device=device_obj)
        assert model_obj.device == device_obj
    
    def test_forward_pass(self, domain_configs, sample_data):
        """Test forward pass through the model."""
        model = PretrainableGNN(
            pretrain_domain_configs=domain_configs,
            hidden_dim=256,
            num_layers=3,  # Smaller for faster testing
            dropout_rate=0.2,
            device='cpu'
        )
        
        output = model(sample_data, domain_name='MUTAG')
        
        # Check output structure
        assert 'node_embeddings' in output
        assert 'graph_embedding' in output
        assert 'h_0' in output  # Initial embeddings for reconstruction
        assert output['node_embeddings'].shape == (20, 256)
        assert output['graph_embedding'].shape == (256,)
        assert output['h_0'].shape == (20, 256)
    
    def test_node_masking_functionality(self, domain_configs, sample_data):
        """Test node feature masking functionality."""
        model = PretrainableGNN(domain_configs, device='cpu')
        
        # Test masking with target computation
        masked_data, mask_indices, target_h0 = model.apply_node_masking(
            sample_data, domain_name='MUTAG', mask_rate=0.15, compute_targets=True
        )
        
        # Check that correct number of nodes are masked
        expected_mask_count = int(20 * 0.15)  # 15% of 20 nodes = 3 nodes
        assert len(mask_indices) == expected_mask_count
        assert target_h0.shape == (expected_mask_count, 256)  # h_0 embeddings are 256-dim
        
        # Check that masked nodes have the mask token
        mask_token = model.get_mask_token('MUTAG')
        for idx in mask_indices:
            assert torch.allclose(masked_data.x[idx], mask_token)
        
        # Test masking without target computation (memory efficiency)
        masked_data_eff, mask_indices_eff, target_h0_eff = model.apply_node_masking(
            sample_data, domain_name='MUTAG', mask_rate=0.15, compute_targets=False
        )
        
        assert target_h0_eff is None  # Should not compute targets
        assert len(mask_indices_eff) == expected_mask_count
    
    def test_augmentation_functionality(self, domain_configs, sample_data):
        """Test augmentation functionality."""
        model = PretrainableGNN(domain_configs, enable_augmentations=True, device='cpu')
        
        # Test creating augmented views
        aug_views = model.create_augmented_views(sample_data, num_views=2)
        
        assert len(aug_views) == 2
        assert all(isinstance(view, Data) for view in aug_views)
        assert all(view.x.shape[1] == sample_data.x.shape[1] for view in aug_views)
        
        # Test with augmentations disabled
        model_no_aug = PretrainableGNN(domain_configs, enable_augmentations=False, device='cpu')
        
        with pytest.warns(UserWarning, match="Augmentations are disabled"):
            aug_views_disabled = model_no_aug.create_augmented_views(sample_data, num_views=2)
        
        assert len(aug_views_disabled) == 2
        # Should return copies of original data
        assert all(torch.allclose(view.x, sample_data.x) for view in aug_views_disabled)
    
    def test_mask_token_retrieval(self, domain_configs):
        """Test mask token retrieval functionality."""
        model = PretrainableGNN(domain_configs, device='cpu')
        
        # Test valid mask token retrieval
        mutag_mask = model.get_mask_token('MUTAG')
        assert mutag_mask.shape == (7,)
        assert isinstance(mutag_mask, torch.nn.Parameter)
        
        # Test invalid domain error
        with pytest.raises(ValueError, match="Unknown domain"):
            model.get_mask_token('INVALID_DOMAIN')
    
    def test_zero_masking_edge_case(self, domain_configs):
        """Test edge case where no nodes are masked."""
        model = PretrainableGNN(domain_configs, device='cpu')
        
        # Create very small graph where 15% rounds to 0
        tiny_data = Data(
            x=torch.randn(2, 7),  # Only 2 nodes
            edge_index=torch.tensor([[0, 1], [1, 0]])
        )
        
        masked_data, mask_indices, target_h0 = model.apply_node_masking(
            tiny_data, domain_name='MUTAG', mask_rate=0.15
        )
        
        # Should return empty tensors when no masking occurs
        assert len(mask_indices) == 0
        assert len(target_h0) == 0
        assert torch.allclose(masked_data.x, tiny_data.x)
    
    def test_unknown_domain_error(self, domain_configs, sample_data):
        """Test that unknown domain raises appropriate error."""
        model = PretrainableGNN(domain_configs, device='cpu')
        
        with pytest.raises(ValueError, match="Unknown domain"):
            model(sample_data, domain_name='UNKNOWN')
    
    def test_get_head_method(self, domain_configs):
        """Test head retrieval method."""
        model = PretrainableGNN(domain_configs, device='cpu')
        
        # Test valid head retrieval
        link_head = model.get_head('link_pred')
        assert isinstance(link_head, DotProductDecoder)
        
        # Test invalid head error
        with pytest.raises(ValueError, match="Unknown head"):
            model.get_head('invalid_head')
    
    def test_gradient_reversal_integration(self, domain_configs):
        """Test gradient reversal integration."""
        model = PretrainableGNN(domain_configs, device='cpu')
        embeddings = torch.randn(32, 256, requires_grad=True)
        lambda_val = 0.5
        
        reversed_embeddings = model.apply_gradient_reversal(embeddings, lambda_val)
        
        # Should be identity in forward pass
        assert torch.allclose(reversed_embeddings, embeddings)
    
    def test_domain_list_method(self, domain_configs):
        """Test domain list retrieval."""
        model = PretrainableGNN(domain_configs, device='cpu')
        
        domains = model.get_domain_list()
        
        assert set(domains) == set(['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES'])
    
    def test_model_info(self, domain_configs):
        """Test model information retrieval."""
        model = PretrainableGNN(domain_configs, device='cpu')
        
        info = model.get_model_info()
        
        # Check that all expected keys are present
        expected_keys = {
            'num_domains', 'domains', 'hidden_dim', 'num_layers', 
            'dropout_rate', 'device', 'total_parameters', 
            'trainable_parameters', 'heads', 'augmentations_enabled'
        }
        assert set(info.keys()) == expected_keys
        
        # Check some values
        assert info['num_domains'] == 4
        assert info['hidden_dim'] == 256
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        assert info['augmentations_enabled'] is True


class TestFullPretrainModel:
    """Test the complete pre-training model with all domains."""
    
    def test_create_full_pretrain_model(self):
        """Test creating the full pre-training model."""
        model = create_full_pretrain_model(device='cpu')
        
        # Should have all 6 domains (4 pre-training + 2 additional)
        assert model.num_domains == 6
        expected_domains = {'MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES', 'FRANKENSTEIN', 'PTC_MR'}
        assert set(model.get_domain_list()) == expected_domains
        
        # Check that all mask tokens are created with correct dimensions
        assert model.mask_tokens['MUTAG'].shape == (7,)
        assert model.mask_tokens['PROTEINS'].shape == (4,)  # Updated: actual is 4
        assert model.mask_tokens['NCI1'].shape == (37,)
        assert model.mask_tokens['ENZYMES'].shape == (21,)  # Updated: actual is 21
        assert model.mask_tokens['FRANKENSTEIN'].shape == (780,)
        assert model.mask_tokens['PTC_MR'].shape == (18,)  # Updated: actual is 18
    
    def test_full_model_forward_pass(self):
        """Test forward pass with different domains."""
        model = create_full_pretrain_model(device='cpu', hidden_dim=64, num_layers=2)
        
        # Test with different domain sizes
        test_cases = [
            ('MUTAG', torch.randn(10, 7)),
            ('PROTEINS', torch.randn(15, 4)),  # Updated: actual is 4
            ('FRANKENSTEIN', torch.randn(5, 780)),  # Large feature dimension
            ('PTC_MR', torch.randn(8, 18))  # Updated: actual is 18
        ]
        
        for domain_name, features in test_cases:
            data = Data(
                x=features,
                edge_index=torch.randint(0, features.shape[0], (2, features.shape[0] * 2))
            )
            
            output = model(data, domain_name=domain_name)
            
            assert output['node_embeddings'].shape == (features.shape[0], 64)
            assert output['graph_embedding'].shape == (64,)
            assert output['h_0'].shape == (features.shape[0], 64)


class TestIntegration:
    """Integration tests for the complete architecture."""
    
    def test_end_to_end_forward_pass(self):
        """Test complete forward pass through all components."""
        # Setup model
        domain_configs = {
            'MUTAG': {'dim_in': 7},
            'ENZYMES': {'dim_in': 18}
        }
        model = PretrainableGNN(domain_configs, hidden_dim=64, num_layers=2, device='cpu')
        
        # Create sample data
        data = Data(
            x=torch.randn(10, 7),
            edge_index=torch.randint(0, 10, (2, 15))
        )
        
        # Forward pass
        embeddings = model(data, domain_name='MUTAG')
        
        # Test all heads can process the embeddings
        node_emb = embeddings['node_embeddings']
        graph_emb = embeddings['graph_embedding']
        h_0 = embeddings['h_0']
        
        # Test each head
        mask_head = model.get_head('node_feat_mask')
        reconstructed = mask_head(node_emb)
        assert reconstructed.shape == node_emb.shape
        
        link_head = model.get_head('link_pred')
        edge_scores = link_head(node_emb, data.edge_index)
        assert edge_scores.shape == (15,)
        
        contrast_head = model.get_head('node_contrast')
        projected = contrast_head(node_emb)
        assert projected.shape == (10, 128)
        
        graph_disc = model.get_head('graph_contrast')
        disc_scores = graph_disc(node_emb, graph_emb.unsqueeze(0).expand(10, -1))
        assert disc_scores.shape == (10,)
        
        prop_head = model.get_head('graph_prop')
        properties = prop_head(graph_emb.unsqueeze(0))
        assert properties.shape == (1, 3)
        
        domain_head = model.get_head('domain_adv')
        domain_pred = domain_head(graph_emb.unsqueeze(0))
        assert domain_pred.shape == (1, 2)  # 2 domains
    
    def test_masking_integration_workflow(self):
        """Test the complete masking workflow for node feature reconstruction."""
        # Setup model
        domain_configs = {'MUTAG': {'dim_in': 7}}
        model = PretrainableGNN(domain_configs, hidden_dim=64, num_layers=2, device='cpu')
        
        # Create sample data
        data = Data(
            x=torch.randn(20, 7),
            edge_index=torch.randint(0, 20, (2, 30))
        )
        
        # Apply masking (this computes original h_0 and returns it as targets)
        masked_data, mask_indices, target_h0 = model.apply_node_masking(
            data, domain_name='MUTAG', mask_rate=0.15
        )
        
        # Forward pass with masked data
        embeddings = model(masked_data, domain_name='MUTAG')
        
        # Get reconstruction head and reconstruct masked nodes
        mask_head = model.get_head('node_feat_mask')
        
        if len(mask_indices) > 0:
            # Reconstruct the initial embeddings (h_0) for masked nodes
            masked_node_final_embeddings = embeddings['node_embeddings'][mask_indices]
            reconstructed_h0 = mask_head(masked_node_final_embeddings)
            
            # The reconstruction should match the shape of target h_0
            assert reconstructed_h0.shape == target_h0.shape
            assert target_h0.shape == (len(mask_indices), 64)  # 64 is hidden_dim
            
            # In actual training, the MSE loss would be:
            # loss = F.mse_loss(reconstructed_h0, target_h0.detach())
            # This trains the model to reconstruct the original h_0 embeddings
    
    def test_augmentation_integration_workflow(self):
        """Test the complete augmentation workflow for contrastive learning."""
        model = create_full_pretrain_model(device='cpu', hidden_dim=64, num_layers=2)
        
        # Create sample data
        data = Data(
            x=torch.randn(12, 7),
            edge_index=torch.randint(0, 12, (2, 20))
        )
        
        # Create augmented views for contrastive learning
        aug_views = model.create_augmented_views(data, num_views=2)
        
        # Process both views through the model
        embeddings1 = model(aug_views[0], domain_name='MUTAG')
        embeddings2 = model(aug_views[1], domain_name='MUTAG')
        
        # Get contrastive projection head
        contrast_head = model.get_head('node_contrast')
        
        # Project embeddings for contrastive learning
        proj1 = contrast_head(embeddings1['node_embeddings'])
        proj2 = contrast_head(embeddings2['node_embeddings'])
        
        # Shapes might be different due to subgraph sampling, but projection dim should be consistent
        assert proj1.shape[1] == proj2.shape[1] == 128  # Projection dimension
        assert proj1.shape[0] == aug_views[0].x.shape[0]  # Match augmented view 1
        assert proj2.shape[0] == aug_views[1].x.shape[0]  # Match augmented view 2
        
        # Both should have at least some nodes
        assert proj1.shape[0] > 0
        assert proj2.shape[0] > 0
        
        # In actual training, these would be used for NT-Xent loss
        # with positive pairs being corresponding nodes across views 