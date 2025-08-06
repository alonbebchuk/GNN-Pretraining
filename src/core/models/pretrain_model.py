import torch
import torch.nn as nn
try:
    # Try relative imports first (for package usage)
    from .gnn import InputEncoder, GIN_Backbone
    from .heads import MLPHead, DotProductDecoder, BilinearDiscriminator
    from ..layers import GradientReversalLayer
    from ...training.augmentations import GraphAugmentor, create_default_augmentor
except ImportError:
    # Fallback to absolute imports (for script usage)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from core.models.gnn import InputEncoder, GIN_Backbone
    from core.models.heads import MLPHead, DotProductDecoder, BilinearDiscriminator
    from core.layers import GradientReversalLayer
    from training.augmentations import GraphAugmentor, create_default_augmentor
from typing import Dict, Optional, Union, Tuple, Any
import warnings


class PretrainableGNN(nn.Module):
    """
    Meta-model for multi-domain GNN pre-training.
    
    This model assembles all components needed for pre-training:
    - Domain-specific input encoders
    - Shared GNN backbone
    - Task-specific prediction heads
    - Gradient reversal layer for domain-adversarial training
    - Learnable [MASK] tokens for node feature masking
    - Graph augmentation capabilities for contrastive learning
    """
    
    def __init__(self, 
                 pretrain_domain_configs: Dict[str, Dict[str, int]], 
                 hidden_dim: int = 256, 
                 num_layers: int = 5, 
                 dropout_rate: float = 0.2,
                 device: Optional[Union[str, torch.device]] = None,
                 enable_augmentations: bool = True):
        """
        Initialize the pretrainable GNN model.
        
        Args:
            pretrain_domain_configs: Dict mapping domain names to their configs
                                   e.g., {'MUTAG': {'dim_in': 7}, 'PROTEINS': {'dim_in': 1}}
            hidden_dim: Hidden dimension for the shared representation
            num_layers: Number of GIN layers in the backbone
            dropout_rate: Dropout rate for regularization
            device: Device to place the model on ('cpu', 'cuda', or torch.device)
            enable_augmentations: Whether to enable graph augmentations
            
        Raises:
            ValueError: If invalid parameters are provided
        """
        super(PretrainableGNN, self).__init__()
        
        # Input validation
        self._validate_inputs(pretrain_domain_configs, hidden_dim, num_layers, dropout_rate)
        
        # Device handling
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        self.hidden_dim = hidden_dim
        self.num_domains = len(pretrain_domain_configs)
        self.domain_configs = pretrain_domain_configs
        self.dropout_rate = dropout_rate
        
        # --- Domain-specific Input Encoders ---
        self.input_encoders = nn.ModuleDict()
        for domain_name, config in pretrain_domain_configs.items():
            self.input_encoders[domain_name] = InputEncoder(
                dim_in=config['dim_in'],
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate
            )
        
        # --- Domain-specific [MASK] tokens for node feature masking ---
        self.mask_tokens = nn.ParameterDict()
        for domain_name, config in pretrain_domain_configs.items():
            # Each domain needs its own [MASK] token with the correct input dimension
            # Use Xavier uniform initialization for better training dynamics
            mask_token = torch.empty(config['dim_in'])
            nn.init.xavier_uniform_(mask_token.unsqueeze(0))
            self.mask_tokens[domain_name] = nn.Parameter(mask_token)
        
        # --- Shared GNN Backbone ---
        self.gnn_backbone = GIN_Backbone(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
        
        # --- Task-specific Prediction Heads ---
        self.heads = nn.ModuleDict({
            # Node feature masking head (reconstruction)
            'node_feat_mask': MLPHead(
                dim_in=hidden_dim,
                dim_hidden=hidden_dim,
                dim_out=hidden_dim,  # Reconstruct 256-dim embeddings
                dropout_rate=dropout_rate
            ),
            
            # Link prediction head (non-parametric)
            'link_pred': DotProductDecoder(),
            
            # Node contrastive learning projection head
            'node_contrast': MLPHead(
                dim_in=hidden_dim,
                dim_hidden=hidden_dim,
                dim_out=128,  # Project to 128-dim for contrastive learning
                dropout_rate=dropout_rate
            ),
            
            # Graph contrastive learning discriminator
            'graph_contrast': BilinearDiscriminator(
                dim1=hidden_dim,  # Node embeddings
                dim2=hidden_dim   # Graph embeddings
            ),
            
            # Graph property prediction head
            'graph_prop': MLPHead(
                dim_in=hidden_dim,
                dim_hidden=hidden_dim,
                dim_out=3,  # Predict 3 properties: num_nodes, num_edges, avg_clustering
                dropout_rate=dropout_rate
            ),
            
            # Domain adversarial classifier (no dropout for sharpest signal)
            'domain_adv': MLPHead(
                dim_in=hidden_dim,
                dim_hidden=128,
                dim_out=self.num_domains,
                dropout_rate=0.0  # No dropout for strongest classifier
            )
        })
        
        # --- Gradient Reversal Layer ---
        self.grl = GradientReversalLayer()
        
        # --- Graph Augmentation ---
        if enable_augmentations:
            self.augmentor = create_default_augmentor()
        else:
            self.augmentor = None
        
        # Move to device
        self.to(self.device)
    
    def _validate_inputs(self, domain_configs: Dict, hidden_dim: int, 
                        num_layers: int, dropout_rate: float) -> None:
        """Validate input parameters."""
        if not domain_configs:
            raise ValueError("pretrain_domain_configs cannot be empty")
        
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        
        for domain_name, config in domain_configs.items():
            if 'dim_in' not in config:
                raise ValueError(f"Domain {domain_name} missing 'dim_in' in config")
            if config['dim_in'] <= 0:
                raise ValueError(f"Domain {domain_name} dim_in must be positive")
    
    def apply_node_masking(self, data, domain_name: str, mask_rate: float = 0.15, 
                          compute_targets: bool = True):
        """
        Apply node feature masking for the node feature masking pre-training task.
        
        Args:
            data: PyTorch Geometric Data object
            domain_name: Name of the domain
            mask_rate: Fraction of nodes to mask (default: 15%)
            compute_targets: Whether to compute reconstruction targets (set False for efficiency)
            
        Returns:
            Tuple of (masked_data, mask_indices, target_h0):
                - masked_data: Data object with masked node features
                - mask_indices: Indices of masked nodes
                - target_h0: Original h_0 embeddings for masked nodes (None if compute_targets=False)
        """
        if domain_name not in self.mask_tokens:
            raise ValueError(f"Unknown domain: {domain_name}. Available domains: {list(self.mask_tokens.keys())}")
        
        # Move data to the correct device
        data = data.to(self.device)
        
        # Compute original h_0 embeddings only if needed (memory efficiency)
        target_h0 = None
        if compute_targets:
            encoder = self.input_encoders[domain_name]
            with torch.no_grad():
                original_h0 = encoder(data.x)
        
        # Clone the data to avoid modifying the original
        masked_data = data.clone()
        num_nodes = data.x.shape[0]
        num_mask = int(num_nodes * mask_rate)
        
        if num_mask == 0:
            # If no nodes to mask, return original data
            return masked_data, torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        
        # Randomly select nodes to mask
        mask_indices = torch.randperm(num_nodes, device=self.device)[:num_mask]
        
        # Store original h_0 embeddings for reconstruction targets
        if compute_targets:
            target_h0 = original_h0[mask_indices].clone()
        
        # Replace selected node features with the learnable [MASK] token
        mask_token = self.mask_tokens[domain_name]
        masked_data.x[mask_indices] = mask_token.unsqueeze(0).expand(num_mask, -1)
        
        return masked_data, mask_indices, target_h0
    
    def create_augmented_views(self, data, num_views: int = 2):
        """
        Create multiple augmented views of the graph for contrastive learning.
        
        Args:
            data: PyTorch Geometric Data object
            num_views: Number of augmented views to create
            
        Returns:
            List of augmented data objects
        """
        if self.augmentor is None:
            warnings.warn("Augmentations are disabled. Returning copies of original data.")
            return [data.clone() for _ in range(num_views)]
        
        return [self.augmentor(data) for _ in range(num_views)]
    
    def forward(self, data, domain_name: str):
        """
        Forward pass through the model.
        
        Args:
            data: PyTorch Geometric Data object containing:
                  - data.x: Node features
                  - data.edge_index: Edge connectivity
            domain_name: Name of the domain (must be in input_encoders keys)
            
        Returns:
            Dictionary containing:
                - 'node_embeddings': Final node embeddings (num_nodes, hidden_dim)
                - 'graph_embedding': Graph-level embedding (hidden_dim,)
                - 'h_0': Initial embeddings for masking reconstruction
        """
        # Input validation
        if domain_name not in self.input_encoders:
            raise ValueError(f"Unknown domain: {domain_name}. Available domains: {list(self.input_encoders.keys())}")
        
        # Move data to the correct device
        data = data.to(self.device)
        
        # 1. Select domain-specific encoder
        encoder = self.input_encoders[domain_name]
        
        # 2. Encode domain-specific features to shared representation
        h_0 = encoder(data.x)
        
        # 3. Process with shared GNN backbone
        final_node_embeddings = self.gnn_backbone(h_0, data.edge_index)
        
        # 4. Compute graph-level summary embedding (mean pooling)
        graph_summary_embedding = final_node_embeddings.mean(dim=0)
        
        return {
            'node_embeddings': final_node_embeddings,
            'graph_embedding': graph_summary_embedding,
            'h_0': h_0  # Include initial embeddings for masking reconstruction
        }
    
    def get_head(self, head_name: str):
        """
        Get a specific prediction head.
        
        Args:
            head_name: Name of the head to retrieve
            
        Returns:
            The requested prediction head module
        """
        if head_name not in self.heads:
            raise ValueError(f"Unknown head: {head_name}. Available heads: {list(self.heads.keys())}")
        
        return self.heads[head_name]
    
    def apply_gradient_reversal(self, embeddings, lambda_val):
        """
        Apply gradient reversal to embeddings for domain-adversarial training.
        
        Args:
            embeddings: Input embeddings tensor
            lambda_val: Scaling factor for gradient reversal
            
        Returns:
            Embeddings with gradient reversal applied
        """
        return self.grl(embeddings, lambda_val)
    
    def get_domain_list(self):
        """
        Get list of available domains.
        
        Returns:
            List of domain names
        """
        return list(self.input_encoders.keys())
    
    def get_mask_token(self, domain_name: str):
        """
        Get the learnable [MASK] token for a specific domain.
        
        Args:
            domain_name: Name of the domain
            
        Returns:
            The [MASK] token parameter for the domain
        """
        if domain_name not in self.mask_tokens:
            raise ValueError(f"Unknown domain: {domain_name}. Available domains: {list(self.mask_tokens.keys())}")
        
        return self.mask_tokens[domain_name]
    
    def validate_model_architecture(self) -> Dict[str, Any]:
        """
        Comprehensive validation of model architecture and components.
        
        Returns:
            Dictionary with validation results and model statistics
        """
        validation_results = {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'components': {},
            'issues': [],
            'architecture_valid': True
        }
        
        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            validation_results['total_parameters'] = total_params
            validation_results['trainable_parameters'] = trainable_params
            
            # Validate input encoders
            encoder_params = {}
            for domain, encoder in self.input_encoders.items():
                params = sum(p.numel() for p in encoder.parameters())
                encoder_params[domain] = params
                
                # Check encoder dimensions
                if hasattr(encoder, 'linear') and encoder.linear.out_features != self.hidden_dim:
                    validation_results['issues'].append(f"Input encoder {domain} output dim mismatch")
                    validation_results['architecture_valid'] = False
            
            validation_results['components']['input_encoders'] = encoder_params
            
            # Validate GNN backbone
            backbone_params = sum(p.numel() for p in self.gnn_backbone.parameters())
            validation_results['components']['gnn_backbone'] = backbone_params
            
            # Validate heads
            head_params = {}
            for head_name, head in self.heads.items():
                params = sum(p.numel() for p in head.parameters())
                head_params[head_name] = params
            
            validation_results['components']['heads'] = head_params
            
            # Validate mask tokens
            mask_token_info = {}
            for domain, token in self.mask_tokens.items():
                mask_token_info[domain] = {
                    'shape': list(token.shape),
                    'requires_grad': token.requires_grad
                }
                
                # Check mask token dimension consistency
                expected_dim = self.domain_configs[domain]['dim_in']
                if token.shape[0] != expected_dim:
                    validation_results['issues'].append(f"Mask token {domain} dimension mismatch: {token.shape[0]} vs {expected_dim}")
                    validation_results['architecture_valid'] = False
            
            validation_results['components']['mask_tokens'] = mask_token_info
            
            # Check device consistency
            model_device = next(self.parameters()).device
            device_consistent = True
            
            for name, param in self.named_parameters():
                if param.device != model_device:
                    validation_results['issues'].append(f"Parameter {name} on wrong device: {param.device} vs {model_device}")
                    device_consistent = False
                    validation_results['architecture_valid'] = False
            
            validation_results['device_consistent'] = device_consistent
            validation_results['model_device'] = str(model_device)
            
            # Memory usage estimation (rough)
            param_memory_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
            validation_results['estimated_memory_mb'] = param_memory_mb
            
        except Exception as e:
            validation_results['issues'].append(f"Validation error: {str(e)}")
            validation_results['architecture_valid'] = False
        
        return validation_results
    
    def print_model_summary(self):
        """Print a comprehensive model summary."""
        validation_results = self.validate_model_architecture()
        
        print("\n" + "="*60)
        print("           MODEL ARCHITECTURE SUMMARY")
        print("="*60)
        
        print(f"Total Parameters: {validation_results['total_parameters']:,}")
        print(f"Trainable Parameters: {validation_results['trainable_parameters']:,}")
        print(f"Estimated Memory: {validation_results['estimated_memory_mb']:.1f} MB")
        print(f"Device: {validation_results['model_device']}")
        print(f"Architecture Valid: {'✅' if validation_results['architecture_valid'] else '❌'}")
        
        print(f"\nComponent Breakdown:")
        for component, info in validation_results['components'].items():
            if isinstance(info, dict):
                print(f"  {component}:")
                for sub_name, value in info.items():
                    if isinstance(value, dict):
                        print(f"    {sub_name}: {value}")
                    else:
                        print(f"    {sub_name}: {value:,} params")
            else:
                print(f"  {component}: {info:,} params")
        
        if validation_results['issues']:
            print(f"\n⚠️  Issues Found ({len(validation_results['issues'])}):")
            for issue in validation_results['issues']:
                print(f"    - {issue}")
        else:
            print(f"\n🎉 No issues found!")
        
        print("="*60)

    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model configuration and statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'num_domains': self.num_domains,
            'domains': self.get_domain_list(),
            'hidden_dim': self.hidden_dim,
            'num_layers': self.gnn_backbone.num_layers,
            'dropout_rate': self.dropout_rate,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'heads': list(self.heads.keys()),
            'augmentations_enabled': self.augmentor is not None
        }


# Convenience function to create model with complete domain configuration
def create_full_pretrain_model(device: Optional[Union[str, torch.device]] = None, 
                              **kwargs) -> PretrainableGNN:
    """
    Create a PretrainableGNN model with the complete domain configuration from the research plan.
    
    Args:
        device: Device to place the model on
        **kwargs: Additional arguments to pass to PretrainableGNN
        
    Returns:
        PretrainableGNN model with all domains configured
    """
    # Complete domain configuration from the research plan
    complete_domain_configs = {
        # Pre-training domains (updated with actual TUDataset dimensions)
        'MUTAG': {'dim_in': 7},
        'PROTEINS': {'dim_in': 4},  # Updated: actual is 4, not 1
        'NCI1': {'dim_in': 37},
        'ENZYMES': {'dim_in': 21},  # Updated: actual is 21, not 18
        
        # Additional TUDatasets (for completeness)
        'FRANKENSTEIN': {'dim_in': 780},
        'PTC_MR': {'dim_in': 18},  # Updated: actual is 18, not 19
        
        # Planetoid datasets (for reference, though not used in pre-training)
        # 'Cora': {'dim_in': 1433},
        # 'CiteSeer': {'dim_in': 3703}
    }
    
    return PretrainableGNN(
        pretrain_domain_configs=complete_domain_configs,
        device=device,
        **kwargs
    ) 