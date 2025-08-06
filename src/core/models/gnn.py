import torch
import torch.nn as nn
from torch_geometric.nn import GINConv


class InputEncoder(nn.Module):
    """
    Domain-specific input encoder that maps raw features to shared hidden space.
    
    Architecture: Linear -> LayerNorm -> ReLU -> Dropout
    This creates a standardized 256-dimensional representation from domain-specific features.
    """
    
    def __init__(self, dim_in: int, hidden_dim: int, dropout_rate: float):
        """
        Initialize the input encoder.
        
        Args:
            dim_in: Input feature dimension (domain-specific)
            hidden_dim: Output hidden dimension (shared across domains)
            dropout_rate: Dropout probability
        """
        super(InputEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
    
    def forward(self, x):
        """
        Encode domain-specific features to shared representation.
        
        Args:
            x: Input features of shape (num_nodes, dim_in)
            
        Returns:
            Encoded features of shape (num_nodes, hidden_dim)
        """
        return self.encoder(x)


class GINLayer(nn.Module):
    """
    Single GIN layer with residual connections and post-activation normalization.
    
    This implements the exact architecture from the research plan:
    1. GIN convolution with learnable epsilon
    2. Residual connection
    3. Post-activation: LayerNorm -> ReLU -> Dropout
    """
    
    def __init__(self, hidden_dim: int, dropout_rate: float):
        """
        Initialize a single GIN layer.
        
        Args:
            hidden_dim: Hidden dimension (same for input and output)
            dropout_rate: Dropout probability
        """
        super(GINLayer, self).__init__()
        
        # GIN MLP: Linear -> ReLU -> Linear
        gin_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GIN convolution with learnable epsilon
        self.gin_conv = GINConv(gin_mlp, train_eps=True)
        
        # Post-activation layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, h, edge_index):
        """
        Forward pass through the GIN layer.
        
        Args:
            h: Node embeddings of shape (num_nodes, hidden_dim)
            edge_index: Edge indices of shape (2, num_edges)
            
        Returns:
            Updated node embeddings of shape (num_nodes, hidden_dim)
        """
        # 1. GIN convolution (aggregation + MLP transformation)
        h_conv = self.gin_conv(h, edge_index)
        
        # 2. Residual connection
        h_res = h_conv + h
        
        # 3. Post-activation layers
        h_out = self.layer_norm(h_res)
        h_out = self.relu(h_out)
        h_out = self.dropout(h_out)
        
        return h_out


class GIN_Backbone(nn.Module):
    """
    The main GNN backbone consisting of stacked GIN layers.
    
    This is the shared component that processes all domains after input encoding.
    Default configuration uses 5 layers as specified in the research plan.
    """
    
    def __init__(self, num_layers: int, hidden_dim: int, dropout_rate: float):
        """
        Initialize the GIN backbone.
        
        Args:
            num_layers: Number of GIN layers to stack
            hidden_dim: Hidden dimension throughout the backbone
            dropout_rate: Dropout probability for each layer
        """
        super(GIN_Backbone, self).__init__()
        
        # Stack of GIN layers
        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, dropout_rate) 
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def forward(self, h, edge_index):
        """
        Forward pass through all GIN layers.
        
        Args:
            h: Initial node embeddings of shape (num_nodes, hidden_dim)
            edge_index: Edge indices of shape (2, num_edges)
            
        Returns:
            Final node embeddings of shape (num_nodes, hidden_dim)
        """
        # Pass through each layer sequentially
        for layer in self.layers:
            h = layer(h, edge_index)
        
        return h 