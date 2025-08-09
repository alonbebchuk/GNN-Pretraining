import torch.nn as nn
from torch_geometric.nn import GINConv

from src.common import DROPOUT_RATE, GNN_HIDDEN_DIM, GNN_NUM_LAYERS


class InputEncoder(nn.Module):
    """
    Domain-specific input encoder that maps raw features to shared hidden space.

    Architecture: Linear -> LayerNorm -> ReLU -> Dropout
    This creates a standardized representation from domain-specific features.
    """

    def __init__(self, dim_in: int):
        """
        Initialize the input encoder.

        Args:
            dim_in: Input feature dimension (domain-specific)
        """
        super(InputEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dim_in, GNN_HIDDEN_DIM),
            nn.LayerNorm(GNN_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE)
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

    This implements the following architecture:
    1. GIN convolution with learnable epsilon
    2. Residual connection
    3. Post-activation: LayerNorm -> ReLU -> Dropout
    """

    def __init__(self):
        """
        Initialize a single GIN layer.
        """
        super(GINLayer, self).__init__()

        # GIN MLP: Linear -> ReLU -> Linear
        gin_mlp = nn.Sequential(
            nn.Linear(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM)
        )

        # GIN convolution with learnable epsilon
        self.gin_conv = GINConv(gin_mlp, train_eps=True)

        # Post-activation layers
        self.layer_norm = nn.LayerNorm(GNN_HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=DROPOUT_RATE)

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
    """

    def __init__(self):
        """
        Initialize the GIN backbone.
        """
        super(GIN_Backbone, self).__init__()

        # Stack of GIN layers
        self.layers = nn.ModuleList([
            GINLayer()
            for _ in range(GNN_NUM_LAYERS)
        ])

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
