import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """
    General-purpose 2-layer MLP for classification/regression tasks.

    Architecture: Linear -> ReLU -> Dropout -> Linear
    This is used as the standard prediction head throughout the system.
    """

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout_rate: float):
        """
        Initialize the MLP head.

        Args:
            dim_in: Input dimension
            dim_hidden: Hidden layer dimension
            dim_out: Output dimension
            dropout_rate: Dropout probability
        """
        super(MLPHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, dim_in)

        Returns:
            Output tensor of shape (batch_size, dim_out)
        """
        return self.mlp(x)


class DotProductDecoder(nn.Module):
    """
    Non-parametric decoder for link prediction.

    Computes edge probabilities using dot product between node embeddings:
    p(edge) = sigmoid(h_u^T * h_v)

    IMPORTANT: This module only scores provided node pairs. The training script
    is responsible for negative sampling and creating the edge_index tensor.
    """

    def __init__(self):
        """Initialize the dot product decoder."""
        super(DotProductDecoder, self).__init__()

    def forward(self, h, edge_index):
        """
        Compute edge probabilities for given node pairs.

        Args:
            h: Node embeddings of shape (num_nodes, embedding_dim)
            edge_index: Edge indices of shape (2, num_edges) representing node pairs

        Returns:
            Edge probabilities of shape (num_edges,)
        """
        # Get embeddings for source and target nodes
        h_src = h[edge_index[0]]  # Shape: (num_edges, embedding_dim)
        h_dst = h[edge_index[1]]  # Shape: (num_edges, embedding_dim)

        # Compute dot product and apply sigmoid
        edge_scores = torch.sigmoid((h_src * h_dst).sum(dim=-1))

        return edge_scores


class BilinearDiscriminator(nn.Module):
    """
    Parametric discriminator for InfoGraph-style contrastive learning.

    Computes scores for (node_embedding, graph_embedding) pairs using:
    D(x, y) = sigmoid(x^T * W * y)
    """

    def __init__(self, dim1: int, dim2: int):
        """
        Initialize the bilinear discriminator.

        Args:
            dim1: Dimension of the first input (e.g., node embeddings)
            dim2: Dimension of the second input (e.g., graph embeddings)
        """
        super(BilinearDiscriminator, self).__init__()

        # Bilinear transformation matrix (no bias)
        self.W = nn.Linear(dim1, dim2, bias=False)

    def forward(self, x, y):
        """
        Compute discriminator scores for input pairs.

        Args:
            x: First input tensor of shape (N, dim1)
            y: Second input tensor of shape (N, dim2)

        Returns:
            Scores of shape (N,)
        """
        # Apply bilinear transformation: x^T * W * y
        # This is equivalent to (W(x) * y).sum(dim=-1)
        transformed_x = self.W(x)  # Shape: (N, dim2)
        scores = torch.sigmoid((transformed_x * y).sum(dim=-1))

        return scores
