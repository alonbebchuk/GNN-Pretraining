import torch
import torch.nn as nn

from src.common import DOMAIN_ADV_HEAD_HIDDEN_DIM, DOMAIN_ADV_HEAD_OUT_DIM, DROPOUT_RATE, GNN_HIDDEN_DIM


class MLPHead(nn.Module):
    """
    General-purpose 2-layer MLP for classification/regression tasks.

    Architecture: Linear -> ReLU -> Dropout -> Linear
    This is used as the standard prediction head throughout the system.
    """

    def __init__(self, dim_hidden: int = GNN_HIDDEN_DIM, dim_out: int = GNN_HIDDEN_DIM):
        """
        Initialize the MLP head.

        Args:
            dim_hidden: Hidden dimension (defaults to GNN_HIDDEN_DIM)
            dim_out: Output dimension (defaults to GNN_HIDDEN_DIM)
        """
        super(MLPHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(GNN_HIDDEN_DIM, dim_hidden),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, GNN_HIDDEN_DIM)

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

    def __init__(self):
        """
        Initialize the bilinear discriminator.
        """
        super(BilinearDiscriminator, self).__init__()

        # Bilinear transformation matrix (no bias)
        self.W = nn.Linear(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, bias=False)

    def forward(self, x, y):
        """
        Compute discriminator scores for input pairs.

        Args:
            x: First input tensor of shape (N, GNN_HIDDEN_DIM)
            y: Second input tensor of shape (N, GNN_HIDDEN_DIM)

        Returns:
            Scores of shape (N,)
        """
        # Apply bilinear transformation: x^T * W * y
        scores = torch.sigmoid((self.W(x) * y).sum(dim=-1))

        return scores


class DomainClassifierHead(nn.Module):
    """
    Specialized MLP head for domain adversarial training.

    This head does NOT use dropout to ensure it is as strong as possible,
    providing the sharpest training signal for the GNN backbone.

    Architecture: Linear -> ReLU -> Linear
    """

    def __init__(self):
        """
        Initialize the domain classifier head.

        Args:
            dim_hidden: Hidden dimension
            dim_out: Output dimension (number of pretrain domains)
        """
        super(DomainClassifierHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(GNN_HIDDEN_DIM, DOMAIN_ADV_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(DOMAIN_ADV_HEAD_HIDDEN_DIM, DOMAIN_ADV_HEAD_OUT_DIM)
        )

    def forward(self, x):
        """
        Forward pass through the domain classifier.

        Args:
            x: Input tensor of shape (batch_size, GNN_HIDDEN_DIM)

        Returns:
            Domain logits of shape (batch_size, num_pretrain_domains)
        """
        return self.classifier(x)
