import torch
import torch.nn as nn

from src.common import DOMAIN_ADV_HEAD_HIDDEN_DIM, DOMAIN_ADV_HEAD_OUT_DIM, DROPOUT_RATE, GNN_HIDDEN_DIM
from src.model.layers import GradientReversalLayer


class MLPHead(nn.Module):
    """
    General-purpose 2-layer MLP for classification/regression tasks.

    Architecture: Linear -> ReLU -> Dropout -> Linear
    This is used as the standard prediction head throughout the system.
    """

    def __init__(self, dim_hidden: int = GNN_HIDDEN_DIM, dim_out: int = GNN_HIDDEN_DIM) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(self) -> None:
        """Initialize the dot product decoder."""
        super(DotProductDecoder, self).__init__()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute edge probabilities for given node pairs.

        Args:
            h: Node embeddings of shape (num_nodes, embedding_dim)
            edge_index: Edge indices of shape (2, num_edges) representing node pairs

        Returns:
            Edge probabilities of shape (num_edges,)
        """
        h_src = h[edge_index[0]]
        h_dst = h[edge_index[1]]

        edge_scores = torch.sigmoid((h_src * h_dst).sum(dim=-1))

        return edge_scores


class BilinearDiscriminator(nn.Module):
    """
    Parametric discriminator for InfoGraph-style contrastive learning.

    Computes scores for (node_embedding, graph_embedding) pairs using:
    D(x, y) = sigmoid(x^T * W * y)
    """

    def __init__(self) -> None:
        """
        Initialize the bilinear discriminator.
        """
        super(BilinearDiscriminator, self).__init__()

        self.W = nn.Linear(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator scores for input pairs.

        Args:
            x: Node embeddings tensor of shape (N, GNN_HIDDEN_DIM)
            y: Graph embeddings tensor of shape (G, GNN_HIDDEN_DIM) where G is number of graphs

        Returns:
            Scores of shape (N, G) - scores between each node and each graph
        """
        # x: (N, D), y: (G, D)
        # We want to compute scores for all (node, graph) pairs
        x_expanded = x.unsqueeze(1)  # (N, 1, D)
        y_expanded = y.unsqueeze(0)  # (1, G, D)
        
        # Broadcast and compute element-wise multiplication
        x_transformed = self.W(x_expanded)  # (N, 1, D)
        scores = torch.sigmoid((x_transformed * y_expanded).sum(dim=-1))  # (N, G)

        return scores


class DomainClassifierHead(nn.Module):
    """
    Specialized MLP head for domain adversarial training with integrated gradient reversal.

    This head applies gradient reversal before classification to encourage domain-invariant
    features. It does NOT use dropout to ensure it is as strong as possible,
    providing the sharpest training signal for the GNN backbone.

    Architecture: GradientReversalLayer -> Linear -> ReLU -> Linear
    """

    def __init__(self) -> None:
        """
        Initialize the domain classifier head with integrated gradient reversal.
        """
        super(DomainClassifierHead, self).__init__()

        self.grl = GradientReversalLayer()

        self.classifier = nn.Sequential(
            nn.Linear(GNN_HIDDEN_DIM, DOMAIN_ADV_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(DOMAIN_ADV_HEAD_HIDDEN_DIM, DOMAIN_ADV_HEAD_OUT_DIM)
        )

    def forward(self, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        """
        Forward pass through the domain classifier with gradient reversal.

        Args:
            x: Input tensor of shape (batch_size, GNN_HIDDEN_DIM)
            lambda_val: Scaling factor for gradient reversal (from GRL scheduler)

        Returns:
            Domain logits of shape (batch_size, num_pretrain_domains)
        """
        x_reversed = self.grl(x, lambda_val)

        return self.classifier(x_reversed)
