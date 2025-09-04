from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.data_setup import PRETRAIN_TUDATASETS
from src.models.gnn import DROPOUT_RATE, GNN_HIDDEN_DIM

CONTRASTIVE_PROJ_DIM = 128
DOMAIN_CLASSIFIER_DROPOUT_RATE = 0.5
DOMAIN_CLASSIFIER_HIDDEN_DIM = 128
GRAPH_PROP_HIDDEN_DIM = 512


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_val, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lambda_val):
        return GradientReversalFunction.apply(x, lambda_val)


class MLPHead(nn.Module):
    def __init__(self, dims: List[int], dropout_rates: List[float] = None) -> None:
        super().__init__()

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                dropout_rate = dropout_rates[i] if dropout_rates is not None else DROPOUT_RATE
                layers.append(nn.Dropout(dropout_rate))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MLPLinkPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = MLPHead([3 * GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, 1])

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_src = h[edge_index[0]]
        h_dst = h[edge_index[1]]

        h_sum = h_src + h_dst
        h_product = h_src * h_dst
        h_diff = torch.abs(h_src - h_dst)

        edge_features = torch.cat([h_sum, h_product, h_diff], dim=1)
        return torch.sigmoid(self.predictor(edge_features).squeeze(-1))


class DomainClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.grl = GradientReversalLayer()

        dims = [GNN_HIDDEN_DIM, DOMAIN_CLASSIFIER_HIDDEN_DIM, len(PRETRAIN_TUDATASETS)]
        dropout_rates = [DOMAIN_CLASSIFIER_DROPOUT_RATE]
        self.classifier = MLPHead(dims, dropout_rates=dropout_rates)

    def forward(self, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        x_reversed = self.grl(x, lambda_val)
        return self.classifier(x_reversed)
