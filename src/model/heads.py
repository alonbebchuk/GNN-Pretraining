from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.data_setup import PRETRAIN_TUDATASETS
from src.model.gnn import DROPOUT_RATE, GNN_HIDDEN_DIM
from src.model.layers import GradientReversalLayer

CONTRASTIVE_PROJ_DIM = 128
DOMAIN_ADV_HEAD_HIDDEN_DIM = 128
DOMAIN_ADV_HEAD_OUT_DIM = len(PRETRAIN_TUDATASETS)
GRAPH_PROP_HEAD_HIDDEN_DIM = 512


class MLPHead(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(DROPOUT_RATE))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DotProductDecoder(nn.Module):
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((h[edge_index[0]] * h[edge_index[1]]).sum(dim=-1))


class BilinearDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W = nn.Linear(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        y = F.dropout(y, p=DROPOUT_RATE, training=self.training)
        x_transformed = self.W(x.unsqueeze(1))
        return torch.sigmoid((x_transformed * y.unsqueeze(0)).sum(dim=-1))


class BilinearLinkPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W = nn.Linear(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, bias=False)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_src = F.dropout(h[edge_index[0]], p=DROPOUT_RATE, training=self.training)
        h_dst = F.dropout(h[edge_index[1]], p=DROPOUT_RATE, training=self.training)
        return torch.sigmoid((self.W(h_src) * h_dst).sum(dim=-1))


class DomainClassifierHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grl = GradientReversalLayer()
        self.linear1 = nn.Linear(GNN_HIDDEN_DIM, DOMAIN_ADV_HEAD_HIDDEN_DIM)
        self.linear2 = nn.Linear(DOMAIN_ADV_HEAD_HIDDEN_DIM, DOMAIN_ADV_HEAD_OUT_DIM)

    def forward(self, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        x = self.grl(x, lambda_val)
        x = F.relu(self.linear1(x))
        return self.linear2(x)
