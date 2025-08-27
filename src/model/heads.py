import torch
import torch.nn as nn

from src.model.gnn import GNN_DROPOUT_RATE, GNN_HIDDEN_DIM
from src.model.layers import GradientReversalLayer

CONTRASTIVE_PROJ_DIM = 128
GRAPH_PROP_HEAD_HIDDEN_DIM = 512
DOMAIN_ADV_HEAD_HIDDEN_DIM = 128
DOMAIN_ADV_HEAD_OUT_DIM = 4


class MLPHead(nn.Module):
    def __init__(self, dim_hidden: int = GNN_HIDDEN_DIM, dim_out: int = GNN_HIDDEN_DIM) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(GNN_HIDDEN_DIM, dim_hidden),
            nn.ReLU(),
            nn.Dropout(p=GNN_DROPOUT_RATE),
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DotProductDecoder(nn.Module):
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_src = h[edge_index[0]]
        h_dst = h[edge_index[1]]
        return torch.sigmoid((h_src * h_dst).sum(dim=-1))


class BilinearDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W = nn.Linear(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, bias=False)
        self.dropout = nn.Dropout(p=GNN_DROPOUT_RATE)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        y = self.dropout(y)
        x_expanded = x.unsqueeze(1)
        y_expanded = y.unsqueeze(0)
        x_transformed = self.W(x_expanded)
        return torch.sigmoid((x_transformed * y_expanded).sum(dim=-1))


class DomainClassifierHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(GNN_HIDDEN_DIM, DOMAIN_ADV_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(DOMAIN_ADV_HEAD_HIDDEN_DIM, DOMAIN_ADV_HEAD_OUT_DIM)
        )

    def forward(self, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        x_reversed = self.grl(x, lambda_val)
        return self.classifier(x_reversed)
