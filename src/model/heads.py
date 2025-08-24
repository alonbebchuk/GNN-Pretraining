import torch
import torch.nn as nn
from src.model.layers import GradientReversalLayer
from src.common import DOMAIN_ADV_HEAD_HIDDEN_DIM, DOMAIN_ADV_HEAD_OUT_DIM, GNN_HIDDEN_DIM


class MLPHead(nn.Module):
    def __init__(self, dropout_rate: float, dim_hidden: int = GNN_HIDDEN_DIM, dim_out: int = GNN_HIDDEN_DIM) -> None:
        super(MLPHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(GNN_HIDDEN_DIM, dim_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DotProductDecoder(nn.Module):
    def __init__(self) -> None:
        super(DotProductDecoder, self).__init__()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_src = h[edge_index[0]]
        h_dst = h[edge_index[1]]

        edge_scores = torch.sigmoid((h_src * h_dst).sum(dim=-1))

        return edge_scores


class BilinearDiscriminator(nn.Module):
    def __init__(self, dropout_rate: float) -> None:
        super(BilinearDiscriminator, self).__init__()

        self.W = nn.Linear(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        y = self.dropout(y)

        x_expanded = x.unsqueeze(1)
        y_expanded = y.unsqueeze(0)

        x_transformed = self.W(x_expanded)
        scores = torch.sigmoid((x_transformed * y_expanded).sum(dim=-1))

        return scores


class DomainClassifierHead(nn.Module):
    def __init__(self) -> None:
        super(DomainClassifierHead, self).__init__()

        self.grl = GradientReversalLayer()

        self.classifier = nn.Sequential(
            nn.Linear(GNN_HIDDEN_DIM, DOMAIN_ADV_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(DOMAIN_ADV_HEAD_HIDDEN_DIM, DOMAIN_ADV_HEAD_OUT_DIM)
        )

    def forward(self, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        x_reversed = self.grl(x, lambda_val)

        return self.classifier(x_reversed)
