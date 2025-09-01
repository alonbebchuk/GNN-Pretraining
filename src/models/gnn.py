import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

DROPOUT_RATE = 0.2
GNN_HIDDEN_DIM = 256
GNN_NUM_LAYERS = 5


class InputEncoder(nn.Module):
    def __init__(self, dim_in: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, GNN_HIDDEN_DIM)
        self.batch_norm = nn.BatchNorm1d(GNN_HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GINLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gin_conv = GINConv(
            nn.Sequential(
                nn.Linear(GNN_HIDDEN_DIM, 2 * GNN_HIDDEN_DIM),
                nn.BatchNorm1d(2 * GNN_HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(2 * GNN_HIDDEN_DIM, GNN_HIDDEN_DIM)
            ),
            train_eps=True
        )
        self.batch_norm = nn.BatchNorm1d(GNN_HIDDEN_DIM)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_out = self.gin_conv(h, edge_index) + h
        h_out = F.relu(self.batch_norm(h_out))
        return F.dropout(h_out, p=DROPOUT_RATE, training=self.training)


class GINBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([GINLayer() for _ in range(GNN_NUM_LAYERS)])

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, edge_index)
        return h
