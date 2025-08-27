import torch
import torch.nn as nn
from torch_geometric.nn import GINConv

GNN_HIDDEN_DIM = 256
GNN_NUM_LAYERS = 3
GNN_DROPOUT_RATE = 0.2


class InputEncoder(nn.Module):
    def __init__(self, dim_in: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, GNN_HIDDEN_DIM),
            nn.LayerNorm(GNN_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(p=GNN_DROPOUT_RATE)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class GINLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        gin_mlp = nn.Sequential(
            nn.Linear(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM)
        )
        self.gin_conv = GINConv(gin_mlp, train_eps=True)
        self.layer_norm = nn.LayerNorm(GNN_HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=GNN_DROPOUT_RATE)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_conv = self.gin_conv(h, edge_index)
        h_res = h_conv + h
        h_out = self.layer_norm(h_res)
        h_out = self.relu(h_out)
        return self.dropout(h_out)


class GIN_Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([GINLayer() for _ in range(GNN_NUM_LAYERS)])

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, edge_index)
        return h
