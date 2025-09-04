from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from src.data.data_setup import DOMAIN_DIMENSIONS
from src.data.graph_properties import GRAPH_PROPERTY_DIM
from src.models.gnn import GINBackbone, GNN_HIDDEN_DIM, InputEncoder
from src.models.heads import (
    CONTRASTIVE_PROJ_DIM,
    GRAPH_PROP_HIDDEN_DIM,
    DomainClassifierHead,
    MLPLinkPredictor,
    MLPHead,
)

MASK_TOKEN_INIT_STD = 0.1
NODE_FEATURE_MASKING_MASK_RATE = 0.15
NODE_FEATURE_MASKING_MIN_NUM_NODES = 3


class PretrainableGNN(nn.Module):
    def __init__(self, device: torch.device, domain_names: List[str], task_names: List[str]) -> None:
        super().__init__()
        self.device = device

        self.input_encoders = nn.ModuleDict({
            domain_name: InputEncoder(dim_in=DOMAIN_DIMENSIONS[domain_name])
            for domain_name in domain_names
        })

        self.mask_token = nn.Parameter(torch.zeros(GNN_HIDDEN_DIM))
        nn.init.normal_(self.mask_token, std=MASK_TOKEN_INIT_STD)

        self.gnn_backbone = GINBackbone()

        self.heads = nn.ModuleDict()
        for task_name in task_names:
            if task_name == 'node_feat_mask':
                self.heads[task_name] = nn.ModuleDict({
                    domain_name: MLPHead([GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, GNN_HIDDEN_DIM])
                    for domain_name in domain_names
                })
            elif task_name == 'link_pred':
                self.heads[task_name] = MLPLinkPredictor()
            elif task_name == 'node_contrast':
                self.heads[task_name] = nn.ModuleDict({
                    domain_name: MLPHead([GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, CONTRASTIVE_PROJ_DIM])
                    for domain_name in domain_names
                })
            elif task_name == 'graph_contrast':
                self.heads[task_name] = nn.ModuleDict({
                    domain_name: MLPHead([2 * GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, CONTRASTIVE_PROJ_DIM])
                    for domain_name in domain_names
                })
            elif task_name == 'graph_prop':
                self.heads[task_name] = nn.ModuleDict({
                    domain_name: MLPHead([GNN_HIDDEN_DIM, GRAPH_PROP_HIDDEN_DIM, GRAPH_PROPERTY_DIM])
                    for domain_name in domain_names
                })
            elif task_name == 'domain_adv':
                self.heads[task_name] = DomainClassifierHead()

        self.to(self.device)

    def apply_node_masking(self, batch: Batch, domain_name: str, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            original_h0 = self.input_encoders[domain_name](batch.x)

        mask_indices = []
        for i in range(batch.num_graphs):
            start_idx = batch.ptr[i].item()
            end_idx = batch.ptr[i + 1].item()
            graph_size = end_idx - start_idx

            if graph_size >= NODE_FEATURE_MASKING_MIN_NUM_NODES:
                num_mask = max(1, int(graph_size * NODE_FEATURE_MASKING_MASK_RATE))
                graph_indices = torch.randperm(graph_size, generator=generator)[:num_mask].to(self.device)
                mask_indices.append(graph_indices + start_idx)

        if mask_indices:
            mask_indices = torch.cat(mask_indices)
            masked_h0 = original_h0.clone()
            masked_h0[mask_indices] = self.mask_token.expand(len(mask_indices), -1)
            return masked_h0, mask_indices, original_h0[mask_indices].detach()
        else:
            return original_h0, torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, original_h0.size(1), device=self.device)

    def forward(self, batch: Batch, domain_name: str) -> torch.Tensor:
        h_0 = self.input_encoders[domain_name](batch.x)
        return self.gnn_backbone(h_0, batch.edge_index)

    def forward_with_h0(self, h_0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gnn_backbone(h_0, edge_index)

    def get_head(self, task_name: str, domain_name: Optional[str] = None) -> nn.Module:
        head = self.heads[task_name]
        return head[domain_name] if domain_name is not None else head
