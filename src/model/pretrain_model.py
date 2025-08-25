from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from src.data.data_setup import DOMAIN_DIMENSIONS
from src.data.graph_properties import GRAPH_PROPERTY_DIM
from src.model.gnn import GIN_Backbone, GNN_HIDDEN_DIM, InputEncoder
from src.model.heads import (
    CONTRASTIVE_PROJ_DIM,
    GRAPH_PROP_HEAD_HIDDEN_DIM,
    BilinearDiscriminator,
    DomainClassifierHead,
    DotProductDecoder,
    MLPHead,
)

MASK_TOKEN_INIT_MEAN = 0.0
MASK_TOKEN_INIT_STD = 0.02
NODE_FEATURE_MASKING_MASK_RATE = 0.15


class PretrainableGNN(nn.Module):
    def __init__(self, device: torch.device, domain_names: list[str], task_names: list[str], dropout_rate: float) -> None:
        super(PretrainableGNN, self).__init__()

        self.device = device
        self.dropout_rate = dropout_rate

        self.input_encoders = nn.ModuleDict()
        for domain_name in domain_names:
            dim_in = DOMAIN_DIMENSIONS[domain_name]
            self.input_encoders[domain_name] = InputEncoder(dim_in=dim_in)

        self.mask_token = nn.Parameter(torch.zeros(GNN_HIDDEN_DIM))
        nn.init.normal_(self.mask_token, mean=MASK_TOKEN_INIT_MEAN, std=MASK_TOKEN_INIT_STD)

        self.gnn_backbone = GIN_Backbone()

        self.heads = nn.ModuleDict()
        for task_name in task_names:
            if task_name == 'node_feat_mask':
                dom_heads = nn.ModuleDict()
                for domain_name in domain_names:
                    dom_heads[domain_name] = MLPHead(self.dropout_rate)
                self.heads[task_name] = dom_heads
            elif task_name == 'link_pred':
                self.heads[task_name] = DotProductDecoder()
            elif task_name == 'node_contrast':
                dom_heads = nn.ModuleDict()
                for domain_name in domain_names:
                    dom_heads[domain_name] = MLPHead(self.dropout_rate, dim_out=CONTRASTIVE_PROJ_DIM)
                self.heads[task_name] = dom_heads
            elif task_name == 'graph_contrast':
                dom_heads = nn.ModuleDict()
                for domain_name in domain_names:
                    dom_heads[domain_name] = BilinearDiscriminator(self.dropout_rate)
                self.heads[task_name] = dom_heads
            elif task_name == 'graph_prop':
                dom_heads = nn.ModuleDict()
                for domain_name in domain_names:
                    dom_heads[domain_name] = MLPHead(self.dropout_rate, dim_hidden=GRAPH_PROP_HEAD_HIDDEN_DIM, dim_out=GRAPH_PROPERTY_DIM)
                self.heads[task_name] = dom_heads
            elif task_name == 'domain_adv':
                self.heads[task_name] = DomainClassifierHead()

        self.to(self.device)

    def apply_node_masking(self, batch: Batch, domain_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = batch.to(self.device)

        encoder = self.input_encoders[domain_name]
        with torch.no_grad():
            original_h0 = encoder(batch.x)

        all_mask_indices = []

        for i in range(batch.num_graphs):
            start_idx = batch.ptr[i].item()
            end_idx = batch.ptr[i + 1].item()
            graph_num_nodes = end_idx - start_idx

            num_nodes_to_mask = int(graph_num_nodes * NODE_FEATURE_MASKING_MASK_RATE)
            if num_nodes_to_mask > 0:
                graph_mask_indices = torch.randperm(graph_num_nodes, device=self.device)[:num_nodes_to_mask]
                global_mask_indices = graph_mask_indices + start_idx
                all_mask_indices.append(global_mask_indices)

        mask_indices = torch.cat(all_mask_indices, dim=0)

        masked_h0 = original_h0.clone()
        masked_h0[mask_indices] = self.mask_token.unsqueeze(0).expand(len(mask_indices), -1)

        target_h0 = original_h0[mask_indices].clone()

        return masked_h0, mask_indices, target_h0

    def forward(self, batch: Batch, domain_name: str) -> torch.Tensor:
        batch = batch.to(self.device)

        encoder = self.input_encoders[domain_name]

        h_0 = encoder(batch.x)

        final_node_embeddings = self.gnn_backbone(h_0, batch.edge_index)

        return final_node_embeddings

    def forward_with_h0(self, h_0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gnn_backbone(h_0, edge_index)

    def get_head(self, task_name: str, domain_name: Optional[str] = None) -> nn.Module:
        head = self.heads[task_name]
        if domain_name is not None:
            return head[domain_name]
        else:
            return head
