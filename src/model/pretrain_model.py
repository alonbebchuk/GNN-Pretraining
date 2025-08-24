import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch_geometric.data import Batch
from src.model.gnn import InputEncoder, GIN_Backbone
from src.model.heads import MLPHead, DotProductDecoder, BilinearDiscriminator, DomainClassifierHead
from src.common import (
    DOMAIN_DIMENSIONS,
    GNN_HIDDEN_DIM,
    NODE_FEATURE_MASKING_MASK_RATE,
    MASK_TOKEN_INIT_MEAN,
    MASK_TOKEN_INIT_STD,
    GRAPH_PROPERTY_DIM,
    CONTRASTIVE_PROJ_DIM,
    GRAPH_PROP_HEAD_HIDDEN_DIM,
)


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
        nn.init.normal_(self.mask_token,mean=MASK_TOKEN_INIT_MEAN, std=MASK_TOKEN_INIT_STD)

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

        num_nodes = batch.x.shape[0]

        mask_indices = (torch.rand(num_nodes, device=self.device) < NODE_FEATURE_MASKING_MASK_RATE).nonzero(as_tuple=True)[0]

        target_h0 = original_h0[mask_indices].clone()

        masked_h0 = original_h0.clone()
        masked_h0[mask_indices] = self.mask_token.unsqueeze(0).expand(mask_indices.size(0), -1)

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
