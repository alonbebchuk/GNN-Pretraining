from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from src.data.data_setup import DOMAIN_DIMENSIONS, TASK_TYPES, NUM_CLASSES
from src.models.gnn import GINBackbone, GNN_HIDDEN_DIM, InputEncoder
from src.models.heads import BilinearPredictor, MLPHead

FINETUNE_HIDDEN_DIM = 128


class FinetuneGNN(nn.Module):
    def __init__(self, device: torch.device, domain_name: str, freeze_backbone: bool) -> None:
        super().__init__()

        self.device = device
        self.freeze_backbone = freeze_backbone
        self.task_type = TASK_TYPES[domain_name]
        self.is_in_domain = domain_name == 'ENZYMES'

        self.input_encoder = InputEncoder(DOMAIN_DIMENSIONS[domain_name])
        self.gnn_backbone = GINBackbone()

        num_classes = NUM_CLASSES[domain_name]
        if self.task_type == 'graph_classification':
            self.classification_head = MLPHead([GNN_HIDDEN_DIM, FINETUNE_HIDDEN_DIM, num_classes])
        elif self.task_type == 'node_classification':
            self.classification_head = MLPHead([GNN_HIDDEN_DIM, num_classes])
        elif self.task_type == 'link_prediction':
            self.classification_head = BilinearPredictor()

        self.to(self.device)

        if self.freeze_backbone:
            for param in self.gnn_backbone.parameters():
                param.requires_grad = False

        if self.is_in_domain:
            for param in self.input_encoder.parameters():
                param.requires_grad = False

    def forward(self, batch: Batch, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_0 = self.input_encoder(batch.x)
        node_embeddings = self.gnn_backbone(h_0, batch.edge_index)

        if self.task_type == 'graph_classification':
            graph_embeddings = global_mean_pool(node_embeddings, batch.batch)
            return self.classification_head(graph_embeddings)
        elif self.task_type == 'node_classification':
            return self.classification_head(node_embeddings)
        elif self.task_type == 'link_prediction':
            return self.classification_head(node_embeddings, edge_index)

    def get_optimizer_param_groups(self, lr_backbone: float, lr_head: float) -> List[Dict[str, Any]]:
        param_groups = []

        if not self.freeze_backbone:
            param_groups.append({
                'params': self.gnn_backbone.parameters(),
                'lr': lr_backbone,
                'name': 'backbone'
            })

        if not self.is_in_domain:
            param_groups.append({
                'params': self.input_encoder.parameters(),
                'lr': lr_backbone,
                'name': 'backbone'
            })

        param_groups.append({
            'params': self.classification_head.parameters(),
            'lr': lr_head,
            'name': 'head'
        })
        return param_groups


def load_pretrained_weights(finetune_model: FinetuneGNN, pretrained_checkpoint_path: str) -> None:
    checkpoint = torch.load(pretrained_checkpoint_path, map_location=finetune_model.device)
    pretrained_state_dict = checkpoint['model_state_dict']
    finetune_state_dict = finetune_model.state_dict()

    for key, value in pretrained_state_dict.items():
        if key.startswith('gnn_backbone.') and key in finetune_state_dict:
            finetune_state_dict[key] = value

    if finetune_model.is_in_domain:
        encoder_prefix = f'input_encoders.ENZYMES.'
        for key, value in pretrained_state_dict.items():
            if key.startswith(encoder_prefix):
                finetune_key = key.replace(encoder_prefix, 'input_encoder.')
                if finetune_key in finetune_state_dict:
                    finetune_state_dict[finetune_key] = value

    finetune_model.load_state_dict(finetune_state_dict, strict=False)


def create_finetune_model(device: torch.device, domain_name: str, freeze_backbone: bool, pretrained_checkpoint_path: Optional[str] = None) -> FinetuneGNN:
    model = FinetuneGNN(device, domain_name, freeze_backbone)
    if pretrained_checkpoint_path is not None:
        load_pretrained_weights(model, pretrained_checkpoint_path)
    return model
