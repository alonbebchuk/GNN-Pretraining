from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from src.data.data_setup import DOMAIN_DIMENSIONS, TASK_TYPES, NUM_CLASSES
from src.models.gnn import GINBackbone, GNN_HIDDEN_DIM, InputEncoder
from src.models.heads import BilinearLinkPredictor, MLPHead

FINETUNE_HIDDEN_DIM = 128
IN_DOMAIN_DATASETS = {'ENZYMES'}
OUT_DOMAIN_DATASETS = {'PTC_MR', 'Cora', 'CiteSeer', 'Cora_LP', 'CiteSeer_LP'}


class FinetuneGNN(nn.Module):
    def __init__(
        self,
        domain_name: str,
        freeze_backbone: bool = False,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()
        
        self.domain_name = domain_name
        self.device = device
        self.freeze_backbone = freeze_backbone
        self.task_type = TASK_TYPES[domain_name]
        self.num_classes = NUM_CLASSES[domain_name]
        self.is_in_domain = domain_name in IN_DOMAIN_DATASETS
        
        self.input_encoder = InputEncoder(DOMAIN_DIMENSIONS[domain_name])
        self.gnn_backbone = GINBackbone()
        
        if self.task_type == 'graph_classification':
            self.classification_head = MLPHead([GNN_HIDDEN_DIM, FINETUNE_HIDDEN_DIM, self.num_classes])
        elif self.task_type == 'node_classification':
            self.classification_head = MLPHead([GNN_HIDDEN_DIM, self.num_classes])
        elif self.task_type == 'link_prediction':
            self.classification_head = BilinearLinkPredictor()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
        self.to(self.device)
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self) -> None:
        for param in self.gnn_backbone.parameters():
            param.requires_grad = False
        if self.is_in_domain:
            for param in self.input_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, batch_or_data, edge_index_for_prediction: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_0 = self.input_encoder(batch_or_data.x)
        node_embeddings = self.gnn_backbone(h_0, batch_or_data.edge_index)
        
        if self.task_type == 'graph_classification':
            graph_embeddings = global_mean_pool(node_embeddings, batch_or_data.batch)
            return self.classification_head(graph_embeddings)
        elif self.task_type == 'node_classification':
            return self.classification_head(node_embeddings)
        elif self.task_type == 'link_prediction':
            if edge_index_for_prediction is None:
                raise ValueError("edge_index_for_prediction required for link prediction")
            return self.classification_head(node_embeddings, edge_index_for_prediction)
        raise ValueError(f"Unknown task type: {self.task_type}")
    
    def get_optimizer_param_groups(self, lr_backbone: float, lr_head: float) -> list:
        param_groups = []
        
        if not self.freeze_backbone:
            backbone_params = list(self.gnn_backbone.parameters())
            if not self.is_in_domain:
                backbone_params.extend(self.input_encoder.parameters())
            param_groups.append({'params': backbone_params, 'lr': lr_backbone, 'name': 'backbone'})
        
        param_groups.append({'params': self.classification_head.parameters(), 'lr': lr_head, 'name': 'head'})
        return param_groups


def load_pretrained_weights(finetune_model: FinetuneGNN, pretrained_checkpoint_path: str) -> None:
    checkpoint = torch.load(pretrained_checkpoint_path, map_location=finetune_model.device)
    pretrained_state_dict = checkpoint['model_state_dict']
    finetune_state_dict = finetune_model.state_dict()
    
    for key in pretrained_state_dict:
        if key.startswith('gnn_backbone.') and key in finetune_state_dict:
            finetune_state_dict[key] = pretrained_state_dict[key]
    
    if finetune_model.is_in_domain:
        encoder_prefix = f'input_encoders.{finetune_model.domain_name}.'
        for key in pretrained_state_dict:
            if key.startswith(encoder_prefix):
                finetune_key = key.replace(encoder_prefix, 'input_encoder.')
                if finetune_key in finetune_state_dict:
                    finetune_state_dict[finetune_key] = pretrained_state_dict[key]
    
    finetune_model.load_state_dict(finetune_state_dict, strict=False)


def create_finetune_model(
    domain_name: str,
    pretrained_checkpoint_path: Optional[str] = None,
    freeze_backbone: bool = False,
    device: torch.device = torch.device('cuda')
) -> FinetuneGNN:
    model = FinetuneGNN(domain_name, freeze_backbone, device)
    if pretrained_checkpoint_path is not None:
        load_pretrained_weights(model, pretrained_checkpoint_path)
    return model
