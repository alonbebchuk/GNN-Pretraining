from typing import Optional

import torch
import torch.nn as nn
import wandb
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from src.data.data_setup import DOMAIN_DIMENSIONS, NUM_CLASSES, TASK_TYPES
from src.models.gnn import GINBackbone, GNN_HIDDEN_DIM, InputEncoder
from src.models.heads import MLPLinkPredictor, MLPHead
from src.pretrain.pretrain import OUTPUT_DIR, PROJECT_NAME

FINETUNE_HIDDEN_DIM = 128

LR_BACKBONE = 1e-4
LR_FINETUNE = 1e-3


class FinetuneGNN(nn.Module):
    def __init__(self, device: torch.device, domain_name: str, finetune_strategy: str) -> None:
        super().__init__()

        self.device = device
        self.domain_name = domain_name
        self.input_encoder = InputEncoder(dim_in=DOMAIN_DIMENSIONS[domain_name])
        self.gnn_backbone = GINBackbone()

        num_classes = NUM_CLASSES[domain_name]
        task_type = TASK_TYPES[domain_name]
        if task_type == 'graph_classification':
            self.classification_head = MLPHead([GNN_HIDDEN_DIM, FINETUNE_HIDDEN_DIM, num_classes])
        elif task_type == 'node_classification':
            self.classification_head = MLPHead([GNN_HIDDEN_DIM, num_classes])
        elif task_type == 'link_prediction':
            self.classification_head = MLPLinkPredictor()

        self.param_groups = []

        if domain_name == 'ENZYMES':
            for param in self.input_encoder.parameters():
                param.requires_grad = False
        else:
            self.param_groups.append({
                'params': self.input_encoder.parameters(),
                'lr': LR_FINETUNE,
                'name': 'encoder'
            })

        if finetune_strategy == 'linear_probe':
            for param in self.gnn_backbone.parameters():
                param.requires_grad = False
        else:
            self.param_groups.append({
                'params': self.gnn_backbone.parameters(),
                'lr': LR_BACKBONE,
                'name': 'backbone'
            })

        self.param_groups.append({
            'params': self.classification_head.parameters(),
            'lr': LR_FINETUNE,
            'name': 'head'
        })

        self.to(self.device)

    def forward(self, batch: Batch, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_0 = self.input_encoder(batch.x)
        node_embeddings = self.gnn_backbone(h_0, batch.edge_index)

        task_type = TASK_TYPES[self.domain_name]
        if task_type == 'graph_classification':
            graph_embeddings = global_mean_pool(node_embeddings, batch.batch)
            return self.classification_head(graph_embeddings)
        elif task_type == 'node_classification':
            return self.classification_head(node_embeddings)
        elif task_type == 'link_prediction':
            return self.classification_head(node_embeddings, edge_index)


def get_pretrained_model_path(pretrained_scheme: str, seed: int) -> str:
    model_name = f"model_{pretrained_scheme}_{seed}"
    model_path = OUTPUT_DIR / f"{model_name}.pt"

    if not model_path.exists():
        api = wandb.Api()
        full_artifact_name = f"{PROJECT_NAME}/{model_name}:latest"
        artifact = api.artifact(full_artifact_name)
        artifact.download(root=str(OUTPUT_DIR))

    return str(model_path)


def load_pretrained_weights(finetune_model: FinetuneGNN, pretrained_scheme: str, seed: int) -> None:
    pretrained_path = get_pretrained_model_path(pretrained_scheme, seed)
    checkpoint = torch.load(pretrained_path, map_location=finetune_model.device, weights_only=False)
    pretrained_state_dict = checkpoint['model_state_dict']
    finetune_state_dict = finetune_model.state_dict()

    for key, value in pretrained_state_dict.items():
        if key.startswith('gnn_backbone.') and key in finetune_state_dict:
            finetune_state_dict[key] = value

    if finetune_model.domain_name == 'ENZYMES':
        encoder_prefix = f'input_encoders.ENZYMES.'
        for key, value in pretrained_state_dict.items():
            if key.startswith(encoder_prefix):
                finetune_key = key.replace(encoder_prefix, 'input_encoder.')
                if finetune_key in finetune_state_dict:
                    finetune_state_dict[finetune_key] = value

    finetune_model.load_state_dict(finetune_state_dict, strict=False)


def create_finetune_model(device: torch.device, cfg) -> FinetuneGNN:
    model = FinetuneGNN(device, cfg.domain_name, cfg.finetune_strategy)
    if cfg.pretrained_scheme != 'b1':
        load_pretrained_weights(model, cfg.pretrained_scheme, cfg.seed)
    return model
