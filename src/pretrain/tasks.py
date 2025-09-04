from abc import ABC, abstractmethod
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import batched_negative_sampling, to_undirected

from src.data.graph_properties import GRAPH_PROPERTY_DIM
from src.models.gnn import GNN_HIDDEN_DIM
from src.models.pretrain_model import PretrainableGNN
from src.pretrain.augmentations import GraphAugmentor
from src.pretrain.schedulers import TemperatureScheduler, GRLScheduler, INITIAL_TEMP, FINAL_TEMP

HARD_NEGATIVE_RATIO = 0.3
MIN_HARD_NEGATIVES = 8


class HardNegativeMiner:
    def mine_hard_negatives_symmetric(self, embeddings: Tensor, positive_pairs: List[Tuple[int, int]]) -> Tensor:
        embeddings_norm = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())

        batch_size = len(positive_pairs)
        device = embeddings.device

        positive_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        identity_mask = torch.eye(similarity_matrix.size(0), device=device, dtype=torch.bool)

        for i, j in positive_pairs:
            if i < similarity_matrix.size(0) and j < similarity_matrix.size(1):
                positive_mask[i, j] = True
                positive_mask[j, i] = True

        negative_mask = ~(positive_mask | identity_mask)

        hard_negative_indices = []
        for i in range(len(embeddings)):
            if negative_mask[i].sum() == 0:
                continue

            negative_similarities = similarity_matrix[i][negative_mask[i]]
            valid_negative_indices = torch.where(negative_mask[i])[0]

            if len(negative_similarities) == 0:
                continue

            num_hard = max(MIN_HARD_NEGATIVES, int(len(negative_similarities) * HARD_NEGATIVE_RATIO))
            num_hard = min(num_hard, len(negative_similarities))

            _, hard_relative_indices = torch.topk(negative_similarities, num_hard, largest=True)
            hard_absolute_indices = valid_negative_indices[hard_relative_indices]
            hard_negative_indices.append(hard_absolute_indices)

        return hard_negative_indices


class BasePretrainTask(ABC):
    def __init__(self, model: PretrainableGNN) -> None:
        self.model = model

    @abstractmethod
    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator) -> Tuple[Tensor, Dict[str, Tensor]]:
        raise NotImplementedError


class NodeFeatureMaskingTask(BasePretrainTask):
    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = self.model.device
        total_loss = torch.tensor(0.0, device=device)
        total_size = torch.tensor(0, device=device, dtype=torch.long)
        per_domain_losses = {}

        for domain_name, batch in domain_batches.items():
            masked_h0, mask_indices, target_h0 = self.model.apply_node_masking(batch, domain_name, generator)

            if mask_indices.size(0) > 0:
                h_final = self.model.forward_with_h0(masked_h0, batch.edge_index)
                reconstructed_h0 = self.model.get_head('node_feat_mask', domain_name)(h_final[mask_indices])

                loss = F.mse_loss(reconstructed_h0, target_h0, reduction='sum')
                size = torch.tensor(mask_indices.size(0) * GNN_HIDDEN_DIM, device=device, dtype=torch.long)
                total_loss += loss
                total_size += size
                per_domain_losses[domain_name] = loss / size
            else:
                per_domain_losses[domain_name] = torch.tensor(0.0, device=device)

        if total_size > 0:
            total_loss /= total_size
        return total_loss, per_domain_losses


class LinkPredictionTask(BasePretrainTask):
    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = self.model.device
        total_loss = torch.tensor(0.0, device=device)
        total_size = torch.tensor(0, device=device, dtype=torch.long)
        per_domain_losses = {}
        decoder = self.model.get_head('link_pred')

        for domain_name, batch in domain_batches.items():
            pos_edges = batch.edge_index
            neg_edges = batched_negative_sampling(
                edge_index=to_undirected(pos_edges),
                batch=batch.batch,
                num_neg_samples=pos_edges.size(1)
            )
            combined_edges = torch.cat([pos_edges, neg_edges], dim=1)
            labels = torch.cat([
                torch.ones(pos_edges.size(1), device=device, dtype=torch.float32),
                torch.zeros(neg_edges.size(1), device=device, dtype=torch.float32),
            ], dim=0)

            h_final = self.model(batch, domain_name)
            probs = decoder(h_final, combined_edges)
            loss = F.binary_cross_entropy(probs, labels, reduction='sum')
            size = torch.tensor(labels.size(0), device=device, dtype=torch.long)
            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / size

        total_loss /= total_size
        return total_loss, per_domain_losses


class NodeContrastiveTask(BasePretrainTask):
    def __init__(self, model, temperature_scheduler: TemperatureScheduler):
        super().__init__(model)
        self.temperature_scheduler = temperature_scheduler
        self.hard_negative_miner = HardNegativeMiner()

    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = self.model.device
        total_loss = torch.tensor(0.0, device=device)
        total_size = torch.tensor(0, device=device, dtype=torch.long)
        per_domain_losses = {}

        current_temperature = self.temperature_scheduler()

        for domain_name, batch in domain_batches.items():
            batch_v1, batch_v2, mask_1_list, mask_2_list = GraphAugmentor.create_two_views(batch, generator)

            h1 = self.model(batch_v1, domain_name)
            h2 = self.model(batch_v2, domain_name)

            h1_common_list = []
            h2_common_list = []

            for graph_idx, (mask_1, mask_2) in enumerate(zip(mask_1_list, mask_2_list)):
                graph_nodes_1 = (batch_v1.batch == graph_idx)
                graph_nodes_2 = (batch_v2.batch == graph_idx)

                h1_graph = h1[graph_nodes_1]
                h2_graph = h2[graph_nodes_2]

                h1_common_graph = h1_graph[mask_1]
                h2_common_graph = h2_graph[mask_2]

                h1_common_list.append(h1_common_graph)
                h2_common_list.append(h2_common_graph)

            if not h1_common_list or not h2_common_list:
                per_domain_losses[domain_name] = torch.tensor(0.0, device=device)
                continue

            h1_common = torch.cat(h1_common_list, dim=0)
            h2_common = torch.cat(h2_common_list, dim=0)

            if h1_common.size(0) < 2 or h2_common.size(0) < 2:
                per_domain_losses[domain_name] = torch.tensor(0.0, device=device)
                continue

            proj_head = self.model.get_head('node_contrast', domain_name)
            z1 = proj_head(h1_common)
            z2 = proj_head(h2_common)

            loss, size = self._simclr_nt_xent(z1, z2, current_temperature)

            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / size

        if total_size > 0:
            total_loss /= total_size

        return total_loss, per_domain_losses

    def _simclr_nt_xent(self, z1: Tensor, z2: Tensor, temperature: float) -> Tuple[Tensor, torch.Tensor]:
        device = z1.device
        N = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)

        sim_matrix = (z @ z.T) / temperature

        mask = torch.eye(2 * N, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        pos_indices = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device)
        ])

        loss = F.cross_entropy(sim_matrix, pos_indices, reduction='sum')
        size = torch.tensor(2 * N, device=device, dtype=torch.long)
        return loss, size


class GraphContrastiveTask(BasePretrainTask):
    def __init__(self, model, temperature_scheduler: TemperatureScheduler = None):
        super().__init__(model)
        self.temperature_scheduler = temperature_scheduler
        self.hard_negative_miner = HardNegativeMiner()

    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = self.model.device
        total_loss = torch.tensor(0.0, device=device)
        total_size = torch.tensor(0, device=device, dtype=torch.long)
        per_domain_losses = {}

        current_temperature = self.temperature_scheduler()

        for domain_name, batch in domain_batches.items():
            unique_graphs = torch.unique(batch.batch)
            if len(unique_graphs) < 2:
                per_domain_losses[domain_name] = torch.tensor(0.0, device=device)
                continue

            batch_v1, batch_v2, _, _ = GraphAugmentor.create_two_views(batch, generator)

            h1 = self.model(batch_v1, domain_name)
            h2 = self.model(batch_v2, domain_name)

            s1_mean = global_mean_pool(h1, batch_v1.batch)
            s1_max = global_max_pool(h1, batch_v1.batch)
            s1 = torch.cat([s1_mean, s1_max], dim=1)

            s2_mean = global_mean_pool(h2, batch_v2.batch)
            s2_max = global_max_pool(h2, batch_v2.batch)
            s2 = torch.cat([s2_mean, s2_max], dim=1)

            proj_head = self.model.get_head('graph_contrast', domain_name)
            z1 = proj_head(s1)
            z2 = proj_head(s2)

            loss, size = self._graph_contrastive_loss(z1, z2, current_temperature)

            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / size

        if total_size > 0:
            total_loss /= total_size

        return total_loss, per_domain_losses


    def _graph_contrastive_loss(self, z1: Tensor, z2: Tensor, temperature: float) -> Tuple[Tensor, Tensor]:
        device = z1.device
        N = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)

        sim_matrix = torch.mm(z, z.t()) / temperature

        mask = torch.eye(2 * N, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        pos_indices = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device)
        ])

        loss = F.cross_entropy(sim_matrix, pos_indices, reduction='sum')
        size = torch.tensor(2 * N, device=device, dtype=torch.long)

        return loss, size


class GraphPropertyPredictionTask(BasePretrainTask):
    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = self.model.device
        total_loss = torch.tensor(0.0, device=device)
        total_size = torch.tensor(0, device=device, dtype=torch.long)
        per_domain_losses = {}

        for domain_name, batch in domain_batches.items():
            h = self.model(batch, domain_name)
            graph_emb = global_mean_pool(h, batch.batch)

            preds = self.model.get_head('graph_prop', domain_name)(graph_emb)
            labels = batch.graph_properties.to(torch.float32).to(device)
            labels = labels.view(graph_emb.size(0), GRAPH_PROPERTY_DIM)

            loss = F.mse_loss(preds, labels, reduction='sum')
            size = torch.tensor(graph_emb.size(0) * GRAPH_PROPERTY_DIM, device=device, dtype=torch.long)
            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / size

        total_loss /= total_size
        return total_loss, per_domain_losses


class DomainAdversarialTask(BasePretrainTask):
    def __init__(self, model: PretrainableGNN, grl_scheduler: GRLScheduler = None) -> None:
        super().__init__(model)
        self.domain_to_idx = {name: i for i, name in enumerate(self.model.input_encoders.keys())}
        self.grl_scheduler = grl_scheduler

    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = self.model.device
        total_loss = torch.tensor(0.0, device=device)
        total_size = torch.tensor(0, device=device, dtype=torch.long)
        per_domain_losses = {}

        lambda_val = self.grl_scheduler() if self.grl_scheduler is not None else 0.0

        for domain_name, batch in domain_batches.items():
            h = self.model(batch, domain_name)
            graph_emb = global_mean_pool(h, batch.batch)

            logits = self.model.get_head('domain_adv')(graph_emb, lambda_val)
            labels = torch.full((graph_emb.size(0),), self.domain_to_idx[domain_name], device=device, dtype=torch.long)

            loss = F.cross_entropy(logits, labels, reduction='sum')
            size = torch.tensor(labels.size(0), device=device, dtype=torch.long)
            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / size

        total_loss /= total_size
        return total_loss, per_domain_losses
