from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import batched_negative_sampling, to_undirected

from src.data.graph_properties import GRAPH_PROPERTY_DIM
from src.models.gnn import GNN_HIDDEN_DIM
from src.models.pretrain_model import PretrainableGNN
from src.pretrain.augmentations import GraphAugmentor

NODE_CONTRASTIVE_TEMPERATURE = 0.1


class BasePretrainTask(ABC):
    def __init__(self, model: PretrainableGNN) -> None:
        self.model = model

    @abstractmethod
    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator, **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        raise NotImplementedError


class NodeFeatureMaskingTask(BasePretrainTask):
    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator, **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
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
    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator, **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
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
                num_neg_samples=pos_edges.size(1),
                generator=generator
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
    def _simclr_nt_xent(self, z1: Tensor, z2: Tensor) -> Tuple[Tensor, torch.Tensor]:
        device = z1.device
        N = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)

        sim_matrix = (z @ z.T) / NODE_CONTRASTIVE_TEMPERATURE

        mask = torch.eye(2 * N, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        pos_indices = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device)
        ])

        loss = F.cross_entropy(sim_matrix, pos_indices, reduction='sum')
        size = torch.tensor(2 * N, device=device, dtype=torch.long)
        return loss, size

    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator, **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = self.model.device
        total_loss = torch.tensor(0.0, device=device)
        total_size = torch.tensor(0, device=device, dtype=torch.long)
        per_domain_losses = {}

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

            h1_common = torch.cat(h1_common_list, dim=0)
            h2_common = torch.cat(h2_common_list, dim=0)

            proj_head = self.model.get_head('node_contrast', domain_name)
            z1 = proj_head(h1_common)
            z2 = proj_head(h2_common)

            loss, size = self._simclr_nt_xent(z1, z2)
            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / size

        total_loss /= total_size
        return total_loss, per_domain_losses


class GraphContrastiveTask(BasePretrainTask):
    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator, **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = self.model.device
        total_loss = torch.tensor(0.0, device=device)
        total_size = torch.tensor(0, device=device, dtype=torch.long)
        per_domain_losses = {}

        for domain_name, batch in domain_batches.items():
            h = self.model(batch, domain_name)
            s = global_mean_pool(h, batch.batch)

            disc = self.model.get_head('graph_contrast', domain_name)

            pos_scores = []
            neg_scores = []

            unique_graphs = torch.unique(batch.batch)
            for graph_id in unique_graphs:
                graph_mask = (batch.batch == graph_id)
                graph_nodes = h[graph_mask]

                pos_summary = s[graph_id]
                pos_pairs_score = disc(graph_nodes, pos_summary.unsqueeze(0).repeat(graph_nodes.size(0), 1))
                pos_scores.append(pos_pairs_score.flatten())

                other_graphs = unique_graphs[unique_graphs != graph_id]
                neg_summaries = s[other_graphs]
                for neg_summary in neg_summaries:
                    neg_pairs_score = disc(graph_nodes, neg_summary.unsqueeze(0).repeat(graph_nodes.size(0), 1))
                    neg_scores.append(neg_pairs_score.flatten())

            pos_scores = torch.cat(pos_scores)
            neg_scores = torch.cat(neg_scores)

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(pos_scores.size(0), device=device, dtype=torch.float32),
                torch.zeros(neg_scores.size(0), device=device, dtype=torch.float32)
            ])

            loss = F.binary_cross_entropy(scores, labels, reduction='sum')
            size = torch.tensor(labels.size(0), device=device, dtype=torch.long)
            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / size

        total_loss /= total_size
        return total_loss, per_domain_losses


class GraphPropertyPredictionTask(BasePretrainTask):
    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator, **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = self.model.device
        total_loss = torch.tensor(0.0, device=device)
        total_size = torch.tensor(0, device=device, dtype=torch.long)
        per_domain_losses = {}

        for domain_name, batch in domain_batches.items():
            h = self.model(batch, domain_name)
            graph_emb = global_mean_pool(h, batch.batch)

            preds = self.model.get_head('graph_prop', domain_name)(graph_emb)
            labels = batch.graph_properties.to(torch.float32).to(device)

            loss = F.mse_loss(preds, labels, reduction='sum')
            size = torch.tensor(graph_emb.size(0) * GRAPH_PROPERTY_DIM, device=device, dtype=torch.long)
            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / size

        total_loss /= total_size
        return total_loss, per_domain_losses


class DomainAdversarialTask(BasePretrainTask):
    def __init__(self, model: PretrainableGNN) -> None:
        super().__init__(model)
        self.domain_to_idx = {name: i for i, name in enumerate(self.model.input_encoders.keys())}

    def compute_loss(self, domain_batches: Dict[str, Batch], generator: torch.Generator, **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = self.model.device
        embeddings = []
        labels = []

        for domain_name, batch in domain_batches.items():
            h = self.model(batch, domain_name)
            graph_emb = global_mean_pool(h, batch.batch)

            domain_idx = self.domain_to_idx[domain_name]
            domain_labels = torch.full((graph_emb.size(0),), domain_idx, device=device, dtype=torch.long)

            embeddings.append(graph_emb)
            labels.append(domain_labels)

        all_embeddings = torch.cat(embeddings, dim=0)
        all_labels = torch.cat(labels, dim=0)

        logits = self.model.get_head('domain_adv')(all_embeddings)

        total_loss = F.cross_entropy(logits, all_labels)
        return total_loss, None
