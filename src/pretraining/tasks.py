from abc import ABC, abstractmethod
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import batched_negative_sampling

from src.common import (
    NODE_CONTRASTIVE_TEMPERATURE,
    NUM_NEGATIVE_SAMPLES,
)
from src.pretraining.augmentations import GraphAugmentor


class BasePretrainTask(ABC):
    """
    Abstract base class for all pre-training tasks.

    Each task computes a scalar loss tensor given a batches of graphs by domain.

    Expected batch format: Dict[str, Batch] (domain_name -> Batch)
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @abstractmethod
    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tensor:
        """Compute and return a single scalar loss tensor for the batch."""
        raise NotImplementedError

    # ----------------------------
    # Helper utilities for tasks
    # ----------------------------
    @staticmethod
    def _mean_pool_node_embeddings(node_embeddings: Tensor) -> Tensor:
        """Mean-pool node embeddings to a single graph embedding [GNN_HIDDEN_DIM]."""
        return node_embeddings.mean(dim=0)


class NodeFeatureMaskingTask(BasePretrainTask):
    """
    Reconstruct original h0 embeddings of masked nodes.
    Loss: MSE between reconstructed h0 and target h0 for masked nodes, averaged over batch.
    """

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tensor:
        device = self.model.device
        per_domain_losses: List[Tensor] = []

        for domain_name, batch in batches_by_domain.items():
            masked_h0, mask_indices, target_h0 = self.model.apply_node_masking(batch, domain_name)

            h_final = self.model.forward_with_h0(masked_h0.to(device), batch.edge_index.to(device))
            reconstructed_h0 = self.model.get_head('node_feat_mask', domain_name)(h_final[mask_indices])
            loss = F.mse_loss(reconstructed_h0, target_h0)
            per_domain_losses.append(loss)

        return torch.stack(per_domain_losses).mean()


class LinkPredictionTask(BasePretrainTask):
    """
    Binary edge classification using dot-product decoder.
    For each positive edge, sample one negative edge uniformly from non-edges.
    Loss: BCE averaged over batch.
    """

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tensor:
        device = self.model.device
        decoder = self.model.get_head('link_pred')
        per_domain_losses: List[Tensor] = []

        for domain_name, batch in batches_by_domain.items():
            pos_edges = batch.edge_index.to(device)
            neg_edges = batched_negative_sampling(
                edge_index=pos_edges,
                batch=batch.batch.to(device),
                num_neg_samples=NUM_NEGATIVE_SAMPLES,
            ).to(device)
            combined_edges = torch.cat([pos_edges, neg_edges], dim=1)
            labels = torch.cat([
                torch.ones(pos_edges.size(1), device=device),
                torch.zeros(neg_edges.size(1), device=device),
            ], dim=0)

            h_final = self.model(batch, domain_name)
            probs = decoder(h_final, combined_edges)
            loss = F.binary_cross_entropy(probs, labels)
            per_domain_losses.append(loss)

        return torch.stack(per_domain_losses).mean()


class NodeContrastiveTask(BasePretrainTask):
    """
    SimCLR-style node-level contrastive learning with canonical in-view + cross-view negatives.
    Two augmented views per graph. Positives are the same node across views.
    Negatives are ALL other embeddings from both views (2N-2 negatives for batch size N).
    
    Uses the canonical SimCLR approach: concatenate embeddings from both views and 
    compute a single large similarity matrix for maximum negative sampling.
    """

    def _simclr_nt_xent(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Compute canonical SimCLR NT-Xent loss with in-view + cross-view negatives.
        
        Implementation:
        1. Concatenate z1 and z2: z = [z1; z2] with shape [2N, D]
        2. Compute similarity matrix: sim = (z @ z.T) / temperature with shape [2N, 2N]
        3. For anchor i in first half, positive is i+N; for anchor i+N, positive is i
        4. All other 2N-2 embeddings serve as negatives
        
        Args:
            z1: Tensor [N, D] - projections from first view
            z2: Tensor [N, D] - projections from second view
        Returns:
            Scalar loss tensor
        """
        N = z1.size(0)
        device = z1.device
        
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
        
        labels = pos_indices
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tensor:
        per_domain_z1: List[Tensor] = []
        per_domain_z2: List[Tensor] = []

        for domain_name, batch in batches_by_domain.items():
            batch_v1, batch_v2 = GraphAugmentor.create_two_views(batch)

            h1 = self.model(batch_v1, domain_name)
            h2 = self.model(batch_v2, domain_name)

            proj_head = self.model.get_head('node_contrast', domain_name)
            per_domain_z1.append(proj_head(h1))
            per_domain_z2.append(proj_head(h2))

        z1_cat = torch.cat(per_domain_z1, dim=0)
        z2_cat = torch.cat(per_domain_z2, dim=0)
        loss = self._simclr_nt_xent(z1_cat, z2_cat)
        return loss


class GraphContrastiveTask(BasePretrainTask):
    """
    InfoGraph-style contrastive task between graph summary vector and node embeddings.
    Positives: (s_g, h_v) for v in graph g. Negatives: (s_g, h_u) for u in other graphs in the batch.
    Loss: BCE.
    """

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tensor:
        per_domain_losses: List[Tensor] = []

        for domain_name, batch in batches_by_domain.items():
            device = self.model.device
            h = self.model(batch, domain_name)
            batch_vec = batch.batch.to(device)
            s = global_mean_pool(h, batch_vec)
            G = s.size(0)
            N = h.size(0)

            disc = self.model.get_head('graph_contrast', domain_name)
            Hw = disc.W(h)
            logits = Hw @ s.T
            probs = torch.sigmoid(logits)

            labels = torch.zeros_like(probs)
            labels[torch.arange(N, device=device), batch_vec] = 1.0

            per_domain_losses.append(F.binary_cross_entropy(probs, labels))

        return torch.stack(per_domain_losses).mean() if per_domain_losses else torch.tensor(0.0, device=self.model.device)


class GraphPropertyPredictionTask(BasePretrainTask):
    """
    Predict pre-computed, standardized graph-level properties.
    Graph embedding via mean pooling; MLP head outputs GRAPH_PROPERTY_DIM properties.
    Loss: MSE averaged over batch.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tensor:
        per_domain_losses: List[Tensor] = []

        for domain_name, batch in batches_by_domain.items():
            h = self.model(batch, domain_name)
            graph_emb = global_mean_pool(h, batch.batch)
            preds = self.model.get_head('graph_prop', domain_name)(graph_emb)
            target_mat = batch.graph_properties.to(self.model.device).to(torch.float32)
            per_domain_losses.append(F.mse_loss(preds, target_mat))

        return torch.stack(per_domain_losses).mean()


class DomainAdversarialTask(BasePretrainTask):
    """
    Domain classification with Gradient Reversal Layer.
    Graph embeddings are fed through GRL then classified by the domain head.
    Loss: Cross-Entropy.
    Note: The trainer should apply a negative weight to this loss when combining.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.domain_to_idx = {name: i for i, name in enumerate(self.model.input_encoders.keys())}

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tensor:
        lambda_val = kwargs.get('lambda_val', 0.0)

        graph_emb_list: List[Tensor] = []
        label_list: List[Tensor] = []

        for domain_name, batch in batches_by_domain.items():
            h = self.model(batch, domain_name)
            s = global_mean_pool(h, batch.batch)
            graph_emb_list.append(s)
            domain_idx = self.domain_to_idx[domain_name]
            label_list.append(torch.full((s.size(0),), domain_idx, device=self.model.device, dtype=torch.long))

        graph_emb = torch.cat(graph_emb_list, dim=0)
        domain_labels = torch.cat(label_list, dim=0)

        logits = self.model.get_head('domain_adv')(graph_emb, lambda_val)
        return F.cross_entropy(logits, domain_labels)
