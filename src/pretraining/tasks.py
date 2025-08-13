from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import batched_negative_sampling

from src.common import (
    NODE_CONTRASTIVE_TEMPERATURE,
    CONTRASTIVE_SYMMETRY_COEF,
    NUM_NEGATIVE_SAMPLES,
)
from src.pretraining.augmentations import GraphAugmentor


class BasePretrainTask(ABC):
    """
    Abstract base class for all pre-training tasks.

    Each task computes a scalar loss tensor given a batches of graphs by domain.

    Expected batch format: Dict[str, Batch] (domain_name -> Batch)
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs) -> Tensor:
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

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs) -> Tensor:
        # Domain-specific reconstruction head
        per_domain_losses: List[Tensor] = []

        for domain_name, batch in batches_by_domain.items():
            device = self.model.device
            masked_h0, mask_indices, target_h0 = self.model.apply_node_masking(batch, domain_name)

            # Forward from masked h_0
            h_final = self.model.forward_with_h0(masked_h0.to(device), batch.edge_index.to(device))
            reconstructed_h0 = self.model.get_head('node_feat_mask', domain_name)(h_final[mask_indices])
            per_domain_losses.append(F.mse_loss(reconstructed_h0, target_h0))

        return torch.stack(per_domain_losses).mean()


class LinkPredictionTask(BasePretrainTask):
    """
    Binary edge classification using dot-product decoder.
    For each positive edge, sample one negative edge uniformly from non-edges.
    Loss: BCE averaged over batch.
    """

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs) -> Tensor:
        decoder = self.model.get_head('link_pred')
        per_domain_losses: List[Tensor] = []

        for domain_name, batch in batches_by_domain.items():
            device = self.model.device
            pos_edges = batch.edge_index.to(device)

            # Compute embeddings
            h_final = self.model(batch, domain_name)

            # Sample negative edges
            neg_edges = batched_negative_sampling(
                edge_index=pos_edges,
                batch=batch.batch.to(device), # Provides node-to-graph mapping
                num_neg_samples=NUM_NEGATIVE_SAMPLES,
            ).to(device)

            # Combine positive and negative edges
            combined_edges = torch.cat([pos_edges, neg_edges], dim=1)
            
            # Create corresponding labels
            labels = torch.cat([
                torch.ones(pos_edges.size(1), device=device),
                torch.zeros(neg_edges.size(1), device=device),
            ], dim=0)

            # Get probabilities and compute the loss
            probs = decoder(h_final, combined_edges)
            loss = F.binary_cross_entropy(probs, labels)
            per_domain_losses.append(loss)

        # Average the loss across all domains
        return torch.stack(per_domain_losses).mean()


class NodeContrastiveTask(BasePretrainTask):
    """
    GraphCL-style node-level contrastive learning with NT-Xent loss.
    Two augmented views per graph. Positives are the same node across views.
    Negatives are all other nodes across the batch.
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.augmentor = GraphAugmentor()

    def _nt_xent(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Compute symmetric NT-Xent (InfoNCE) loss between two sets of projections.

        Args:
            z1: Tensor [N, D]
            z2: Tensor [N, D]
        Returns:
            Scalar loss tensor
        """
        if z1.size(0) == 0:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

        # Normalize to use cosine similarity
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        logits_12 = (z1 @ z2.T) / NODE_CONTRASTIVE_TEMPERATURE  # [N, N]
        logits_21 = (z2 @ z1.T) / NODE_CONTRASTIVE_TEMPERATURE  # [N, N]

        targets = torch.arange(z1.size(0), device=z1.device)
        loss_12 = F.cross_entropy(logits_12, targets)
        loss_21 = F.cross_entropy(logits_21, targets)
        return CONTRASTIVE_SYMMETRY_COEF * (loss_12 + loss_21)

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs) -> Tensor:
        # Domain-specific projection head
        z1_cat_all: List[Tensor] = []
        z2_cat_all: List[Tensor] = []

        for domain_name, batch in batches_by_domain.items():
            data_list = batch.to_data_list()
            if not data_list:
                continue

            # Create two augmented views per-graph in Python, then batch once
            view1_list: List[Data] = []
            view2_list: List[Data] = []
            pairs_idx1: List[Tensor] = []
            pairs_idx2: List[Tensor] = []
            for g in data_list:
                v1, v2 = self.augmentor.create_augmented_pair(g)
                i1, i2 = GraphAugmentor.get_contrastive_pairs(v1, v2)
                view1_list.append(v1)
                view2_list.append(v2)
                pairs_idx1.append(i1)
                pairs_idx2.append(i2)

            batch_v1: Batch = Batch.from_data_list(view1_list)
            batch_v2: Batch = Batch.from_data_list(view2_list)

            h1 = self.model(batch_v1, domain_name)
            h2 = self.model(batch_v2, domain_name)

            # Compute per-graph start offsets once, then vectorize index building
            ptr1: Tensor = batch_v1.ptr  # [G+1]
            ptr2: Tensor = batch_v2.ptr  # [G+1]
            if ptr1.numel() <= 1:
                continue

            # Filter out empty pairs and collect lengths
            valid = [(i1, i2, g_idx) for g_idx, (i1, i2) in enumerate(zip(pairs_idx1, pairs_idx2)) if i1.numel() > 0 and i2.numel() > 0]
            if not valid:
                continue

            i1_list, i2_list, g_idx_list = zip(*valid)
            i1_cat = torch.cat(i1_list, dim=0)
            i2_cat = torch.cat(i2_list, dim=0)
            g_idx_tensor = torch.tensor(g_idx_list, device=h1.device, dtype=torch.long)

            # Build offsets for each element using repeat_interleave
            len_i1 = torch.tensor([t.numel() for t in i1_list], device=h1.device, dtype=torch.long)
            len_i2 = torch.tensor([t.numel() for t in i2_list], device=h2.device, dtype=torch.long)
            off1 = ptr1[g_idx_tensor].repeat_interleave(len_i1)
            off2 = ptr2[g_idx_tensor].repeat_interleave(len_i2)

            global_i1 = off1 + i1_cat
            global_i2 = off2 + i2_cat

            proj_head = self.model.get_head('node_contrast', domain_name)
            z1 = proj_head(h1[global_i1])
            z2 = proj_head(h2[global_i2])

            z1_cat_all.append(z1)
            z2_cat_all.append(z2)

        if not z1_cat_all:
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        z1_cat = torch.cat(z1_cat_all, dim=0)
        z2_cat = torch.cat(z2_cat_all, dim=0)
        return self._nt_xent(z1_cat, z2_cat)


class GraphContrastiveTask(BasePretrainTask):
    """
    InfoGraph-style contrastive task between graph summary vector and node embeddings.
    Positives: (s_g, h_v) for v in graph g. Negatives: (s_g, h_u) for u in other graphs in the batch.
    Loss: BCE.
    """

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs) -> Tensor:
        # Domain-specific discriminator
        per_domain_losses: List[Tensor] = []

        for domain_name, batch in batches_by_domain.items():
            device = self.model.device
            # Node embeddings and per-graph summaries
            h = self.model(batch, domain_name)  # [N_total, D]
            s = global_mean_pool(h, batch.batch)  # [G, D]

            # Build losses per graph using vectorized indexing where practical
            per_graph_losses: List[Tensor] = []
            ptr: Tensor = batch.ptr
            num_graphs = ptr.numel() - 1
            for g in range(num_graphs):
                start = int(ptr[g].item())
                end = int(ptr[g+1].item())
                if end <= start:
                    continue
                pos_nodes = h[start:end]
                pos_s = s[g].unsqueeze(0).expand(pos_nodes.size(0), -1)
                pos_labels = torch.ones(pos_nodes.size(0), device=device)

                # Negatives: nodes from all other graphs
                neg_left = h[:start]
                neg_right = h[end:]
                neg_nodes = torch.cat([t for t in [neg_left, neg_right] if t.numel() > 0], dim=0)
                neg_s = s[g].unsqueeze(0).expand(neg_nodes.size(0), -1)
                neg_labels = torch.zeros(neg_nodes.size(0), device=device)

                x_pairs = torch.cat([pos_nodes, neg_nodes], dim=0)
                y_pairs = torch.cat([pos_s, neg_s], dim=0)
                labels = torch.cat([pos_labels, neg_labels], dim=0)

                scores = self.model.get_head('graph_contrast', domain_name)(x_pairs, y_pairs)
                per_graph_losses.append(F.binary_cross_entropy(scores, labels))

            if per_graph_losses:
                per_domain_losses.append(torch.stack(per_graph_losses).mean())

        return torch.stack(per_domain_losses).mean() if per_domain_losses else torch.tensor(0.0, device=self.model.device)


class GraphPropertyPredictionTask(BasePretrainTask):
    """
    Predict pre-computed, standardized graph-level properties.
    Graph embedding via mean pooling; MLP head outputs GRAPH_PROPERTY_DIM properties.
    Loss: MSE averaged over batch.
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs) -> Tensor:
        # Domain-specific graph property head

        per_domain_losses: List[Tensor] = []
        for domain_name, batch in batches_by_domain.items():
            h = self.model(batch, domain_name)
            graph_emb = global_mean_pool(h, batch.batch)
            preds = self.model.get_head('graph_prop', domain_name)(graph_emb)
            target_mat = batch.graph_properties.to(self.model.device).to(torch.float32)
            per_domain_losses.append(F.mse_loss(preds, target_mat))

        return torch.stack(per_domain_losses).mean() if per_domain_losses else torch.tensor(0.0, device=self.model.device)


class DomainAdversarialTask(BasePretrainTask):
    """
    Domain classification with Gradient Reversal Layer.
    Graph embeddings are fed through GRL then classified by the domain head.
    Loss: Cross-Entropy.
    Note: The trainer should apply a negative weight to this loss when combining.
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.domain_to_idx = {name: i for i, name in enumerate(self.model.input_encoders.keys())}

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs) -> Tensor:
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

        grl_out = self.model.apply_gradient_reversal(graph_emb, lambda_val)
        logits = self.model.get_head('domain_adv')(grl_out)
        return F.cross_entropy(logits, domain_labels)
