from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from src.common import NODE_CONTRASTIVE_TEMPERATURE
from src.pretraining.augmentations import GraphAugmentor
from src.pretraining.schedulers import GRLLambdaScheduler


class BasePretrainTask(ABC):
    """
    Abstract base class for all pre-training tasks.

    Each task computes a scalar loss tensor given a batch of graphs.

    Expected batch format: Sequence[Tuple[Data, str]] (graph, domain_name)
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def compute_loss(self, batch: Sequence) -> Tensor:
        """Compute and return a single scalar loss tensor for the batch."""
        raise NotImplementedError

    # ----------------------------
    # Helper utilities for tasks
    # ----------------------------
    @staticmethod
    def _unpack_batch(batch: Sequence) -> Tuple[List[Data], List[str]]:
        graphs: List[Data] = []
        domains: List[str] = []

        for item in batch:
            g, d = item
            graphs.append(g)
            domains.append(d)
        return graphs, domains

    @staticmethod
    def _mean_pool_node_embeddings(node_embeddings: Tensor) -> Tensor:
        """Mean-pool node embeddings to a single graph embedding [GNN_HIDDEN_DIM]."""
        return node_embeddings.mean(dim=0)


class NodeFeatureMaskingTask(BasePretrainTask):
    """
    Reconstruct original h0 embeddings of masked nodes.
    Loss: MSE between reconstructed h0 and target h0 for masked nodes, averaged over batch.
    """

    def compute_loss(self, batch: Sequence) -> Tensor:
        graphs, domains = self._unpack_batch(batch)

        mse_losses: List[Tensor] = []
        head = self.model.get_head('node_feat_mask')

        for graph, domain_name in zip(graphs, domains):
            masked_graph, mask_indices, target_h0 = self.model.apply_node_masking(graph, domain_name)

            # Forward pass on masked graph to get final embeddings
            h_final = self.model(masked_graph, domain_name)

            # Select masked node embeddings and reconstruct original h0
            predicted_h_final = h_final[mask_indices]
            reconstructed_h0 = head(predicted_h_final)

            # MSE loss for this graph
            mse_loss = F.mse_loss(reconstructed_h0, target_h0)
            mse_losses.append(mse_loss)

        return torch.stack(mse_losses).mean()


class LinkPredictionTask(BasePretrainTask):
    """
    Binary edge classification using dot-product decoder.
    For each positive edge, sample one negative edge uniformly from non-edges.
    Loss: BCE averaged over batch.
    """

    def compute_loss(self, batch: Sequence) -> Tensor:
        graphs, domains = self._unpack_batch(batch)

        bce_losses: List[Tensor] = []
        decoder = self.model.get_head('link_pred')

        for graph, domain_name in zip(graphs, domains):
            # Positive edges
            pos_edge_index: Tensor = graph.edge_index.to(self.model.device)

            # Negative edges: 1:1 ratio with positives
            num_pos = pos_edge_index.size(1)
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=graph.x.size(0),
                num_neg_samples=num_pos,
            ).to(self.model.device)

            # Combine edges and labels
            combined_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            labels = torch.cat([
                torch.ones(num_pos, device=self.model.device),
                torch.zeros(neg_edge_index.size(1), device=self.model.device),
            ], dim=0)

            # Node embeddings from graph
            h_final = self.model(graph, domain_name)

            probs = decoder(h_final, combined_edge_index)
            bce_loss = F.binary_cross_entropy(probs, labels)
            bce_losses.append(bce_loss)

        return torch.stack(bce_losses).mean()


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
        return 0.5 * (loss_12 + loss_21)

    def compute_loss(self, batch: Sequence) -> Tensor:
        graphs, domains = self._unpack_batch(batch)

        head = self.model.get_head('node_contrast')

        # Collect projected embeddings across the whole batch
        proj_view1_list: List[Tensor] = []
        proj_view2_list: List[Tensor] = []

        for graph, domain_name in zip(graphs, domains):
            # Create two augmented views
            view1, view2 = self.augmentor.create_augmented_pair(graph)

            # Encode both views
            h1 = self.model(view1, domain_name)
            h2 = self.model(view2, domain_name)

            # Determine positive pairs via overlapping nodes
            idx1, idx2 = GraphAugmentor.get_contrastive_pairs(view1, view2)

            # Select and project embeddings
            z1 = head(h1[idx1])
            z2 = head(h2[idx2])

            proj_view1_list.append(z1)
            proj_view2_list.append(z2)

        z1_all = torch.cat(proj_view1_list, dim=0)
        z2_all = torch.cat(proj_view2_list, dim=0)

        return self._nt_xent(z1_all, z2_all)


class GraphContrastiveTask(BasePretrainTask):
    """
    InfoGraph-style contrastive task between graph summary vector and node embeddings.
    Positives: (s_g, h_v) for v in graph g. Negatives: (s_g, h_u) for u in other graphs in the batch.
    Loss: BCE.
    """

    def compute_loss(self, batch: Sequence) -> Tensor:
        graphs, domains = self._unpack_batch(batch)

        discriminator = self.model.get_head('graph_contrast')

        all_nodes: List[Tensor] = []
        graph_summaries: List[Tensor] = []
        # (start, end) per graph in all_nodes
        graph_node_slices: List[Tuple[int, int]] = []

        # Encode each graph, record node embeddings and summary
        start = 0
        for graph, domain_name in zip(graphs, domains):
            h = self.model(graph, domain_name)  # [N_g, D]
            s = self._mean_pool_node_embeddings(h)  # [D]

            end = start + h.size(0)
            graph_node_slices.append((start, end))
            start = end

            all_nodes.append(h)
            graph_summaries.append(s)

        H_all = torch.cat(all_nodes, dim=0)  # [M, D]

        per_graph_losses: List[Tensor] = []
        for (start, end), s in zip(graph_node_slices, graph_summaries):
            # Positive pairs: nodes from this graph with its summary
            pos_nodes = H_all[start:end]  # [N_g, D]
            pos_s = s.unsqueeze(0).expand(pos_nodes.size(0), -1)  # [N_g, D]
            pos_labels = torch.ones(pos_nodes.size(0), device=self.model.device)

            # Negative pairs: nodes from other graphs with this summary
            neg_left = H_all[:start]
            neg_right = H_all[end:]

            neg_nodes = torch.cat([t for t in [neg_left, neg_right] if t.numel() > 0], dim=0)
            neg_s = s.unsqueeze(0).expand(neg_nodes.size(0), -1)
            neg_labels = torch.zeros(neg_nodes.size(0), device=self.model.device)

            x_pairs = torch.cat([pos_nodes, neg_nodes], dim=0)
            y_pairs = torch.cat([pos_s, neg_s], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)

            scores = discriminator(x_pairs, y_pairs)
            per_graph_losses.append(F.binary_cross_entropy(scores, labels))

        return torch.stack(per_graph_losses).mean()


class GraphPropertyPredictionTask(BasePretrainTask):
    """
    Predict pre-computed, standardized graph-level properties.
    Graph embedding via mean pooling; MLP head outputs 15-D properties.
    Loss: MSE averaged over batch.
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def compute_loss(self, batch: Sequence, **kwargs) -> Tensor:
        graphs, domains = self._unpack_batch(batch)

        head = self.model.get_head('graph_prop')

        preds: List[Tensor] = []
        targets: List[Tensor] = []

        for graph, domain_name in zip(graphs, domains):
            h = self.model(graph, domain_name)
            s = self._mean_pool_node_embeddings(h).unsqueeze(0)  # [1, D]
            pred = head(s).squeeze(0)  # [15]

            # Precomputed standardized properties
            target_props = graph.graph_properties.to(self.model.device).to(torch.float32)

            preds.append(pred)
            targets.append(target_props)

        pred_mat = torch.stack(preds, dim=0)
        target_mat = torch.stack(targets, dim=0)

        return F.mse_loss(pred_mat, target_mat)


class DomainAdversarialTask(BasePretrainTask):
    """
    Domain classification with Gradient Reversal Layer.
    Graph embeddings are fed through GRL then classified by the domain head.
    Loss: Cross-Entropy.
    Note: The trainer should apply a negative weight to this loss when combining.
    """

    def __init__(self, model: nn.Module, lambda_scheduler: GRLLambdaScheduler):
        super().__init__(model)
        self.lambda_scheduler = lambda_scheduler
        self.domain_to_idx = {name: i for i, name in enumerate(self.model.input_encoders.keys())}

    def _get_lambda_val(self) -> float:
        return float(self.lambda_scheduler())

    def compute_loss(self, batch: Sequence) -> Tensor:
        graphs, domains = self._unpack_batch(batch)

        lambda_val: float = self._get_lambda_val()

        # Map domain_name -> index based on model's encoder order
        domain_labels = torch.tensor([self.domain_to_idx[d] for d in domains], device=self.model.device, dtype=torch.long)

        # Encode each graph and compute graph embeddings
        graph_emb_list: List[Tensor] = []
        for graph, domain_name in zip(graphs, domains):
            h = self.model(graph, domain_name)
            s = self._mean_pool_node_embeddings(h)
            graph_emb_list.append(s)

        graph_emb = torch.stack(graph_emb_list, dim=0)  # [B, D]

        # Apply GRL
        grl_out = self.model.apply_gradient_reversal(graph_emb, lambda_val)

        # Domain head
        logits = self.model.get_head('domain_adv')(grl_out)  # [B, num_domains]

        return F.cross_entropy(logits, domain_labels)
