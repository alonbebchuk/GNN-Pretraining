from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import batched_negative_sampling

from src.common import (
    NODE_CONTRASTIVE_TEMPERATURE,
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
    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute loss for the batch.

        Returns:
            Tuple of (total_loss, per_domain_losses) where:
            - total_loss: Scalar tensor with the aggregated loss
            - per_domain_losses: Dict mapping domain names to their individual scalar losses
        """
        raise NotImplementedError


class NodeFeatureMaskingTask(BasePretrainTask):
    """
    Reconstruct original h0 embeddings of masked nodes.
    Loss: MSE between reconstructed h0 and target h0 for masked nodes, averaged over batch.
    """

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        total_loss = 0.0
        total_size = 0
        per_domain_losses = {}
        device = self.model.device

        for domain_name, batch in batches_by_domain.items():
            masked_h0, mask_indices, target_h0 = self.model.apply_node_masking(batch, domain_name)

            h_final = self.model.forward_with_h0(masked_h0.to(device), batch.edge_index.to(device))
            reconstructed_h0 = self.model.get_head('node_feat_mask', domain_name)(h_final[mask_indices])
            loss = F.mse_loss(reconstructed_h0, target_h0, reduction='sum')
            size = mask_indices.size(0)
            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / size

        total_loss /= total_size
        return total_loss, per_domain_losses


class LinkPredictionTask(BasePretrainTask):
    """
    Binary edge classification using dot-product decoder.
    For each positive edge, sample one negative edge uniformly from non-edges.
    Loss: BCE averaged over batch.
    """

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        total_loss = 0.0
        total_size = 0
        per_domain_losses = {}
        device = self.model.device
        decoder = self.model.get_head('link_pred')

        for domain_name, batch in batches_by_domain.items():
            pos_edges = batch.edge_index.to(device)
            neg_edges = batched_negative_sampling(
                edge_index=pos_edges,
                batch=batch.batch.to(device),
            ).to(device)
            combined_edges = torch.cat([pos_edges, neg_edges], dim=1)
            labels = torch.cat([
                torch.ones(pos_edges.size(1), device=device),
                torch.zeros(neg_edges.size(1), device=device),
            ], dim=0)

            h_final = self.model(batch, domain_name)
            probs = decoder(h_final, combined_edges)
            loss = F.binary_cross_entropy(probs, labels, reduction='sum')
            size = labels.size(0)
            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / size

        total_loss /= total_size
        return total_loss, per_domain_losses


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
        device = z1.device
        N = z1.size(0)
        
        # Handle edge cases: empty or single node batches
        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0
        if N == 1:
            # For single node, return small loss to avoid division by zero
            return torch.tensor(0.1, device=device, requires_grad=True), 2

        # Ensure z1 and z2 have the same size
        if z1.size(0) != z2.size(0):
            min_size = min(z1.size(0), z2.size(0))
            z1 = z1[:min_size]
            z2 = z2[:min_size]
            N = min_size
            if N == 0:
                return torch.tensor(0.0, device=device, requires_grad=True), 0
            if N == 1:
                return torch.tensor(0.1, device=device, requires_grad=True), 2

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

        # Ensure pos_indices are within bounds
        max_valid_idx = sim_matrix.size(1) - 1
        pos_indices = torch.clamp(pos_indices, 0, max_valid_idx)

        loss = F.cross_entropy(sim_matrix, pos_indices, reduction='sum')
        size = 2 * N
        return loss, size

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        total_loss = 0.0
        total_size = 0
        per_domain_losses = {}

        for domain_name, batch in batches_by_domain.items():
            # Skip empty batches
            if batch.x.size(0) == 0:
                per_domain_losses[domain_name] = torch.tensor(0.0, device=batch.x.device)
                continue
                
            batch_v1, batch_v2 = GraphAugmentor.create_two_views(batch)
            
            # Skip if augmentation resulted in empty batches
            if batch_v1.x.size(0) == 0 or batch_v2.x.size(0) == 0:
                per_domain_losses[domain_name] = torch.tensor(0.0, device=batch.x.device)
                continue

            h1 = self.model(batch_v1, domain_name)
            h2 = self.model(batch_v2, domain_name)

            proj_head = self.model.get_head('node_contrast', domain_name)
            z1 = proj_head(h1)
            z2 = proj_head(h2)

            loss, size = self._simclr_nt_xent(z1, z2)
            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / max(size, 1)  # Avoid division by zero

        if total_size == 0:
            # Return a small loss if no valid batches were processed
            device = next(iter(batches_by_domain.values())).x.device
            return torch.tensor(0.1, device=device, requires_grad=True), per_domain_losses
            
        total_loss /= total_size
        return total_loss, per_domain_losses


class GraphContrastiveTask(BasePretrainTask):
    """
    InfoGraph-style contrastive task between graph summary vector and node embeddings.
    Positives: (s_g, h_v) for v in graph g. Negatives: (s_g, h_u) for u in other graphs in the batch.
    Loss: BCE.
    """

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        total_loss = 0.0
        total_size = 0
        per_domain_losses = {}
        device = self.model.device

        for domain_name, batch in batches_by_domain.items():
            h = self.model(batch, domain_name)
            batch_vec = batch.batch.to(device)
            s = global_mean_pool(h, batch_vec)
            N = h.size(0)

            disc = self.model.get_head('graph_contrast', domain_name)
            probs = disc(h, s)  # Shape: (N, G) where G is number of graphs in batch

            # Create labels: each node should be positive with its own graph
            labels = torch.zeros_like(probs)
            labels[torch.arange(N, device=device), batch_vec] = 1.0

            loss = F.binary_cross_entropy(probs, labels, reduction='sum')
            size = labels.numel()
            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / size

        total_loss /= total_size
        return total_loss, per_domain_losses


class GraphPropertyPredictionTask(BasePretrainTask):
    """
    Predict pre-computed, standardized graph-level properties.
    Graph embedding via mean pooling; MLP head outputs GRAPH_PROPERTY_DIM properties.
    Loss: MSE averaged over batch.
    """

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        total_loss = 0.0
        total_size = 0
        per_domain_losses = {}
        device = self.model.device

        for domain_name, batch in batches_by_domain.items():
            # Check if this batch has graph_properties
            if not hasattr(batch, 'graph_properties'):
                # Skip this domain if it doesn't have graph properties
                # This can happen if the dataset wasn't processed with graph properties
                per_domain_losses[domain_name] = torch.tensor(0.0, device=device)
                continue
                
            h = self.model(batch, domain_name)
            batch_vec = batch.batch.to(device)
            graph_emb = global_mean_pool(h, batch_vec)

            preds = self.model.get_head('graph_prop', domain_name)(graph_emb)

            # Reshape labels to match predictions shape
            # batch.graph_properties is flattened: [batch_size * num_properties]
            # preds has shape: [batch_size, num_properties]
            labels = batch.graph_properties.to(device).to(torch.float32)
            batch_size = graph_emb.size(0)
            num_properties = preds.size(1)
            
            # Ensure labels can be reshaped correctly
            expected_size = batch_size * num_properties
            if labels.numel() != expected_size:
                raise ValueError(f"Label size mismatch: expected {expected_size}, got {labels.numel()}")
            
            labels = labels.view(batch_size, num_properties)

            loss = F.mse_loss(preds, labels, reduction='sum')
            size = batch_size
            total_loss += loss
            total_size += size
            per_domain_losses[domain_name] = loss / size

        if total_size == 0:
            # No domains had graph properties, return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True), per_domain_losses
            
        total_loss /= total_size
        return total_loss, per_domain_losses


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

    def compute_loss(self, batches_by_domain: Dict[str, Batch], **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        total_loss = 0.0
        lambda_val = kwargs.get('lambda_val', 0.0)
        embeddings = []
        labels = []
        device = self.model.device

        for domain_name, batch in batches_by_domain.items():
            h = self.model(batch, domain_name)
            batch_vec = batch.batch.to(device)
            graph_emb = global_mean_pool(h, batch_vec)

            domain_idx = self.domain_to_idx[domain_name]
            domain_labels = torch.full((graph_emb.size(0),), domain_idx, device=device, dtype=torch.long)

            embeddings.append(graph_emb)
            labels.append(domain_labels)

        all_embeddings = torch.cat(embeddings, dim=0)
        all_labels = torch.cat(labels, dim=0)

        logits = self.model.get_head('domain_adv')(all_embeddings, lambda_val)

        total_loss = F.cross_entropy(logits, all_labels)
        return total_loss, None
