"""
Enhanced Link Prediction Implementation for GNN Pre-training.

This module provides sophisticated link prediction capabilities including:
- Advanced negative sampling strategies
- Temporal and structural aware sampling
- Comprehensive evaluation metrics
- Support for different graph types (homogeneous, heterogeneous)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, degree, to_undirected
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
from dataclasses import dataclass
from enum import Enum


class SamplingStrategy(Enum):
    """Different negative sampling strategies."""
    UNIFORM = "uniform"
    DEGREE_BIASED = "degree_biased"
    STRUCTURAL = "structural"
    COMMUNITY_AWARE = "community_aware"
    TEMPORAL = "temporal"


@dataclass
class LinkPredictionConfig:
    """Configuration for link prediction task."""
    negative_sampling_ratio: float = 1.0
    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    max_edges_per_graph: int = 1000  # Limit for efficiency
    min_edges_per_graph: int = 10
    use_edge_features: bool = False
    temporal_split_ratio: float = 0.8  # For temporal graphs
    structural_radius: int = 2  # For structural sampling
    community_detection: bool = False  # Enable community-aware sampling


class AdvancedNegativeSampler:
    """
    Advanced negative sampling strategies for link prediction.
    """
    
    def __init__(self, config: LinkPredictionConfig):
        self.config = config
        self.node_degrees = {}  # Cache for node degrees
        self.communities = {}   # Cache for community assignments
    
    def sample_negative_edges(self, 
                             edge_index: torch.Tensor, 
                             num_nodes: int,
                             num_neg_samples: Optional[int] = None,
                             batch: Optional[torch.Tensor] = None,
                             node_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample negative edges using the specified strategy.
        
        Args:
            edge_index: Positive edge indices [2, num_edges]
            num_nodes: Total number of nodes
            num_neg_samples: Number of negative samples to generate
            batch: Batch assignment for graphs
            node_features: Node features for structural sampling
            
        Returns:
            Negative edge indices [2, num_neg_samples]
        """
        if num_neg_samples is None:
            num_neg_samples = int(edge_index.shape[1] * self.config.negative_sampling_ratio)
        
        strategy = self.config.sampling_strategy
        
        if strategy == SamplingStrategy.UNIFORM:
            return self._uniform_sampling(edge_index, num_nodes, num_neg_samples, batch)
        elif strategy == SamplingStrategy.DEGREE_BIASED:
            return self._degree_biased_sampling(edge_index, num_nodes, num_neg_samples, batch)
        elif strategy == SamplingStrategy.STRUCTURAL:
            return self._structural_sampling(edge_index, num_nodes, num_neg_samples, batch, node_features)
        elif strategy == SamplingStrategy.COMMUNITY_AWARE:
            return self._community_aware_sampling(edge_index, num_nodes, num_neg_samples, batch)
        else:
            # Fallback to uniform sampling
            return self._uniform_sampling(edge_index, num_nodes, num_neg_samples, batch)
    
    def _uniform_sampling(self, edge_index: torch.Tensor, num_nodes: int, 
                         num_neg_samples: int, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard uniform negative sampling."""
        return negative_sampling(edge_index, num_nodes, num_neg_samples, force_undirected=True)
    
    def _degree_biased_sampling(self, edge_index: torch.Tensor, num_nodes: int,
                               num_neg_samples: int, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Degree-biased negative sampling.
        Higher degree nodes are more likely to be sampled as negatives.
        """
        device = edge_index.device
        
        # Compute node degrees
        node_degrees = degree(edge_index[0], num_nodes) + degree(edge_index[1], num_nodes)
        node_degrees = node_degrees.float()
        
        # Create probability distribution based on degrees
        degree_probs = node_degrees / node_degrees.sum()
        
        # Sample negative edges
        neg_edges = []
        attempts = 0
        max_attempts = num_neg_samples * 10
        
        # Convert positive edges to set for faster lookup
        pos_edge_set = set()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            pos_edge_set.add((min(u, v), max(u, v)))
        
        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            # Sample nodes based on degree distribution
            src_nodes = torch.multinomial(degree_probs, num_neg_samples - len(neg_edges), replacement=True)
            dst_nodes = torch.multinomial(degree_probs, num_neg_samples - len(neg_edges), replacement=True)
            
            for src, dst in zip(src_nodes, dst_nodes):
                if src != dst:  # No self-loops
                    edge = (min(src.item(), dst.item()), max(src.item(), dst.item()))
                    if edge not in pos_edge_set:
                        neg_edges.append([src.item(), dst.item()])
                        if len(neg_edges) >= num_neg_samples:
                            break
            
            attempts += 1
        
        if len(neg_edges) < num_neg_samples:
            # Fallback to uniform sampling for remaining edges
            remaining = num_neg_samples - len(neg_edges)
            uniform_neg = self._uniform_sampling(edge_index, num_nodes, remaining, batch)
            for i in range(uniform_neg.shape[1]):
                neg_edges.append([uniform_neg[0, i].item(), uniform_neg[1, i].item()])
        
        return torch.tensor(neg_edges[:num_neg_samples], device=device).t()
    
    def _structural_sampling(self, edge_index: torch.Tensor, num_nodes: int,
                           num_neg_samples: int, batch: Optional[torch.Tensor] = None,
                           node_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Structural negative sampling based on node similarity and local structure.
        """
        device = edge_index.device
        
        if node_features is None:
            # Fallback to degree-biased sampling
            return self._degree_biased_sampling(edge_index, num_nodes, num_neg_samples, batch)
        
        # Compute node similarities
        node_similarities = torch.mm(F.normalize(node_features), F.normalize(node_features).t())
        
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1  # Make undirected
        
        # Sample negative edges based on structural similarity
        neg_edges = []
        attempts = 0
        max_attempts = num_neg_samples * 5
        
        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            # Sample nodes with higher probability for similar but unconnected pairs
            similarity_scores = node_similarities.flatten()
            connectivity_mask = (1 - adj).flatten()  # 1 for non-connected pairs
            
            # Combine similarity and non-connectivity
            sampling_probs = similarity_scores * connectivity_mask
            sampling_probs = F.softmax(sampling_probs, dim=0)
            
            # Sample edge indices
            sampled_indices = torch.multinomial(sampling_probs, min(num_neg_samples - len(neg_edges), 100), replacement=True)
            
            for idx in sampled_indices:
                src = idx // num_nodes
                dst = idx % num_nodes
                
                if src != dst and adj[src, dst] == 0:  # Valid negative edge
                    neg_edges.append([src.item(), dst.item()])
                    if len(neg_edges) >= num_neg_samples:
                        break
            
            attempts += 1
        
        if len(neg_edges) < num_neg_samples:
            # Fallback to uniform sampling
            remaining = num_neg_samples - len(neg_edges)
            uniform_neg = self._uniform_sampling(edge_index, num_nodes, remaining, batch)
            for i in range(uniform_neg.shape[1]):
                neg_edges.append([uniform_neg[0, i].item(), uniform_neg[1, i].item()])
        
        return torch.tensor(neg_edges[:num_neg_samples], device=device).t()
    
    def _community_aware_sampling(self, edge_index: torch.Tensor, num_nodes: int,
                                 num_neg_samples: int, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Community-aware negative sampling.
        Sample negatives both within and across communities.
        """
        # For now, implement a simplified version using node clustering
        # In practice, you might use more sophisticated community detection
        
        device = edge_index.device
        
        # Simple community detection based on local clustering coefficient
        communities = self._detect_communities(edge_index, num_nodes)
        
        neg_edges = []
        intra_community_ratio = 0.3  # 30% intra-community negatives
        
        # Sample intra-community negatives
        intra_samples = int(num_neg_samples * intra_community_ratio)
        inter_samples = num_neg_samples - intra_samples
        
        # Intra-community sampling
        for community in communities.values():
            if len(community) > 1:
                community_nodes = list(community)
                for _ in range(min(intra_samples // len(communities), len(community) * (len(community) - 1) // 2)):
                    src, dst = np.random.choice(community_nodes, 2, replace=False)
                    if not self._edge_exists(edge_index, src, dst):
                        neg_edges.append([src, dst])
        
        # Inter-community sampling
        community_list = list(communities.values())
        for _ in range(inter_samples):
            if len(community_list) > 1:
                comm1, comm2 = np.random.choice(len(community_list), 2, replace=False)
                src = np.random.choice(list(community_list[comm1]))
                dst = np.random.choice(list(community_list[comm2]))
                
                if not self._edge_exists(edge_index, src, dst):
                    neg_edges.append([src, dst])
        
        # Fill remaining with uniform sampling
        if len(neg_edges) < num_neg_samples:
            remaining = num_neg_samples - len(neg_edges)
            uniform_neg = self._uniform_sampling(edge_index, num_nodes, remaining, batch)
            for i in range(uniform_neg.shape[1]):
                neg_edges.append([uniform_neg[0, i].item(), uniform_neg[1, i].item()])
        
        return torch.tensor(neg_edges[:num_neg_samples], device=device).t()
    
    def _detect_communities(self, edge_index: torch.Tensor, num_nodes: int) -> Dict[int, set]:
        """Simple community detection based on connected components."""
        # This is a simplified implementation
        # In practice, you might use more sophisticated algorithms like Louvain
        
        adj_list = {i: set() for i in range(num_nodes)}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].add(dst)
            adj_list[dst].add(src)
        
        visited = set()
        communities = {}
        community_id = 0
        
        for node in range(num_nodes):
            if node not in visited:
                community = set()
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        community.add(current)
                        stack.extend(neighbor for neighbor in adj_list[current] if neighbor not in visited)
                
                communities[community_id] = community
                community_id += 1
        
        return communities
    
    def _edge_exists(self, edge_index: torch.Tensor, src: int, dst: int) -> bool:
        """Check if an edge exists between two nodes."""
        for i in range(edge_index.shape[1]):
            if (edge_index[0, i] == src and edge_index[1, i] == dst) or \
               (edge_index[0, i] == dst and edge_index[1, i] == src):
                return True
        return False


class LinkPredictionEvaluator:
    """
    Comprehensive evaluation metrics for link prediction.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset evaluation state."""
        self.predictions = []
        self.targets = []
        self.edge_types = []  # For heterogeneous graphs
    
    def add_batch(self, predictions: torch.Tensor, targets: torch.Tensor, 
                  edge_types: Optional[torch.Tensor] = None):
        """
        Add a batch of predictions and targets.
        
        Args:
            predictions: Predicted edge probabilities [num_edges]
            targets: True edge labels [num_edges]
            edge_types: Edge types for heterogeneous evaluation [num_edges]
        """
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())
        
        if edge_types is not None:
            self.edge_types.append(edge_types.detach().cpu())
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive link prediction metrics.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.predictions:
            return {}
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.predictions, dim=0).numpy()
        all_targets = torch.cat(self.targets, dim=0).numpy()
        
        metrics = {}
        
        # Basic classification metrics
        binary_preds = (all_preds > 0.5).astype(int)
        
        metrics['accuracy'] = np.mean(binary_preds == all_targets)
        metrics['precision'] = self._safe_precision(all_targets, binary_preds)
        metrics['recall'] = self._safe_recall(all_targets, binary_preds)
        metrics['f1_score'] = self._safe_f1(all_targets, binary_preds)
        
        # Ranking metrics
        if len(np.unique(all_targets)) > 1:  # Need both classes for AUC
            metrics['auc_roc'] = roc_auc_score(all_targets, all_preds)
            metrics['auc_pr'] = average_precision_score(all_targets, all_preds)
        
        # Threshold-specific metrics
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            thresh_preds = (all_preds > threshold).astype(int)
            metrics[f'precision_at_{threshold}'] = self._safe_precision(all_targets, thresh_preds)
            metrics[f'recall_at_{threshold}'] = self._safe_recall(all_targets, thresh_preds)
        
        # Top-k metrics
        for k in [10, 50, 100]:
            if len(all_preds) >= k:
                metrics[f'precision_at_{k}'] = self._precision_at_k(all_targets, all_preds, k)
                metrics[f'recall_at_{k}'] = self._recall_at_k(all_targets, all_preds, k)
        
        # Heterogeneous metrics (if edge types available)
        if self.edge_types:
            all_edge_types = torch.cat(self.edge_types, dim=0).numpy()
            type_metrics = self._compute_type_specific_metrics(all_targets, all_preds, all_edge_types)
            metrics.update(type_metrics)
        
        return metrics
    
    def _safe_precision(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Compute precision safely handling edge cases."""
        tp = np.sum((targets == 1) & (predictions == 1))
        fp = np.sum((targets == 0) & (predictions == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _safe_recall(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Compute recall safely handling edge cases."""
        tp = np.sum((targets == 1) & (predictions == 1))
        fn = np.sum((targets == 1) & (predictions == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _safe_f1(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Compute F1 score safely handling edge cases."""
        precision = self._safe_precision(targets, predictions)
        recall = self._safe_recall(targets, predictions)
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _precision_at_k(self, targets: np.ndarray, predictions: np.ndarray, k: int) -> float:
        """Compute precision at top-k predictions."""
        top_k_indices = np.argsort(predictions)[-k:]
        top_k_targets = targets[top_k_indices]
        return np.mean(top_k_targets)
    
    def _recall_at_k(self, targets: np.ndarray, predictions: np.ndarray, k: int) -> float:
        """Compute recall at top-k predictions."""
        top_k_indices = np.argsort(predictions)[-k:]
        top_k_targets = targets[top_k_indices]
        total_positives = np.sum(targets)
        return np.sum(top_k_targets) / total_positives if total_positives > 0 else 0.0
    
    def _compute_type_specific_metrics(self, targets: np.ndarray, predictions: np.ndarray, 
                                     edge_types: np.ndarray) -> Dict[str, float]:
        """Compute metrics for each edge type separately."""
        type_metrics = {}
        unique_types = np.unique(edge_types)
        
        for edge_type in unique_types:
            type_mask = edge_types == edge_type
            type_targets = targets[type_mask]
            type_preds = predictions[type_mask]
            
            if len(type_targets) > 0:
                binary_preds = (type_preds > 0.5).astype(int)
                
                type_metrics[f'type_{edge_type}_accuracy'] = np.mean(binary_preds == type_targets)
                type_metrics[f'type_{edge_type}_precision'] = self._safe_precision(type_targets, binary_preds)
                type_metrics[f'type_{edge_type}_recall'] = self._safe_recall(type_targets, binary_preds)
                
                if len(np.unique(type_targets)) > 1:
                    type_metrics[f'type_{edge_type}_auc'] = roc_auc_score(type_targets, type_preds)
        
        return type_metrics


class EnhancedLinkPredictor(nn.Module):
    """
    Enhanced link prediction model with multiple prediction strategies.
    """
    
    def __init__(self, hidden_dim: int, prediction_type: str = 'dot_product',
                 use_edge_features: bool = False, edge_feature_dim: int = 0):
        """
        Initialize enhanced link predictor.
        
        Args:
            hidden_dim: Hidden dimension of node embeddings
            prediction_type: Type of prediction ('dot_product', 'mlp', 'bilinear', 'hadamard')
            use_edge_features: Whether to use edge features
            edge_feature_dim: Dimension of edge features
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.prediction_type = prediction_type
        self.use_edge_features = use_edge_features
        
        if prediction_type == 'dot_product':
            # Simple dot product decoder
            self.predictor = None  # No parameters needed
        elif prediction_type == 'mlp':
            # MLP-based prediction
            input_dim = hidden_dim * 2
            if use_edge_features:
                input_dim += edge_feature_dim
            
            self.predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif prediction_type == 'bilinear':
            # Bilinear transformation
            self.predictor = nn.Bilinear(hidden_dim, hidden_dim, 1)
        elif prediction_type == 'hadamard':
            # Hadamard product + MLP
            input_dim = hidden_dim
            if use_edge_features:
                input_dim += edge_feature_dim
            
            self.predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
    
    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict edge probabilities.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_features: Edge features [num_edges, edge_feature_dim]
            
        Returns:
            Edge prediction logits [num_edges]
        """
        src_embeddings = node_embeddings[edge_index[0]]  # [num_edges, hidden_dim]
        dst_embeddings = node_embeddings[edge_index[1]]  # [num_edges, hidden_dim]
        
        if self.prediction_type == 'dot_product':
            # Simple dot product
            logits = torch.sum(src_embeddings * dst_embeddings, dim=1)
        
        elif self.prediction_type == 'mlp':
            # Concatenate embeddings
            edge_repr = torch.cat([src_embeddings, dst_embeddings], dim=1)
            
            if self.use_edge_features and edge_features is not None:
                edge_repr = torch.cat([edge_repr, edge_features], dim=1)
            
            logits = self.predictor(edge_repr).squeeze(1)
        
        elif self.prediction_type == 'bilinear':
            # Bilinear transformation
            logits = self.predictor(src_embeddings, dst_embeddings).squeeze(1)
        
        elif self.prediction_type == 'hadamard':
            # Hadamard product
            edge_repr = src_embeddings * dst_embeddings
            
            if self.use_edge_features and edge_features is not None:
                edge_repr = torch.cat([edge_repr, edge_features], dim=1)
            
            logits = self.predictor(edge_repr).squeeze(1)
        
        return logits


def create_enhanced_link_prediction_task(config: LinkPredictionConfig) -> Dict[str, any]:
    """
    Create enhanced link prediction task components.
    
    Args:
        config: Link prediction configuration
        
    Returns:
        Dictionary containing sampler, evaluator, and predictor
    """
    sampler = AdvancedNegativeSampler(config)
    evaluator = LinkPredictionEvaluator()
    
    # Create predictor based on config
    predictor = EnhancedLinkPredictor(
        hidden_dim=256,  # This should match your model's hidden dimension
        prediction_type='mlp',  # More sophisticated than dot product
        use_edge_features=config.use_edge_features
    )
    
    return {
        'sampler': sampler,
        'evaluator': evaluator,
        'predictor': predictor,
        'config': config
    }


# Example usage and testing
if __name__ == '__main__':
    # Example configuration
    config = LinkPredictionConfig(
        negative_sampling_ratio=1.0,
        sampling_strategy=SamplingStrategy.DEGREE_BIASED,
        max_edges_per_graph=500
    )
    
    # Create components
    components = create_enhanced_link_prediction_task(config)
    print("Enhanced link prediction components created successfully!")
    
    # Test with dummy data
    num_nodes = 100
    num_edges = 200
    
    # Create dummy graph
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    node_embeddings = torch.randn(num_nodes, 256)
    
    # Test negative sampling
    sampler = components['sampler']
    neg_edges = sampler.sample_negative_edges(edge_index, num_nodes, num_edges)
    
    print(f"Generated {neg_edges.shape[1]} negative edges")
    
    # Test predictor
    predictor = components['predictor']
    all_edges = torch.cat([edge_index, neg_edges], dim=1)
    predictions = predictor(node_embeddings, all_edges)
    
    print(f"Generated {len(predictions)} edge predictions")
    
    # Test evaluator
    evaluator = components['evaluator']
    targets = torch.cat([torch.ones(num_edges), torch.zeros(neg_edges.shape[1])])
    evaluator.add_batch(torch.sigmoid(predictions), targets)
    
    metrics = evaluator.compute_metrics()
    print(f"Computed {len(metrics)} evaluation metrics")
    print("Sample metrics:", {k: f"{v:.4f}" for k, v in list(metrics.items())[:5]}) 