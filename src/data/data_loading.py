"""
Data loading and batch processing for GNN training pipeline.

This module provides domain-balanced sampling, custom batch collation,
and efficient data loading for multi-domain graph pre-training.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.data import Data, Batch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import random
from collections import defaultdict
import logging


class GraphDataset(Dataset):
    """
    Dataset for loading processed graph data for training.
    
    This dataset handles loading pre-processed graphs from disk and
    provides access to graphs with their domain labels.
    """
    
    def __init__(self, processed_data_dir: Union[str, Path], 
                 dataset_names: List[str], 
                 split: str = 'train'):
        """
        Initialize the graph dataset.
        
        Args:
            processed_data_dir: Directory containing processed datasets
            dataset_names: List of dataset names to load (e.g., ['MUTAG', 'PROTEINS'])
            split: Data split to load ('train', 'val', 'test')
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.dataset_names = dataset_names
        self.split = split
        
        # Load all graphs and create domain mappings
        self.graphs = []
        self.domain_labels = []
        self.domain_to_idx = {name: idx for idx, name in enumerate(dataset_names)}
        self.idx_to_domain = {idx: name for idx, name in enumerate(dataset_names)}
        
        # Statistics
        self.graphs_per_domain = defaultdict(int)
        
        self._load_graphs()
        
        logging.info(f"Loaded {len(self.graphs)} graphs from {len(dataset_names)} domains")
        for domain, count in self.graphs_per_domain.items():
            logging.info(f"  {domain}: {count} graphs")
    
    def _load_graphs(self):
        """Load graphs from processed data files."""
        for domain_name in self.dataset_names:
            # Load data and splits - using the structure from data_setup.py
            try:
                data_path = self.processed_data_dir / f"{domain_name}_graphs.pt"
                splits_path = self.processed_data_dir / f"{domain_name}_splits.pt"
                
                if not data_path.exists() or not splits_path.exists():
                    logging.warning(f"Missing data files for {domain_name}: data={data_path.exists()}, splits={splits_path.exists()}")
                    continue
                
                # Load data - using weights_only=False for graph data objects (they contain custom classes)
                # This is safe for our own saved data files
                graphs = torch.load(data_path, weights_only=False)
                splits = torch.load(splits_path, weights_only=False)
                
                if self.split not in splits:
                    logging.warning(f"Split '{self.split}' not found for {domain_name}")
                    continue
                
                # Get indices for the requested split
                split_indices = splits[self.split]
                
                # Add graphs from this split
                domain_idx = self.domain_to_idx[domain_name]
                for idx in split_indices:
                    if idx < len(graphs):
                        graph = graphs[idx]
                        # Ensure graph is a proper Data object
                        if not isinstance(graph, Data):
                            # Convert if needed
                            if hasattr(graph, 'x') and hasattr(graph, 'edge_index'):
                                graph = Data(x=graph.x, edge_index=graph.edge_index, 
                                           y=getattr(graph, 'y', None))
                        
                        self.graphs.append(graph)
                        self.domain_labels.append(domain_idx)
                        self.graphs_per_domain[domain_name] += 1
                
            except Exception as e:
                logging.error(f"Error loading {domain_name}: {str(e)}")
                continue
    
    def __len__(self) -> int:
        """Return the total number of graphs."""
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Tuple[Data, int]:
        """
        Get a graph and its domain label.
        
        Args:
            idx: Index of the graph
            
        Returns:
            Tuple of (graph, domain_label)
        """
        return self.graphs[idx], self.domain_labels[idx]
    
    def get_domain_name(self, domain_idx: int) -> str:
        """Get domain name from domain index."""
        return self.idx_to_domain[domain_idx]
    
    def get_domain_indices(self, domain_name: str) -> List[int]:
        """Get all indices for graphs from a specific domain."""
        domain_idx = self.domain_to_idx[domain_name]
        return [i for i, label in enumerate(self.domain_labels) if label == domain_idx]


class DomainBalancedSampler(Sampler):
    """
    Custom sampler that ensures balanced representation from all domains in each batch.
    
    This sampler implements the domain-balanced sampling strategy described in the
    research plan: first select a domain uniformly, then sample a graph from that domain.
    """
    
    def __init__(self, dataset: GraphDataset, batch_size: int, 
                 drop_last: bool = True, shuffle: bool = True):
        """
        Initialize the domain-balanced sampler.
        
        Args:
            dataset: GraphDataset to sample from
            batch_size: Size of each batch
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to shuffle the data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Group indices by domain
        self.domain_indices = defaultdict(list)
        for idx, domain_label in enumerate(dataset.domain_labels):
            domain_name = dataset.get_domain_name(domain_label)
            self.domain_indices[domain_name].append(idx)
        
        self.domains = list(self.domain_indices.keys())
        self.num_domains = len(self.domains)
        
        if self.num_domains == 0:
            raise ValueError("No domains found in dataset")
        
        # Calculate number of batches
        total_samples = len(dataset)
        if self.drop_last:
            self.num_batches = total_samples // batch_size
        else:
            self.num_batches = (total_samples + batch_size - 1) // batch_size
        
        logging.info(f"DomainBalancedSampler: {self.num_domains} domains, {self.num_batches} batches")
    
    def __iter__(self):
        """Generate batches with domain-balanced sampling."""
        # Shuffle indices within each domain if requested
        if self.shuffle:
            for domain in self.domains:
                random.shuffle(self.domain_indices[domain])
        
        # Create iterators for each domain
        domain_iterators = {}
        for domain in self.domains:
            indices = self.domain_indices[domain]
            if self.shuffle:
                # Create a shuffled, repeating iterator
                domain_iterators[domain] = self._create_repeating_iterator(indices)
            else:
                domain_iterators[domain] = iter(indices * 1000)  # Repeat many times
        
        # Generate individual indices (DataLoader will batch them)
        total_samples = self.num_batches * self.batch_size
        
        for _ in range(total_samples):
            # Uniformly select a domain
            domain = random.choice(self.domains)
            
            # Sample from that domain
            try:
                idx = next(domain_iterators[domain])
                yield idx
            except StopIteration:
                # Recreate iterator if exhausted
                indices = self.domain_indices[domain]
                if self.shuffle:
                    random.shuffle(indices)
                domain_iterators[domain] = iter(indices)
                idx = next(domain_iterators[domain])
                yield idx
    
    def _create_repeating_iterator(self, indices):
        """Create an iterator that repeats the indices indefinitely with shuffling."""
        while True:
            shuffled_indices = indices.copy()
            random.shuffle(shuffled_indices)
            for idx in shuffled_indices:
                yield idx
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return self.num_batches


def graph_collate_fn(batch: List[Tuple[Data, int]]) -> Tuple[List[Data], torch.Tensor]:
    """
    Custom collate function for multi-domain graphs.
    
    Since graphs from different domains have different feature dimensions,
    we can't batch them into a single disconnected graph. Instead, we return
    a list of individual graphs with their domain labels.
    
    Args:
        batch: List of (Data, domain_label) tuples
        
    Returns:
        Tuple of (list_of_graphs, domain_labels_tensor)
    """
    graphs, domain_labels = zip(*batch)
    
    # Convert domain labels to tensor
    domain_labels = torch.tensor(domain_labels, dtype=torch.long)
    
    # Return list of individual graphs (can't batch due to different feature dims)
    return list(graphs), domain_labels


def negative_sampling(edge_index: torch.Tensor, num_nodes: int, 
                     num_neg_samples: Optional[int] = None,
                     batch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Generate negative edges for link prediction.
    
    Args:
        edge_index: Positive edge indices [2, num_edges]
        num_nodes: Total number of nodes
        num_neg_samples: Number of negative samples (default: same as positive)
        batch: Batch assignment for nodes (for batched graphs)
        
    Returns:
        Negative edge indices [2, num_neg_samples]
    """
    if num_neg_samples is None:
        num_neg_samples = edge_index.shape[1]
    
    # Convert to set for fast lookup
    pos_edges = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        pos_edges.add((src, dst))
        pos_edges.add((dst, src))  # Add reverse edge for undirected graphs
    
    neg_edges = []
    attempts = 0
    max_attempts = num_neg_samples * 10  # Avoid infinite loops
    
    while len(neg_edges) < num_neg_samples and attempts < max_attempts:
        # Sample random node pairs
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        
        # Skip self-loops and existing edges
        if src == dst or (src, dst) in pos_edges:
            attempts += 1
            continue
        
        # If using batched graphs, ensure nodes are from the same graph
        if batch is not None:
            if batch[src] != batch[dst]:
                attempts += 1
                continue
        
        neg_edges.append([src, dst])
        attempts += 1
    
    if len(neg_edges) == 0:
        # Fallback: create some negative edges even if not perfect
        neg_edges = [[0, 1]] if num_nodes > 1 else [[0, 0]]
    
    return torch.tensor(neg_edges, dtype=torch.long).t()


def compute_graph_properties(data: Data) -> torch.Tensor:
    """
    Compute graph-level properties for auxiliary supervision.
    
    Args:
        data: PyTorch Geometric Data object
        
    Returns:
        Tensor of graph properties [num_nodes, num_edges, avg_clustering]
    """
    num_nodes = data.x.shape[0] if data.x is not None else 0
    num_edges = data.edge_index.shape[1] if data.edge_index is not None else 0
    
    # Compute average clustering coefficient (simplified version)
    # For efficiency, we'll use a simple approximation
    if num_nodes > 0 and num_edges > 0:
        # Average degree as a proxy for clustering
        avg_degree = (2 * num_edges) / num_nodes
        # Normalize to [0, 1] range (rough approximation)
        avg_clustering = min(avg_degree / num_nodes, 1.0)
    else:
        avg_clustering = 0.0
    
    return torch.tensor([num_nodes, num_edges, avg_clustering], dtype=torch.float)


def create_data_loaders(processed_data_dir: Union[str, Path],
                       dataset_names: List[str],
                       batch_size: int = 32,
                       domain_balanced_sampling: bool = True,
                       num_workers: int = 4,
                       pin_memory: bool = True,
                       shuffle: bool = True,
                       drop_last: bool = True) -> Dict[str, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        processed_data_dir: Directory containing processed datasets
        dataset_names: List of dataset names to load
        batch_size: Batch size for data loaders
        domain_balanced_sampling: Whether to use domain-balanced sampling
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        Dictionary containing 'train' and 'val' data loaders
    """
    data_loaders = {}
    
    for split in ['train', 'val']:
        # Create dataset
        dataset = GraphDataset(
            processed_data_dir=processed_data_dir,
            dataset_names=dataset_names,
            split=split
        )
        
        if len(dataset) == 0:
            logging.warning(f"No data found for split '{split}'")
            continue
        
        # Create sampler
        if domain_balanced_sampling and split == 'train':
            sampler = DomainBalancedSampler(
                dataset=dataset,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=shuffle
            )
            # When using custom sampler, disable DataLoader's shuffle
            loader_shuffle = False
        else:
            sampler = None
            loader_shuffle = shuffle
        
        # Create data loader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=loader_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=graph_collate_fn
        )
        
        data_loaders[split] = data_loader
        
        logging.info(f"Created {split} data loader: {len(data_loader)} batches")
    
    return data_loaders


if __name__ == '__main__':
    # Test the data loading system
    import sys
    sys.path.append('.')
    
    # Test configuration
    processed_data_dir = 'data/processed'
    dataset_names = ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES']
    
    try:
        # Create data loaders
        data_loaders = create_data_loaders(
            processed_data_dir=processed_data_dir,
            dataset_names=dataset_names,
            batch_size=8,
            domain_balanced_sampling=True,
            num_workers=0  # Use 0 for testing
        )
        
        if 'train' in data_loaders:
            # Test a few batches
            train_loader = data_loaders['train']
            print(f"Train loader: {len(train_loader)} batches")
            
            for i, (batch_data, domain_labels) in enumerate(train_loader):
                print(f"Batch {i}: {batch_data.num_graphs} graphs, domains: {domain_labels}")
                
                # Test negative sampling
                neg_edges = negative_sampling(
                    batch_data.edge_index, 
                    batch_data.num_nodes,
                    batch=batch_data.batch
                )
                print(f"  Generated {neg_edges.shape[1]} negative edges")
                
                # Test graph properties
                if i == 0:  # Test on first graph only
                    first_graph = batch_data.get_example(0)
                    props = compute_graph_properties(first_graph)
                    print(f"  Graph properties: {props}")
                
                if i >= 2:  # Test only first few batches
                    break
        
        print("Data loading test completed successfully!")
        
    except Exception as e:
        print(f"Data loading test failed: {str(e)}")
        import traceback
        traceback.print_exc() 