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
    Compute comprehensive graph-level properties for auxiliary supervision.
    
    This function computes 15 graph properties following scientific standards:
    1. num_nodes - Number of nodes
    2. num_edges - Number of edges  
    3. density - Graph density
    4. avg_degree - Average node degree
    5. degree_variance - Variance of node degrees
    6. max_degree - Maximum node degree
    7. global_clustering - Global clustering coefficient
    8. transitivity - Transitivity (alternative clustering measure)
    9. num_triangles - Total number of triangles
    10. num_components - Number of connected components
    11. diameter_approx - Approximate diameter (capped for efficiency)
    12. assortativity_approx - Approximate degree assortativity
    13. density_weighted_degree - Density-weighted average degree
    14. edge_connectivity_ratio - Ratio of actual to maximum possible edges
    15. degree_centralization - Degree centralization measure
    
    Args:
        data: PyTorch Geometric Data object
        
    Returns:
        Tensor of 15 comprehensive graph properties
    """
    from torch_geometric.utils import degree, to_networkx
    import networkx as nx
    import torch
    
    # Handle completely empty data
    if data.x is None:
        return torch.zeros(15, dtype=torch.float)
    
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1] if data.edge_index is not None else 0
    
    # Handle empty graph (no nodes)
    if num_nodes == 0:
        return torch.zeros(15, dtype=torch.float)
    
    # Handle single node case
    if num_nodes == 1:
        properties = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return torch.tensor(properties, dtype=torch.float)
    
    # For undirected graphs, divide edges by 2
    if data.edge_index.shape[1] > 0:
        # Check if graph is undirected (has both (i,j) and (j,i) for most edges)
        edge_set = set()
        reverse_edges = 0
        for i in range(min(100, data.edge_index.shape[1])):  # Sample check
            edge = (data.edge_index[0, i].item(), data.edge_index[1, i].item())
            reverse_edge = (edge[1], edge[0])
            if reverse_edge in edge_set:
                reverse_edges += 1
            edge_set.add(edge)
        
        is_undirected = reverse_edges > len(edge_set) * 0.8  # Heuristic check
        if is_undirected:
            num_edges = num_edges // 2
    
    # 1-3: Basic structural properties
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0
    
    # 4-6: Degree statistics
    degrees = degree(data.edge_index[0], num_nodes=num_nodes).float()
    avg_degree = degrees.mean().item()
    degree_variance = degrees.var().item() if degrees.numel() > 1 else 0.0
    max_degree = degrees.max().item()
    
    # 7-9: Clustering and triangle properties
    try:
        # For efficiency, limit graph size for complex computations
        if num_nodes <= 1000:
            # Convert to NetworkX for accurate clustering computation
            G = to_networkx(data, to_undirected=True)
            G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
            
            # Global clustering coefficient
            global_clustering = nx.average_clustering(G) if len(G.edges()) > 0 else 0.0
            
            # Transitivity
            transitivity = nx.transitivity(G)
            
            # Triangle count
            triangles = sum(nx.triangles(G).values()) // 3  # Each triangle counted 3 times
            num_triangles = float(triangles)
            
        else:
            # Approximations for large graphs
            # Simple clustering approximation
            if num_edges > 0:
                expected_triangles = (avg_degree ** 2) * num_nodes / 6
                global_clustering = min(expected_triangles / (num_nodes * avg_degree), 1.0) if avg_degree > 0 else 0.0
            else:
                global_clustering = 0.0
            
            transitivity = global_clustering  # Use same approximation
            num_triangles = global_clustering * num_nodes * avg_degree if avg_degree > 0 else 0.0
            
    except Exception:
        # Fallback to simple approximations
        global_clustering = min(avg_degree / num_nodes, 1.0) if num_nodes > 0 and avg_degree > 0 else 0.0
        transitivity = global_clustering
        num_triangles = 0.0
    
    # 10: Connected components
    try:
        if num_nodes <= 1000:
            G = to_networkx(data, to_undirected=True)
            num_components = float(nx.number_connected_components(G))
        else:
            # Approximation: assume mostly connected for large graphs
            num_components = 1.0 if num_edges > 0 else float(num_nodes)
    except Exception:
        num_components = 1.0 if num_edges > 0 else float(num_nodes)
    
    # 11: Approximate diameter (capped for efficiency)
    try:
        if num_nodes <= 500 and num_edges > 0:
            G = to_networkx(data, to_undirected=True)
            if nx.is_connected(G):
                diameter_approx = float(nx.diameter(G))
            else:
                # Use largest component diameter
                largest_cc = max(nx.connected_components(G), key=len)
                if len(largest_cc) > 1:
                    diameter_approx = float(nx.diameter(G.subgraph(largest_cc)))
                else:
                    diameter_approx = 0.0
        else:
            # Approximation based on graph structure
            if num_edges > 0:
                diameter_approx = min(num_nodes - 1, max(3.0, torch.log2(torch.tensor(float(num_nodes))).item()))
            else:
                diameter_approx = 0.0
    except Exception:
        diameter_approx = min(num_nodes - 1, 6.0) if num_edges > 0 else 0.0
    
    # 12: Approximate assortativity
    try:
        if num_nodes <= 1000 and num_edges > 0:
            G = to_networkx(data, to_undirected=True)
            assortativity_approx = nx.degree_assortativity_coefficient(G)
            if torch.isnan(torch.tensor(assortativity_approx)):
                assortativity_approx = 0.0
        else:
            # Simple approximation: high degree variance suggests disassortative mixing
            assortativity_approx = -min(degree_variance / (avg_degree + 1e-8), 1.0) if avg_degree > 0 else 0.0
    except Exception:
        assortativity_approx = 0.0
    
    # 13: Density-weighted degree
    density_weighted_degree = avg_degree * density
    
    # 14: Edge connectivity ratio
    edge_connectivity_ratio = num_edges / max(num_nodes - 1, 1) if num_nodes > 1 else 0.0
    
    # 15: Degree centralization
    if max_degree > 0 and num_nodes > 2:
        degree_centralization = ((max_degree - avg_degree) * num_nodes) / ((num_nodes - 1) * (num_nodes - 2))
        degree_centralization = min(degree_centralization, 1.0)
    else:
        degree_centralization = 0.0
    
    # Compile all properties
    properties = [
        float(num_nodes),           # 1
        float(num_edges),           # 2  
        density,                    # 3
        avg_degree,                 # 4
        degree_variance,            # 5
        max_degree,                 # 6
        global_clustering,          # 7
        transitivity,               # 8
        num_triangles,              # 9
        num_components,             # 10
        diameter_approx,            # 11
        assortativity_approx,       # 12
        density_weighted_degree,    # 13
        edge_connectivity_ratio,    # 14
        degree_centralization       # 15
    ]
    
    # Ensure all properties are finite
    properties = [p if torch.isfinite(torch.tensor(p)) else 0.0 for p in properties]
    
    return torch.tensor(properties, dtype=torch.float)


def create_data_loaders(processed_data_dir: Union[str, Path],
                       dataset_names: List[str],
                       batch_size: int = 32,
                       domain_balanced_sampling: bool = True,
                       num_workers: int = 4,
                       pin_memory: bool = True,
                       shuffle: bool = True,
                       drop_last: bool = True,
                       seed: Optional[int] = None) -> Dict[str, DataLoader]:
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

    # Reproducibility utilities (optional import path depending on execution context)
    try:
        from ..infrastructure.reproducibility import seed_worker, get_generator
    except Exception:
        try:
            from infrastructure.reproducibility import seed_worker, get_generator
        except Exception:
            seed_worker = None  # type: ignore
            get_generator = lambda s: None  # type: ignore
    
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
            collate_fn=graph_collate_fn,
            worker_init_fn=seed_worker if seed is not None and seed_worker is not None else None,
            generator=get_generator(seed) if seed is not None else None
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