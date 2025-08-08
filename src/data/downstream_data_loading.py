"""
Downstream data loading utilities for fine-tuning evaluation.

This module handles loading and preprocessing data for downstream tasks:
- Graph classification (ENZYMES, FRANKENSTEIN, PTC_MR)
- Node classification (Cora, CiteSeer)
- Link prediction (Cora, CiteSeer) - Enhanced implementation
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.model_selection import train_test_split


class DownstreamGraphDataset(Dataset):
    """Dataset for downstream graph classification tasks."""
    
    def __init__(self, graphs: List[Data], labels: List[int], split_indices: List[int]):
        """
        Initialize downstream dataset.
        
        Args:
            graphs: List of graph data objects
            labels: List of graph labels
            split_indices: Indices for this split
        """
        self.graphs = [graphs[i] for i in split_indices]
        self.labels = [labels[i] for i in split_indices]
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


class DownstreamNodeDataset(Dataset):
    """Dataset for downstream node classification tasks."""
    
    def __init__(self, data: Data, split_mask: torch.Tensor):
        """
        Initialize node dataset.
        
        Args:
            data: Single graph with node features and labels
            split_mask: Boolean mask for this split
        """
        self.data = data
        self.split_mask = split_mask
        self.node_indices = torch.where(split_mask)[0]
    
    def __len__(self):
        return len(self.node_indices)
    
    def __getitem__(self, idx):
        # For node classification, we return the full graph but indicate which nodes to use
        return self.data, self.node_indices[idx]


class EnhancedLinkPredictionDataset(Dataset):
    """Enhanced dataset for link prediction tasks."""
    
    def __init__(self, data: Data, edge_split: str = 'train'):
        """
        Initialize link prediction dataset.
        
        Args:
            data: Graph data with edge splits
            edge_split: Which edge split to use ('train', 'val', 'test')
        """
        self.data = data
        self.edge_split = edge_split
        
        # Generate positive and negative edges for this split
        self._prepare_edges()
    
    def _prepare_edges(self):
        """Prepare positive and negative edges for the split."""
        try:
            from link_prediction import LinkPredictionConfig, SamplingStrategy, AdvancedNegativeSampler
            
            # Get positive edges for this split
            if hasattr(self.data, f'{self.edge_split}_edge_index'):
                self.pos_edges = getattr(self.data, f'{self.edge_split}_edge_index')
            else:
                # Fallback: use all edges (for compatibility)
                self.pos_edges = self.data.edge_index
            
            # Configure negative sampling
            config = LinkPredictionConfig(
                negative_sampling_ratio=1.0,
                sampling_strategy=SamplingStrategy.STRUCTURAL,
                max_edges_per_graph=min(1000, self.pos_edges.shape[1] * 2)
            )
            
            # Generate negative edges
            sampler = AdvancedNegativeSampler(config)
            self.neg_edges = sampler.sample_negative_edges(
                self.pos_edges, 
                self.data.num_nodes,
                num_neg_samples=self.pos_edges.shape[1],
                node_features=self.data.x if hasattr(self.data, 'x') else None
            )
            
            # Combine positive and negative edges
            self.all_edges = torch.cat([self.pos_edges, self.neg_edges], dim=1)
            self.edge_labels = torch.cat([
                torch.ones(self.pos_edges.shape[1]),
                torch.zeros(self.neg_edges.shape[1])
            ])
            
        except ImportError:
            logging.warning("Enhanced link prediction not available, using basic implementation")
            # Fallback to basic implementation
            self.pos_edges = self.data.edge_index
            # Simple random negative sampling
            num_neg = self.pos_edges.shape[1]
            neg_edges = []
            while len(neg_edges) < num_neg:
                src = torch.randint(0, self.data.num_nodes, (1,))
                dst = torch.randint(0, self.data.num_nodes, (1,))
                if src != dst:
                    neg_edges.append([src.item(), dst.item()])
            
            self.neg_edges = torch.tensor(neg_edges).t()
            self.all_edges = torch.cat([self.pos_edges, self.neg_edges], dim=1)
            self.edge_labels = torch.cat([
                torch.ones(self.pos_edges.shape[1]),
                torch.zeros(self.neg_edges.shape[1])
            ])
    
    def __len__(self):
        return 1  # Single graph
    
    def __getitem__(self, idx):
        # Return graph with edge prediction data
        data_copy = self.data.clone()
        data_copy.edge_index = self.all_edges
        data_copy.edge_labels = self.edge_labels
        data_copy.pos_edge_index = self.pos_edges
        data_copy.neg_edge_index = self.neg_edges
        
        return data_copy, 0  # Dummy domain label


def load_downstream_data(processed_data_dir: Union[str, Path], 
                        dataset_name: str,
                        task_type: str,
                        batch_size: int = 32,
                        num_workers: int = 4,
                        pin_memory: bool = True,
                        seed: Optional[int] = None) -> Dict[str, DataLoader]:
    """
    Load downstream task data from processed files.
    
    Args:
        processed_data_dir: Directory containing processed data
        dataset_name: Name of the dataset (e.g., 'ENZYMES', 'Cora')
        task_type: Type of task ('graph_classification', 'node_classification', 'link_prediction')
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    processed_data_dir = Path(processed_data_dir)
    
    # Load processed data
    data_file = processed_data_dir / f"{dataset_name}_graphs.pt"
    splits_file = processed_data_dir / f"{dataset_name}_splits.pt"
    
    if not data_file.exists() or not splits_file.exists():
        raise FileNotFoundError(f"Processed data not found for {dataset_name}. Run data_setup.py first.")
    
    graphs = torch.load(data_file, weights_only=False)
    splits = torch.load(splits_file, weights_only=False)
    
    data_loaders = {}

    # Reproducibility utilities
    try:
        from ..infrastructure.reproducibility import seed_worker, get_generator
    except Exception:
        try:
            from infrastructure.reproducibility import seed_worker, get_generator
        except Exception:
            seed_worker = None  # type: ignore
            get_generator = lambda s: None  # type: ignore
    
    if task_type == 'graph_classification':
        # Extract labels from graphs
        labels = [graph.y.item() if graph.y.dim() == 0 else graph.y[0].item() for graph in graphs]
        
        # Create datasets for each split
        for split_name in ['train', 'val', 'test']:
            if split_name in splits:
                dataset = DownstreamGraphDataset(graphs, labels, splits[split_name])
                
                # Use batch_size=1 for very small datasets to avoid empty batches
                effective_batch_size = min(batch_size, len(dataset)) if len(dataset) > 0 else 1
                
                data_loaders[split_name] = DataLoader(
                    dataset,
                    batch_size=effective_batch_size,
                    shuffle=(split_name == 'train'),
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=False,  # Keep all samples for evaluation
                    worker_init_fn=seed_worker if seed is not None and seed_worker is not None else None,
                    generator=get_generator(seed) if seed is not None else None
                )
    
    elif task_type in ['node_classification', 'link_prediction']:
        # For node/link tasks, we typically have a single large graph
        if len(graphs) == 0:
            raise ValueError(f"No graph data found for {dataset_name}")
        
        # Use the first (and typically only) graph
        graph_data = graphs[0]
        
        if task_type == 'node_classification':
            # Create datasets using node masks
            for split_name in ['train', 'val', 'test']:
                if split_name in splits:
                    # Convert indices to mask
                    mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
                    if len(splits[split_name]) > 0:
                        mask[splits[split_name]] = True
                    
                    dataset = DownstreamNodeDataset(graph_data, mask)
                    
                    # For node classification, typically use batch_size=1 (full-batch)
                    data_loaders[split_name] = DataLoader(
                        dataset,
                        batch_size=1,  # Full-batch for node classification
                        shuffle=False,  # Order doesn't matter for full-batch
                        num_workers=0,  # No multiprocessing for single large graph
                        pin_memory=pin_memory,
                        worker_init_fn=seed_worker if seed is not None and seed_worker is not None else None,
                        generator=get_generator(seed) if seed is not None else None
                    )
        
        elif task_type == 'link_prediction':
            # Enhanced link prediction data loading
            # Split edges temporally (80% train, 10% val, 10% test)
            num_edges = graph_data.edge_index.shape[1]
            if num_edges == 0:
                logging.warning(f"No edges found in {dataset_name} for link prediction")
                return {}
            
            # Create edge splits
            perm = torch.randperm(num_edges)
            train_size = int(0.8 * num_edges)
            val_size = int(0.1 * num_edges)
            
            edge_splits = {
                'train': perm[:train_size],
                'val': perm[train_size:train_size + val_size],
                'test': perm[train_size + val_size:]
            }
            
            # Create datasets for each split
            for split_name, edge_indices in edge_splits.items():
                if len(edge_indices) > 0:
                    # Create subgraph with edges for this split
                    split_data = graph_data.clone()
                    split_data.edge_index = graph_data.edge_index[:, edge_indices]
                    
                    # Add split information
                    setattr(split_data, f'{split_name}_edge_index', split_data.edge_index)
                    
                    # Create enhanced dataset
                    dataset = EnhancedLinkPredictionDataset(split_data, split_name)
                    
                    data_loaders[split_name] = DataLoader(
                        dataset,
                        batch_size=1,  # Full-batch for link prediction
                        shuffle=False,
                        num_workers=0,
                        pin_memory=pin_memory,
                        worker_init_fn=seed_worker if seed is not None and seed_worker is not None else None,
                        generator=get_generator(seed) if seed is not None else None
                    )
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Log dataset statistics
    logging.info(f"Loaded {dataset_name} for {task_type}:")
    for split_name, loader in data_loaders.items():
        logging.info(f"  {split_name}: {len(loader.dataset)} samples, {len(loader)} batches")
    
    return data_loaders


def create_downstream_data_loaders(config: Dict, seed: Optional[int] = None) -> Dict[str, DataLoader]:
    """
    Create data loaders for downstream tasks based on configuration.
    
    Args:
        config: Downstream task configuration
        
    Returns:
        Dictionary containing data loaders
    """
    return load_downstream_data(
        processed_data_dir='data/processed',
        dataset_name=config['dataset_name'],
        task_type=config['task_type'],
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        seed=seed
    )


def get_dataset_info(dataset_name: str) -> Dict[str, Union[int, str]]:
    """
    Get comprehensive dataset information including task type, classes, and domain status.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset information
    """
    dataset_info = {
        # TU Datasets (in-domain for pre-training)
        'MUTAG': {
            'input_dim': 7,
            'num_classes': 2, 
            'task_type': 'graph_classification',
            'in_domain': True,
            'domain_name': 'MUTAG'
        },
        'PROTEINS': {
            'input_dim': 4,  # Corrected from plan.md
            'num_classes': 2, 
            'task_type': 'graph_classification',
            'in_domain': True,
            'domain_name': 'PROTEINS'
        },
        'NCI1': {
            'input_dim': 37,
            'num_classes': 2, 
            'task_type': 'graph_classification',
            'in_domain': True,
            'domain_name': 'NCI1'
        },
        'ENZYMES': {
            'input_dim': 21,  # Corrected from plan.md
            'num_classes': 6, 
            'task_type': 'graph_classification',
            'in_domain': True,
            'domain_name': 'ENZYMES'
        },
        
        # TU Datasets (out-of-domain)
        'FRANKENSTEIN': {
            'input_dim': 780,
            'num_classes': 2, 
            'task_type': 'graph_classification',
            'in_domain': False,
            'domain_name': None
        },
        'PTC_MR': {
            'input_dim': 18,  # Corrected from plan.md
            'num_classes': 2, 
            'task_type': 'graph_classification',
            'in_domain': False,
            'domain_name': None
        },
        
        # Planetoid datasets (out-of-domain)
        'Cora': {
            'input_dim': 1433,
            'num_classes': 7, 
            'task_type': 'node_classification',
            'in_domain': False,
            'domain_name': None
        },
        'CiteSeer': {
            'input_dim': 3703,
            'num_classes': 6, 
            'task_type': 'node_classification',
            'in_domain': False,
            'domain_name': None
        },
    }
    
    if dataset_name not in dataset_info:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_info.keys())}")
    
    return dataset_info[dataset_name]


def create_out_of_domain_input_encoder(dataset_name: str, hidden_dim: int = 256, dropout_rate: float = 0.2):
    """
    Create a new input encoder for out-of-domain datasets.
    
    Args:
        dataset_name: Name of the out-of-domain dataset
        hidden_dim: Hidden dimension for the encoder
        dropout_rate: Dropout rate
        
    Returns:
        New InputEncoder instance
    """
    try:
        from src.models.gnn import InputEncoder
    except ImportError:
        from models.gnn import InputEncoder
    
    dataset_info = get_dataset_info(dataset_name)
    input_dim = dataset_info['input_dim']
    
    return InputEncoder(
        dim_in=input_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    )


def validate_task_compatibility(dataset_name: str, task_type: str) -> bool:
    """
    Validate that the specified task type is compatible with the dataset.
    
    Args:
        dataset_name: Name of the dataset
        task_type: Requested task type
        
    Returns:
        True if compatible, False otherwise
    """
    dataset_info = get_dataset_info(dataset_name)
    expected_task_type = dataset_info['task_type']
    
    # Allow link prediction as a valid task for any graph dataset
    if task_type == 'link_prediction' and expected_task_type in ['graph_classification', 'node_classification']:
        return True
    
    return task_type == expected_task_type


def get_recommended_batch_size(dataset_name: str, task_type: str) -> int:
    """
    Get recommended batch size based on dataset and task type.
    
    Args:
        dataset_name: Name of the dataset
        task_type: Type of task
        
    Returns:
        Recommended batch size
    """
    dataset_info = get_dataset_info(dataset_name)
    
    # Node classification typically uses full-batch
    if dataset_info['task_type'] == 'node_classification':
        return 1
    
    # Graph classification can use larger batches
    elif dataset_info['task_type'] == 'graph_classification':
        # Smaller batch for large feature dimensions
        if dataset_info['input_dim'] > 500:
            return 16
        else:
            return 32
    
    # Link prediction
    elif task_type == 'link_prediction':
        return 16  # Moderate batch size for link prediction
    
    return 32  # Default 