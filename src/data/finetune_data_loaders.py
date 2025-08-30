from typing import Dict, List, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.data.data_setup import PROCESSED_DIR, TASK_TYPES
from src.data.pretrain_data_loaders import GraphDataset


class NodeDataset(Dataset):
    def __init__(self, data: Data, indices: NDArray[np.int64]) -> None:
        self.data = data
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        node_idx = self.indices[idx]
        return self.data, node_idx, self.data.y[node_idx]


class LinkPredictionDataset(Dataset):
    def __init__(self, data: Data, split_edges: Dict[str, torch.Tensor], split: str) -> None:
        self.data = data
        self.split = split

        if split == 'train':
            self.edges = split_edges['train_pos']
            self.labels = torch.ones(self.edges.size(1))
        else:
            self.pos_edges = split_edges[f'{split}_pos']
            self.neg_edges = split_edges[f'{split}_neg']
            self.edges = torch.cat([self.pos_edges, self.neg_edges], dim=1)
            self.labels = torch.cat([
                torch.ones(self.pos_edges.size(1)),
                torch.zeros(self.neg_edges.size(1))
            ])

    def __len__(self) -> int:
        return self.edges.size(1)

    def __getitem__(self, idx: int) -> tuple:
        edge = self.edges[:, idx]
        label = self.labels[idx]
        return self.data, edge, label


def _collate_node_batch(batch: List[tuple]) -> tuple:
    data = batch[0][0]
    node_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return data, node_indices, labels


def _collate_link_batch(batch: List[tuple]) -> tuple:
    data = batch[0][0]
    edges = torch.stack([item[1] for item in batch], dim=1)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float)
    return data, edges, labels


def create_graph_classification_loader(domain_name: str, split: str, batch_size: int, generator: torch.Generator) -> PyGDataLoader:
    domain_dir = PROCESSED_DIR / domain_name

    graphs = torch.load(domain_dir / "data.pt")
    splits = torch.load(domain_dir / "splits.pt")
    split_indices = splits[split]

    dataset = GraphDataset(graphs, split_indices)
    return PyGDataLoader(dataset, batch_size=batch_size, generator=generator)


def create_node_classification_loader(domain_name: str, split: str, batch_size: int, generator: torch.Generator) -> torch.utils.data.DataLoader:
    domain_dir = PROCESSED_DIR / domain_name

    graphs = torch.load(domain_dir / "data.pt")
    splits = torch.load(domain_dir / "splits.pt")
    data = graphs[0]
    split_indices = splits[split]

    dataset = NodeDataset(data, split_indices)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, generator=generator, collate_fn=_collate_node_batch)


def create_link_prediction_loader(domain_name: str, split: str, batch_size: int, generator: torch.Generator) -> torch.utils.data.DataLoader:
    domain_dir = PROCESSED_DIR / domain_name

    graphs = torch.load(domain_dir / "data.pt")
    splits = torch.load(domain_dir / "splits.pt")
    data = graphs[0]
    split_indices = splits[split]

    dataset = LinkPredictionDataset(data, split_indices, split)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, generator=generator, collate_fn=_collate_link_batch)


def create_finetune_data_loader(domain_name: str, split: str, batch_size: int, generator: torch.Generator) -> Union[PyGDataLoader, torch.utils.data.DataLoader]:
    task_type = TASK_TYPES[domain_name]

    if task_type == 'graph_classification':
        return create_graph_classification_loader(domain_name, split, batch_size, generator)
    elif task_type == 'node_classification':
        return create_node_classification_loader(domain_name, split, batch_size, generator)
    elif task_type == 'link_prediction':
        return create_link_prediction_loader(domain_name, split, batch_size, generator)
