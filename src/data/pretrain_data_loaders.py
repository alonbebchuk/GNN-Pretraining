from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.data.data_setup import PROCESSED_DIR

BATCH_SIZE = 32


class GraphDataset(Dataset):
    def __init__(self, graphs: List[Data], indices: NDArray[np.int64]) -> None:
        self.graphs = graphs
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[self.indices[idx]]


class BalancedMultiDomainSampler:
    def __init__(self, domain_datasets: Dict[str, GraphDataset], generator: torch.Generator) -> None:
        self.domain_datasets = domain_datasets
        self.generator = generator
        self.samples_per_domain = BATCH_SIZE // len(domain_datasets)
        self.num_steps = max(len(dataset) for dataset in domain_datasets.values()) // self.samples_per_domain

    def __iter__(self) -> Iterator[Dict[str, Batch]]:
        for _ in range(self.num_steps):
            domain_batches = {}
            for domain, dataset in self.domain_datasets.items():
                indices = torch.randint(0, len(dataset), (self.samples_per_domain,), generator=self.generator)
                graphs = [dataset[idx.item()] for idx in indices]
                domain_batches[domain] = Batch.from_data_list(graphs)

            yield domain_batches

    def __len__(self) -> int:
        return self.num_steps


def _load_graph_properties_for_split(domain_dir: Path, graphs: List[Data], split_indices: NDArray[np.int64]) -> None:
    properties_path = domain_dir / "graph_properties.pt"
    graph_properties = torch.load(properties_path)
    for idx in split_indices:
        graphs[idx].graph_properties = graph_properties[idx]


def create_val_data_loader(domain_name: str, generator: torch.Generator) -> PyGDataLoader:
    domain_dir = PROCESSED_DIR / domain_name

    graphs = torch.load(domain_dir / "data.pt")
    splits = torch.load(domain_dir / "splits.pt")
    split_indices = splits["val"]
    _load_graph_properties_for_split(domain_dir, graphs, split_indices)

    dataset = GraphDataset(graphs, split_indices)
    return PyGDataLoader(dataset, batch_size=BATCH_SIZE, generator=generator)


def create_train_data_loader(domains: List[str], generator: torch.Generator) -> torch.utils.data.DataLoader:
    domain_datasets = {}

    for domain in domains:
        domain_dir = PROCESSED_DIR / domain

        graphs = torch.load(domain_dir / "data.pt")
        splits = torch.load(domain_dir / "splits.pt")
        split_indices = splits["train"]
        _load_graph_properties_for_split(domain_dir, graphs, split_indices)

        domain_datasets[domain] = GraphDataset(graphs, split_indices)

    sampler = BalancedMultiDomainSampler(domain_datasets, generator)
    return torch.utils.data.DataLoader(sampler)
