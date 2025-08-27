import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.data.data_setup import PROCESSED_DIR

BATCH_SIZE = 32
NUM_WORKERS = 4


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DomainSplitDataset(Dataset):
    def __init__(self, graphs: List[Data], indices: NDArray[np.int64]) -> None:
        self.graphs = graphs
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[self.indices[idx]]


def _load_graph_properties_for_split(domain_dir: Path, graphs: List[Data], split_idx: NDArray[np.int64]) -> None:
    properties_path = domain_dir / "graph_properties.pt"
    if properties_path.exists():
        graph_properties = torch.load(properties_path)
        for idx in split_idx:
            graphs[idx].graph_properties = graph_properties[idx]


def create_data_loader(domain_name: str, split: str, generator: torch.Generator) -> PyGDataLoader:
    domain_dir = PROCESSED_DIR / domain_name
    data_path = domain_dir / "data.pt"
    splits_path = domain_dir / "splits.pt"

    graphs = torch.load(data_path)
    splits = torch.load(splits_path)
    split_idx = splits[split]

    _load_graph_properties_for_split(domain_dir, graphs, split_idx)

    ds = DomainSplitDataset(graphs, split_idx)

    loader = PyGDataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    return loader


class BalancedMultiDomainSampler:
    def __init__(self, domain_datasets: Dict[str, DomainSplitDataset], generator: torch.Generator):
        self.domain_datasets = domain_datasets
        self.generator = generator

        self.samples_per_domain = BATCH_SIZE // len(domain_datasets)
        self.num_steps = max(len(dataset) for dataset in domain_datasets.values()) // self.samples_per_domain

    def __iter__(self):
        for _ in range(self.num_steps):
            batch_items = []

            for domain, dataset in self.domain_datasets.items():
                indices = torch.randint(0, len(dataset), (self.samples_per_domain,), generator=self.generator)
                for idx in indices:
                    batch_items.append((dataset[idx.item()], domain))

            yield batch_items

    def __len__(self):
        return self.num_steps


def _collate_domain_batches(batch_items: List[Tuple[Data, str]]) -> Dict[str, Batch]:
    domain_graphs = {}

    for graph, domain in batch_items:
        if domain not in domain_graphs:
            domain_graphs[domain] = []
        domain_graphs[domain].append(graph)

    batches_by_domain = {}
    for domain, graphs in domain_graphs.items():
        batches_by_domain[domain] = Batch.from_data_list(graphs)

    return batches_by_domain


def create_pretrain_data_loader(domains: List[str], generator: torch.Generator) -> torch.utils.data.DataLoader:
    domain_datasets = {}

    for domain in domains:
        domain_dir = PROCESSED_DIR / domain
        data_path = domain_dir / "data.pt"
        splits_path = domain_dir / "splits.pt"

        graphs = torch.load(data_path)
        splits = torch.load(splits_path)
        split_idx = splits["train"]

        _load_graph_properties_for_split(domain_dir, graphs, split_idx)

        domain_datasets[domain] = DomainSplitDataset(graphs, split_idx)

    sampler = BalancedMultiDomainSampler(domain_datasets, generator)

    loader = torch.utils.data.DataLoader(
        dataset=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_seed_worker,
        collate_fn=_collate_domain_batches
    )

    return loader
