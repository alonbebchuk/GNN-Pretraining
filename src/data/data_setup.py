import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import negative_sampling, to_undirected
from tqdm import tqdm

from src.data.graph_properties import GraphPropertyCalculator

MIN_SCALE = -3.0
MAX_SCALE = 3.0
RANDOM_SEED = 42
VAL_FRACTION = 0.1
VAL_TEST_FRACTION = 0.2
VAL_TEST_SPLIT_RATIO = 0.5

CONTINUOUS_TUDATASETS = ['PROTEINS', 'ENZYMES']
DOWNSTREAM_TUDATASETS = ['ENZYMES', 'PTC_MR']
PRETRAIN_TUDATASETS = ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES']
TUDATASETS = ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES', 'PTC_MR']

PLANETOID_DATASETS = ['Cora', 'CiteSeer']

DOMAIN_DIMENSIONS = {
    'MUTAG': 7,
    'PROTEINS': 4,
    'NCI1': 37,
    'ENZYMES': 21,
    'PTC_MR': 18,
    'Cora_NC': 1433,
    'CiteSeer_NC': 3703,
    'Cora_LP': 1433,
    'CiteSeer_LP': 3703
}

NUM_CLASSES = {
    'ENZYMES': 6,
    'PTC_MR': 2,
    'Cora_NC': 7,
    'CiteSeer_NC': 6,
    'Cora_LP': 2,
    'CiteSeer_LP': 2
}

TASK_TYPES = {
    'ENZYMES': 'graph_classification',
    'PTC_MR': 'graph_classification',
    'Cora_NC': 'node_classification',
    'CiteSeer_NC': 'node_classification',
    'Cora_LP': 'link_prediction',
    'CiteSeer_LP': 'link_prediction'
}

DATA_ROOT_DIR = Path(__file__).parent.parent.parent / 'data'
RAW_DIR = DATA_ROOT_DIR / 'raw'
PROCESSED_DIR = DATA_ROOT_DIR / 'processed'


def save_processed_data(dataset_name: str, data: List[Data], splits: Dict[str, torch.Tensor], graph_properties: Optional[torch.Tensor] = None) -> None:
    save_dir = PROCESSED_DIR / dataset_name
    os.makedirs(save_dir, exist_ok=True)
    torch.save(data, save_dir / 'data.pt')
    torch.save(splits, save_dir / 'splits.pt')
    if graph_properties is not None:
        torch.save(graph_properties, save_dir / 'graph_properties.pt')


def process_tudatasets() -> None:
    calculator = GraphPropertyCalculator()
    for name in tqdm(TUDATASETS, desc="Processing TU datasets"):
        dataset = TUDataset(root=RAW_DIR, name=name, use_node_attr=True)
        dataset_list = list(dataset)
        num_graphs = len(dataset_list)
        needs_pretrain = name in PRETRAIN_TUDATASETS
        needs_downstream = name in DOWNSTREAM_TUDATASETS

        if needs_downstream:
            labels = dataset.y.numpy()
            sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=VAL_TEST_FRACTION, random_state=RANDOM_SEED)
            train_idx, val_test_idx = next(sss_train_val.split(np.arange(num_graphs), labels))
            val_test_labels = labels[val_test_idx]

            if name in CONTINUOUS_TUDATASETS:
                train_X_list = [dataset_list[i].x.detach().cpu() for i in train_idx]
                train_X = torch.cat(train_X_list, dim=0).numpy()
                scaler = StandardScaler()
                scaler.fit(train_X)
                scaler.scale_[scaler.scale_ == 0] = 1.0
                for g in dataset_list:
                    X = g.x.detach().cpu().numpy()
                    X_scaled = scaler.transform(X)
                    X_scaled = np.clip(X_scaled, MIN_SCALE, MAX_SCALE)
                    g.x = torch.from_numpy(X_scaled).to(g.x.dtype)

            sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=VAL_TEST_SPLIT_RATIO, random_state=RANDOM_SEED)
            val_idx_rel, test_idx_rel = next(sss_val_test.split(np.arange(len(val_test_idx)), val_test_labels))
            val_idx, test_idx = val_test_idx[val_idx_rel], val_test_idx[test_idx_rel]

            splits = {
                'train': torch.tensor(train_idx, dtype=torch.long),
                'val': torch.tensor(val_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)
            }
            graph_properties = calculator.compute_and_standardize_for_dataset(dataset_list, train_idx) if needs_pretrain else None
            save_processed_data(name, dataset_list, splits, graph_properties)

        elif needs_pretrain:
            ss_train_val = ShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
            train_idx, val_idx = next(ss_train_val.split(np.arange(num_graphs)))

            splits = {
                'train': torch.tensor(train_idx, dtype=torch.long),
                'val': torch.tensor(val_idx, dtype=torch.long),
            }
            graph_properties = calculator.compute_and_standardize_for_dataset(dataset_list, train_idx)
            save_processed_data(name, dataset_list, splits, graph_properties)


def create_link_prediction_splits(data: Data) -> Dict[str, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    num_edges = data.num_edges
    num_val_test = int(num_edges * VAL_TEST_FRACTION)
    num_val = int(num_val_test * VAL_TEST_SPLIT_RATIO)

    perm = torch.randperm(num_edges, generator=generator)
    train_edges = data.edge_index[:, perm[num_val_test:]]
    val_test_edges = data.edge_index[:, perm[:num_val_test]]
    val_test_neg_edges = negative_sampling(
        edge_index=to_undirected(train_edges),
        num_nodes=data.num_nodes,
        num_neg_samples=num_val_test
    )

    return {
        'train_pos': train_edges,
        'val_pos': val_test_edges[:, :num_val],
        'val_neg': val_test_neg_edges[:, :num_val],
        'test_pos': val_test_edges[:, num_val:],
        'test_neg': val_test_neg_edges[:, num_val:]
    }


def process_planetoid_datasets() -> None:
    for name in tqdm(PLANETOID_DATASETS, desc="Processing Planetoid datasets"):
        dataset = Planetoid(root=RAW_DIR, name=name, transform=NormalizeFeatures())
        data = dataset[0]

        nc_splits = {
            'train': torch.where(data.train_mask)[0],
            'val': torch.where(data.val_mask)[0],
            'test': torch.where(data.test_mask)[0]
        }
        save_processed_data(f"{name}_NC", [data], nc_splits)

        lp_splits = create_link_prediction_splits(data)
        save_processed_data(f"{name}_LP", [data], lp_splits)


def main() -> None:
    os.makedirs(DATA_ROOT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    process_tudatasets()
    process_planetoid_datasets()


if __name__ == "__main__":
    main()
