import copy
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.transforms import NormalizeFeatures
from tqdm import tqdm

from src.data.graph_properties import GraphPropertyCalculator

ALL_TUDATASETS = ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES', 'FRANKENSTEIN', 'PTC_MR']
PRETRAIN_TUDATASETS = ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES']
DOWNSTREAM_TUDATASETS = ['ENZYMES', 'FRANKENSTEIN', 'PTC_MR']
OVERLAP_TUDATASETS = ['ENZYMES']
ALL_PLANETOID_DATASETS = ['Cora', 'CiteSeer']

FEATURE_TYPES = {
    'MUTAG': 'categorical',
    'PROTEINS': 'categorical',
    'NCI1': 'categorical',
    'ENZYMES': 'continuous',
    'FRANKENSTEIN': 'categorical',
    'PTC_MR': 'categorical',
    'Cora': 'bow',
    'CiteSeer': 'bow'
}

DOMAIN_DIMENSIONS = {
    'MUTAG': 7,
    'PROTEINS': 4,
    'NCI1': 37,
    'ENZYMES': 21,
    'FRANKENSTEIN': 780,
    'PTC_MR': 18,
    'Cora': 1433,
    'CiteSeer': 3703
}

DATASET_SIZES = {
    'MUTAG': 188,
    'PROTEINS': 1113,
    'NCI1': 4110,
    'ENZYMES': 600,
    'FRANKENSTEIN': 4337,
    'PTC_MR': 344,
    'Cora': 1,
    'CiteSeer': 1
}

RANDOM_SEED = 0
VAL_FRACTION = 0.1
VAL_TEST_FRACTION = 0.2
VAL_TEST_SPLIT_RATIO = 0.5
CONTINUOUS_FEATURE_SCALE_FACTOR = 0.5

DATA_ROOT_DIR = '/kaggle/working/gnn-pretraining/data'
RAW_DIR = Path(DATA_ROOT_DIR) / 'raw'
PROCESSED_DIR = Path(DATA_ROOT_DIR) / 'processed'


def apply_feature_preprocessing(dataset: List[Data], train_idx: NDArray[np.int64], dataset_name: str) -> None:
    if FEATURE_TYPES.get(dataset_name) == 'continuous':
        train_X_list = [dataset[i].x.detach().cpu() for i in train_idx]
        train_X = torch.cat(train_X_list, dim=0).numpy()

        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(train_X)
        scaler.scale_[scaler.scale_ == 0] = 1.0

        for g in dataset:
            X = g.x.detach().cpu().numpy()
            X_scaled = scaler.transform(X) * CONTINUOUS_FEATURE_SCALE_FACTOR
            g.x = torch.from_numpy(X_scaled).to(g.x.dtype)


def save_processed_data(dataset_name: str, data: List[Data], splits: Dict[str, torch.Tensor], graph_properties: torch.Tensor = None) -> None:
    save_dir = PROCESSED_DIR / dataset_name
    os.makedirs(save_dir, exist_ok=True)

    torch.save(data, save_dir / 'data.pt')
    torch.save(splits, save_dir / 'splits.pt')
    if graph_properties is not None:
        torch.save(graph_properties, save_dir / 'graph_properties.pt')


def process_tudatasets() -> None:
    calculator = GraphPropertyCalculator()

    for name in tqdm(ALL_TUDATASETS, desc="Processing TU datasets"):
        dataset = TUDataset(root=RAW_DIR, name=name, use_node_attr=True)

        needs_pretrain = name in PRETRAIN_TUDATASETS
        needs_downstream = name in DOWNSTREAM_TUDATASETS

        if needs_downstream:
            processed_dataset = copy.deepcopy(dataset)
            num_graphs = len(processed_dataset)
            labels = processed_dataset.y.numpy()
            sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=VAL_TEST_FRACTION, random_state=RANDOM_SEED)
            train_idx, val_test_idx = next(sss_train_val.split(np.arange(num_graphs), labels))
            val_test_labels = labels[val_test_idx]
            sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=VAL_TEST_SPLIT_RATIO, random_state=RANDOM_SEED)
            val_idx_rel, test_idx_rel = next(sss_val_test.split(np.arange(len(val_test_idx)), val_test_labels))
            val_idx, test_idx = val_test_idx[val_idx_rel], val_test_idx[test_idx_rel]

            apply_feature_preprocessing(processed_dataset, train_idx, name)

            splits = {
                'train': torch.tensor(train_idx, dtype=torch.long),
                'val': torch.tensor(val_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)
            }

            if needs_pretrain:
                graph_properties = calculator.compute_and_standardize_for_dataset(processed_dataset, train_idx)
                save_processed_data(name, list(processed_dataset), splits, graph_properties)
            else:
                save_processed_data(name, list(processed_dataset), splits)

        elif needs_pretrain:
            processed_dataset = copy.deepcopy(dataset)
            num_graphs = len(processed_dataset)
            ss_train_val = ShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
            train_idx, val_idx = next(ss_train_val.split(np.arange(num_graphs)))

            apply_feature_preprocessing(processed_dataset, train_idx, name)

            splits = {
                'train': torch.tensor(train_idx, dtype=torch.long),
                'val': torch.tensor(val_idx, dtype=torch.long),
            }
            graph_properties = calculator.compute_and_standardize_for_dataset(processed_dataset, train_idx)
            save_processed_data(name, list(processed_dataset), splits, graph_properties)


def process_planetoid_datasets() -> None:
    for name in tqdm(ALL_PLANETOID_DATASETS, desc="Processing Planetoid datasets"):
        dataset = Planetoid(root=RAW_DIR, name=name, transform=NormalizeFeatures())
        data = dataset[0]

        splits = {
            'train': torch.where(data.train_mask)[0],
            'val': torch.where(data.val_mask)[0],
            'test': torch.where(data.test_mask)[0]
        }
        save_processed_data(name, [data], splits)


def main() -> None:
    os.makedirs(DATA_ROOT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    process_tudatasets()
    process_planetoid_datasets()


if __name__ == "__main__":
    main()
