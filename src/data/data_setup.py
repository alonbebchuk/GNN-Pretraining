import os
import copy
import logging
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from tqdm import tqdm
from src.data.graph_properties import GraphPropertyCalculator
from src.common import (
    RANDOM_SEED,
    PRETRAIN_TUDATASETS,
    DOWNSTREAM_TUDATASETS,
    OVERLAP_TUDATASETS,
    ALL_TUDATASETS,
    ALL_PLANETOID_DATASETS,
    FEATURE_TYPES,
    VAL_TEST_FRACTION,
    VAL_TEST_SPLIT_RATIO,
    VAL_FRACTION,
    TUDATASET_USE_NODE_ATTR,
)
from typing import Sequence, List, Dict
from sklearn.preprocessing import StandardScaler

# --- Configuration (centralized in src.common) -------------------------------

# Define paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

# --- Helper Functions --------------------------------------------------------


def apply_feature_preprocessing(dataset: Sequence[Data], train_idx: Sequence[int], dataset_name: str) -> None:
    """Apply appropriate preprocessing based on feature type."""
    feature_type = FEATURE_TYPES.get(dataset_name)

    if feature_type == 'continuous':
        # Z-score standardization for continuous features
        train_indices = list(train_idx) if not hasattr(train_idx, 'tolist') else train_idx.tolist()

        train_X_list = [dataset[i].x.detach().cpu() for i in train_indices]
        train_X = torch.cat(train_X_list, dim=0).numpy()

        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(train_X)

        for g in dataset:
            X = g.x.detach().cpu().numpy()
            X_scaled = scaler.transform(X)
            g.x = torch.from_numpy(X_scaled).to(g.x.dtype)

        logging.info(f"Applied z-score standardization to {dataset_name}")

    elif feature_type == 'categorical':
        # Categorical features are already one-hot encoded by PyG datasets
        logging.info(f"Categorical features for {dataset_name} are already one-hot encoded by PyG")

    elif feature_type == 'bow':
        # BoW features are handled by NormalizeFeatures transform in Planetoid loading
        logging.info(f"BoW features for {dataset_name} handled by NormalizeFeatures transform")

    else:
        logging.warning(f"Unknown feature type '{feature_type}' for dataset {dataset_name}")


def attach_standardized_graph_properties(dataset: Sequence[Data], train_idx: Sequence[int], calculator: GraphPropertyCalculator) -> None:
    """Compute standardized graph properties and attach per-graph tensor at `graph_properties`."""
    props = calculator.compute_and_standardize_for_dataset(dataset, train_idx)
    for i, g in enumerate(dataset):
        g.graph_properties = props[i]


def save_processed_data(dataset_name: str, data: List[Data], splits: Dict[str, torch.Tensor]) -> None:
    """Save processed data and splits to disk."""
    save_dir = PROCESSED_DIR / dataset_name
    os.makedirs(save_dir, exist_ok=True)

    torch.save(data, save_dir / 'data.pt')
    torch.save(splits, save_dir / 'splits.pt')

    logging.info(f"Successfully processed and saved '{dataset_name}'.")


# --- Core Processing Functions -----------------------------------------------

def process_tudatasets() -> None:
    """Download, process, and split TU datasets."""
    logging.info("Processing TU datasets...")

    calculator = GraphPropertyCalculator()

    for name in tqdm(ALL_TUDATASETS, desc="Processing TU datasets"):
        # Download dataset
        logging.info(f"Downloading {name}...")
        dataset = TUDataset(root=RAW_DIR, name=name, use_node_attr=TUDATASET_USE_NODE_ATTR)
        logging.info(f"Downloaded {name}: {len(dataset)} graphs")

        # Overlap datasets: create a single canonical 80/10/10 split used by downstream,
        # then derive a pretraining-only val from within the canonical train
        if name in OVERLAP_TUDATASETS:
            canonical_ds = copy.deepcopy(dataset)
            num_graphs = len(canonical_ds)
            labels = canonical_ds.y.numpy()
            sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=VAL_TEST_FRACTION, random_state=RANDOM_SEED)
            train_idx, val_test_idx = next(sss_train_val.split(np.arange(num_graphs), labels))
            val_test_labels = labels[val_test_idx]
            sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=VAL_TEST_SPLIT_RATIO, random_state=RANDOM_SEED)
            val_idx_rel, test_idx_rel = next(sss_val_test.split(np.arange(len(val_test_idx)), val_test_labels))
            val_idx, test_idx = val_test_idx[val_idx_rel], val_test_idx[test_idx_rel]

            # Apply feature preprocessing
            apply_feature_preprocessing(canonical_ds, train_idx, name)

            # Downstream: save normalized graphs and canonical splits
            downstream_splits = {
                'train': torch.tensor(train_idx, dtype=torch.long),
                'val': torch.tensor(val_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)
            }
            save_processed_data(f"{name}_downstream", list(canonical_ds), downstream_splits)

            # Build pretraining dataset as a subset of canonical train only
            canonical_train_graphs = [copy.deepcopy(dataset[int(i)]) for i in train_idx]
            ss_ptrain = ShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
            tr_rel, va_rel = next(ss_ptrain.split(np.arange(len(canonical_train_graphs))))

            # Apply feature preprocessing
            apply_feature_preprocessing(canonical_train_graphs, tr_rel, name)

            # Attach standardized graph properties per-graph
            attach_standardized_graph_properties(canonical_train_graphs, tr_rel, calculator)

            pretrain_splits = {
                'train': torch.tensor(tr_rel, dtype=torch.long),
                'val': torch.tensor(va_rel, dtype=torch.long),
            }
            save_processed_data(f"{name}_pretrain", list(canonical_train_graphs), pretrain_splits)

        else:
            # Pre-training splits (90/10) for pure pretrain-only datasets
            if name in PRETRAIN_TUDATASETS:
                pretrain_dataset = copy.deepcopy(dataset)
                num_graphs = len(pretrain_dataset)
                ss_train_val = ShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
                train_idx, val_idx = next(ss_train_val.split(np.arange(num_graphs)))

                # Apply feature preprocessing
                apply_feature_preprocessing(pretrain_dataset, train_idx, name)

                # Attach standardized graph properties per-graph
                attach_standardized_graph_properties(pretrain_dataset, train_idx, calculator)

                pretrain_splits = {
                    'train': torch.tensor(train_idx, dtype=torch.long),
                    'val': torch.tensor(val_idx, dtype=torch.long),
                }
                save_processed_data(f"{name}_pretrain", list(pretrain_dataset), pretrain_splits)

            # Downstream splits (80/10/10) for pure downstream-only datasets
            if name in DOWNSTREAM_TUDATASETS:
                downstream_dataset = copy.deepcopy(dataset)
                num_graphs = len(downstream_dataset)
                labels = downstream_dataset.y.numpy()
                sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=VAL_TEST_FRACTION, random_state=RANDOM_SEED)
                train_idx, val_test_idx = next(sss_train_val.split(np.arange(num_graphs), labels))
                val_test_labels = labels[val_test_idx]
                sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=VAL_TEST_SPLIT_RATIO, random_state=RANDOM_SEED)
                val_idx_rel, test_idx_rel = next(sss_val_test.split(np.arange(len(val_test_idx)), val_test_labels))
                val_idx, test_idx = val_test_idx[val_idx_rel], val_test_idx[test_idx_rel]

                # Apply feature preprocessing
                apply_feature_preprocessing(downstream_dataset, train_idx, name)

                downstream_splits = {
                    'train': torch.tensor(train_idx, dtype=torch.long),
                    'val': torch.tensor(val_idx, dtype=torch.long),
                    'test': torch.tensor(test_idx, dtype=torch.long)
                }
                save_processed_data(f"{name}_downstream", list(downstream_dataset), downstream_splits)


def process_planetoid_datasets() -> None:
    """Process Planetoid datasets with row-normalization for BoW features (via NormalizeFeatures)."""
    logging.info("Processing Planetoid datasets...")

    for name in tqdm(ALL_PLANETOID_DATASETS, desc="Processing Planetoid datasets"):
        # Download dataset with NormalizeFeatures applied at load time
        logging.info(f"Downloading {name}...")
        dataset = Planetoid(root=RAW_DIR, name=name, transform=NormalizeFeatures())
        data = dataset[0]
        logging.info(f"Downloaded {name}: {data.num_nodes} nodes, {data.num_edges} edges")

        splits = {
            'train': torch.where(data.train_mask)[0],
            'val': torch.where(data.val_mask)[0],
            'test': torch.where(data.test_mask)[0]
        }
        save_processed_data(f"{name}_downstream", [data], splits)

# --- Main Execution ----------------------------------------------------------


def main() -> None:
    """Main function to run data setup process."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    logging.info(f"Raw data: {RAW_DIR}")
    logging.info(f"Processed data: {PROCESSED_DIR}")

    # Process datasets
    process_tudatasets()
    process_planetoid_datasets()
    logging.info("Data processing completed!")


if __name__ == "__main__":
    main()
