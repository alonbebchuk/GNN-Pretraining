import os
import copy
import logging
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from torch_geometric.datasets import Planetoid, TUDataset
from tqdm import tqdm
from src.pretraining.graph_properties import GraphPropertyCalculator

# --- Configuration -----------------------------------------------------------

SEED = 0

PRETRAIN_TUDATASETS = ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES']
DOWNSTREAM_TUDATASETS = ['ENZYMES', 'FRANKENSTEIN', 'PTC_MR']
ALL_TUDATASETS = sorted(list(set(PRETRAIN_TUDATASETS + DOWNSTREAM_TUDATASETS)))
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

# Define paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

# --- Helper Functions --------------------------------------------------------


def apply_preprocessing(dataset, train_idx, dataset_name):
    """Apply preprocessing based on feature type."""
    feature_type = FEATURE_TYPES[dataset_name]

    if feature_type == 'continuous':
        # Z-score normalization for continuous features
        if isinstance(train_idx, torch.Tensor):
            train_idx = train_idx.tolist()
        train_graphs = [dataset[i] for i in train_idx]
        all_train_x = torch.cat([g.x for g in train_graphs if g.x is not None], dim=0)

        # Compute statistics only from training set
        mean = all_train_x.mean(dim=0)
        std = all_train_x.std(dim=0, unbiased=True)
        std[std < 1e-8] = 1.0

        # Apply normalization to all graphs using training statistics
        for g in dataset:
            g.x = (g.x - mean) / std

        logging.info(f"Applied z-score normalization to {dataset_name}")

    elif feature_type == 'bow':
        # Row-normalize Bag-of-Words features
        for g in dataset:
            row_sum = g.x.sum(dim=1, keepdim=True)
            row_sum[row_sum == 0] = 1
            g.x = g.x / row_sum

        logging.info(f"Applied row normalization to {dataset_name}")


def attach_standardized_graph_properties(dataset, train_idx, dataset_name):
    """Compute and attach standardized 15-D graph property vectors for pretraining datasets only."""
    if isinstance(train_idx, torch.Tensor):
        train_idx = train_idx.tolist()

    calculator = GraphPropertyCalculator()

    # Collect properties for training graphs
    train_props_list = []
    for idx in train_idx:
        props = calculator(dataset[idx]).to(torch.float32)
        train_props_list.append(props)

    train_props = torch.stack(train_props_list, dim=0)  # [N_train, 15]
    prop_mean = train_props.mean(dim=0)
    prop_std = train_props.std(dim=0, unbiased=True)
    prop_std[prop_std < 1e-8] = 1.0

    # Attach standardized properties to all graphs using train stats
    for g in dataset:
        props = calculator(g).to(torch.float32)
        g.graph_properties = (props - prop_mean) / prop_std

    logging.info(f"Computed and standardized graph properties for {dataset_name}")


def save_processed_data(dataset_name, data, splits):
    """Save processed data and splits to disk."""
    save_dir = PROCESSED_DIR / dataset_name
    os.makedirs(save_dir, exist_ok=True)

    torch.save(data, save_dir / 'data.pt')
    torch.save(splits, save_dir / 'splits.pt')

    logging.info(f"Successfully processed and saved '{dataset_name}'.")


# --- Core Processing Functions -----------------------------------------------

def process_tudatasets():
    """Download, process, and split TU datasets."""
    logging.info("Processing TU datasets...")

    for name in tqdm(ALL_TUDATASETS, desc="Processing TU datasets"):
        # Download dataset
        logging.info(f"Downloading {name}...")
        dataset = TUDataset(root=RAW_DIR, name=name, use_node_attr=True)
        logging.info(f"Downloaded {name}: {len(dataset)} graphs")

        # Pre-training splits (80/20)
        if name in PRETRAIN_TUDATASETS:
            pretrain_dataset = copy.deepcopy(dataset)
            num_graphs = len(pretrain_dataset)
            ss_train_val = ShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
            train_idx, val_test_idx = next(ss_train_val.split(np.arange(num_graphs)))
            ss_val_test = ShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
            val_idx_rel, test_idx_rel = next(ss_val_test.split(np.arange(len(val_test_idx))))
            val_idx, test_idx = val_test_idx[val_idx_rel], val_test_idx[test_idx_rel]

            # Apply preprocessing
            apply_preprocessing(pretrain_dataset, train_idx, name)

            # Attach standardized graph properties
            attach_standardized_graph_properties(pretrain_dataset, train_idx, name)

            pretrain_splits = {
                'train': torch.tensor(train_idx, dtype=torch.long),
                'val': torch.tensor(val_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)
            }
            save_processed_data(f"{name}_pretrain", list(pretrain_dataset), pretrain_splits)

        # Downstream splits (80/10/10)
        if name in DOWNSTREAM_TUDATASETS:
            downstream_dataset = copy.deepcopy(dataset)
            num_graphs = len(downstream_dataset)
            labels = downstream_dataset.y.numpy()
            sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
            train_idx, val_test_idx = next(sss_train_val.split(np.arange(num_graphs), labels))
            val_test_labels = labels[val_test_idx]
            sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
            val_idx_rel, test_idx_rel = next(sss_val_test.split(np.arange(len(val_test_idx)), val_test_labels))
            val_idx, test_idx = val_test_idx[val_idx_rel], val_test_idx[test_idx_rel]

            # Apply preprocessing
            apply_preprocessing(downstream_dataset, train_idx, name)

            downstream_splits = {
                'train': torch.tensor(train_idx, dtype=torch.long),
                'val': torch.tensor(val_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)
            }
            save_processed_data(f"{name}_downstream", list(downstream_dataset), downstream_splits)


def process_planetoid_datasets():
    """Process Planetoid datasets with row-normalization for BoW features."""
    logging.info("Processing Planetoid datasets...")

    for name in tqdm(ALL_PLANETOID_DATASETS, desc="Processing Planetoid datasets"):
        # Download dataset
        logging.info(f"Downloading {name}...")
        dataset = Planetoid(root=RAW_DIR, name=name)
        data = dataset[0]
        logging.info(f"Downloaded {name}: {data.num_nodes} nodes, {data.num_edges} edges")

        train_idx = torch.where(data.train_mask)[0]

        # Apply preprocessing
        apply_preprocessing([data], train_idx, name)

        splits = {
            'train': torch.where(data.train_mask)[0],
            'val': torch.where(data.val_mask)[0],
            'test': torch.where(data.test_mask)[0]
        }
        save_processed_data(f"{name}_downstream", [data], splits)

# --- Main Execution ----------------------------------------------------------


def main():
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
