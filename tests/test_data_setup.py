import pytest
import torch
from pathlib import Path

# --- Test Configuration -----------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

PRETRAIN_TUDATASETS = ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES']
DOWNSTREAM_TUDATASETS = ['ENZYMES', 'FRANKENSTEIN', 'PTC_MR']
PLANETOID_DATASETS = ['Cora', 'CiteSeer']

ALL_TUDATASETS = sorted(list(set(PRETRAIN_TUDATASETS + DOWNSTREAM_TUDATASETS)))
ALL_PLANETOID_DATASETS = PLANETOID_DATASETS

# Generate a list of all expected processed dataset names for parametrization
PROCESSED_DATASET_NAMES = []
for name in ALL_TUDATASETS:
    if name in PRETRAIN_TUDATASETS:
        PROCESSED_DATASET_NAMES.append(f"{name}_pretrain")
    if name in DOWNSTREAM_TUDATASETS:
        PROCESSED_DATASET_NAMES.append(f"{name}_downstream")
for name in ALL_PLANETOID_DATASETS:
    PROCESSED_DATASET_NAMES.append(f"{name}_downstream")


# --- Basic Setup Tests ------------------------------------------------------

def test_directories_exist():
    """Tests if the fundamental data directories were created."""
    assert RAW_DIR.exists(), "The 'data/raw' directory does not exist."
    assert PROCESSED_DIR.exists(), "The 'data/processed' directory does not exist."

def test_raw_data_downloaded():
    """
    Tests if raw data folders were created for each dataset.
    This is a proxy for successful download by torch_geometric.
    """
    all_datasets = ALL_TUDATASETS + ALL_PLANETOID_DATASETS
    for name in all_datasets:
        # TUDataset creates a folder with the dataset name.
        # Planetoid creates a folder and then a 'raw' subfolder.
        raw_path = RAW_DIR / name
        assert raw_path.exists(), f"Raw data directory not found for '{name}'"


# --- Parametrized Tests for Processed Data -----------------------------------
# These tests run once for every single processed dataset version.

@pytest.mark.parametrize("dataset_name", PROCESSED_DATASET_NAMES)
def test_processed_files_exist(dataset_name):
    """Tests that each processed dataset has all three required files."""
    processed_path = PROCESSED_DIR / dataset_name
    assert processed_path.exists(), f"Processed directory not found for '{dataset_name}'"
    
    data_file = processed_path / 'data.pt'
    splits_file = processed_path / 'splits.pt'
    scaler_file = processed_path / 'scaler.pt'
    
    assert data_file.exists(), f"'data.pt' is missing for '{dataset_name}'"
    assert splits_file.exists(), f"'splits.pt' is missing for '{dataset_name}'"
    assert scaler_file.exists(), f"'scaler.pt' is missing for '{dataset_name}'"

@pytest.mark.parametrize("dataset_name", PROCESSED_DATASET_NAMES)
def test_processed_content_is_valid(dataset_name):
    """
    Performs a deep check on the content of the processed files to ensure
    they are loadable and have the correct structure and consistency.
    """
    processed_path = PROCESSED_DIR / dataset_name
    
    # --- 1. Load Data ---
    try:
        data = torch.load(processed_path / 'data.pt')
        splits = torch.load(processed_path / 'splits.pt')
        scaler = torch.load(processed_path / 'scaler.pt')
    except Exception as e:
        pytest.fail(f"Failed to load .pt files for '{dataset_name}'. Error: {e}")

    # --- 2. Validate Data and Splits Consistency ---
    is_tu_dataset = isinstance(data, list)
    
    if is_tu_dataset:
        # Case: TUDataset (list of Data objects)
        num_graphs = len(data)
        assert num_graphs > 0, f"Data list for '{dataset_name}' is empty."
        assert 'train' in splits, f"Missing 'train' split for '{dataset_name}'"
        assert 'val' in splits, f"Missing 'val' split for '{dataset_name}'"
        
        # Check that splits are non-empty and indices are valid
        all_indices = []
        for split_name, indices in splits.items():
            assert len(indices) > 0, f"Split '{split_name}' is empty for '{dataset_name}'"
            assert indices.max() < num_graphs, f"Split index out of bounds for '{dataset_name}'"
            all_indices.append(indices)
        
        # For downstream tasks, check for test split and full coverage
        if "downstream" in dataset_name:
            assert 'test' in splits, f"Missing 'test' split for downstream '{dataset_name}'"
            # Check that all graphs are covered by a split
            assert len(torch.unique(torch.cat(all_indices))) == num_graphs

    else:
        # Case: Planetoid (single Data object)
        num_nodes = data.num_nodes
        assert num_nodes > 0, f"Data object for '{dataset_name}' has no nodes."
        assert hasattr(data, 'x') and data.x is not None, f"Node features 'x' missing for '{dataset_name}'"
        
        for split_name, mask in splits.items():
            assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool, f"Split '{split_name}' is not a boolean tensor for '{dataset_name}'"
            assert len(mask) == num_nodes, f"Mask length for '{split_name}' does not match num_nodes for '{dataset_name}'"
            assert mask.sum() > 0, f"Split mask '{split_name}' is empty for '{dataset_name}'"

    # --- 3. Validate Scaler ---
    is_continuous = "ENZYMES_downstream" in dataset_name # As per the plan
    if is_continuous:
        assert isinstance(scaler, dict), f"Scaler for '{dataset_name}' should be a dict, but is not."
        assert 'mean' in scaler and 'std' in scaler, f"Scaler for '{dataset_name}' is missing 'mean' or 'std' keys."
        assert isinstance(scaler['mean'], torch.Tensor)
        assert isinstance(scaler['std'], torch.Tensor)
    else:
        assert scaler is None, f"Scaler for non-continuous dataset '{dataset_name}' should be None."