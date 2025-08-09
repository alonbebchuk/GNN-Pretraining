"""
Utilities for strict reproducibility across Python, NumPy, PyTorch, and DataLoader workers.

This module centralizes seed handling to ensure consistent results across runs and
provides helpers for seeding DataLoader workers and creating seeded generators.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

SPLIT_SEED = 7
SEEDS = [42, 84, 126]


def set_seed(seed: int) -> None:
    """
    Set global seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Random seed to set
    """
    # Python & numpy
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch CPU/CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # PyTorch backend determinism controls
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Seed function for DataLoader workers.

    PyTorch initializes each worker with a base seed accessible via torch.initial_seed().
    We derive NumPy/Python seeds from it to ensure disjoint, deterministic RNG streams.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """
    Create a torch.Generator seeded with the given seed, or None if seed is None.
    """
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def get_split_seed() -> int:
    """
    Return the fixed global seed to use for creating dataset splits.

    This is intentionally centralized to guarantee that all preprocessing
    and split generation steps are reproducible across machines and runs.
    """
    return SPLIT_SEED


def get_run_seeds() -> list[int]:
    """
    Return the fixed list of seeds used in this project.

    We standardize on SEEDS for statistical robustness and
    comparability. Environment overrides are intentionally disabled.
    """
    return SEEDS
