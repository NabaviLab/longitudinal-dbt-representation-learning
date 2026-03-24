"""Reproducibility helpers (random seeds and determinism flags)."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """Set seeds for common RNG sources used in ML experiments.

    This function attempts to seed Python's `random`, NumPy (if installed), and PyTorch
    (if installed).

    Args:
        seed: Seed value.
        deterministic_torch: If True and PyTorch is installed, sets common flags to
            encourage deterministic behavior (may reduce performance).

    Returns:
        None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
