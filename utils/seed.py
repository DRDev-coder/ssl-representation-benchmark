"""
Seed Utilities
==============
Deterministic training support for reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
        deterministic: If True, forces fully deterministic behavior
                       (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+ deterministic algorithms
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        print("Deterministic mode enabled (may reduce performance)")
    else:
        torch.backends.cudnn.benchmark = True

    print(f"Random seed set to {seed}")
