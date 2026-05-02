"""
Seed utilities for reproducible experiments.
"""
import random
import numpy as np
import torch

# Standard experiment seeds as specified
EXPERIMENT_SEEDS = [40, 41, 42, 43, 44]


def set_all_seeds(seed: int):
    """
    Set all random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_torch_generator(seed: int) -> torch.Generator:
    """
    Create a torch Generator with the given seed.

    Args:
        seed: Integer seed value

    Returns:
        torch.Generator with the seed set
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
