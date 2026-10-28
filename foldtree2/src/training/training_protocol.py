"""Shared, deterministic protein split matching test_monodecoders.ipynb."""

import torch
from torch.utils.data import Subset


def held_out_split(dataset, fraction=0.1, seed=7):
    """Reserve the beginning of a locally seeded permutation for validation."""
    if len(dataset) < 2 or not 0 < fraction < 1:
        raise ValueError('Need at least two structures and a split fraction in (0, 1)')
    count = min(max(1, int(round(len(dataset) * fraction))), len(dataset) - 1)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    return Subset(dataset, indices[count:]), Subset(dataset, indices[:count])
