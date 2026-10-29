"""Backward-compatibility shim for ``foldtree2.src.distance_restraints``.

This module moved to ``foldtree2.src.distance.distance_restraints`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.distance.distance_restraints import *  # noqa: F401,F403
from foldtree2.src.distance.distance_restraints import (  # noqa: F401
    DISTANCE_EDGES,
    F,
    PairwiseDistanceHead,
    ResidueContextLayer,
    distance_labels,
    distribution_restraint_loss,
    eligible_pairs,
    math,
    nn,
    torch,
    validate_edges,
    validate_pairs,
)
