"""Backward-compatibility shim for ``foldtree2.src.contact_dot_experiments``.

This module moved to ``foldtree2.src.distance.contact_dot_experiments`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.distance.contact_dot_experiments import *  # noqa: F401,F403
from foldtree2.src.distance.contact_dot_experiments import (  # noqa: F401
    F,
    PairAdjustedDotHead,
    PairwiseDistanceHead,
    SymmetricCrossDotHead,
    _FlexibleDotHead,
    copy,
    math,
    nn,
    torch,
    validate_pairs,
)
