"""Backward-compatibility shim for ``foldtree2.src.contact_bias_experiment``.

This module moved to ``foldtree2.src.distance.contact_bias_experiment`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.distance.contact_bias_experiment import *  # noqa: F401,F403
from foldtree2.src.distance.contact_bias_experiment import (  # noqa: F401
    F,
    PairwiseDistanceHead,
    ResidueBiasDistanceHead,
    math,
    nn,
    torch,
    validate_pairs,
)
