"""Backward-compatibility shim for ``foldtree2.src.contact_levers``.

This module moved to ``foldtree2.src.distance.contact_levers`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.distance.contact_levers import *  # noqa: F401,F403
from foldtree2.src.distance.contact_levers import (  # noqa: F401
    ExclusiveSelfAttention,
    F,
    InspectablePairHead,
    PairwiseDistanceHead,
    RelativeBiasXSA,
    XSAPairHead,
    math,
    nn,
    torch,
    validate_pairs,
)
