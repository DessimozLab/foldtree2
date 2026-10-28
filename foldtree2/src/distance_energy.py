"""Backward-compatibility shim for ``foldtree2.src.distance_energy``.

This module moved to ``foldtree2.src.distance.distance_energy`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.distance.distance_energy import *  # noqa: F401,F403
from foldtree2.src.distance.distance_energy import (  # noqa: F401
    F,
    distance_energy,
    fit_reference,
    interval_log_weights,
    math,
    sample_contact_strata,
    separation_groups,
    torch,
    validate_edges,
    validate_probabilities,
)
