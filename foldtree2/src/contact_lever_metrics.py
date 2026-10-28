"""Backward-compatibility shim for ``foldtree2.src.contact_lever_metrics``.

This module moved to ``foldtree2.src.distance.contact_lever_metrics`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.distance.contact_lever_metrics import *  # noqa: F401,F403
from foldtree2.src.distance.contact_lever_metrics import (  # noqa: F401
    contact_metrics,
    math,
    per_protein_range_metrics,
    summarize_ranges,
    torch,
)
