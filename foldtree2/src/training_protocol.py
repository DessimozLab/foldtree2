"""Backward-compatibility shim for ``foldtree2.src.training_protocol``.

This module moved to ``foldtree2.src.training.training_protocol`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.training.training_protocol import *  # noqa: F401,F403
from foldtree2.src.training.training_protocol import (  # noqa: F401
    Subset,
    held_out_split,
    torch,
)
