"""Backward-compatibility shim for ``foldtree2.src.folding_refiner``.

This module moved to ``foldtree2.src.models.folding_refiner`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.folding_refiner import *  # noqa: F401,F403
from foldtree2.src.models.folding_refiner import (  # noqa: F401
    L,
    QuaternionFoldingRefiner,
    torch,
)
