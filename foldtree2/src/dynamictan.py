"""Backward-compatibility shim for ``foldtree2.src.dynamictan``.

This module moved to ``foldtree2.src.models.dynamictan`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.dynamictan import *  # noqa: F401,F403
from foldtree2.src.models.dynamictan import (  # noqa: F401
    DynamicTanh,
    F,
    convert_ln_to_dyt,
    nn,
    torch,
)
