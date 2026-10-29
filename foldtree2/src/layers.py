"""Backward-compatibility shim for ``foldtree2.src.layers``.

This module moved to ``foldtree2.src.models.layers`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.layers import *  # noqa: F401,F403
from foldtree2.src.models.layers import (  # noqa: F401
    Learnable_Positional_Encoding,
    Position_MLP,
    SIRENLayer,
    SwiGLU,
    nn,
    torch,
)
