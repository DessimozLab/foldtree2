"""Backward-compatibility shim for ``foldtree2.src.chebconv``.

This module moved to ``foldtree2.src.models.chebconv`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.chebconv import *  # noqa: F401,F403
from foldtree2.src.models.chebconv import (  # noqa: F401
    F,
    MessagePassing,
    OptPairTensor,
    Parameter,
    StableChebConv,
    Tensor,
    Union,
    add_self_loops,
    get_laplacian,
    nn,
    remove_self_loops,
    torch,
)
