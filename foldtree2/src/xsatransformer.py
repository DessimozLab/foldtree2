"""Backward-compatibility shim for ``foldtree2.src.xsatransformer``.

This module moved to ``foldtree2.src.models.xsatransformer`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.xsatransformer import *  # noqa: F401,F403
from foldtree2.src.models.xsatransformer import (  # noqa: F401
    Adj,
    Callable,
    ExclusiveSelfAttention,
    F,
    Linear,
    MessagePassing,
    OptTensor,
    Optional,
    PairTensor,
    SparseTensor,
    SwiGLU,
    Tensor,
    TransformerConv,
    Tuple,
    Union,
    XSATransformerConv,
    XSATransformerEncoderLayer,
    _get_activation_fn,
    copy,
    math,
    nn,
    overload,
    softmax,
    torch,
)
