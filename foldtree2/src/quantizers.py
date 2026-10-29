"""Backward-compatibility shim for ``foldtree2.src.quantizers``.

This module moved to ``foldtree2.src.models.quantizers`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.quantizers import *  # noqa: F401,F403
from foldtree2.src.models.quantizers import (  # noqa: F401
    F,
    VectorQuantizer,
    VectorQuantizerEMA,
    cosine_anneal,
    diversity_regularization,
    diversity_regularization_soft,
    entropy_regularization,
    js_to_target,
    kl_divergence_regularization,
    kl_to_target,
    laplace_smooth,
    math,
    nn,
    np,
    pack_inorder,
    pyg_to_padded,
    second_order_entropy_rate_masked,
    soft_code_probs_from_distances,
    torch,
    usage_entropy_from_qpad,
)
