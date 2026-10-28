"""Backward-compatibility shim for ``foldtree2.src.se3_struct_decoder``.

This module moved to ``foldtree2.src.models.se3_struct_decoder`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.se3_struct_decoder import *  # noqa: F401,F403
from foldtree2.src.models.se3_struct_decoder import (  # noqa: F401
    AttentionPooling,
    F,
    GATConv,
    GCNConv,
    GotenNet,
    L,
    ManifoldHyperConnections,
    Position_MLP,
    QuaternionFoldingRefiner,
    StableGotenNet,
    StableHighDegreeNorm,
    StagedTransformerRefiner,
    TransformerConv,
    Transformer_Geometry_Decoder,
    _compute_local_frame_from_ca,
    _frame_outputs_from_coords,
    _frame_outputs_from_rt_pred,
    _gotennet_backend,
    _safe_gotennet_spherical_harmonics,
    _sincos_logits_to_angles,
    enable_stable_tensor_norm,
    enable_tensor_ff_input_norm,
    equivariant_ca_frame_rotmat,
    global_mean_pool,
    importlib,
    math,
    nn,
    rotation_matrix_to_quaternion,
    se3_denoiser,
    struct_transformer_decoder,
    torch,
    types,
)
