"""Backward-compatibility shim for ``foldtree2.src.mono_decoders``.

This module moved to ``foldtree2.src.models.mono_decoders`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.mono_decoders import *  # noqa: F401,F403
from foldtree2.src.models.mono_decoders import (  # noqa: F401
    AttentionPooling,
    CNN_AA_Decoder,
    CNN_geo_Decoder,
    CoarseCAStepDecoder,
    EPS,
    F,
    GATv2Conv,
    GraphNorm,
    HeteroConv,
    HeteroData,
    HeteroGAE_geo_Decoder,
    JumpingKnowledge,
    L,
    MFConv,
    ManifoldHyperConnections,
    MultiMonoDecoder,
    SAGEConv,
    StableChebConv,
    TransformerConv,
    Transformer_AA_Decoder,
    Transformer_Foldx_Decoder,
    Transformer_Geometry_Decoder,
    XSATransformerConv,
    XSATransformerEncoderLayer,
    _frame_outputs_from_rt_pred,
    _normalize_quaternion,
    _quaternion_to_rotation_matrix,
    _shared_quaternion_to_rotation_matrix,
    _sincos_logits_to_angles,
    copy,
    datadir,
    importlib,
    integrate_ca_steps,
    load_model,
    nn,
    np,
    save_model,
    scipy,
    torch,
)
