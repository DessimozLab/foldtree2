"""Backward-compatibility shim for ``foldtree2.src.decoder_feature_fusion``.

This module moved to ``foldtree2.src.models.decoder_feature_fusion`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.decoder_feature_fusion import *  # noqa: F401,F403
from foldtree2.src.models.decoder_feature_fusion import (  # noqa: F401
    DecoderFeatureFusion,
    DecoderFusionRefiner,
    MODES,
    build_model,
    checked_features,
    enable_tensor_ff_input_norm,
    forward,
    nn,
    soft_contact_context,
    torch,
)
