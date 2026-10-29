"""Backward-compatibility shim for ``foldtree2.src.sidechain_refiner``.

This module moved to ``foldtree2.src.models.sidechain_refiner`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.sidechain_refiner import *  # noqa: F401,F403
from foldtree2.src.models.sidechain_refiner import (  # noqa: F401
    AA_CHI_COUNTS,
    AA_HAS_CB,
    AA_ORDER,
    AA_TO_INDEX,
    F,
    L,
    MAX_CHI,
    MAX_CHI_BY_AA,
    SidechainRefiner,
    cb_direction_loss,
    chi_sincos_loss,
    nn,
    sidechain_total_loss,
    torch,
)
