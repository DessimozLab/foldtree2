"""Backward-compatibility shim for ``foldtree2.src.test_chebconv_rework``.

This module moved to ``foldtree2.src.models.test_chebconv_rework`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.test_chebconv_rework import *  # noqa: F401,F403
from foldtree2.src.models.test_chebconv_rework import (  # noqa: F401
    Data,
    StableChebConv,
    nn,
    os,
    sys,
    test_lazy_initialization,
    torch,
)
