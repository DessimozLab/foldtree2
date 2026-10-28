"""Backward-compatibility shim for ``foldtree2.src.manifold_hyper_connections``.

This module moved to ``foldtree2.src.models.manifold_hyper_connections`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.manifold_hyper_connections import *  # noqa: F401,F403
from foldtree2.src.models.manifold_hyper_connections import (  # noqa: F401
    F,
    ManifoldHyperConnections,
    nn,
    torch,
)
