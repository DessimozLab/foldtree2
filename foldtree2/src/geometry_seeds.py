"""Backward-compatibility shim for ``foldtree2.src.geometry_seeds``.

This module moved to ``foldtree2.src.models.geometry_seeds`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.geometry_seeds import *  # noqa: F401,F403
from foldtree2.src.models.geometry_seeds import (  # noqa: F401
    equivariant_ca_frame_rotmat,
    lift_seed_with_local_normals,
    math,
    torch,
)
