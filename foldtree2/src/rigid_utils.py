"""Backward-compatibility shim for ``foldtree2.src.rigid_utils``.

This module moved to ``foldtree2.src.models.rigid_utils`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.rigid_utils import *  # noqa: F401,F403
from foldtree2.src.models.rigid_utils import (  # noqa: F401
    Any,
    Callable,
    Optional,
    Rigid,
    Rotation,
    Sequence,
    Tuple,
    _get_quat,
    _to_mat,
    annotations,
    identity_quats,
    identity_rot_mats,
    identity_trans,
    invert_quat,
    invert_rot_mat,
    lru_cache,
    np,
    quat_multiply,
    quat_multiply_by_vec,
    quat_to_rot,
    rot_matmul,
    rot_to_quat,
    rot_vec_mul,
    torch,
)
