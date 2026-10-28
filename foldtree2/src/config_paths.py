"""Backward-compatibility shim for ``foldtree2.src.config_paths``.

This module moved to ``foldtree2.src.graphs.config_paths`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.graphs.config_paths import *  # noqa: F401,F403
from foldtree2.src.graphs.config_paths import (  # noqa: F401
    Optional,
    Path,
    resolve_aapropcsv_path,
)
