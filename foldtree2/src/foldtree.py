"""Backward-compatibility shim for ``foldtree2.src.foldtree``.

This module moved to ``foldtree2.src.trees.foldtree`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.trees.foldtree import *  # noqa: F401,F403
from foldtree2.src.trees.foldtree import (  # noqa: F401
    argparse,
    corecut,
    foldseek2tree,
    np,
    os,
    pd,
    structblob2tree,
)
