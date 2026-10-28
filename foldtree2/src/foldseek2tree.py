"""Backward-compatibility shim for ``foldtree2.src.foldseek2tree``.

This module moved to ``foldtree2.src.trees.foldseek2tree`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.trees.foldseek2tree import *  # noqa: F401,F403
from foldtree2.src.trees.foldseek2tree import (  # noqa: F401
    MDS_smooth,
    Tajima_dist,
    cdist,
    consensustree,
    datadir,
    distmat_to_txt,
    kernelfun,
    np,
    os,
    pd,
    postprocess,
    read_dbfiles3di,
    runFastme,
    runFoldseek_allvall,
    runFoldseek_allvall_EZsearch,
    runFoldseekdb,
    runQuicktree,
    runargs,
    shlex,
    structblob2tree,
    subprocess,
    toytree,
)
