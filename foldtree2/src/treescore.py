"""Backward-compatibility shim for ``foldtree2.src.treescore``.

This module moved to ``foldtree2.src.trees.treescore`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.trees.treescore import *  # noqa: F401,F403
from foldtree2.src.trees.treescore import (  # noqa: F401
    compute_red_score,
    compute_sum_dist_to_desc_leaves,
    exp_score,
    frac_score,
    getTaxOverlap,
    getTaxOverlap_root,
    label_leaves,
    labelwRED,
    make_lineages,
    standard_treedraw,
    sum_rootscore,
    toyplot,
    toytree,
)
