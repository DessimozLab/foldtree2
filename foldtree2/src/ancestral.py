"""Backward-compatibility shim for ``foldtree2.src.ancestral``.

This module moved to ``foldtree2.src.trees.ancestral`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.trees.ancestral import *  # noqa: F401,F403
from foldtree2.src.trees.ancestral import (  # noqa: F401
    Path,
    ancestral_command,
    ancestral_states_to_fasta,
    run_ancestral,
    subprocess,
)
