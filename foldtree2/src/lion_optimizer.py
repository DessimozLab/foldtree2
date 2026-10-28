"""Backward-compatibility shim for ``foldtree2.src.lion_optimizer``.

This module moved to ``foldtree2.src.training.lion_optimizer`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.training.lion_optimizer import *  # noqa: F401,F403
from foldtree2.src.training.lion_optimizer import (  # noqa: F401
    Lion,
    Tuple,
    torch,
)
