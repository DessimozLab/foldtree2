"""Backward-compatibility shim for ``foldtree2.src.contact_cross_dot_perturbed``.

This module moved to ``foldtree2.src.distance.contact_cross_dot_perturbed`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.distance.contact_cross_dot_perturbed import *  # noqa: F401,F403
from foldtree2.src.distance.contact_cross_dot_perturbed import (  # noqa: F401
    PerturbedCrossDotHead,
    SymmetricCrossDotHead,
    torch,
)
