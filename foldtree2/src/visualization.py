"""Backward-compatibility shim for ``foldtree2.src.visualization``.

This module moved to ``foldtree2.src.training.visualization`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.training.visualization import *  # noqa: F401,F403
from foldtree2.src.training.visualization import (  # noqa: F401
    Color,
    create_reconstruction_figure,
    np,
    plot_loss_curves,
    plot_metric_curve,
    plot_multiple_curves,
    plt,
    torch,
    visualize_batch_reconstructions,
)
