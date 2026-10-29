"""Backward-compatibility shim for ``foldtree2.src.ddp_training``.

This module moved to ``foldtree2.src.training.ddp_training`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.training.ddp_training import *  # noqa: F401,F403
from foldtree2.src.training.ddp_training import (  # noqa: F401
    DDP,
    DataLoader,
    DistributedSampler,
    F,
    GradScaler,
    StructureDataset,
    aa_reconstruction_loss,
    angles_reconstruction_loss,
    autocast,
    cleanup_ddp,
    dist,
    np,
    recon_loss_diag,
    setup_ddp,
    torch,
    tqdm,
    train_distributed,
)
