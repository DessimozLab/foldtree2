"""Backward-compatibility shim for ``foldtree2.src.encoder``.

This module moved to ``foldtree2.src.models.encoder`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.models.encoder import *  # noqa: F401,F403
from foldtree2.src.models.encoder import (  # noqa: F401
    AttentionPooling,
    DataLoader,
    F,
    FoldcompStructureDataset,
    GATv2Conv,
    GraphNorm,
    JumpingKnowledge,
    L,
    RESIDUE_TRACK_ORDER,
    SAGEConv,
    StructureDataset,
    TransformerConv,
    XSATransformerConv,
    _foldcomp_chunk_producer_process,
    _load_foldcomp_ids,
    copy,
    datadir,
    importlib,
    load_encoded_fasta,
    load_model,
    mk1_Encoder,
    mp,
    nn,
    np,
    os,
    pd,
    pickle,
    queue,
    residue_features_for_encoder,
    save_model,
    threading,
    torch,
    tqdm,
)
