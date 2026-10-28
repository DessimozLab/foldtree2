"""Backward-compatibility shim for ``foldtree2.src.download_utils``.

This module moved to ``foldtree2.src.data.download_utils`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.data.download_utils import *  # noqa: F401,F403
from foldtree2.src.data.download_utils import (  # noqa: F401
    List,
    MODEL_VERSION_CANDIDATES,
    Optional,
    Tuple,
    _candidate_urls,
    download_pdb_requests,
    download_pdb_subprocess,
    download_structures,
    mp,
    np,
    os,
    pd,
    subprocess,
    tqdm,
    verify_downloads,
)
