"""Backward-compatibility shim for ``foldtree2.src.AFDB_tools``.

This module moved to ``foldtree2.src.data.AFDB_tools`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.data.AFDB_tools import *  # noqa: F401,F403
from foldtree2.src.data.AFDB_tools import (  # noqa: F401
    PDBParser,
    StringIO,
    chunk,
    descr,
    describe,
    filter_plddt,
    get_amino_acid_sequence,
    grab_entries,
    grab_struct,
    np,
    os,
    pd,
    pdb_folder_to_fasta,
    requests,
    res2fasta,
    seq1,
    sys,
    time,
    unirequest_tab,
    wget,
)
