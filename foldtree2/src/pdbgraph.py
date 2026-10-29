"""Backward-compatibility shim for ``foldtree2.src.pdbgraph``.

This module moved to ``foldtree2.src.graphs.pdbgraph`` and is re-exported here
so existing imports, tests, notebooks, docs and saved checkpoints keep
working. New code should import the canonical location.
"""
from foldtree2.src.graphs.pdbgraph import *  # noqa: F401,F403
from foldtree2.src.graphs.pdbgraph import (  # noqa: F401
    ComplexDataset,
    Data,
    Dataset,
    EPS,
    HeteroData,
    PDB,
    PDB2PyG,
    PDBParser,
    StructureDataset,
    _init_pdb2pyg_worker,
    _process_pdb_worker,
    _process_pdb_worker_queue,
    cdist,
    datadir,
    deepcopy,
    get_interaction_object,
    h5py,
    interaction_matrix_energy,
    json,
    load_structure,
    mp,
    np,
    os,
    pd,
    pebble,
    plt,
    pr,
    pydssp,
    resolve_aapropcsv_path,
    scipy,
    sparse,
    time,
    torch,
    torch_geometric,
    tqdm,
    traceback,
    warnings,
)
