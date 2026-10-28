# Structure graphs

PDB and Foldcomp structures to PyTorch Geometric residue graphs.

[Source map](../README.md) | [Project](../../../README.md) |
[Encoding CLI](../../../docs/cli/encoding.md)

- [pdbgraphmk2](./pdbgraphmk2.py): current builder. `PDB2PyG`, `StructureDataset`
  and `FoldcompStructureDataset` convert structures to heterogeneous residue
  graphs with the packaged feature schema. Use this for new pipelines and
  production inference.
- [config_paths](./config_paths.py): `resolve_aapropcsv_path` resolves the
  packaged amino-acid property CSV. The path is fixed for the new subpackage
  layout; use it so the feature schema lines up across builders.
- [pdbgraph](./pdbgraph.py): original builder, legacy. Still imported by some
  `learn_*` scripts and notebooks and role-compatible with mk2, but it uses an
  older feature schema. Do not mix with mk2 in one pipeline.
- [pdbgraph_polars](./pdbgraph_polars.py): Polars variant with no active
  importers. Kept for reference; do not use for production inference.

## When to use this

Use `pdbgraphmk2` to turn PDB or Foldcomp-compressed structures into residue
graphs for training or inference. The graph feature schema is a compatibility
boundary: a model trained on one builder's schema expects that exact schema at
inference. Never swap feature schemas between `pdbgraphmk2`, `pdbgraph` and the
Polars variant for a trained model, and never mix builders in one dataset.

Entry points: the [pdbs-to-graphs](../../scripts/pdbs_to_graphs.py) CLI and
[encode_pdbs](../../encode_pdbs.py) batch-encode structures into graphs. The
modern structure-to-tree path is driven by
[ft2treebuilder](../../ft2treebuilder.py); see the
[treebuilding guide](../../../docs/cli/treebuilding.md).