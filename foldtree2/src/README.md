# Source module map

[Project](../../README.md) | [CLI guides](../../docs/cli/README.md) |
[Notebook catalogue](../notebooks/README.md) | [Experiments](../../docs/experiments.md)

## Follow the structure-to-tree path

1. [ft2treebuilder](../ft2treebuilder.py) orchestrates preprocessing, encoding,
   alignment, inference and optional rooting/ancestral analysis.
2. [pdbgraphmk2](graphs/pdbgraphmk2.py) converts PDB/Foldcomp structures to
   heterogeneous residue graphs; [config_paths](graphs/config_paths.py) resolves
   packaged properties.
3. [encoder](models/encoder.py) builds continuous embeddings and discrete code
   indices; [quantizers](models/quantizers.py) implements codebooks and training
   penalties.
4. [makesubmat](../makesubmat.py) estimates model-specific matrices from reference
   alignments. [Native wrappers](../mafft_wrappers.py) invoke packaged helpers.
5. [ancestral](trees/ancestral.py) shares native marginal reconstruction/FASTA
   export between the tree CLI and benchmark scripts.

Graph schemas, codebooks, maps and matrices form one compatibility boundary.
Equal feature widths or alphabet sizes alone do not establish compatibility.
Preserve trusted bundle hashes and exact state order. Never strip arbitrary
whitespace from structural-token FASTA.

## Subpackages

The source tree is organized into importable domain subpackages, each with its
own README describing contracts, status and entry points. Import modules from
their canonical subpackage locations (`foldtree2.src.graphs.pdbgraphmk2`,
`foldtree2.src.models.encoder`, ...); there are no top-level aliases in this
directory.

- [graphs](graphs/README.md): PDB/Foldcomp structures to PyG residue graphs;
  [pdbgraphmk2](graphs/pdbgraphmk2.py) is the current builder, [pdbgraph](graphs/pdbgraph.py)
  and [pdbgraph_polars](graphs/pdbgraph_polars.py) are legacy/reference.
- [models](models/README.md): neural encoders, codebooks and geometry decoders for
  structure-token learning; [encoder](models/encoder.py) and
  [mono_decoders](models/mono_decoders.py) are the main entry points.
- [losses](losses/README.md): geometry and reconstruction objectives, units and
  masks, plus optional upstream losses.
- [distance](distance/README.md): pairwise-distance and contact supervision built
  around [distance_restraints](distance/distance_restraints.py).
- [trees](trees/README.md): phylogenetic tree inference, scoring and
  ancestral-state reconstruction.
- [data](data/README.md): dataset acquisition and reference downloads.
- [training](training/README.md): deterministic splits, distributed training,
  optional optimizers and diagnostics.

## Documentation conventions

Every Python module in this directory and its subpackages has a purpose
docstring. Public interfaces should describe non-obvious input/output shapes,
geometry units/frame direction, masks, device behavior and side effects. Follow
existing NumPy/Google docstring style within a module. Inline comments explain
numerical/protocol constraints, not obvious assignments. Preserve upstream
copyright/license notices.

When adding a module, update the relevant subpackage README and its
notebook/CLI links. Document experimental dependencies explicitly rather than
adding them implicitly to the base inference install. See the [maintenance
checklist](../../docs/documentation_maintenance.md).
