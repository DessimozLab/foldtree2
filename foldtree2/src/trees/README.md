# Phylogenetic trees

Tree inference, scoring and ancestral-state reconstruction from structure tokens.

[Source map](../README.md) | [Project](../../../README.md) |
[Treebuilding guide](../../../docs/cli/treebuilding.md)

- [foldseek2tree](./foldseek2tree.py): historical FoldTree pipeline — foldseek
  tokens to tree. Kept for reproducibility of older runs; not the learned-alphabet
  path.
- [foldtree](./foldtree.py): historical. Known broken: it imports a missing
  `corecut` module. Do not use; kept for reference only.
- [treescore](./treescore.py): taxonomy/tree drawing and scoring. Used by the TCS
  benchmark; requires frozen taxa/taxonomy. See [TCS benchmark](../../../docs/curnagl_tcs.md).
- [ancestral](./ancestral.py): ancestral-state reconstruction and FASTA export;
  shared between the treebuilding CLI (`ft2treebuilder`) and the benchmark scripts.

## When to use this

The modern structure-to-tree path runs through the
[foldtree2 CLI](../../ft2treebuilder.py), which encodes structures, aligns, builds
the tree and optionally roots and reconstructs ancestral states via `ancestral`;
see the [treebuilding guide](../../../docs/cli/treebuilding.md). `foldseek2tree`
is the historical foldseek-token path and `foldtree` is broken (missing `corecut`)
— do not run either for new work. `treescore` backs the TCS benchmark
([curnagl_tcs](../../../docs/curnagl_tcs.md)) and expects a frozen taxonomy; record
the taxa set and scoring protocol in benchmark provenance.