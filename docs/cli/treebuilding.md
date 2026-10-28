# Tree building: foldtree2 / ft2treebuilder

[CLI index](README.md) | [Project quick start](../../README.md#quick-start-build-a-tree-from-structures) |
[User guide](../user_guide.md) | [Implementation](../../foldtree2/ft2treebuilder.py) |
[Source map](../../foldtree2/src/README.md)

The two names invoke the same pipeline:
PDB -> mk2 graph -> encoder tokens -> matrix-aware MAFFT -> multistate RAxML-NG.
Foldseek is not required when using prepared matrices.

```bash
foldtree2 --help
foldtree2 --encoder /path/to/bundle/encoder.pt \
  --decoder /path/to/bundle/decoder.pt \
  --mafftmat /path/to/bundle/mafftmat.mtx \
  --submat /path/to/bundle/submat.txt \
  --charmaps /path/to/bundle/pair_counts.pkl \
  --structures "/path/to/structures/*.pdb" \
  --device cpu --ncores 4 --outdir results/new_tree
```

Paths above are placeholders for five matching, trusted artifacts. Checkpoints
and pickle maps can execute Python during loading: never load untrusted bundles.
Do not mix codebooks, decoders or matrix generations, even with equal state counts.
There is no automatic model download in this CLI.

Inputs need at least four usable structures, one selected protein/chain per
file, complete backbone coordinates where possible and unique simple filename
stems. Quote the glob so the CLI receives one pattern. Pre-extract desired chains
from multichain structures. Use `cuda:0` only after checking CUDA visibility.
MAFFT/RAxML remain CPU programs.

The Newick tree is `*_.raxml.bestTree`; the encoded/aligned FASTAs and RAxML
logs/models are retained in the output directory. Default inference is unrooted
and has no bootstrap support. `--bs` requests bootstrap analysis; `--root` requests
MAD rooting; `--ancestral` requests experimental ancestral processing. Inspect
logs and requested outputs rather than treating an existing file as completion.

Always use a new output directory for changed inputs or bundles. `--redo`
controls RAxML reuse, not full-pipeline cache invalidation, and the CLI's
`--overwrite` is not consistently forwarded. This limitation remains a release
consideration, not a promise of safe overwrite behavior.

For interactive and benchmark examples, see the
[notebook catalogue](../../foldtree2/notebooks/README.md).
