# Command-line reference

[Project](../../README.md) | [Installation](../installation.md) |
[Notebooks](../../foldtree2/notebooks/README.md) |
[Source modules](../../foldtree2/src/README.md)

Install the package before using entry points. These guides describe executable
interfaces; notebook workflows may have different configuration and caches.

| Commands | Guide | Intended use |
| --- | --- | --- |
| `foldtree2`, `ft2treebuilder` | [Tree building](treebuilding.md) | PDB structures to a maximum-likelihood tree; same CLI under both names. |
| `pdbs-to-graphs`, `foldcomp2fasta` | [Graph and token encoding](encoding.md) | Training graphs or encoded sequences, not a tree. |
| `makesubmat` | [Substitution matrices](matrices.md) | Build a matching MAFFT/RAxML matrix bundle from reference alignments. |
| `tune-monodecoder-bayes`, module-based trainers | [Training](training.md) | Train/evaluate alphabets; not needed with a prepared bundle. |
| `hex2maffttext`, `maffttext2hex`, `raxml-ng`, `mad` | [Native tools](native_tools.md) | Token conversion, inference and rooting helpers. |

Most Python entry points accept `--help`; native helpers do not share argparse
conventions. Research benchmark scripts stay in the checkout and are documented
in the [experiment guide](../experiments.md), not installed as console commands.
