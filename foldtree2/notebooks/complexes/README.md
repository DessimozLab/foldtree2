# Complex and pairwise research

[Notebook catalogue / prerequisites](../README.md) |
[Source map](../../src/README.md) | [pairwise models](../../src/models/pairwise.py) |
[Supported monomer tree CLI](../../../docs/cli/treebuilding.md)

These historical exploratory notebooks are not required for building a tree.
They have sparse inline introductions and have not been rerun as release tests.
Review every path/cell before execution and supply your own complex structures,
interaction labels and any optional FoldX installation/data.

| Notebook | Scope | Expected artifacts / caveats |
| --- | --- | --- |
| [complex_graph_trainingdata](complex_graph_trainingdata.ipynb) | Complex graph/training-data exploration. | Graph datasets; verify chain/pair identities and feature layout before training. |
| [pairwise_foldx](pairwise_foldx.ipynb) | FoldX-related pairwise interaction experiments. | Pairwise features/labels and analysis; optional licensed external tool is not supplied. |
| [simple_pairwise](simple_pairwise.ipynb) | Simpler pairwise graph/model exploration. | Training/diagnostic outputs; not a validated phylogenetic benchmark. |

Record paired-sample splits to avoid protein/family leakage. Do not imply that
complex graph schemas or pairwise checkpoints can be substituted into the
monomer encoder CLI without explicit compatibility checks.
