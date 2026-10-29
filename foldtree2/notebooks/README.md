# Notebook catalogue

[Project / build a tree](../../README.md) | [CLI guides](../../docs/cli/README.md) |
[Source modules](../src/README.md) | [Experiment protocols](../../docs/experiments.md) |
[Final results checklist](../../docs/final_benchmarks.md)

Notebooks are research entry points, **not** required for normal tree inference.
They are kept in the repository and excluded from pip/conda runtime packages.
For repeatable sweeps prefer the extracted scripts in the experiment guide.
An existing cell output is historical evidence, not proof that the current
notebook was rerun or that its full benchmark completed.

## Before running

1. Install FoldTree2 in a dedicated environment and select its Jupyter kernel.
   Install Jupyter separately; it is not an inference dependency.
2. Inspect all path/configuration cells before execution. Historical notebooks
   may contain workstation paths, downloads, long training loops or external
   tool calls. Do not "Run All" against existing production folders.
3. Provide your own PDB/Foldcomp/HDF5 datasets and trusted, schema-matched model
   bundle. Notebook paths do not identify public model downloads.
4. Use a new output directory; record source, data/bundle hashes, seeds, hardware,
   requested versus completed coverage and any smoke limits.
5. GPU training, optional geometry packages and external alignment/tree tools
   are notebook-specific. Legacy ProDy, OpenFold, FoldX and complex workflows
   do not define the supported mk2 structure-to-tree path.

## Start by task

| Notebook | Purpose and required inputs | Status / preferred route |
| --- | --- | --- |
| [Foldtree2Colab](Foldtree2Colab.ipynb) | Historical hosted walkthrough; needs structures, model files and external programs. | Not a verified one-click hosted deployment. Use the [tree CLI](../../docs/cli/treebuilding.md). |
| [monomer_graph_trainingdata](monomer_graph_trainingdata.ipynb) | Inspect graph construction from monomer structures and feature tables. | Verify converter/schema; use [pdbs-to-graphs](../../docs/cli/encoding.md) for current mk2 graphs. |
| [makesubmat](makesubmat.ipynb) | Interactive reference-alignment counting and matrix generation. | Needs reference graphs, encoder and Foldseek alignments; use [makesubmat CLI](../../docs/cli/matrices.md) for repeatable jobs. |
| [create_fams_submat](create_fams_submat.ipynb) | Historical reference-family preparation for matrices. | Inspect download paths and selection logic; not the frozen OMA evaluation cohort. |
| [show_residues](show_residues.ipynb) | Residue/structure visualization. | Needs local structures and notebook plotting dependencies; diagnostic only. |

### Specialized notebooks

- [Alphabet, geometry and conversion experiments](experiments/README.md)
- [Information, species-tree and topology benchmarks](benchmarks/README.md)
- [Complex and pairwise research](complexes/README.md)

## Interpretation and maintenance

Do not pair independently aligned columns for cross-alphabet MI. Separate
native alignment/tree analyses from controlled common-column projections,
and distinguish dataset-generation pilots from scored tree benchmarks.
Protein-held-out training evaluation is not a phylogenetic accuracy result.

When adding a notebook, add its purpose, inputs, computational side effects,
outputs and script/source links to the appropriate catalogue. Clear obsolete
outputs only when deliberately editing/rerunning that notebook. Follow the
[documentation maintenance checklist](../../docs/documentation_maintenance.md).
