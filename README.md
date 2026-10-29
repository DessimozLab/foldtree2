<p align="center">
  <img src="https://raw.githubusercontent.com/DessimozLab/foldtree2/main/logo.png" alt="FoldTree2 Logo" width="300"/>
</p>

# FoldTree2

Infer maximum-likelihood phylogenetic trees from protein structures. FoldTree2
encodes structures into a learned discrete alphabet, aligns the tokens with
MAFFT, and estimates a tree with RAxML-NG using model-specific matrices.

```text
Your PDB structures -> residue graphs -> trained encoder -> structural tokens
                   -> MAFFT alignment -> RAxML-NG -> Newick tree
```

**Start here:** [install](docs/installation.md), obtain a
[matching model bundle](docs/user_guide.md#prerequisites), then build a tree below.
The supported release target is **Linux x86_64, glibc >=2.34**; macOS, Windows and ARM are not
supported by the bundled native executables. Model weights and datasets are
distributed separately from the Python/conda code package.

## Quick start: build a tree from structures

Use at least four usable PDB files, unique filename stems, and a new output folder.
The following illustrates an existing local 30-character bundle, not a public
download or a claim that 30 states is optimal:

```bash
foldtree2 \
  --encoder models/production/30char_minimal_decoder/final_30char_contacts_aa_encoder_full_epoch_52.pt \
  --decoder models/production/30char_minimal_decoder/final_30char_contacts_aa_decoder_full_epoch_52.pt \
  --mafftmat models/production/30char_minimal_decoder/final_30char_contacts_aa_encoder_full_epoch_52_mafftmat.mtx \
  --submat models/production/30char_minimal_decoder/final_30char_contacts_aa_encoder_full_epoch_52_submat.txt \
  --charmaps models/production/30char_minimal_decoder/final_30char_contacts_aa_encoder_full_epoch_52_pair_counts.pkl \
  --structures "/path/to/structures/*.pdb" \
  --device cpu --ncores 4 --outdir results/new_run
```

The final inferred tree is `results/new_run/*_.raxml.bestTree` in Newick format.
It is unrooted and does not automatically contain bootstrap support.
[Tree CLI guide](docs/cli/treebuilding.md) covers inputs, outputs, rooting,
ancestral inference and safe reruns.

## Install

For development, start with a CPU environment and install from the checkout:

```bash
conda create -n foldtree2-cpu -c conda-forge -c bioconda python=3.10 pip mafft
conda activate foldtree2-cpu
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
foldtree2 --help
```

See the [installation guide](docs/installation.md) for validation status, CUDA,
PyPI and conda release installation, executable checks and troubleshooting.
Registry publishing is a release gate, not an assertion that the current
development revision is already available there.

## Other inference and training

- [Encoded FASTA from Foldcomp databases](docs/cli/encoding.md), without building trees.
- [Native ancestral reconstruction](docs/experiments.md#native-ancestral-character-reconstruction);
  optional AA decoding is a distinct, experimental step.
- [Custom alphabet training and matrix estimation](docs/training.md).
- [Notebook catalogue](foldtree2/notebooks/README.md), including geometry and
  complex-model experiments that are **not** required for tree building.

## Documentation

- [Installation and troubleshooting](docs/installation.md)
- [Model bundles, inputs, tree inference, and outputs](docs/user_guide.md)
- [Custom training and matrix building](docs/training.md)
- [Running information and phylogenetic-gain experiments](docs/experiments.md)
- [Final benchmark results to compile](docs/final_benchmarks.md)
- [Notebook catalogue and prerequisites](foldtree2/notebooks/README.md)
- [Source module map and conventions](foldtree2/src/README.md)
- [CLI reference index](docs/cli/README.md)
- [Release procedure and remaining gates](docs/releasing.md)
- [Production preparation and developer release checklist](docs/production_alphabet_readiness.md)
- [Updating documentation as experiments are added or completed](docs/documentation_maintenance.md)
- [Representation conversion details](docs/representation_conversion_guide.md)
- [Staged geometry training](docs/staged_geometry_training.md)
- [SE(3) residue refiner validation and training](docs/se3_validation.md)
- [SE(3) validation results and learning gates](docs/se3_validation_report.md)
- [SE(3) Lightning rollout on Clariden](docs/se3_clariden.md)
- [Manuscript methods framework](docs/manuscript_methods_framework.md)

## Command-line tools

`foldtree2` and `ft2treebuilder` are aliases for tree inference. `pdbs-to-graphs`
creates training datasets; `makesubmat` estimates matrices. `hex2maffttext` and
`maffttext2hex` convert token representations. `raxml-ng` and `mad` wrap bundled
native tools; MAFFT itself must be installed separately. Most Python CLIs have
`--help`; native/conversion tools have their own usage conventions.

## License and contact

Project code: [MIT License](LICENSE.txt). Bundled/derived third-party components
retain their own licenses; see [release licensing gates](docs/releasing.md#licensing).
Dave Moi (<dmoi@unil.ch>).
