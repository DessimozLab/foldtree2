<p align="center">
  <img src="logo.png" alt="FoldTree2 Logo" width="300"/>
</p>

# FoldTree2

Infer maximum-likelihood phylogenetic trees from protein structures. FoldTree2
encodes structures into a learned discrete alphabet, aligns the tokens with
MAFFT, and estimates a tree with RAxML-NG using model-specific matrices.

## Install

From a checkout of this repository, start with a CPU environment:

```bash
conda create -n foldtree2-cpu -c conda-forge -c bioconda python=3.10 pip mafft
conda activate foldtree2-cpu
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
foldtree2 --help
```

See the [installation guide](docs/installation.md) for validation status, CUDA,
executable checks, and troubleshooting. Models and benchmark datasets are not
installed by pip. Bundled native tools have been used on Linux; other platforms
are not validated here. A Linux C++ runtime workaround was needed in the fresh
installation check; see the guide if `foldtree2 --help` raises an ABI import error.

## Build a tree with a local model bundle

Obtain a trusted, matching bundle following the [user guide](docs/user_guide.md).
The existing 30-character bundle is an example, not a claim that 30 is optimal.
Use at least four usable PDB files, unique filename stems, and a new output folder.

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
It is unrooted and does not automatically contain bootstrap support. See the
[user guide](docs/user_guide.md) for input handling, outputs, optional rooting
and reconstruction, and rerun limitations.

## Documentation

- [Installation and troubleshooting](docs/installation.md)
- [Model bundles, inputs, tree inference, and outputs](docs/user_guide.md)
- [Custom training and matrix building](docs/training.md)
- [Running information and phylogenetic-gain experiments](docs/experiments.md)
- [Production preparation and developer release checklist](docs/production_alphabet_readiness.md)
- [Updating documentation as experiments are added or completed](docs/documentation_maintenance.md)
- [Representation conversion details](docs/representation_conversion_guide.md)
- [Staged geometry training](docs/staged_geometry_training.md)
- [Manuscript methods framework](docs/manuscript_methods_framework.md)

## Command-line tools

`foldtree2` and `ft2treebuilder` are aliases for tree inference. `pdbs-to-graphs`
creates training datasets; `makesubmat` estimates matrices. `hex2maffttext` and
`maffttext2hex` convert token representations. `raxml-ng` and `mad` wrap bundled
native tools; MAFFT itself must be installed separately. Most Python CLIs have
`--help`; native/conversion tools have their own usage conventions.

## License and contact

MIT License: [LICENSE.txt](LICENSE.txt). Dave Moi (<dmoi@unil.ch>).
