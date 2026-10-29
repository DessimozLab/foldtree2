# Alphabet and geometry experiments

[Notebook catalogue / prerequisites](../README.md) |
[Source module map](../../src/README.md) |
[Training](../../../docs/training.md) | [CLI index](../../../docs/cli/README.md)

All notebooks require an installed environment, reviewed path cells and separate
output directories. Most training notebooks need compatible precomputed HDF5
graphs and substantial GPU time. Names beginning with `test_` do not imply
automated or cheap unit tests.

| Notebook | Inputs, purpose and outputs | Code / protocol |
| --- | --- | --- |
| [test_monodecoders](test_monodecoders.ipynb) | Historical alphabet architecture/training and reconstruction inspection; HDF5 graphs -> checkpoints/plots. Production scripts use a corrected held-out split rather than the notebook's later loader override. | [encoder](../../src/encoder.py), [mono_decoders](../../src/mono_decoders.py), [production protocol](../../../docs/production_alphabet_readiness.md) |
| [test_monodecoders_config](test_monodecoders_config.ipynb) | Configuration-oriented version of alphabet experiments; inspect paths/config before training. | [Training CLI](../../../docs/cli/training.md) |
| [test_lightning](test_lightning.ipynb) | Lightning training/log inspection; may launch GPU work. TensorBoard logs are inputs to visual inspection, not final evaluation. | [learn_lightning](../../learn_lightning.py), [ddp_training](../../src/ddp_training.py) |
| [test_makesubmat_manual](test_makesubmat_manual.ipynb) | Encoder/reference graph/Foldseek records -> matrices, count histories and convergence plots; uses the shared production monitor. | [makesubmat](../../makesubmat.py), [matrix CLI](../../../docs/cli/matrices.md) |
| [test_fapeloss](test_fapeloss.ipynb) | Geometry representation conversion, noise robustness and loss/gradient diagnostics. | [conversion guide](../../../docs/representation_conversion_guide.md), [losses](../../src/losses/README.md) |
| [test_monodecoders_folding](<test_monodecoders _folding.ipynb>) | Geometry-focused decoder/reconstruction experiments; not a production alphabet recipe. | [mono_decoders](../../src/mono_decoders.py), [folding_refiner](../../src/folding_refiner.py) |
| [train_frozen_encoder_geometry_refiner](train_frozen_encoder_geometry_refiner.ipynb) | Frozen encoder/decoder plus compatible graphs -> staged transformer geometry-refiner checkpoints and diagnostics. | [staged geometry training](../../../docs/staged_geometry_training.md) |
| [inspect_frozen_encoder_geometry_learning](inspect_frozen_encoder_geometry_learning.ipynb) | Notebook-local geometry training and inspection; distinguish training fit from held-out reconstruction. | [SE(3) validation](../../../docs/se3_validation.md), [se3_struct_decoder](../../src/se3_struct_decoder.py) |
| [test_foldcompconversion](test_foldcompconversion.ipynb) | Foldcomp database/encoder -> streaming conversion throughput and encoded FASTA checks; can scan a large database. | [foldcomp2fasta CLI](../../../docs/cli/encoding.md), [pdbgraphmk2](../../src/pdbgraphmk2.py) |
| [test_embedding](test_embedding.ipynb) | Local model/structures -> embedding and reconstruction exploration. | [encoder](../../src/encoder.py), [visualization](../../src/visualization.py) |
| [test_fourier](test_fourier.ipynb) | Structure/distance-matrix FFT exploration and plots. | Research diagnostic, not a validated tree-inference route. |
| [test_prody](test_prody.ipynb) | Historical ProDy interaction/graph experiments. | [Legacy pdbgraph](../../src/pdbgraph.py); not mk2 CLI conversion. |
| [chaos_game](chaos_game.ipynb) | Exploratory sequence/alphabet visualization. | Inspect cell inputs and assumptions; not a benchmark completion artifact. |

Geometry notebooks may require GotenNet or other research dependencies beyond
the base install. Use the [dated experiment records](../../../docs/experiments.md#external-machines)
for pass/fail gates. A bounded learning failure must remain visible, not be
replaced with an attractive reconstruction figure.
