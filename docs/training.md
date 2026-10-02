# Custom training and matrix building

[Back to README](../README.md)

## Prepare graphs

```bash
pdbs-to-graphs /path/to/pdb_directory /path/to/training_graphs.h5 --verbose
```

This recursively searches PDBs using the older converter. Verify populated HDF5
groups and conversion logs. Graphs, checkpoints, and inference must share feature
definitions, not just alphabet size. Retained production encoders expect 857
features; approved new 10/50 training also uses 857 features from the notebook's
precomputed `/mnt/data2/datasets/ft2_train_final.h5`. The earlier 865-feature
training attempt is superseded; its checkpoints are preserved separately.
Arbitrary graph datasets are not interchangeable.

## Train

```bash
python -m foldtree2.learn_monodecoder \
  --dataset /path/to/training_graphs.h5 \
  --model-name my_custom_model --num-embeddings 30 \
  --epochs 100 --batch-size 8 --hidden-size 100 \
  --embedding-dim 20 --device cpu --output-dir models/my_custom_model
```

This demonstrates the interface, not the production recipe or a recommended CPU
workload. Training generally benefits from a GPU. Copy
[production_alphabet.yaml](../configs/production_alphabet.yaml), change dataset
and output paths, then pass `--config /path/to/custom_config.yaml` for that recipe.
Do not modify configuration underneath a running job.

The production recipe uses a local seed-7 permutation with its first 10%
(rounded) reserved for validation, matching the notebook's split cell. Unlike
the notebook's later loader override, training never includes validation
structures. Model seed is 0. Training uses batch size 8 and 20 loader workers;
validation uses batch size 1 with no workers. AdamW weight decay is 0.01 and
the plateau scheduler monitors summed training AA loss, as in the notebook.
`--preflight-only` checks a single stored graph/model forward pass without
training or creating checkpoints. Training and selected-pair evaluation share
the exact same split implementation.

An alternative Lightning trainer supports distributed training:

```bash
python -m foldtree2.learn_lightning \
  --dataset /path/to/training_graphs.h5 \
  --model-name my_lightning_model --num-embeddings 30 \
  --epochs 100 --batch-size 8 --learning-rate 1e-4 \
  --output-dir models/my_lightning_model --clip-grad
```

Consult each trainer's `--help` for defaults, GPU flags, and resume behavior;
they do not share every option. Preserve configs, dataset versions, seeds, and
logs. Evaluate the matched best encoder/decoder on the reserved split and inspect
reconstruction accuracy/state occupancy before promotion. A checkpoint created
during training does not mean the run completed.

## Matrices from prepared references

Inputs: trained encoder, compatible reference-structure graphs, and headerless
Foldseek files at `/path/to/reference_data/struct_align/<family>/allvall.csv`.
Graph accessions must match alignment identifiers. Model name is the complete
encoder filename stem without `.pt`:

```bash
makesubmat \
  --modelname my_custom_model_best_encoder \
  --modeldir models/my_custom_model --datadir /path/to/reference_data \
  --dataset /path/to/reference_graphs.h5 --device cpu --encode_alns \
  --monitor-convergence --require-convergence --save-history --plot \
  --update-interval 25 --convergence-threshold 0.01 --convergence-patience 5
```

Outputs include `_mafftmat.mtx`, `_submat.txt`, `_pair_counts.pkl`, metrics, and
plots. Repeated state pairs are counted individually. Files are sorted/shuffled
with seed 42; every reference is used unless explicitly limited. The gate requires
five stable updates with new pairs, at least 100 files/10,000 pairs, finite scores,
and all-state coverage. Absolute Frobenius-change threshold: 0.01. First snapshots
or unchanged counts cannot establish convergence. Inspect reference quality too:
numerical convergence does not prove unbiased estimates.

On failure, `--require-convergence` exits nonzero but diagnostics/staged files
may remain. Do not treat them as accepted production matrices. Add evidence or
investigate the failure. The [production workflow](production_alphabet_readiness.md)
only promotes converged, validated matrices and preserves originals.

Optional `--download_structs`, `--convert_to_pyg`, and `--align_structs` prepare
reference data; they are unnecessary for prepared inputs and may require network,
substantial storage, and Foldseek. Use a separate workspace rather than silently
including large downloads in a basic matrix command.
