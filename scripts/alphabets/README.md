# Learned-alphabet experiments and substitution-matrix references

Production-alphabet experiments, checkpoint evaluation and substitution-matrix reference management for the 3Di-phylo alphabet program.

[Scripts home](../README.md) | [Project](../../README.md) | [Source modules](../../foldtree2/src/README.md) | [SE3 toolkit](../se3/README.md)

The 3Di-phylo alphabet program learns a structural alphabet whose substitution
matrix the phylogenetic pipeline uses for distance scoring. This package sweeps
encoder/decoder pairs over reserved splits, evaluates checkpoints, and manages
the substitution-matrix references the downstream pipeline reads from the
configuration directory — keeping trained and downloaded matrices in sync with
the alphabet version the benchmarks actually compare against.

## Modules
- [production_alphabet_experiments.py](./production_alphabet_experiments.py): production-alphabet experiment driver; sweeps encoder/decoder pairs over reserved splits.
- [prepare_production_alphabets.py](./prepare_production_alphabets.py): shared preparation helper; invokes `evaluate_alphabet_checkpoint.py` on reserved splits.
- [evaluate_alphabet_checkpoint.py](./evaluate_alphabet_checkpoint.py): evaluate an encoder/decoder pair on a reserved split; called by `prepare_production_alphabets.py`.
- [rebuild_downloaded_matrix_references.py](./rebuild_downloaded_matrix_references.py): rebuild downloaded substitution-matrix references.
- [extend_matrix_references.py](./extend_matrix_references.py): shared substitution-matrix reference extension.
- [promote_nonconverged_matrix.py](./promote_nonconverged_matrix.py): promote a non-converged matrix checkpoint as a reference baseline.
- [alphabet_information_metrics.py](./alphabet_information_metrics.py): shared alphabet information metrics imported by the drivers.

## Workflow
The production readiness track is documented in
[production_alphabet_readiness.md](../../docs/production_alphabet_readiness.md),
with the experiment log in [docs/experiments.md](../../docs/experiments.md).
`production_alphabet_experiments.py` drives the sweep;
`prepare_production_alphabets.py` prepares the run and evaluates checkpoints via
`evaluate_alphabet_checkpoint.py` on the reserved split. Matrix references land
under [3diphy_submats](../../foldtree2/config/3diphy_submats), the configuration
directory the downstream phylogenetic pipeline reads;
`rebuild_downloaded_matrix_references.py` and `extend_matrix_references.py` keep
that directory in sync with the latest trained and downloaded matrices.

## Caveats
The shared helpers (`prepare_production_alphabets.py`,
`extend_matrix_references.py`, `alphabet_information_metrics.py`) are imported
by the drivers via `scripts.alphabets.<name>`; compose on them rather than
re-deriving the metrics or matrix IO. Pin the alphabet version and split seed in
provenance before comparing information gains across runs — the 3Di-phylo
information metric is sensitive to the encoder checkpoint, so an unpinned
comparison conflates alphabet and checkpoint effects.

## Status
The production-alphabet sweep is the maintained experiment driver; the matrix
reference helpers keep `3diphy_submats` in sync. A promoted non-converged matrix
is a labelled baseline, not a production release — the readiness doc tracks
which matrices are production and which are baselines. Rebuild a matrix
reference from the exact encoder checkpoint that produced it; a mismatched
checkpoint silently drifts the substitution scores the phylogenetic pipeline
reads.