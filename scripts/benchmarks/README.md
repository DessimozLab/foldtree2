# Benchmark drivers and phylogenetic-information helpers

Persistent benchmark queue, cohort and supermatrix drivers plus the shared 3Di-alignment and phylogenetic-information-gain helpers they import.

[Scripts home](../README.md) | [Project](../../README.md) | [Source modules](../../foldtree2/src/README.md) | [SE3 toolkit](../se3/README.md)

These drivers measure phylogenetic-information gains across cohorts, the species
supermatrix and learned alphabets, using the shared 3Di-alignment and
phylogenetic-information-gain helpers as their backbone. The local queue keeps
a persistent priority-ordered backlog on the workstation so long benchmark
sweeps can drain unattended rather than blocking an interactive session.

## Modules
- [run_local_benchmark_queue.py](./run_local_benchmark_queue.py): persistent workstation benchmark queue; drains queued cohorts in priority order.
- [benchmark_cohort.py](./benchmark_cohort.py): benchmark a single cohort of alignments and trees.
- [compare_oma_representations.py](./compare_oma_representations.py): compare OMA-based sequence representations head-to-head.
- [benchmark_species_supermatrix.py](./benchmark_species_supermatrix.py): benchmark over the concatenated species supermatrix.
- [alphabet_information_benchmark.py](./alphabet_information_benchmark.py): information-theoretic benchmark across learned alphabets.
- [optuna_phylo_information_search.py](./optuna_phylo_information_search.py): Optuna-driven search over phylogenetic-information hyperparameters.
- [verify_benchmark_configs.py](./verify_benchmark_configs.py): validate benchmark cohort and config files before queuing.
- [three_di_ml.py](./three_di_ml.py): shared native 3Di-alignment/ML helper imported by the drivers.
- [phylogenetic_information_gain.py](./phylogenetic_information_gain.py): shared phylogenetic-information-gain CLI imported by the drivers.

## Workflow
The drivers above are the entry points; the shared helpers
(`three_di_ml.py`, `phylogenetic_information_gain.py`) are imported by them via
`scripts.benchmarks.<name>`, so prefer composing on those rather than
duplicating their alignment or accounting. `run_local_benchmark_queue.py` keeps
a persistent queue on the workstation and drains cohorts through
`benchmark_cohort.py`; larger sweeps over the species supermatrix or across
learned alphabets go through `benchmark_species_supermatrix.py` and
`alphabet_information_benchmark.py` respectively, with
`optuna_phylo_information_search.py` steering hyperparameters. Evidence from
these runs lands in [docs/experiments.md](../../docs/experiments.md) and the
dated [final benchmarks](../../docs/final_benchmarks.md) report; the queue
itself writes intermediate artifacts under `runs/` (gitignored).

## Caveats
Match the alphabet, split seed and cohort manifest against the pinned evidence
copy before quoting a number — benchmark deltas are only meaningful under a
shared provenance, and phylogenetic-information gains are sensitive to the
alignment alphabet and site filter. Configs are checked by
`verify_benchmark_configs.py` before the queue accepts a cohort; do not bypass
that gate when adding new entries, and re-run it after editing any cohort
manifest the queue has already drained.

## Status
The local queue is the maintained entry point for unattended benchmark sweeps;
the cohort and supermatrix drivers compose on the shared helpers rather than
re-implementing alignment or accounting. Evidence is pinned in `docs/` and not
regenerated in place.