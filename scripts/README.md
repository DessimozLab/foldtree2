# Scripts — experiment drivers, benchmarks and HPC ops

[Source map](../foldtree2/src/README.md) | [SE3 package](se3/README.md) | [Project](../README.md)

The flat `scripts/` directory holds experiment drivers, benchmarks, HPC ops
and release tooling run directly with `python scripts/<file>.py` against an
editable install. SE(3) validation and investigation work lives in the
[se3/](se3/README.md) package; this file covers the flat scripts only.

## Keep — current workflow

### Contact and distance experiments
- [train_contact_levers.py](./train_contact_levers.py): live contact training driver, launched by `queue_contact_*.sh`.
- [experiment_contact_objectives.py](./experiment_contact_objectives.py): sweep contact objective choices.
- [experiment_contact_dot_variants.py](./experiment_contact_dot_variants.py): sweep dot-product contact variants.
- [experiment_contact_residue_bias.py](./experiment_contact_residue_bias.py): sweep residue-level contact bias.
- [experiment_contact_cross_perturbed.py](./experiment_contact_cross_perturbed.py): cross-perturbed contact experiments.
- [calibrate_contact_bias.py](./calibrate_contact_bias.py): calibrate contact bias on a screen.
- [compare_spherical_contact_screen.py](./compare_spherical_contact_screen.py): compare spherical contact screens.
- [export_contact_operating_point.py](./export_contact_operating_point.py): export a chosen operating point. One screen→calibrate→export pipeline.

### Benchmarks
- [run_local_benchmark_queue.py](./run_local_benchmark_queue.py): run the local benchmark queue.
- [compare_oma_representations.py](./compare_oma_representations.py): compare OMA representations.
- [benchmark_species_supermatrix.py](./benchmark_species_supermatrix.py): benchmark over the species supermatrix.
- [alphabet_information_benchmark.py](./alphabet_information_benchmark.py): alphabet information benchmark.
- [phylogenetic_information_gain.py](./phylogenetic_information_gain.py): shared phylogenetic information gain helper.
- [optuna_phylo_information_search.py](./optuna_phylo_information_search.py): Optuna search over phylogenetic information.
- [benchmark_cohort.py](./benchmark_cohort.py): benchmark over a cohort.
- [three_di_ml.py](./three_di_ml.py): shared 3D-ML helper.

### Curnagl / TCS ops
- [curnagl_tcs.py](./curnagl_tcs.py): base Curnagl TCS driver.
- [supervise_curnagl_tcs.py](./supervise_curnagl_tcs.py): supervise a TCS campaign.
- [advance_curnagl_tcs.py](./advance_curnagl_tcs.py): advance a TCS run.
- [check_curnagl_tcs.py](./check_curnagl_tcs.py): check TCS run status.
- [check_curnagl_raxml_affinity.py](./check_curnagl_raxml_affinity.py): check RAxML affinity on Curnagl.
- [configure_curnagl_tcs.py](./configure_curnagl_tcs.py): configure a TCS run.
- [install_curnagl_tcs_layer.py](./install_curnagl_tcs_layer.py): install the Curnagl TCS conda layer.
- [isolate_curnagl_tcs.py](./isolate_curnagl_tcs.py): isolate a TCS run.
- [curnagl_physical_step.py](./curnagl_physical_step.py): run a single physical TCS step.
- [resume_curnagl_affinity.py](./resume_curnagl_affinity.py): resume an affinity run.
- [resume_curnagl_keepgoing.py](./resume_curnagl_keepgoing.py): resume a keepgoing run.

### CATH families
- [rebuild_cath_families.py](./rebuild_cath_families.py): rebuild CATH family inputs.
- [generate_cath_families.py](./generate_cath_families.py): generate CATH families.
- [retain_cath_subsets.py](./retain_cath_subsets.py): retain CATH subsets.
- [supervise_cath_generation.py](./supervise_cath_generation.py): supervise CATH generation.
- [audit_cath_chains.py](./audit_cath_chains.py): shared CATH chain audit helper.

### Alphabets and matrices
- [production_alphabet_experiments.py](./production_alphabet_experiments.py): production alphabet experiment driver.
- [prepare_production_alphabets.py](./prepare_production_alphabets.py): shared production-alphabet preparation; invokes `evaluate_alphabet_checkpoint.py` on reserved splits.
- [extend_matrix_references.py](./extend_matrix_references.py): shared matrix-reference extension.
- [rebuild_downloaded_matrix_references.py](./rebuild_downloaded_matrix_references.py): rebuild downloaded matrix references.
- [promote_nonconverged_matrix.py](./promote_nonconverged_matrix.py): promote a non-converged matrix.
- [evaluate_alphabet_checkpoint.py](./evaluate_alphabet_checkpoint.py): evaluate an encoder/decoder pair on a reserved split; called by `prepare_production_alphabets.py`.

### Ancestral
- [ancestral_state_uncertainty.py](./ancestral_state_uncertainty.py): ancestral-state uncertainty estimates.
- [ancestral_inspection.py](./ancestral_inspection.py): ancestral reconstruction inspection.

### CI and release
- [check_documentation.py](./check_documentation.py): documentation checks.
- [check_distribution.py](./check_distribution.py): distribution checks.
- [stage_conda_source.py](./stage_conda_source.py): stage the conda source distribution.
- [geometry_mlflow.py](./geometry_mlflow.py): geometry MLflow logging; test-covered by [../foldtree2/tests/test_geometry_mlflow.py](../foldtree2/tests/test_geometry_mlflow.py).
- `test_conda_packaging.sh`: conda packaging test (shell, see "Not code" below).

## Shared helpers
The backbone other scripts import from: [benchmark_common.py](./benchmark_common.py)
(10 importers), [phylogenetic_information_gain.py](./phylogenetic_information_gain.py)
(4), [prepare_production_alphabets.py](./prepare_production_alphabets.py) (4),
[rebuild_cath_families.py](./rebuild_cath_families.py) (3),
[three_di_ml.py](./three_di_ml.py), [supervise_curnagl_tcs.py](./supervise_curnagl_tcs.py)
and [audit_cath_chains.py](./audit_cath_chains.py). Prefer composing on these
rather than duplicating their IO or accounting.

## Obsolete
The one candidate flagged for removal was
[evaluate_alphabet_checkpoint.py](./evaluate_alphabet_checkpoint.py). A
repo-wide grep for `evaluate_alphabet_checkpoint` (excluding `.git/`,
`runs/`, `build/` and `docs/*_evidence/`) matches only
`prepare_production_alphabets.py`, which invokes it on reserved splits via
`run([sys.executable, '-u', 'scripts/evaluate_alphabet_checkpoint.py', ...])`;
it is therefore not orphaned and stays in Keep above. There are no obsolete
scripts in this directory.

## Not code
`*.sh` files here are queue drivers (`queue_contact_*.sh`) and packaging tests
(`test_conda_packaging.sh`); Slurm/qsub templates live under `docs/` and
`alps/`, not here. Treat the shell drivers as launch glue, not importable
modules.

## Reproducibility
Most drivers write under `runs/` (gitignored); evidence lands in `docs/` via
the report scripts. Pinned historical copies of these scripts exist under
`docs/*_evidence/sources/` and `runs/*/sources` — do not edit those, they
record what actually ran.