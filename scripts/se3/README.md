# SE(3) validation, training and investigation toolkit

[Project](../../README.md) | [Validation report](../../docs/se3_validation.md) | [Investigation docs](../../docs/)

This package holds the SE(3) structure-prediction validation, training and
investigation toolkit. It is an importable namespace package (`scripts.se3`,
see `__init__.py`) run from the repo root against an editable install; produce
a report with `python scripts/se3/<file>.py`. Maintained workflow scripts sit
directly under `scripts/se3/`; closed investigations live under
`scripts/se3/archive/`. Match the loss version, coordinate scale and split
seed in your provenance before comparing runs.

## Maintained workflow

### Validation and gates
- [report_se3_validation.py](./report_se3_validation.py): maintained validation report; writes the dated report under `docs/`.
- [se3_capacity_gate.py](./se3_capacity_gate.py): go/no-go capacity gate before larger cohorts.
- [verify_se3_normalized_debug.py](./verify_se3_normalized_debug.py): checks for the normalized-distance debug configuration.
- [verify_se3_bundle_files.py](./verify_se3_bundle_files.py): integrity checks over decoder bundles and cache artifacts.
- [verify_se3_release.py](./verify_se3_release.py): release-readiness checks.
- [profile_se3_validation.py](./profile_se3_validation.py): timing and memory profiling of the validation step.

### Reconstruction, restraint and distance program
Current program per the [reconstruction plan](../../docs/se3_reconstruction_plan_20261020.md).
- [diagnose_se3_reconstruction.py](./diagnose_se3_reconstruction.py): shared atomic-IO hub (`atomic_json`, `atomic_checkpoint`, `select_panel`); 12+ importers build on it.
- [diagnose_se3_restraints.py](./diagnose_se3_restraints.py): restraint-set diagnostics.
- [reconstruct_distance_restraints.py](./reconstruct_distance_restraints.py): reconstruct distance restraints from decoder outputs.
- [diagnose_distance_learning.py](./diagnose_distance_learning.py): distance-learning curve diagnostics.
- [compare_distance_capacity.py](./compare_distance_capacity.py): compare distance capacity across cohorts.
- [audit_distance_pair_errors.py](./audit_distance_pair_errors.py): audit pair-level distance errors.
- [queue_se3_reconstruction_controls.py](./queue_se3_reconstruction_controls.py): queue reconstruction control runs.
- [queue_se3_remaining_controls.py](./queue_se3_remaining_controls.py): queue the remaining control panels.
- [queue_distance_generalization.py](./queue_distance_generalization.py): queue distance generalization sweeps.
- [queue_distance_data_scaling.py](./queue_distance_data_scaling.py): queue the data-scaling sweep.
- [probe_se3_information.py](./probe_se3_information.py): probe SE3 information content.
- [run_se3_generalization.py](./run_se3_generalization.py): drive a generalization run.
- [run_se3_decoder_fusion.py](./run_se3_decoder_fusion.py): drive the decoder fusion experiment.
- [enrich_se3_decoder_cache.py](./enrich_se3_decoder_cache.py): enrich the decoder cache with additional panels.
- [scale_distance_data.py](./scale_distance_data.py): scale the distance dataset.
- [extend_distance_data.py](./extend_distance_data.py): extend the distance dataset.
- [extend_distance_data_10240.py](./extend_distance_data_10240.py): extend the distance dataset to the 10240 panel.

### Cohort growth and continuation
- [prepare_se3_larger_cohort.py](./prepare_se3_larger_cohort.py): shared `fresh_splits` helper for new cohorts.
- [prepare_se3_expanded_cohort.py](./prepare_se3_expanded_cohort.py): prepare the expanded cohort.
- [prepare_se3_growth_cohort.py](./prepare_se3_growth_cohort.py): prepare the growth cohort.
- [prepare_se3_continuation.py](./prepare_se3_continuation.py): prepare a continuation from a prior checkpoint.
- [stage_se3_growth.py](./stage_se3_growth.py): stage growth inputs.
- [extract_se3_growth_clariden.py](./extract_se3_growth_clariden.py): extract growth runs on Clariden.
- [export_se3_graph_inputs.py](./export_se3_graph_inputs.py): export graph inputs for the growth cohort.
- [export_se3_cache.py](./export_se3_cache.py): export the decoder cache.
- [submit_se3_growth_trial.py](./submit_se3_growth_trial.py): submit a growth trial.
- [submit_se3_growth_extraction.py](./submit_se3_growth_extraction.py): submit growth extraction.
- [submit_se3_continuation.py](./submit_se3_continuation.py): submit a continuation run.
- [submit_se3_long_trial.py](./submit_se3_long_trial.py): submit a long trial.
- [submit_se3_normalized_checks.py](./submit_se3_normalized_checks.py): submit normalized-distance checks.

### Audits and pilots
- [audit_se3_loss.py](./audit_se3_loss.py): audit the SE3 loss terms.
- [audit_se3_balanced_loss.py](./audit_se3_balanced_loss.py): audit the balanced-loss variant.
- [train_se3_revised_pilot.py](./train_se3_revised_pilot.py): revised pilot training run.
- [se3_extraction_parity.py](./se3_extraction_parity.py): extraction parity check.
- [pin_se3_image.py](./pin_se3_image.py): pin the runtime image for reproducibility.
- [se3_budget.py](./se3_budget.py): fcntl-locked Slurm GPU-hour ledger; `reserve()` is called by every submitter above.

## Shared helpers
The package builds on four scripts:
[diagnose_se3_reconstruction.py](./diagnose_se3_reconstruction.py) (atomic IO
for panels and checkpoints), [investigate_se3_scale.py](./investigate_se3_scale.py)
(`load_entries`/`fit_scale`; still live and also used by `audit_se3_loss` and
`train_se3_revised_pilot`), [se3_budget.py](./se3_budget.py) (the GPU-hour
ledger every submitter reserves against) and
[prepare_se3_larger_cohort.py](./prepare_se3_larger_cohort.py) (the
`fresh_splits` cohort helper). Compose on top of these instead of re-deriving
their IO or accounting.

## Archive (closed investigations)
Kept unmodified for reproducibility; new code should not import from
`scripts/se3/archive/`. Evidence for each lives under `docs/se3_*_evidence/`.
- [report_se3_investigation.py](./archive/report_se3_investigation.py): dated reporter for the root 2026-10-09 investigation ([docs](../../docs/se3_investigation_20261009.md)).
- [report_se3_spatial_learning.py](./archive/report_se3_spatial_learning.py): dated reporter for the spatial-learning sub-investigation ([docs](../../docs/se3_spatial_learning_20261009.md)).
- [report_se3_loss_ablation.py](./archive/report_se3_loss_ablation.py): dated reporter for the loss-ablation campaign ([docs](../../docs/se3_loss_ablation_20261009.md)).
- [report_se3_scaling.py](./archive/report_se3_scaling.py): dated reporter for the scaling sub-investigation ([docs](../../docs/se3_investigation_20261009.md)).
- [report_se3_stability.py](./archive/report_se3_stability.py): dated reporter for the medium/stability investigation ([docs](../../docs/se3_medium_stability_20261013.md)).
- [run_se3_loss_ablation.py](./archive/run_se3_loss_ablation.py): loss-ablation campaign driver ([docs](../../docs/se3_loss_ablation_20261009.md)).
- [verify_se3_lightning.py](./archive/verify_se3_lightning.py): Lightning config checks from the gradient-precision investigation ([docs](../../docs/se3_gradient_precision_20261014.md)).
- [verify_se3_loss_checks.py](./archive/verify_se3_loss_checks.py): loss finite-value/gradient guards from the gradient-precision investigation ([docs](../../docs/se3_gradient_precision_20261014.md)).
- [diagnose_se3_medium.py](./archive/diagnose_se3_medium.py): root-cause diagnosis for the medium-failure investigation ([docs](../../docs/se3_medium_failure_20261013.md)).

## When to use / status
Use the maintained workflow above for any new SE3 validation, reconstruction
or cohort work; it supersedes the archived reporters for live runs. The
reconstruction and distance program is current and under active development,
while the loss-ablation, medium-failure, gradient-precision and scaling
investigations are closed and recorded in `docs/`. Prefer these docs for
context before re-running or extending any of these scripts:
[SE3 validation](../../docs/se3_validation.md),
[reconstruction plan](../../docs/se3_reconstruction_plan_20261020.md),
[decoder handoff](../../docs/se3_decoder_handoff.md) and
[larger-local](../../docs/se3_larger_local.md).