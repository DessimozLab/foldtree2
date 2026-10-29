# Contact and pairwise-distance head experiments

Train and calibrate contact-head and pairwise-distance-head variants on top of `foldtree2.src.distance`.

[Scripts home](../README.md) | [Project](../../README.md) | [Source modules](../../foldtree2/src/README.md) | [SE3 toolkit](../se3/README.md)

This package runs the contact-head and pairwise-distance-head experiments on
top of the foldtree2 distance model. The contact signal is the per-residue-pair
prior that biases the distance head; the experiments here sweep its objective,
dot-product variant and residue-level bias across matched small screens, then
calibrate a single bias operating point and export the tuned checkpoint for the
spherical screen.

## Modules
- [train_contact_levers.py](./train_contact_levers.py): main resumable contact-levers trainer; driven by the `queue_contact_*.sh` shell drivers at the scripts root.
- [report_contact_objectives.py](./report_contact_objectives.py): reporting companion to the trainer; aggregates per-objective curves into a dated report.
- [experiment_contact_objectives.py](./experiment_contact_objectives.py): sweep contact objective choices across matched small screens.
- [experiment_contact_dot_variants.py](./experiment_contact_dot_variants.py): sweep dot-product contact-head variants on matched screens.
- [experiment_contact_residue_bias.py](./experiment_contact_residue_bias.py): sweep residue-level contact-bias terms on matched screens.
- [experiment_contact_cross_perturbed.py](./experiment_contact_cross_perturbed.py): cross-perturbed contact screens for robustness checks.
- [calibrate_contact_bias.py](./calibrate_contact_bias.py): pick the contact-bias operating point on a held-out dev subset.
- [compare_spherical_contact_screen.py](./compare_spherical_contact_screen.py): compare saved heads at a fixed recall over the spherical screen.
- [export_contact_operating_point.py](./export_contact_operating_point.py): export the tuned sphere checkpoint for downstream use.

## Workflow
The four `experiment_contact_*` drivers each run a matched small screen of the
pairwise-distance head variants so their learning curves are directly
comparable; keep the loss version, coordinate scale and split seed identical
across a screen before ranking variants. Pick the best-behaved screen, then run
`calibrate_contact_bias.py` on the dev subset to select the bias operating
point, and finally `export_contact_operating_point.py` to freeze that
checkpoint. The full screen→calibrate→export pipeline is one trainer plus its
reporting companion; the model code itself lands in
[foldtree2.src.distance](../../foldtree2/src/distance/README.md), not here.
`compare_spherical_contact_screen.py` ranks the saved heads at a fixed recall
over the spherical screen so the operating-point choice is grounded in a shared
recall boundary rather than a per-head threshold.

## Caveats
Runs write under `runs/` (gitignored); pair each run directory with its
provenance before comparing screens — the contact and distance objectives
behave differently under coordinate rescaling, so record the scale in the run
manifest. These modules compose on `benchmark_common.py` at the scripts root
for artifact and provenance primitives; import them via
`scripts.contact.<name>` rather than re-deriving their IO. The
`queue_contact_*.sh` drivers at the scripts root are the intended entry points
for batched screens; prefer them over invoking the trainer directly, since they
fix the device, seed and queue ordering the report scripts expect.

## Status
The screen→calibrate→export pipeline is the current workflow; the four
experiment drivers are the maintained entry points for variant sweeps. The
trainers write model code into foldtree2.src.distance, so a change to the head
signature belongs there, not in this package.