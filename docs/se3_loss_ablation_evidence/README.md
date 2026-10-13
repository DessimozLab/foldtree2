# Residue loss-ablation evidence

See the [results report](../se3_loss_ablation_20261009.md).

The combined objective passed the bounded gates: 21.36% own-objective training
reduction, better held-out FAPE on 111/128 structures, and no material structural
regression under the existing thresholds. A revised 512/128 pilot remains
required before larger training; this archive does not contain that pilot.

This bounded experiment compares four eight-structure screens and two
predeclared 32-structure, 30-epoch runs: the original objective and combined
soft-FAPE/local-bond objective. Both use the same seed, architecture, lifted
input, cache, batch order and 128 held-out IDs. Holdouts do not select weights,
cases or checkpoints. Epoch selection uses each model's training objective.

`provenance.json` binds input hashes, IDs, software/device, fixed weights and
executed source identities. `sources/` retains the executed files. Subsequent
packaging changes are separately hashed in `packaged_source_manifest.json`.
`initial_gradient_diagnostics.json` contains component values and coordinate
gradient norms before training. `screening.json`, `overfit.json`,
`heldout_pairs.json`, `gate_summary.json`, events, console and learning curves
contain the measurements. Dataset, caches and checkpoint weights stay local;
`checkpoint_inventory.json` retains weight hashes.

Correctness: 76 tests passed, including CUDA coverage. The test suite covers
the soft tail gradient, exact original-loss control, masked NaNs, rigid-motion
invariance, bond gap masking and the existing refiner/frame regressions.

The soft training term does not replace clamped FAPE evaluation. Bond-loss
eligibility excludes target adjacent distances outside [2.5,4.5] Å as a
continuity heuristic. Evaluation retains the original structure/residue masks.
These are CA-level metrics and refiner holdouts, without a production-pretraining
independence claim or an atom-level validation claim.

```bash
python scripts/report_se3_loss_ablation.py --evidence docs/se3_loss_ablation_evidence --output docs/se3_loss_ablation_20261009.md
```
