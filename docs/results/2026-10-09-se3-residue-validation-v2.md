# SE(3) residue refiner validation, 2026-10-09

Protocol: `se3-residue-validation-v2-endpoint-masks`.
Status: correctness validated and bounded experiments complete; the overfit
learning gate failed. Larger training, distributed training and atom refinement
remain deferred.

The source started from revision `8351257`, with the implementation and tests
packaged in the same commit as this record. The
[source manifest](../se3_validation_evidence/source_manifest.json) identifies
the packaged files by SHA-256; per-run provenance separately binds the executed
refiner and loss functions. The
[full report](../se3_validation_report.md) contains learning curves, all metrics,
per-epoch measurements, diagnosis and failed-attempt history.

## Inputs and configuration

The manifest-selected 40-character epoch-40 encoder and decoder remained frozen.
The existing mk2 HDF5 dataset was used directly, without graph regeneration.
Dataset and matched checkpoint hashes are in the report and archived provenance.
All runs used seed 42, float32, residue-level production bottleneck coordinates,
latent/codebook features and geometry contacts. The refiner used width 32,
depth 2, two heads and head dimension 16; AdamW at 5e-4, weight decay 0.01,
gradient clipping 1, microbatch 1 and effective batch size 8. The runner uses no
data-loader workers, atom refinement, uncertainty weighting, gradient
sanitization or silent empty-loss skipping.

The [workflow](../se3_validation.md) provides exact commands. Configurations are
[smoke](../../configs/se3_validation_smoke.yaml),
[overfit](../../configs/se3_validation_overfit.yaml) and
[pilot](../../configs/se3_validation_pilot.yaml).

## Completed counts and results

Smoke completed one training and one validation batch with finite nonzero
gradients. Overfit completed 30 epochs on 32 complete chains of at most 256
residues, evaluated on the same structures. Pilot completed 20 epochs on 512
training and 128 disjoint validation chains of at most 384 residues. Selection
excluded 4, 25 and 209 structures respectively, with reasons and IDs recorded
in provenance. Chains were excluded by eligibility or length rather than cropped.
Validation early stopping used patience five and FAPE minimum improvement 1e-4.

| Outcome | Measured result |
|---|---|
| Overfit training geometry | 3.304416 → 3.295885; 0.26% reduction, below required 20% |
| Overfit FAPE | 0.944481 → 0.946298; worse than unrefined coordinates |
| Pilot training geometry | 3.569278 → 3.384625; 5.17% reduction |
| Pilot held-out FAPE | 0.949674 → 0.948793; improved for 124/128 structures |
| Pilot aligned CA RMSD | 20.9333 → 21.1033 Å; 0.81% increase |
| Pilot TM-score | 0.07718 → 0.08001 |
| Pilot CA lDDT | 0.02588 → 0.03264 |
| Pilot CA bond MAE | 3.6442 → 3.4019 Å |
| Peak pilot GPU memory | 3.31 GiB |
| Primary corrected training time | Overfit 161 seconds; pilot 1,389 seconds |

The pilot meets the operational held-out gate: lower FAPE with no error metric
regressing by more than 5%, and no TM-score/lDDT drop exceeding 0.01 absolute.
The overall gate fails because overfit did not demonstrate the required learning.
FAPE improvement is small and does not establish useful reconstruction quality.
No confidence intervals or claim of independence from production pretraining
are made. Baseline coordinates are the decoder bottleneck supplied to the
refiner; atom bond metrics and a missing production angle head remain unavailable.

## Verification, diagnosis and remaining work

All 59 CPU/CUDA tests passed, retaining the existing 29 geometry/frame tests.
Coverage includes rotation/translation contracts at float32 tolerances of
1e-4, mixed-length batching, padding, masked residues, short/collinear chains,
coincident coordinates and finite gradients. Cached/fresh extraction and full
refiner outputs agreed. Frozen parameters received no gradients; their state
and production checkpoint files remained unchanged. Resume restored optimizer,
scheduler, epoch, global step and RNG state, and rejected incompatible provenance.

GPU 1 was used; unrelated GPU 0 workloads were not interrupted. Warmed
forward/backward profiling without updates used at most 2.98 GiB for the tested
mixed batches. The validated training envelope remains 384 residues with
microbatch 1; allocation measurements do not establish larger-batch training
stability.

The pilot refined trace has mean radius 1.58 Å versus 21.82 Å in targets, and
mean CA spacing 0.44 Å versus 3.84 Å. The uncalibrated bottleneck scale,
normalized degree-1 coordinate projection and absence of a seed-coordinate
residual are plausible causes of poor reconstruction; causal ablations have
not been run. Investigate unit calibration and an identity-preserving update
before increasing capacity or training duration. The staged transformer,
production model status and alphabet benchmark results were not changed.

The [evidence archive](../se3_validation_evidence/README.md) preserves logs,
curves, structure IDs, comparisons and earlier attempts. Large checkpoints,
cache and inputs remain local, with experimental checkpoint hashes catalogued.
