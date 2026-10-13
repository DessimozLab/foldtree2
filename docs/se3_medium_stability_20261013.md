# Normalized medium SE(3) stability gate, 2026-10-13

This experiment checks one full local training epoch and all held-out structures
after the tensor overflow diagnosed in
[the failure investigation](se3_medium_failure_20261013.md).
The gate **passed**: 256 optimizer updates, all 4,096 training structures and
all 512 validation structures completed with no nonfinite events. The final
model parameters, buffers and optimizer state are finite, and the frozen cache
hashes and exact validation split membership were verified after the run.

The trained checkpoint also passed a CUDA rotation/translation equivariance
and backward check on the longest held-out structure, `A0A543CTK2` (255 stored
residues, 210 valid). At `atol=rtol=1e-4`, maximum coordinate error was
`3.81e-6` Å and maximum angle error was `2.38e-7` radians. Gradients were finite
and nonzero. This check transformed the calibrated input coordinates and used
the same valid-residue mask for both predictions.

Elapsed time from trainer readiness through final checkpoints was 927.4 seconds
(15.5 minutes). Peak allocated training GPU memory was 4,726,221,824 bytes
(4.40 GiB), and median update throughput was 5.20 structures/second. Logged
gradient norms, measured before clipping, ranged from 0.728 to 14.08; every
update used clipping at 1. The effective training geometry mean was 4.3992 and
held-out geometry was 4.0946.

| Metric | Frozen-decoder coordinates | Refined coordinates |
| --- | ---: | ---: |
| FAPE | 0.942045 | 0.933578 |
| Aligned CA RMSD, Å | 19.1519 | 16.2529 |
| TM-score | 0.079161 | 0.209439 |
| lDDT | 0.020720 | 0.243617 |
| CA bond MAE, Å | 3.6384 | 1.3549 |
| CA bend MAE, radians | 1.4434 | 0.7519 |
| CA torsion MAE, radians | 1.5486 | 1.5387 |

Metrics use 512 paired structures, except torsions, which use 511 because one
structure has no defined valid torsion. Refined angle MAE is 0.5840 radians;
the raw decoder has no corresponding angle prediction. These improvements
also include the input calibration and normal lift, so they do not isolate
learning attributable to the normalization or this one epoch.

The model uses width 64, depth 4, four heads and head dimension 16, with explicit
`tensor_ff_input_norm: true`. Each equivariant feed-forward MLP input receives a
parameter-free LayerNorm. This changes the architecture but preserves the
initial parameter draw. It is not a repair for previously trained checkpoints.
The original small/medium configs retain their original model functions.

The cache-only Lightning trainer uses seed 42, float32 without TF32, fresh
initialization, one structure per microbatch, global batch 16, AdamW at 5e-4,
and gradient clipping at 1. The same 4,096 training and 512 validation IDs are
used, with at most 256 stored residues per structure. One epoch means 256
optimizer updates and exactly one presentation of every training structure.
The frozen production encoder/decoder are not loaded or called. Checkpoints
are saved every 32 updates and at the complete epoch boundary.

The explicit normalization option is included in checkpoint provenance. Resume
from a configuration with different normalization is rejected even though the
parameter-free LayerNorm does not add state-dictionary keys. A new regression
test verifies that rejection and that normalization remains opt-in.
Warm-start now rejects a normalization mismatch in either direction, including
legacy unnormalized checkpoints, rather than silently accepting identical
parameter shapes.

```bash
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
CUDA_VISIBLE_DEVICES=0 python -m foldtree2.train_se3_lightning \
  --config configs/clariden_se3/medium_stability_local.yaml \
  > runs/se3_clariden/medium_stability_20261013.log 2>&1
python -m scripts.report_se3_stability \
  runs/se3_clariden/medium_stability_20261013 \
  --cache runs/se3_clariden/bundle_4096 \
  --output runs/se3_clariden/medium_stability_summary.json
```

The report checks the completed stream, exact optimizer update sequence,
effective training count, validation count and split membership, finite losses
and nonzero finite gradient norms, finite model and optimizer state, finite
structural metrics, absence of failure artifacts, and unchanged frozen bundle
hashes. Validation records paired frozen-decoder and refined metrics. Those
pairs include the previously selected input calibration and local-normal lift;
they are not a pure measurement of learning from this one epoch.

This gate establishes local one-epoch stability only. It does not establish
long-run or distributed stability, capacity scaling, or independence from
production-model pretraining. The next gate is a bounded Clariden test of this
exact variant, including checkpoint recovery, before a fresh learning trial.
The six-layer rollout remains blocked.

The actual training source hashes match commit `cce64f9` on
`se3-clariden-lightning`. The subsequent warm-start guard changes checkpoint
compatibility code without changing model mathematics. Strict source provenance
requires using the recorded revision for this run's checkpoints; they are not
automatically migrated to the newer revision. Local artifacts, `last.ckpt` and
`best.ckpt` are retained under `runs/se3_clariden/medium_stability_20261013`.
Committed evidence is in
[medium_stability](se3_clariden_evidence/medium_stability/): completion summary,
per-update events, all per-structure validation comparisons, software versions,
configuration, checkpoint hashes and test results. No new cluster job was
submitted for this local gate.

The local environment was missing `tmtools`; the pinned version `0.3.0` was
installed before the validation phase. An initial geometry test failure was
caused by that missing dependency. The rerun passed 46 tests with one CUDA test
skipped in the restricted sandbox. The Lightning suite passed 12 tests with the
existing strict expected failure for the blocked six-layer model.
