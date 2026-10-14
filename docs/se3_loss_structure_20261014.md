# Coordinate-loss experiment, 2026-10-14

The user requested changes to the loss after the learning-rate continuations
showed diminishing structural gains. This implements **opt-in loss supervision**,
not a claim of improved trained structures. The existing soft-FAPE + all-pairs
SmoothL1 distance + CA bond + auxiliary angle objective remains the default.
Production encoder/decoder weights, caches, staged transformer, evaluation FAPE
and evaluation masks are unchanged.

## What changes

`balanced_geometry_v1` exposes independent weights for the existing terms and
adds three coordinate-derived terms:

* **Nearby nonlocal distances:** SmoothL1(predicted distance / 4 Å, target
  distance / 4 Å), for unordered valid pairs whose original row indices differ
  by more than two and whose target CA distance is below 15 Å. Target distances
  below 1e-4 Å are excluded. This is a smooth distance surrogate, not lDDT itself.
* **Backbone bend:** SmoothL1 between consecutive unit-bond dot products.
  It supervises coordinate bends without the singular derivative of acos.
* **Signed torsion:** half the squared distance between predicted and target
  (cos torsion, sin torsion). This detects reflections that distances alone
  cannot distinguish. It supervises coordinate torsions directly; it does not
  make the separate angle head reconstruct the backbone.

Target bonds must join adjacent valid rows and have length in [2.5, 4.5] Å,
matching the existing continuity heuristic. Bends/torsions require all their
bonds to qualify; undefined target torsions are masked. Undefined new terms
contribute differentiable zero with explicit eligible counts. A chain with
no eligible active coordinate supervision fails clearly. The existing FAPE
frame and valid-input checks remain enforced. Predictions cannot avoid a term
by becoming degenerate: eligibility depends only on targets and masks. Unit
vectors use a local 1e-4 normalization epsilon; no global dtype/backend changes.

The combined starting weights are global distance 1, soft FAPE 1, bond 1,
auxiliary angle 0.1, nearby distance 1, bend 0.5 and torsion 0.5. Existing
component normalization remains unchanged: all-pairs SmoothL1 / 10, pseudo-Huber
frame error at 10 Å, bond SmoothL1 in Å, and mean circular angle loss.
Weights are preliminary experimental settings, not validated optimal values.

## Controlled comparison

Four matched four-epoch configurations are prepared under `configs/clariden_se3/`:
`loss_v1_control.yaml`, `loss_v1_local_distances.yaml`,
`loss_v1_backbone.yaml` and `loss_v1_combined.yaml`. They respectively retain
the control, add nearby distance, add bends/torsions, and add all three.
All start weights-only from the same epoch-16 growth checkpoint and reset the
optimizer, RNG initialization and sample stream equally. Same 8,192 training /
512 validation structures, width 64/depth 4, global batch 16, LR 5e-4,
normalization and float64 gradient accumulation/reduction.

Example trainer arguments, inside a prepared Clariden allocation with the new
code archive (not the previous production training archive):

```bash
python -m foldtree2.train_se3_lightning \
  --config configs/clariden_se3/loss_v1_combined.yaml \
  --warm-start /capstor/store/cscs/swissai/a0117/foldtree2/se3/runs/growth_fp64_8192/best.ckpt \
  --allow-loss-change
```

`--allow-loss-change` requires an explicit weights-only warm start. It verifies
the cache/split, frozen identities, preprocessing, architecture, normalization,
training protocol and unchanged model sources. The only historical trainer
source accepted across this boundary is the exact pre-experiment implementation
(hash recorded in code). Existing strict resume checks remain in place. Only
loss settings, epoch endpoint, LR, logging frequency/structural metrics and the
checkpoint selection metric may change; topology/output/cache paths remain
operational fields. All four configurations passed compatibility checks against
the actual saved parent provenance.

All four configurations select checkpoints by **the unchanged evaluation FAPE**.
The default selection criterion remains geometry for existing configurations.
Validation events now report all seven raw loss components plus the fixed
original `control_geometry`; per-structure records retain values and eligible
counts. Update diagnostics explicitly label loss components as the last
microbatch's values, not epoch means. Raw training-objective magnitudes cannot
be compared across differently weighted losses. Evaluate paired FAPE, aligned
CA RMSD, TM-score, lDDT, bonds, bends, torsions and angle error on identical
structures/masks, with the established material-regression limits. Treat this
repeatedly monitored holdout as development data, not an independent test.

## Verification and limits

The [CPU audit](se3_clariden_evidence/loss_structure_v1/audit.json) used eight
retained **training** structures only and the trained epoch-16 checkpoint.
All coordinate-component gradients were finite. One complete model backward
had nonzero finite parameter-gradient norm **7.512584**. No production model
was called, no optimizer step was taken, and no GPU allocation was used.

Mean weighted coordinate-gradient norms on these eight examples:

| Component | Norm |
|---|---:|
| distance | 0.008132 |
| training_fape | 0.049186 |
| bond | 0.094728 |
| local_distance | 0.018891 |
| bend | 0.013822 |
| torsion | 0.051866 |

These measurements support a bounded comparison but do not establish improved
accuracy or balanced gradients across the full dataset. Per-structure gradient
cosines and target-stencil counts are preserved in the audit. Tests cover exact
control compatibility, perfect predictions, independent rigid transforms,
float32 tolerance, reflected backbones, masking/NaNs, gaps, short/collinear
chains, nonfinite-input rejection, finite-difference gradcheck, model loss-path
integration, FAPE checkpoint selection and provenance rejection.

Final verification: **71 passed, one skipped, one expected failure** across
balanced/legacy losses, Lightning integration, refiner correctness and geometry
validation. The GPU-only check was skipped in the restricted sandbox; the
pre-existing depth-six numerical failure remains explicitly expected. The CPU
audit uses float32 model forward/backward and float64 coordinate diagnostics.

At the implementation checkpoint, no Clariden loss-experiment jobs had yet been submitted.

## Authorized Clariden submission

The user authorized queuing the matched loss experiment. Submission uses
`scripts.submit_se3_loss_trials`: a one-node save/reload check, a two-node
comparison, then control, nearby-distance, backbone and combined trials.
Every job depends on successful completion of the preceding job. The one-node
check compares one-update + resumed second update against an uninterrupted
two-update control, including exact model/optimizer state and validation. The
two-node check compares two updates against that one-node control at 1e-4
absolute/relative tolerances. Both use the combined loss and the full original
cache/split. Checks run in the pinned container/venv on allocated compute nodes.

Resources: normal partition; one node/four GPUs for ten minutes (2/3 GPUh),
two nodes/eight GPUs for ten minutes (4/3 GPUh), then two nodes/eight GPUs for
45 minutes per four-epoch trial (6 GPUh each). Total ceiling **26 GPUh**; the
shared ledger would rise from 124 to **150 of 256 GPUh**. Based on prior runs,
each four-epoch trial is expected to take roughly 13–20 minutes; the new loss
overhead remains unmeasured. Credit availability uses the user's existing
sufficient-credit confirmation, not an independently retrieved portal balance.
No architecture expansion, new cache extraction, automatic retries or extra
epochs are included. Parent checkpoint and frozen bundle hashes are checked
before submission; the trainer verifies compatible weights-only initialization.
The control restarts optimizer/sample stream exactly as all three candidates do.

```bash
python3 -m scripts.submit_se3_loss_trials --available-credit 26 --submit
```

## Accepted jobs

All six Slurm preflights and actual submissions succeeded. Code revision
`d7108d96e95e2fa7cf1ce07abdda5fc61abbfb48`, archive SHA-256
`7be890468ea3f459d7c19d13b0bd509c47d4dcd45d99ccc8ed2d8b295406eb7b`,
verified on Clariden before extraction/submission.

| Stage | Job |
|---|---|
| loss_v1_debug_one | 3663702 |
| loss_v1_debug_two | 3663703 |
| loss_v1_control | 3663704 |
| loss_v1_local_distances | 3663705 |
| loss_v1_backbone | 3663706 |
| loss_v1_combined | 3663707 |

The ledger now reserves **150 of 256 GPUh**. All jobs are queued behind the
preceding job with `afterok`; the first check is eligible immediately. Queue
acceptance does not establish passed gates or trained quality.
[Submission receipts and observed queue state](se3_clariden_evidence/loss_structure_v1/submissions.json)
are retained. Fifteen focused loss/launcher tests passed before submission.

## Failed-node investigation and bounded recovery

One-node exact resume and two-node agreement **passed**; model state difference
was zero in both comparisons. Control job **3663704** completed all four epochs
/ 2,048 updates in 14m36s. Its FAPE-selected checkpoint was epoch 3, FAPE
**0.9298599067**, versus the epoch-16 parent's **0.9298138915**. Lower is better:
this control does **not** improve FAPE over its parent. It does remain better
than the raw frozen decoder's FAPE **0.9420447623**. The control's best-checkpoint
CA RMSD was 15.8498 Å, TM-score 0.212973, lDDT 0.260636 and bond MAE 0.902921 Å.
These do not establish a benefit from any new loss, since it is the control.

Nearby-distance job **3663705 failed before a complete optimizer update** after
2m32s. Rank 2 reported `CUDA error: unspecified launch failure` while executing
the model forward. CUDA errors may surface asynchronously, so this location
does not establish the originating operation. The node **nid007494** was then
put into **IDLE+DRAIN** with site health reason **"Failed to allocate 90% of the
GPU memory after job 3663705"** at 21:33:42. This supports a node/runtime fault
as the leading explanation; it is not proof that the loss implementation is
fault-free. Secondary TCPStore/broken-pipe warnings followed rank termination.
No application failure snapshot was captured because Lightning teardown also
failed on the unusable CUDA context. The source, failed log and partial output
are preserved. The deterministic first global batch is retained in the
[failure evidence](se3_clariden_evidence/loss_structure_v1/failure-diagnosis.txt).

A single correction uses identical pinned training code, cache, parent weights
and loss, with a fresh output `loss_v1_local_distances_recovery1` and explicit
exclusion of nid007494. Its **40-minute/two-node** ceiling plus the failed
152-second allocation is at most **5.6712 GPUh**, below the original trial's
**6 GPUh / 45-minute** reservation. The controller records a zero-additional-
reservation child of the existing trial reservation under the durable ledger
lock, refuses a second correction, verifies the failed allocation/health state
and parent checkpoint, and tests Slurm arguments before submission. Total
reserved budget remains **150 of 256 GPUh**. On acceptance it rewires only the
waiting backbone job 3663706 to this recovery; combined job 3663707 already
depends on the backbone job. A second failure stops this correction path for
further diagnosis. No architecture-branch code or mathematical loss change is
part of this recovery.

Correction **3663975** was accepted with a 40-minute ceiling and unchanged
training source revision. Backbone job **3663706** now depends on
`afterok:3663975`; combined **3663707** remains dependent on the backbone job.
The failed attempt used 0.3378 GPUh; the failure-plus-correction ceiling is
5.6712 GPUh against its existing 6-GPUh reservation. The recovery row records
zero additional reserved GPUh and links to that parent reservation, preventing
double allocation. Actual queue/finite-update progress must still be monitored.
Passed check records, completed control events/paired metrics/checkpoint hash,
recovery receipt and paired control-versus-parent FAPE deltas are retained in
the evidence directory. The control-vs-parent descriptive bootstrap interval
uses the repeatedly monitored development holdout, not an independent test.
