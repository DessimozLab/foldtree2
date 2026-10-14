# Paired learning-rate continuation, 2026-10-14

The user authorized the recommended continuation-versus-lower-rate comparison.
Both branches start from growth trial job 3659891's best/final checkpoint at
16 epochs / 8,192 updates. Each trains **16 additional epochs**, ending at
32 epochs / 16,384 total updates. Branch learning rates are **5e-4** and **1e-4**.
All model, loss, frozen cache, structure IDs/masks, normalization, gradient
precision, global batch and topology settings are identical to the parent.
Schedulers remain disabled, to isolate the rate change.

`prepare_se3_continuation` creates explicitly audited checkpoint branches,
preserving model state, AdamW moments/steps, RNG, global batch stream, and prior
best validation geometry. Only optimizer param-group learning rate and declared
training/initialization provenance change; the epoch endpoint becomes 32.
The unchanged trainer uses strict `--resume-from` on these derived checkpoints.
The parent checkpoint hash is
`93ff91d18ff9b4b0df5396889f9cd6c4fc0c91f65f88775caf71d2c90487bb04`.
Branch generation rejects different sources, model/dataset/training settings,
partial-epoch parents, topology changes and unapproved epoch/rate values.
Six focused tests passed, including retained moments/RNG and rejected changes.

Each job requests **two nodes/eight GPUs**, normal partition, two-hour walltime:
**16 GPUh each / 32 GPUh total reserved**. The lower-rate job depends on successful
completion of the original-rate job; both begin from the same parent, not from
each other. No automatic retries or model expansion. Expected runtime based
on the parent is approximately 50 minutes per branch, subject to measurement.
Shared budget reservations would rise from 92 to 124 of the authorized 256 GPUh.
Project credit availability is based on the user's existing sufficient-credit
confirmation, not a fair-share value or independently retrieved balance.

```bash
python -m scripts.prepare_se3_continuation --parent PARENT.ckpt \
  --config configs/clariden_se3/continue_lr5e4.yaml --output continue_lr5e4.ckpt
python -m scripts.prepare_se3_continuation --parent PARENT.ckpt \
  --config configs/clariden_se3/continue_lr1e4.yaml --output continue_lr1e4.ckpt
# On Clariden, after staging branch checkpoints and auditing their hashes:
python3 -m scripts.submit_se3_continuation --available-credit 32 --submit
```

Outputs: `.../se3/runs/continue_lr5e4` and `.../se3/runs/continue_lr1e4`.
Checkpoints and complete-epoch events are saved to durable project store.
Compare both branches against the epoch-16 parent on the identical 512 validation
structures using paired FAPE, aligned CA RMSD, TM-score, lDDT, bond, bend/torsion
and predicted-angle errors. Preserve composite-loss-selected checkpoint metrics
and the full epoch trajectory. A successful candidate should improve FAPE without
>5% error regression or >0.01 TM-score/lDDT regression. The holdout has been
repeatedly monitored and does not establish independent generalization.

## Submitted jobs

The actual jobs are **3661398** (`5e-4`) and **3661399** (`1e-4`), with
`afterok:3661398` on the second. Both Slurm preflights passed. Pinned code
revision: `4bdaab896b9a59f2c18720d3abdb126340d7e1ad`.
Staged archive and both derived checkpoint SHA-256 values were verified on
Clariden before submission. The locked ledger now reserves **124 of 256 GPUh**.
Submission receipts and branch provenance are preserved in
[continuation evidence](se3_clariden_evidence/continuation/submissions.json).
Queue acceptance does not establish completed learning; final comparisons remain
pending execution. Real parent-versus-branch tensor checks confirmed identical
model tensors and all AdamW state tensors, with respective rates 5e-4 and 1e-4.

## Completed comparison

Both jobs completed with exit 0: **3661398** in **50m45s**, **3661399** in
**49m12s** (13.327 combined actual GPUh). Each completed all 16 additional
epochs, reaching epoch 32 / 16,384 updates, with zero logged failure events.
Every epoch consumed 8,192 training structures and evaluated 512 validation
structures. The table compares each branch's composite-validation-geometry
selected checkpoint against the epoch-16 parent, rather than choosing a
different checkpoint per metric.

| Metric | Parent, epoch 16 | 5e-4, best epoch 26 | 1e-4, best epoch 19 |
|---|---:|---:|---:|
| Validation geometry ↓ | 3.630099 | 3.591750 | 3.604795 |
| FAPE ↓ | 0.929814 | 0.929656 | 0.929892 |
| Aligned CA RMSD, Å ↓ | 15.6886 | 15.7013 | 15.6829 |
| TM-score ↑ | 0.214902 | 0.214650 | 0.214742 |
| lDDT ↑ | 0.254775 | 0.258373 | 0.259029 |
| CA bond MAE, Å ↓ | 0.923609 | 0.889364 | 0.899749 |
| CA bend MAE, radians ↓ | 0.708719 | 0.715520 | 0.713215 |
| CA torsion MAE, radians ↓ | 1.520316 | 1.523159 | 1.523453 |
| Predicted angle MAE, radians ↓ | 0.514091 | 0.519928 | 0.513430 |

The original-rate branch improved selected validation geometry **1.06%**; the
lower-rate branch improved it **0.70%**. Final training geometry was 3.516331
and 3.488235 respectively, versus the parent's 3.623650. However, final-epoch
validation geometry was 3.605497 and 3.632564; final FAPE was 0.930327 and
0.930290, both worse than the parent. The lower rate does not demonstrate better
generalization. The original-rate selected checkpoint has marginally better
FAPE, with all mean structural regressions within the declared material bounds.
These tiny differences have no statistical significance claim.

The additional training provides limited structural gains and suggests
diminishing returns for this objective/model. Preserve the parent and both
selected checkpoints; no further training has been submitted. The selected
checkpoint paired metrics, complete event histories and checkpoint SHA-256
values are retained in [completed evidence](se3_clariden_evidence/continuation/completed/).
The newly produced checkpoint tensor contents have not yet been locally loaded
and audited; the hashes confirm durable file identity. Holdouts remain the same
repeatedly monitored 512 structures, with 511 defined torsion measurements.
