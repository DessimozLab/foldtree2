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
