# Longer normalized medium SE(3) trial, 2026-10-14

The user authorized a longer learning trial after the one- and two-node
[gradient precision checks](se3_gradient_precision_20261014.md) passed, including
exact model, optimizer, checkpoint recovery and held-out metric comparisons.
This trial uses the same verified training mathematics with a fresh seed-42
initialization. It does not resume the short debug checkpoints.

Configuration: `configs/clariden_se3/medium_fp64_trial.yaml`. Train on the existing
4,096-structure frozen cache, validate on the same disjoint 512 structures each
epoch, and cap each structure at 256 residues. Run width 64, depth 4, four heads
of dimension 16, parameter-free tensor feed-forward input normalization and
float64 gradient accumulation/reduction. Forward/backward, parameters and AdamW
remain float32. Global batch 16 means one structure per microbatch and two
microsteps per rank across eight ranks. AdamW learning rate is 5e-4, clipping 1,
with no scheduler. Losses, calibration, masks and frozen production pair are
unchanged. This is a fixed 16-epoch trial (4,096 optimizer updates); there is no
automatic extension or subsequent scale-up.

Allocate two GH200 nodes/eight GPUs on `normal` for at most two hours: **16 GPUh
maximum reservation**. The shared rollout ledger remains authoritative; its
prior reservations are 52 GPUh, so this submission brings the conservative
total to 68 of the user-authorized 256 GPUh ceiling. This is reservation
accounting, not a fresh portal credit balance. The user confirmed sufficient
project credit. The trial-only schedule does not authorize all older expansion
stages cumulatively; later stages require reconciliation against the same ledger.

The short two-node check measured approximately 24 global structures/s during
updates. Sixteen full epochs contain 65,536 training structure visits, suggesting
about 45 minutes for updates alone. Validation, startup and checkpointing add
time; the two-hour wall limit is a bound, not a completion guarantee.

Use the pinned ARM extended PyTorch image and already prepared environment
recorded in the precision report. No production checkpoint is replaced, no
graphs regenerated, and no unrelated model or workload changed.

After staging a verified code archive in a fresh revision-named directory:

```bash
python3 -m scripts.submit_se3_long_trial --available-credit 256 --submit
```

Here `256` follows the user's existing credit confirmation and authorized
ceiling; it is not a measured portal balance. The launcher verifies both
stability gates, rejects an existing trial output or duplicate budget reservation,
checks Slurm resource acceptance, then records the actual submission receipt.
It submits once, without automatic requeue or retry.

Durable output is
`/capstor/store/cscs/swissai/a0117/foldtree2/se3/runs/medium_fp64_trial`.
`last.ckpt` is saved every 64 completed optimizer updates and at epoch ends;
`best.ckpt` selects complete-epoch validation geometry. Checkpoints include
optimizer, stream, RNG and provenance. Resume uses the exact pinned source and
configuration with `--resume-from .../last.ckpt` after an explicit bounded
recovery decision. A walltime interruption must not silently extend the budget.

At completion, inspect all epoch means and paired validation files, finite
weights/optimizer state, failure counts, throughput and peak memory. Report
epoch-one versus best training/validation geometry and FAPE, plus baseline versus
refined FAPE, aligned CA RMSD, TM-score, lDDT, backbone bonds and angle errors on
identical structures/masks. Raw-decoder improvements include the pre-existing
coordinate calibration and seed lift; epoch trends establish learning beyond
that preprocessing. These are refiner holdouts, not evidence of independence
from production pretraining. The historical small model used different
normalization and accumulation, so this trial is not a matched capacity gate.
Require held-out FAPE improvement and no material structural regression before
larger training; diagnose failure rather than automatically increasing scale.

Actual submission: job **3657774**, pinned code revision
`1afdd56ad16f23327e1e93cf60293b2240cbf5a4`. Archive SHA-256:
`01c5f9e778c787bf7293ba1a68e94769e320a95ff6f31567255c19a5569a8180`.
The archive completeness check passed for all 20 required dependencies, and
all 12 training source files matched the validated `8f300d6` revision byte for
byte. Configuration checks confirmed 4,096 updates, two microsteps per rank,
both stabilization flags, absence of the debug stop limit, and exact agreement
between launch resources and the 16 GPUh budget. Batch-shell syntax and Python
compilation passed. The Slurm test-only ID `3657773` is not the submitted job.

Initial observed state: **PENDING**, reason **QOSGrpNodeLimit**, no allocated
nodes and no training output yet. Submission is not evidence of successful
training. The durable ledger confirmed 52 prior GPUh and 68 after reservation.
The job is left queued without changing its partition, resources or other jobs.

Log:
`/capstor/store/cscs/swissai/a0117/foldtree2/se3/code/1afdd56ad16f23327e1e93cf60293b2240cbf5a4/logs/medium_fp64_trial-3657774.out`.
Submission receipt:
`/capstor/store/cscs/swissai/a0117/foldtree2/se3/artifacts/medium_fp64_trial-submission.json`.

```bash
squeue -j 3657774
sacct -j 3657774 --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
tail -n 20 /capstor/store/cscs/swissai/a0117/foldtree2/se3/runs/medium_fp64_trial/events.jsonl
```

The event file appears after initialization. Launch evidence is retained in
`docs/se3_clariden_evidence/medium_fp64_trial/submission.json`.
