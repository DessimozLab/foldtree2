# SE(3) Lightning rollout on Clariden

The cache-only Lightning trainer and bounded launch bundle are implemented.
One- and two-node Clariden checks passed, including checkpoint resume.
The small capacity trial completed all 16 epochs; the medium trial failed after
six updates with nonfinite GotenNet invariant features. The capacity gate is
incomplete. See the [medium failure investigation](se3_medium_failure_20261013.md)
for the reproduced tensor overflow and a bounded normalization ablation.
The explicit normalized medium variant has since passed a complete local
4,096-training / 512-validation one-epoch [stability gate](se3_medium_stability_20261013.md).
Its distributed and longer-run gates remain outstanding.
The 128-wide, six-layer variant is **blocked** by nonfinite forward features in
GotenNet 0.3.1. Code and the verified cache have been staged on Clariden. The user confirmed sufficient credit and authorized submission. Slurm fair-share
is not a credit balance.

This trains the residue SE(3) refiner. The staged transformer, production
40-character epoch-40 encoder/decoder, alphabet benchmarks and atom refinement
are unchanged. Holdouts describe refiner splits; production-model pretraining
independence is not claimed.

## Implementation

`python -m foldtree2.train_se3_lightning --config CONFIG` uses Lightning manual
optimization with AdamW, float32, no TF32, clipping at 1, and the previously tested
soft training FAPE, distance, local CA bond and circular angle objective. Clamped
FAPE remains the evaluation metric. Frozen extraction runs locally, before
export; training calls neither production component.

Global batch 16 is fixed. One, two and four GH200 nodes use 4, 8 and 16 ranks and
4, 2 and 1 accumulation microsteps. Every GPU processes one structure at a time;
there is no cross-structure padding. Length-sorted groups of 16 are shuffled by
seed 42 plus epoch and sharded in global sample order. No training or validation
IDs are dropped or duplicated. Validation reduces metric sums and individual
valid counts, including nullable torsion metrics, and writes paired raw-decoder
versus refined measurements per structure.

Each rank's BatchNorm normalizes each training structure independently. Running
statistics are merged in global sample order after each update, so validation
uses the same statistics as serial training. Ordinary rank-local BatchNorm
buffers caused a reproducible serial/DDP validation discrepancy; synchronized
running statistics fix it without changing per-structure training normalization.

The medium model's final high-degree norm overflowed while accumulating squared
tensor features, producing finite outputs but NaN gradients. Its opt-in stable
norm rescales before accumulating the norm, then scales epsilon identically;
this is algebraically the original normalization and stays in float32. Only
these model instances change. The small model retains its earlier norm path.
The large model overflows earlier in its tensor stream; final-norm stabilization
does not resolve it, and the training CLI rejects that architecture.

## Local verification, 2026-10-13

- CPU suite: 86 passed, one CUDA test skipped in the restricted sandbox, one
  strict expected failure for the blocked six-layer model.
- The skipped CUDA equivariance/gradient test was subsequently run on idle GPU 0.
- Two real-cache optimizer updates, global batch 16: serial versus two CPU DDP
  ranks agree to maximum parameter/buffer difference `3.0914e-7`.
- Three validation structures split unevenly across two ranks are each counted
  once. Paired structural metrics are retained in the verification artifacts.
- Interrupted one-rank training resumes mid-epoch to exactly identical weights
  and buffers, including BatchNorm statistics. Interrupted two-rank DDP training
  also resumes to exactly identical weights and buffers.
- Real CUDA forward/backward tests at 32, 128 and 256 residues have finite losses
  and nonzero finite gradients for small and stabilized medium models.
- Medium Lightning smoke: two updates on the full 4,096-structure training
  stream, finite validation losses on all 512 holdouts, peak GPU memory
  3,689,615,872 bytes (3.44 GiB). This is numerical verification, not a learning
  experiment or a measured GH200 capacity limit.

Evidence is in [se3_clariden_evidence](se3_clariden_evidence/README.md); full local
artifacts are under `runs/se3_clariden`. GH200/ARM imports, numerical parity,
NCCL transport and multi-node behavior still need allocated-node checks.

## Cache export and storage

```bash
python -m scripts.export_se3_cache \
  --cohort runs/se3_larger_local_20261013/cohort.json \
  --calibration runs/se3_loss_ablation_20261009/provenance.json \
  --output runs/se3_clariden/bundle_4096
```

Export verifies the source HDF5, matched production checkpoint hashes and frozen
extractor identity, then copies exactly 4,608 cache entries. The bundle is about
481 MiB, with content identity
`0250c8b94e10884553794470f2035355eb044e327feff3e66a72553f0e37e290`.
Its manifest contains split IDs, per-file hashes/lengths, calibration and
production hashes. Cluster loading verifies content without resolving workstation
paths or requiring the original HDF5. Export is deliberately a fresh-directory
operation; entry hashes are checked again after transfer.

Durable root: `/capstor/store/cscs/swissai/a0117/foldtree2/se3`.
Scratch cache: `/iopsstor/scratch/cscs/dmoi/se3/bundle_4096`.
Use a revision-specific code directory and venv, never the old
`/users/dmoi/foldtree2` checkout. Existing `alps/` launchers remain intact.

## Environment and launch

The reviewed extended image is
`jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.02-py3-alps6`.
The unauthenticated mirror metadata request returned HTTP 401. The documented
public GHCR copy was resolved and its config verified as ARM64, with digest
`sha256:3dd0d01a02e58d3c78e4222ce252bd8d177c14392b3d1a05a1f4c9a92454ae40`.
The committed `extended-pinned.toml` uses that public image by digest.
`pin_se3_image.py` defaults to this public registry, keeps its anonymous pull
token in memory, and verifies ARM64 without downloading image layers. Record
resolved Python packages during allocated-node preparation. `alps/se3/requirements.txt` pins the locally validated refiner libraries.
Preparation creates an ARM venv with the base image's packages visible and pins
its existing torch version as a constraint, preventing an accidental CUDA/PyTorch
replacement. Base PyTorch differs from workstation 2.8.0+cu128 and must pass
cluster numerical checks before training.

The extended-image EDF uses `entrypoint=true`, disables the host CXI hook, and
omits the AWS OFI hook. Its launcher uses `--network=disable_rdzv_get --mpi=none`.
These settings follow the [CSCS extended-image documentation](https://docs.cscs.ch/software/alps-extended-images/).
`existing.toml` is a separate inspection-only recipe for the older
`/users/dmoi/pygmk3.sqsh` with its existing cuda12 OFI hook. Do not mix these
networking branches; that image is not yet validated against current pins.

One Slurm task per node runs four torchrun workers, with zero automatic restarts.
Lightning explicitly uses TorchElasticEnvironment to avoid SLURM task-count
misidentification. Model construction happens on CPU before Lightning assigns
CUDA devices. The user account is `dmoi`, project `a0117`.

After credit and environment verification, prepare paths and logs before sbatch:

```bash
export TRAIN_WORKDIR=/capstor/store/cscs/swissai/a0117/foldtree2/se3/code/REVISION
export EDF_PATH="$TRAIN_WORKDIR/alps/se3/extended-pinned.toml"
export TRAIN_VENV=/capstor/store/cscs/swissai/a0117/foldtree2/se3/environments/REVISION
export SOURCE_CACHE=/capstor/store/cscs/swissai/a0117/foldtree2/se3/cache/bundle_4096
export TARGET_CACHE=/iopsstor/scratch/cscs/dmoi/se3/bundle_4096
cd "$TRAIN_WORKDIR"
mkdir -p logs
# CREDIT is the freshly checked portal balance, not a guessed allocation.
python -m scripts.se3_budget --ledger /capstor/store/cscs/swissai/a0117/foldtree2/se3/budget.jsonl \
  --stage prepare_initial --available-credit "$CREDIT" --submit \
  sbatch --account=a0117 --nodes=1 --time=60 alps/se3/prepare.sbatch
```

Once preparation passes, submit stages sequentially with the same budget wrapper:

```bash
sbatch --account=a0117 --partition=debug --nodes=1 --time=30 \
  alps/se3/train.sbatch --config configs/clariden_se3/debug_1nodes.yaml
sbatch --account=a0117 --partition=debug --nodes=2 --time=15 \
  alps/se3/train.sbatch --config configs/clariden_se3/debug_2nodes.yaml
sbatch --account=a0117 --partition=normal --nodes=1 --time=240 \
  alps/se3/train.sbatch --config configs/clariden_se3/small.yaml
sbatch --account=a0117 --partition=normal --nodes=1 --time=240 \
  alps/se3/train.sbatch --config configs/clariden_se3/medium.yaml
```

The direct sbatch examples show framework/resource arguments; the budget wrapper
is the submission route. Inspect logs and `sacct` after every stage. Four-node
checks and production runs remain gated. No automatic requeue or repeated retry
is configured. Announce each allocation's nodes, GPUs, partition, walltime and
maximum GPUh before submitting.

## Gates and maximum budget

| Stage | Nodes | Walltime | Maximum GPUh |
| --- | ---: | ---: | ---: |
| Initial environment/cache preparation | 1 | 1 h | 4 |
| Expanded cache staging, after capacity win | 1 | 1 h | 4 |
| One-node debug | 1 | 30 min | 2 |
| Two-node debug | 2 | 15 min | 2 |
| Four-node debug, conditional | 4 | 10 min | 2.67 |
| Small capacity control | 1 | 4 h | 16 |
| Medium capacity trial | 1 | 4 h | 16 |
| Large capacity trial, currently blocked | 1 | 4 h | 16 |
| Expanded two-node run | 2 | 6 h | 48 |
| Expanded four-node continuation | 4 | 8 h | 128 |

Maximum scheduled allocation is 238.67 GPUh, with 17.33 GPUh reserve, capped at
256 GPUh. Reservations conservatively count full walltime, even on failure.
The ledger locks across concurrent clients and refuses duplicate stage
reservations, ambiguous resource overrides, insufficient credit or ceiling
violations. Corrections require explicit reconciliation; no retry loop exists.
CSCS charges four GPUh per GH200 node-hour and exposes credit through the portal;
see [ML platform policies](https://docs.cscs.ch/platforms/mlp/policies/).

Small and medium trials start fresh, train the identical 4096/512 cohort for
exactly 16 epochs / 4096 updates, LR 5e-4 constant, global batch 16. A walltime
cutoff before completion makes the matched comparison incomplete. Capacity
selection requires at least 1% lower validation geometry, improved clamped FAPE,
and no error metric rising over 5% or TM-score/lDDT declining over 0.01. Use:

```bash
python -m scripts.se3_capacity_gate CONTROL_RUN CANDIDATE_RUN --output gate.json
```

Stop expansion if medium fails. The six-layer run also requires a new passing
numerical gate; its configuration is retained to document intended parameters,
not to authorize launching a known failing model. Do not silently reduce depth,
sanitize gradients or change losses to bypass a gate.

After a capacity win, prepare expanded frozen outputs locally on an idle GPU:

```bash
python -m scripts.prepare_se3_expanded_cohort \
  --config configs/clariden_se3/prepare_expanded_local.yaml --capacity-gate gate.json
```

This retains all 4096 training IDs and adds 4096 fresh training and 1024 new
validation IDs, excluding all previous diagnostic and validation IDs from new
additions. Insufficient eligible structures fail without changing the 256-residue
cap. Export and stage the expanded bundle exactly as above. Warm-start the winning
architecture's weights with a fresh optimizer. Expanded configs share a 30-epoch
horizon; use `--stop-after-epoch 10` for the two-node leg, then resume at four nodes
only after the validation and scaling gates pass. Scheduler factor .5/patience 3,
validation early stopping patience 8/min_delta .001 are preserved across legs.

Benchmark the same checkpoint and global batch stream at two and four nodes,
with 20 warmup and 100 measured updates (`--stop-after-updates 120` in a separate
output). `report_se3_scaling.py` requires the complete window. Four-node efficiency
must reach 60% relative to two nodes. `--profile` writes per-rank PyTorch traces;
update/checkpoint events record walltime, data wait, lengths, gradients, host and
GPU memory. Epoch events report globally reduced training geometry and consumed
structure counts; partial-epoch statistics survive resume. Failures create
per-rank JSON records and propagate to the launcher. GPU utilization and NCCL transport still require allocated-node
monitoring. Small local smoke measurements are not scaling evidence.

## Recovery

Atomic `last.ckpt` publication occurs every 64 consumed updates and at epoch end
(the final batch waits for complete-epoch validation before publication);
`best.ckpt` is published only after complete-epoch validation improves. All ranks
participate in RNG gathering/checkpoint barriers. Checkpoints contain model,
optimizer, scheduler, each rank's torch/CUDA/Python/NumPy RNG, logical epoch,
optimizer update count, consumed global-batch cursor and immutable provenance.

```bash
python -m foldtree2.train_se3_lightning --config CONFIG --resume-from RUN/last.ckpt
python -m foldtree2.train_se3_lightning --config CONFIG --warm-start RUN/best.ckpt
```

Resume starts a fresh Lightning iterator over the remaining stream and explicitly
restores optimizer/scheduler/RNG rather than replaying prefetched iterator state.
Ordinary resume rejects code, dataset/cache, split, architecture, training config
and topology changes. Paths can move without changing content identity.
`--allow-topology-change` permits an epoch-boundary transition with the same
fixed global batch; new rank RNG is seeded and the transition recorded. It does
not claim bitwise equivalence across topologies. A mid-epoch topology change is
rejected. `--warm-start` resets training state and requires compatible model
weights; it is separate from full resume.

## First cluster submission and packaging correction

On 2026-10-13, preparation job `3652348` failed after 305 seconds because the
release archive omitted `alps/se3/requirements.txt`: the repository's generic
`*.txt` ignore rule excluded it from the commit. Cache staging and container
startup succeeded; the base image reported torch `2.11.0a0+eb65b36914.nv26.2`.
The dependent checks `3652401`, `3652402` and capacity jobs `3652424`, `3652426`
were canceled with empty allocated TRES and never consumed GPU allocations.
Training has not started in this attempt.

The correction explicitly tracks the dependency file, checks for it before
preparation, and adds an archive completeness verifier with a regression test
that reproduces the missing-file failure. Use `verify_se3_release.py` on every
new deployment archive before transfer/submission. The first attempt's logs and
accounting remain preserved. One corrected one-hour preparation allocation
uses 4 GPUh of the agreed reserve. Canceled-before-allocation reservations are
released only after recording Slurm proof; this permits rebuilding the same
bounded dependency chain without losing its audit history.

## Successful cluster startup

Deployment `f625bf1` reuses the fully installed ARM environment from `028cc75`.
The inspector runs with `python -m alps.se3.inspect_environment`, keeping the
repository in Python's module path. Environment verification and real CUDA DDP
training succeeded in job `3653704` (one node/four GPUs, 159 seconds), including
checkpoint resume from update 2 through update 4. Job `3653761` passed on two
nodes/eight GPUs in 88 seconds. Both counted every one of the 512 holdouts once;
their update-2 validation geometry differed by approximately `6e-9`.

The initial two-node submission was rejected by Slurm's active-job submit limit;
its reservation was released with no job allocated, then submission succeeded
after the one-node check finished. Completion checks for the capacity trials run
inside the container, where the prepared venv's interpreter and system packages
are available.

Small trial `3653793` is running and has passed 246 finite optimizer updates at
the recorded observation. Medium trial `3653794` is pending its successful
completion. Each trains the identical 4096/512 cohort for 16 epochs / 4096
updates, with a four-hour ceiling. The medium job computes the paired capacity
gate after training; no data expansion or six-layer trial is automatically queued.
Logs live under the revision-specific `logs` directory on project store;
checkpoints/events/paired validation are under `se3/runs/small` and
`se3/runs/medium`. See the cluster startup evidence archive for exact observations.

The ledger conservatively reserves 44 GPUh for the current sequence, including
the full ceilings of both failed preparation jobs; canceled/unsubmitted jobs
have auditable zero-allocation releases. The total rollout ceiling remains
256 GPUh. The extra one-hour preparation correction uses 4 GPUh of the agreed
17.33 GPUh reserve; actual failed preparation runtime totaled 357 seconds.
