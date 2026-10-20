# SE(3) decoder handoff — 2026-10-20

## Start here

The residue-level SE(3) refiner runs and learns reproducibly on Clariden. Keep the
trained **width-64, two-pass, pair-informed, bend-weight-0.5** model as the
incumbent. The later fresh width-80 experiment completed but did not pass the
quality thresholds. The next investigation is objective balance, not another
width increase. Clariden SSH currently fails on the second hop; training and
access are separate issues.

This is a dated handoff, not a live scheduler/credential status. The latest
successful job/metric queries were on October 20; later access diagnosis could
reach Ela but not execute commands on Clariden. No new training was queued for
the diagnosis. No production checkpoint was replaced, and the architecture
branch has not been merged.

## Checkout and change boundaries

| Purpose | Location / revision |
|---|---|
| Active architecture worktree | `/tmp/foldtree2-se3-recycling` |
| Architecture branch | `se3-recycled-refiner`, pushed to `origin` |
| Latest analysis/source revision before this handoff | `93ee35d` |
| Pinned code used for completed capacity recovery | `9b661111538f9288f1eb86a8dc4f78d422efca9c` |
| Main checkout | `/home/dmoi/projects/foldtree2`, branch `dev` |
| Earlier Lightning/loss worktree | `/tmp/foldtree2-clariden`, branch `se3-clariden-lightning` |

The main checkout has many unrelated tracked/untracked edits. Preserve them;
work on the architecture branch or create an isolated worktree from it. Do not
infer that `dev` has the current SE(3) implementation just because it has this
standalone handoff. Paths to code below refer to the architecture branch.
The `/tmp` worktrees and untracked data are machine-local; Git commits are the
portable source of code. Preserve large local artifacts before cleaning `/tmp`.

The user authorized bounded Clariden experiments and previously requested
commit/push/documentation. Conditional scale-up authorization was: “do 1,2 and
3. then if that works do 5”; the bend/recycling gate passed. That is not unlimited
credit or authorization to ignore quality/provenance gates. Original reservation
ceiling remains **256 GPUh**. Keep the separate staged non-SE(3) transformer,
production encoder/decoder and alphabet benchmark results unchanged. Atom-level
refinement and full-data production rollout remain deferred.

## Frozen inputs and current model

Both production components are the matched **40-character epoch-40 pair**, frozen
with no gradients. Input graph tensors come from the existing mk2 HDF5; do not
regenerate graphs or substitute older hardcoded epoch-41 models.

| Identity | SHA256 |
|---|---|
| Encoder | `75d176f4dd11fa1e676c536be953b78a767c40c405b425a49182192fbb6a7fc6` |
| Decoder | `b9642afd20db829a5ce2dd1c908c86df0c351319347fccd2ad1d335b3aa9089d` |
| mk2 HDF5 | `2c970702701a5bf789dd46974ff689f02c0805dcf146746db8d1eba07a2b3673` |
| Existing 8,192/512 frozen bundle | `b2c635e8cf30af851a63eb8f0836ff090fb5707b51523a6b709cca8b707877f3` |

Current validated experiment settings:

- Residue coordinates from the production bottleneck; latent plus codebook
  features. Target coordinates are loss/evaluation inputs only.
- GotenNet-based refiner, hidden width 64, depth 4, four heads, head dimension 16.
  Width 80/depth 4 is executable but not a quality winner. Depth six remains
  blocked by known numerical failures.
- Masked centering and rotation-invariant RMS-radius normalization; restore
  physical units/translation. Residual coordinate gain 32; zero-initialized
  coordinate update; sin/cos angle head.
- Fixed normal lift of 0.1 RMS radius for initial geometry. Learned seed geometry
  was explored but did not pass the bend-quality screen; it is not the incumbent.
- Twelve invariant pair channels: decoder contact cosine, bounded calibrated
  logit/probability, three sequence-separation channels, six current-distance
  RBFs. Calibration temperature `2.4314587116241455`, bias `-1.3124585151672363`.
- Two shared-weight refinement passes, rebuilding geometric contacts/pairs each
  pass. Full autograd through recycling; final-pass loss weights `[0.0, 1.0]`.
- Loss protocol `balanced_geometry_v1`: geometry = distance + smooth training
  FAPE + bond + 0.5 × bend; total adds 0.1 × angle. Local-distance and torsion
  terms are available but have zero weight in these capacity trials.
- Smooth training FAPE uses a pseudo-Huber error with a nonzero tail; evaluation
  FAPE retains the historical clamp. They are different quantities.
- Float32 forward/backward; float64 gradient accumulation, stable tensor norm,
  tensor feed-forward input normalization. Keep safeguards local to the refiner.
  Do not restore global dtype/backend mutations or silent gradient sanitization.
- Seed 42, global batch 16, maximum 256 residues, LR `5e-4`, no scheduler in the
  fresh capacity trials. Two nodes × four GPUs, Lightning DDP. These evolved
  workstation/HPC settings supersede the original small smoke configuration.

The inherited frozen-cache extractor identity records a 384-residue extraction
cap; the selected training cohorts/configurations enforce 256 residues. Preserve
that identity as recorded; do not rewrite it to match the later cohort cap.

## Results and checkpoint to retain

**Incumbent:** `/capstor/store/cscs/swissai/a0117/foldtree2/se3/runs/bend_recycle_v2_bend05/best.ckpt`

SHA256: `ec74e62f44ae58cc4d99ac977e88fb1795dabec8f3e2e342c01d8930885d1145`.

The eight-additional-epoch bend/recycling comparison passed all stability gates
and its predeclared quality gate. At the selected incumbent checkpoint:

| FAPE ↓ | aligned CA RMSD Å ↓ | TM ↑ | lDDT ↑ | Bend MAE rad ↓ | Bond MAE Å ↓ |
|---:|---:|---:|---:|---:|---:|
| 0.920546 | 14.858 | 0.26749 | 0.36311 | 0.37864 | 0.23001 |

Versus the matched longer one-pass control, FAPE improved on 376/512 structures;
paired mean difference −0.003948, bootstrap 95% interval
[−0.005361, −0.002863]. RMSD was 2.56% worse, within the declared 5% tolerance.
The model had a warm-start history; it is not comparable to a fresh four-epoch
run as an isolated width test.

Latest **fresh**, matched capacity trials used 8,192 training / 512 development
structures, four epochs and 2,048 updates each:

| Job | Purpose | Last verified result |
|---:|---|---|
| 3700111 | One-node exact-resume checks, both widths | Completed; bitwise agreement |
| 3700112 | One-node/two-node agreement, both widths | Completed; max state difference 0 |
| 3700114 | Fresh width 64 | Completed, 24m46s; no recorded training failures |
| 3700115 | Fresh width 80 | Completed, 24m47s; no recorded training failures |

| Width | Train geometry, epoch 1 → 4 | Best dev epoch | FAPE ↓ | Bond MAE Å ↓ | Bend MAE rad ↓ |
|---|---|---:|---:|---:|---:|
| 64 | 4.3230 → 3.1054 | 3 | 0.926371 | 0.29524 | 0.40857 |
| 80 | 4.3373 → 3.1836 | 4 | 0.926102 | 0.36154 | 0.45975 |

Width 80 reduces selected-checkpoint mean geometry loss by only 0.315% (required:
≥1%), while bond/bend errors worsen 22.45% / 12.53% (allowed: ≤5%). Its FAPE
advantage is only 0.000269. Aggregate thresholds already reject promotion.
Per-structure capacity comparisons, paired confidence intervals and new
checkpoint hashes still need retrieval; do not claim statistical significance.

Recorded validation components show a possible loss tradeoff: width-64 bond
loss drops 0.7877 → 0.0615, distance drops 1.2267 → 0.9118, but smooth training
FAPE rises 1.8865 → 2.0109. Width 80 shows the same direction. This is not proof
of a faulty FAPE implementation, gradient conflict or clamp saturation. Measure
those directly before changing loss weights.

These are repeatedly monitored structure-level refiner development holdouts.
They are not evidence of independence from production-model pretraining, and
checkpoint-selected bootstrap intervals are not fresh confirmation.

## Data expansion: prepared locally, not trained

Completed local export:
`/tmp/foldtree2-se3-recycling/runs/se3_scaleup_v3/inputs_v3`.

- 10,240 training IDs: all old 8,192 plus 2,048 additions.
- Original 512 development IDs retained; 512 disjoint fresh confirmation IDs.
- 2,561 graph exports: new training + confirmation + one parity structure,
  approximately 2.4 GiB; serialized encoder/decoder and manifest alongside them.
- Export SHA256:
  `2f45d9c5a8b35ef3adb6bf1dcd1b2ec69b89017adf9a5f1cf52c2b66b8d1a617`.
- Logs: `runs/se3_scaleup_v3/prepare_v3.log`; previous failed 16k selection in
  `inputs_v2` / `prepare_v2.log` is preserved. Too few unused structures pass
  filters for 16k; do not relax geometry/length eligibility to force that count.

No expanded frozen-output cache has been extracted/published on Clariden, and
no expanded-data training has been submitted. Confirmation IDs must remain
outside validation/checkpoint selection.

The old `scripts/extract_se3_growth_clariden.py` / growth validation contains
4,096→8,192 assumptions. Do not run it unchanged for this 10,240 + confirmation
extension. Adapt it to explicit added IDs, preserve all retained cache entries
bitwise, verify hashes and frozen parameters, and keep same-runtime parity
strict. Any previously authorized bounded cross-runtime audit is separate from
same-runtime repeatability; do not silently widen tolerances.

A matched data comparison would use five old-data epochs versus four expanded
epochs: 40,960 presentations / 2,560 updates each at global batch 16, starting
from the same incumbent checkpoint and preserving the development split.

## Clariden operations and current access blocker

Read the `clariden` skill at `/home/dmoi/.codex/skills/clariden/SKILL.md`, its
personal-access reference, and `/home/dmoi/cscs-access/{AGENTS.md,README.md}` before
access work. Use `/home/dmoi/.local/bin/cscs-access`; check `status` first. The
saved working MFA reference is already configured. Never display credentials,
OTP values, key contents or token caches. Do not switch MFA entries or recreate
working keys to diagnose this connection closure.

Last access diagnosis: the certificate was valid until October 21, 20:50:31 UTC.
Ela accepts it and executes commands. The Clariden hop reaches SSH and accepts
the certificate offer, then closes before successful authentication/session
confirmation. An IPv4 jump route behaved the same. This isolates the affected
path but not the server-side cause. No outage was established and no support
message was sent. Recheck actual status/expiry rather than treating this dated
certificate observation as current. Existing UNIL VPN/Curnagl jobs are separate;
do not disconnect their shared VPN as an SSH experiment.

| Resource | Location / setting |
|---|---|
| SE(3) root | `/capstor/store/cscs/swissai/a0117/foldtree2/se3` |
| Frozen 8,192 cache | `/iopsstor/scratch/cscs/dmoi/se3/bundle_growth_8192` |
| Pinned capacity source | `<root>/code/9b661111538f9288f1eb86a8dc4f78d422efca9c` |
| Training environment | `<root>/environments/028cc752968f218de4e5837f75b2e7a8a004162a` |
| Capacity run directories | `<root>/runs/scale_capacity_v3r1_width64`, `..._width80` |
| Logs | `<pinned source>/logs/scale_capacity_v3r1_*-<job>.out` |
| Per-run evidence | `events.jsonl`, `validation-<zero-based epoch>.json`, `best.ckpt`, `last.ckpt` |
| Receipts / gates | `<root>/artifacts` |
| Reservation ledger | `<root>/budget.jsonl` |
| Slurm resources | account `a0117`, partition `normal`, four GPUs/node; exclude `nid007494` |
| Pinned container | `ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch@sha256:3dd0d01a02e58d3c78e4222ce252bd8d177c14392b3d1a05a1f4c9a92454ae40` |

Last conservative ledger: **237.666667 / 256 GPUh**, leaving **18.333333 GPUh**.
Credit was confirmed by the user earlier; this is not a freshly read portal
balance. Keep charged reservations and receipts; do not refund failed/unused
rows or repeat one-time recovery controllers. No new allocation was made during
the latest loss/access diagnosis.

Original capacity chain 3675417–3675420 failed at a stale width-80 CLI allowlist;
width 80 never initialized in that attempt. Three dependent jobs never started.
Recovery 3700111/12/14/15 fixed the CLI and completed. Do not misclassify the old
CLI failure as a width-80 numerical failure or rerun either submission command.

## Code map, recovery constraints and verification

| Code | Role |
|---|---|
| `foldtree2/train_se3_lightning.py` | CLI, Lightning training, provenance, stream/RNG/optimizer recovery |
| `foldtree2/se3_cache.py`, `se3_validation.py` | Frozen entries, matched masks, physical geometry and metrics |
| `foldtree2/se3_gradient_accumulation.py` | Float64 accumulation |
| `foldtree2/src/se3_struct_decoder.py`, `pair_gotennet.py` | Equivariant refiner and dense pair adapter |
| `foldtree2/src/pair_features.py`, `recycled_refiner.py` | Invariant pairs and shared two-pass refinement |
| `foldtree2/src/geometry_seeds.py`, `learned_geometry_seed.py` | Fixed and experimental learned seeds |
| `foldtree2/src/losses/se3_ablation.py`, `se3_balanced.py` | Smooth training FAPE and weighted geometry terms |
| `scripts/se3_capacity_gate.py` | Corrected paired comparison and declared-schedule checks |
| `scripts/gate_se3_bend_recycle.py` | Incumbent selection and conditional scale-up gate |
| `configs/clariden_se3/scale_capacity_v3r1_width*.yaml` | Actual completed fresh width-trial configs |
| `alps/se3/train.sbatch`, `extended-pinned.toml` | Validated distributed launcher/container |

`--resume-from` requires exact training/cache/source provenance and restores
optimizer, RNG and stream state. A topology change needs explicit opt-in at an
epoch boundary. `--warm-start` is weights-only and resets optimizer/stream.
Explicit loss/refinement migrations have restricted source allowlists; they do
not permit arbitrary source changes. In particular, the CLI fix changed the
trainer source hash. Continuing the older incumbent using newer code needs its
compatible pinned source or a reviewed migration; do not disable provenance
checks to make it load. Change loss and architecture in separate experiments.

The latest comparison fix is analysis-only (`93ee35d`): the old reporter demanded
16 epochs / 4,096 updates; it now validates the actual declared schedule,
complete sample counts, failure artifacts/events, best/last provenance, paired
IDs, raw baselines, metric eligibility and finite values. Original quality
thresholds remain unchanged. Run it after downloading completed artifacts:

```bash
cd /tmp/foldtree2-se3-recycling
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
python -m scripts.se3_capacity_gate /path/to/scale_capacity_v3r1_width64 \
  /path/to/scale_capacity_v3r1_width80 --output /path/to/comparison.json
```

Latest focused verification: nine capacity-analysis tests and six existing
capacity tests pass. Earlier Lightning checks: 12 passed, one known expected
failure for the blocked six-layer configuration. These are historical scoped
results, not a fresh claim that the whole repository suite passes.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-se3 python -m pytest \
  foldtree2/tests/test_se3_capacity_gate.py \
  foldtree2/tests/test_se3_scale_capacity.py -q
```

Other relevant tests: `test_se3_refiner.py`, `test_se3_validation.py`,
`test_se3_pairs_seed.py`, `test_se3_recycling.py`, `test_se3_balanced_loss.py`,
`test_se3_loss_ablation.py`, `test_se3_lightning.py`. Choose checks appropriate
to changed code. A restricted sandbox may hide NVIDIA devices: per `AGENTS.md`,
check CUDA outside that sandbox in the `foldtree2` environment before claiming
CUDA is unavailable. Do not interrupt unrelated local/cluster workloads.

## Next agent: ordered work

1. Recheck access with the maintained helper. Ela works in the recorded diagnosis;
   investigate the Clariden login/session path without credential cycling or
   changing trust. Preserve the connection evidence if server-side help is needed.
2. Retrieve full artifacts for both completed capacity runs, record checkpoint
   hashes and run the corrected paired analyzer. Keep aggregate failures visible;
   do not promote width 80 based on a small FAPE delta.
3. On a fixed, recorded set of real structures, audit coordinate/model gradient
   norms and cosine similarities by loss term, unclamped frame-error distribution,
   evaluation-clamp saturation, and metrics stratified by length/seed quality.
   These measurements are **not implemented or run yet**. Quantify the suspected
   objective tradeoff before changing loss weights; targets/masks must match.
4. Prepare one controlled continuation of the incumbent or finish the authorized
   old-versus-expanded data preparation. Preserve source compatibility and match
   updates. Fit extraction, checks and training inside the remaining reservation
   ceiling; state resources/cost before any authorized submission. Do not queue
   arbitrary new width/depth/duration sweeps.
5. Evaluate the selected checkpoint on the fresh 512 confirmation structures
   only after model/checkpoint selection. Preserve failed attempts, paired
   metrics, software/checkpoint/cache hashes, resource usage and persistent logs.

## Detailed records (architecture branch)

These pinned links also work from the standalone copy of this handoff on `dev`:

- [Incumbent bend/recycling results](https://github.com/DessimozLab/foldtree2/blob/93ee35d/docs/se3_bend_recycle_v2_results.md)
- [Current capacity/loss/access diagnosis](https://github.com/DessimozLab/foldtree2/blob/93ee35d/docs/se3_clariden_diagnosis_20261020.md)
- [Recorded capacity aggregates and limitations](https://github.com/DessimozLab/foldtree2/blob/93ee35d/docs/se3_clariden_evidence/scale_capacity_v3/completed_aggregates_20261020.json)
- [Capacity recovery, job receipts and budget](https://github.com/DessimozLab/foldtree2/blob/93ee35d/docs/se3_scale_capacity_recovery_20261019.md)
- [Data/capacity plan and export status](https://github.com/DessimozLab/foldtree2/blob/93ee35d/docs/se3_scaleup_v3.md)

Separate work: the Curnagl TCS benchmark has its own keep-going job/supervisor
and taxonomy-mismatch failure. Do not alter or cancel it while working on SE(3).
Its documentation is on `dev` in `docs/curnagl_raxml_affinity_20261019.md`.
