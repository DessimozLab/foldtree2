# Bend supervision, pairs-only recycling and longer training

The user authorized levers 1/2/3, followed by more data and model capacity only
if they work. This experiment retains continuous pair information, removes the
unproven learned seed, and tests direct coordinate-derived bend supervision
alongside one/two shared passes and eight additional training epochs.

| Arm | Passes | Bend weight | Additional epochs |
|---|---:|---:|---:|
| `bend_recycle_v2_one` | 1 | 0 | 8 |
| `bend_recycle_v2_two` | 2 | 0 | 8 |
| `bend_recycle_v2_bend05` | 2 | 0.5 | 8 |
| `bend_recycle_v2_bend20` | 2 | 2.0 | 8 |

Every arm starts from the same trained pairs-only checkpoint, SHA256
`3006316349da3605336ccd516f79c1eb1d5afebbfd66c386ee23ea855072548b`,
from the [previous comparison](se3_pairs_seed_results_20261015.md). It already
completed four epochs; these are eight further epochs with a fresh AdamW
optimizer and sample stream, not an exact optimizer continuation. All arms
retain LR 5e-4, global batch 16, seed/order 42, width 64/depth 4, heads 4, head
dimension 16, max length 256, 8,192 training / 512 validation IDs, float32
forward/backward and float64 accumulation/reduction. There are 4,096 updates
per arm. Data, production weights and graph generation are unchanged.

The existing `balanced_geometry_v1` loss directly penalizes SmoothL1 error of
consecutive bond-direction cosine, with target-selected contiguous CA stencils
(both target bonds 2.5–4.5 Angstrom). It avoids unstable acos gradients. Existing
FAPE/distance/bond/angle terms retain their weights; local-distance and torsion
extra terms remain disabled. Bend weight zero follows the original loss
operation order exactly. The coordinate-derived bend evaluation in radians is
unchanged, including its existing masks; this differs from the angle-head loss.

## Provenance and stability

No training implementation or migration guard was relaxed. The first allocated
check stages explicit **weights-only** initialization artifacts for one and two
passes, using the existing reviewed refinement migration checker. The model
state is identical tensor-for-tensor to the trained parent, and the record
explicitly states that the inherited weights were trained for one pass. These
artifacts carry no optimizer/resume state. Their hashes and parent chain persist
under `artifacts/bend_recycle_v2_initialization/manifest.json`.

Each training arm then crosses only the existing explicit loss-ablation boundary
with `--allow-loss-change`. Loss and pass-count changes remain separate verified
operations. All four arms get exact one-node resume checks and two-node model,
optimizer and validation agreement checks before longer training. A failure
cancels downstream jobs. There are no automatic retries.

Local validation: 50 tests passed across new staging/gradient/quality/submission
checks and existing architecture/recycling/balanced losses. The new tests verify
that actual bend supervision changes coordinate gradients without changing
evaluation FAPE, strict state equality during pass migration, rejection of
incompatible model changes, matched check configs, resource/dependency limits,
and rejection of bend regression or mismatched masks by the scale-up gate.

## Resources and submission

The pinned existing extended container, ARM environment, account `a0117` and
normal partition are reused. `nid007494` remains excluded. One-node checks:
30 minutes / 2 GPUh. Two-node checks: 20 minutes / 2.667 GPUh. Four training jobs:
two nodes / eight GPUs / 60 minutes each, 8 GPUh apiece. Final CPU quality decision
runs inside the container on one allocated node for at most five minutes,
0.333 GPUh. Total maximum **37 GPUh**; rollout reservations rise from 184.667 to
221.667 / 256 GPUh, retaining **34.333 GPUh** for conditional expansion.
Project credit uses the user's prior sufficient-credit confirmation, not a
fresh portal balance or Slurm fair-share value.

```bash
python3 -m scripts.submit_se3_bend_recycle --available-credit 37
python3 -m scripts.submit_se3_bend_recycle --available-credit 37 --submit
```

All commands pass `sbatch --test-only` before any reservation. Jobs form a
sequential `afterok` chain. Outputs: `.../se3/runs/bend_recycle_v2_*`;
receipts/gates: `.../se3/artifacts/bend_recycle_v2_*`; allocation ledger:
`.../se3/budget.jsonl`, under `/capstor/store/cscs/swissai/a0117/foldtree2`.
The new scripts/config dependencies are covered by the release verifier.

## Decision before more data or capacity

The final job records `artifacts/bend_recycle_v2-quality.json`. It verifies all
four complete eight-epoch trials and all eight stability reports. For each
candidate, it evaluates the best-FAPE checkpoint on the pinned 512 IDs, checks
identical raw baselines and residue counts, and compares with the matched longer
one-pass control. It requires:

* Paired FAPE bootstrap 95% upper bound below zero versus the longer control
  (10,000 structure resamples, RNG seed 42).
* FAPE at least as good as the previous best combined two-pass value 0.924550214.
* CA bend error within 5% of the original pre-pair control: at most 0.752317 rad.
  Other error metrics must remain within 5% of both original and matched control.
* TM-score/lDDT lose at most 0.01 against original, previous-best and matched
  control scores; finite metrics and no failure artifacts.

This is a development screening decision conditional on checkpoint/arm selection,
not a fresh confirmation test. The winner, exact checkpoint hash, paired metrics
and eligibility flag are written even if no candidate passes. Failed quality
screens preserve results and **do not** authorize scale-up. The quality job never
submits a larger run merely because training exited successfully.

Step 5 has standing conditional user authorization. Once eligible results exist,
prepare more frozen structures and a larger-width model as separate bounded
comparisons, with fresh confirmation holdouts and their own CUDA/resume/DDP
checks, using the remaining 34.333 GPUh. The current staged inputs contain the
old cohort only; expanding them and validating a larger model must precede
submission. No larger-data/model job is queued before this quality decision.
No production replacement or branch merge occurs automatically.

Access/launch follow the [Clariden skill](/home/dmoi/.codex/skills/clariden/SKILL.md).
Live normal-partition limits were rechecked against
[Clariden documentation](https://docs.cscs.ch/clusters/clariden/).
