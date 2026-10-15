# Clariden pair/seed/recycling results — 2026-10-15

Continuous decoder pairs improve validation quality, and the combined two-pass model has the best FAPE. The learned seed alone is neutral. **The merge gate fails because CA bend-angle error regresses by about 28%.** No merge, new training submission or automatic budget expansion was performed.

All seven jobs completed with exit 0. Each training arm processed 8,192 structures for four epochs / 2,048 updates, with 512 disjoint structure-level validation IDs. Ten per-variant recovery/topology checks passed; model/BatchNorm maximum difference across topologies was zero, and new pair/seed heads updated. No failure artifacts were found. Production components remained frozen through cached inputs.

Code was pinned to `7503bfbf4bfff9c6c040616f4971d803e1762e03`. All arms used the original common loss, original growth epoch-16 parent, identical split/order/seed, LR, optimizer reset and global batch. The control reproduces the prior loss-trial control FAPE. Best checkpoints were selected independently by minimum validation FAPE. Epochs below are one-based.

## Best validation checkpoints

| Variant | Epoch | FAPE ↓ | CA RMSD Å ↓ | TM-score ↑ | lDDT ↑ | CA bond MAE Å ↓ | CA bend MAE rad ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control | 3 | 0.929860 | 15.850 | 0.2130 | 0.2606 | 0.903 | 0.716 |
| Pairs | 4 | 0.926776 | 14.921 | 0.2444 | 0.2650 | 0.305 | 0.927 |
| Learned seed | 1 | 0.929923 | 15.690 | 0.2142 | 0.2553 | 0.927 | 0.712 |
| Pairs + seed | 4 | 0.926326 | 14.949 | 0.2387 | 0.2713 | 0.312 | 0.924 |
| Pairs + seed, two passes | 4 | 0.924550 | 14.913 | 0.2532 | 0.2852 | 0.252 | 0.917 |

The best two-pass model improves FAPE by 0.005310 (0.57%), RMSD by 0.937 Å (5.9%), TM-score by 0.04026, lDDT by 0.02455 and CA bond MAE by 0.651 Å (72.0%) relative to control. However, CA bend error rises from 0.71649 to 0.91660 rad (+27.9%). Pair-only and combined single-pass models also regress bend error, so this tradeoff is already present without recycling. Torsion error changes from 1.52180 to 1.54105 rad (+1.3%); predicted angle-head MAE improves from 0.51448 to 0.50571 rad. Angle-head accuracy and coordinate-derived bend geometry are separate measurements.

The raw frozen-decoder baseline has FAPE 0.942045, CA RMSD 19.152 Å, TM-score 0.07916 and lDDT 0.02072. Its angle-head metric is undefined. Every arm used identical IDs, valid-residue counts and raw baseline records; the comparison script checked these exactly.

## Paired structure comparisons

Differences are candidate minus control; negative FAPE is better. Paired percentile bootstrap: 10,000 resamples of 512 structures, NumPy RNG seed 42. Intervals are pointwise/descriptive and conditional on selected best checkpoints, without adjustment for checkpoint/arm selection.

| Variant | Mean FAPE difference | 95% bootstrap interval | Structures improved / 512 |
|---|---:|---|---:|
| Pairs | -0.003084 | [-0.003758, -0.002445] | 349 |
| Learned seed | +0.000063 | [-0.000404, +0.000429] | 202 |
| Pairs + seed | -0.003534 | [-0.004146, -0.002966] | 390 |
| Pairs + seed, two passes | -0.005310 | [-0.006455, -0.004388] | 397 |

Adding the learned seed to pairs changes FAPE by −0.000450, interval [−0.000849, −0.000076], but lowers mean TM-score versus pairs alone. The learned seed by itself shows no clear FAPE benefit. Adding the second pass to the combined model changes FAPE by −0.001775, interval [−0.002603, −0.001165]. These are useful development signals, not independent replications or a compute-matched comparison. A pairs-only two-pass arm was not included, so the best model does not establish that learned seeding is necessary when recycling.

Absolute structural quality remains limited (best TM-score 0.253); the 512 structures are repeatedly monitored development holdouts. Fresh confirmation structures and multiple seeds are needed before promotion. Independence from production-model pretraining is not claimed.

## Learning and resources

Training geometry by epoch (same objective in every arm):

| Variant | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
|---|---:|---:|---:|---:|
| Control | 3.6199 | 3.6082 | 3.6017 | 3.5911 |
| Pairs | 3.3020 | 3.1459 | 3.0948 | 3.0561 |
| Learned seed | 3.6248 | 3.6136 | 3.6061 | 3.5959 |
| Pairs + seed | 3.2968 | 3.1414 | 3.0869 | 3.0518 |
| Pairs + seed, two passes | 3.3055 | 3.1202 | 3.0401 | 2.9943 |

Validation FAPE by epoch:

| Variant | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
|---|---:|---:|---:|---:|
| Control | 0.930226 | 0.930402 | 0.929860 | 0.930054 |
| Pairs | 0.929550 | 0.928083 | 0.927629 | 0.926776 |
| Learned seed | 0.929923 | 0.930097 | 0.929924 | 0.930093 |
| Pairs + seed | 0.929924 | 0.928318 | 0.927403 | 0.926326 |
| Pairs + seed, two passes | 0.931369 | 0.927051 | 0.925919 | 0.924550 |

All jobs used the normal partition, account `a0117`, the existing pinned extended container and Lightning DDP with float32 model computation / float64 accumulation. Jobs 3664373–3664374 are the stability checks; 3664375–3664379 are the five training arms in table order.

| Variant | Job walltime | Allocated GPUh | Median structures/s | Rank-0 peak allocated GPU GiB |
|---|---:|---:|---:|---:|
| Control | 14m 30s | 1.933 | 49.94 | 4.17 |
| Pairs | 14m 26s | 1.924 | 49.69 | 4.21 |
| Learned seed | 15m 08s | 2.018 | 47.13 | 4.20 |
| Pairs + seed | 15m 24s | 2.053 | 45.74 | 4.25 |
| Pairs + seed, two passes | 24m 52s | 3.316 | 26.10 | 8.28 |

Actual allocation usage, including stability checks, was **13.128 GPUh**, below the 34.667 GPUh reservation. The conservative rollout ledger remains at 184.667 / 256 GPUh; no reservations were silently released. Throughput is the trainer’s logged global structures per optimizer-update second, excluding validation/checkpoint/startup costs; memory is rank-0-reported allocated peak, not a maximum across all ranks. The two-pass run takes 1.71× control walltime and about 1.91× median update time.

## Decision

Keep the architecture branch separate. Pair conditioning is worth retaining for further experiments; learned seeding alone is not an established lever. The existing >5% structural-error-regression bound rejects the bend-angle increase, despite improved FAPE, RMSD, TM-score and lDDT. Do not merge or scale automatically.

The next bounded comparison should preserve the pair signal while addressing coordinate-derived bend geometry, and include pairs-only with two passes to isolate recycling from seed changes. Inspect bend-error distributions and short-range distances before choosing loss weights or a constrained coordinate update. A separate compute-matched control is needed for efficiency claims. No next run has been submitted.

## Retained evidence

* [Metric summary, bootstrap intervals and resource accounting](se3_clariden_evidence/pairs_seed_v1/results/summary.json).
* [Per-structure best-checkpoint metrics, checkpoint hashes, provenance, software and learning curves](se3_clariden_evidence/pairs_seed_v1/results/paired_metrics.json).
* [Training events, gate results and check histories](se3_clariden_evidence/pairs_seed_v1/results/gates_and_training.json).
* [Pinned submission receipts and resource requests](se3_clariden_evidence/pairs_seed_v1/submission.json).

The best combined two-pass checkpoint is `/capstor/store/cscs/swissai/a0117/foldtree2/se3/runs/pairs_seed_v1_pairs_seed_two/best.ckpt`, SHA256 `069c88ce38e92d59ebb2c3ef2f560a8645dcd541dee920e25800b03d0265cee6`. All variants’ hashes are retained in the summary. Access/monitoring followed the [Clariden skill](/home/dmoi/.codex/skills/clariden/SKILL.md).
