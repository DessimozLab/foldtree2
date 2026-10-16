# Bend/recycling follow-up results

All four eight-epoch trials completed with 4,096 updates each, no failure artifacts, and all eight recovery/topology gates passed. The predeclared quality screen **passed**, selecting **bend weight 0.5**. This activates the user’s conditional authorization for bounded data/capacity experiments, without an automatic production replacement or merge.

Best checkpoints selected by validation FAPE on the same 512 development structures:

| Variant | Epoch | FAPE ↓ | RMSD Å ↓ | TM ↑ | lDDT ↑ | Bend rad ↓ | Bond MAE Å ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Longer one-pass control | 8 | 0.924494 | 14.487 | 0.2566 | 0.2800 | 0.926 | 0.305 |
| Two passes | 6 | 0.921751 | 14.564 | 0.2598 | 0.2937 | 0.959 | 0.225 |
| Two passes + bend 0.5 | 8 | 0.920546 | 14.858 | 0.2675 | 0.3631 | 0.379 | 0.230 |
| Two passes + bend 2.0 | 6 | 0.921472 | 15.012 | 0.2624 | 0.3614 | 0.257 | 0.251 |

The winning bend-0.5 model improves FAPE on **376/512** structures versus longer one-pass training. Paired mean difference is **−0.003948**, with percentile bootstrap 95% interval **[−0.005361, −0.002863]** (10,000 structure resamples, seed 42). CA bend error improves by 59% versus that control and is better than the original pre-pair control (0.71649 rad). lDDT improves by 0.08309 and bond MAE by 0.07467 Å. RMSD is 0.371 Å / 2.56% worse than the longer one-pass control; that is inside the predeclared 5% tolerance, not a claim that every metric improves.

Bend weight 2.0 also passes and achieves still lower bend/torsion error, but its best FAPE is worse than weight 0.5. Two passes without bend supervision improve FAPE but fail the bend-regression screen. Longer training alone improves the original pair model from FAPE 0.926776 to 0.924494; bend supervision plus recycling reaches 0.920546.

Checkpoints were selected on repeatedly monitored development holdouts. Bootstrap intervals are conditional on checkpoint/arm selection and do not establish a fresh independent confirmation. Production-pretraining independence is not claimed. The next cohort reserves 512 additional confirmation structures outside training and checkpoint selection.

All jobs 3673916–3673922 completed with exit 0. Allocations consumed **23.578 GPUh**, including checks and the 21-second quality decision, below the 37 GPUh reservation. The conservative reservation ledger remains 221.667 / 256 GPUh.

Winning checkpoint: `/capstor/store/cscs/swissai/a0117/foldtree2/se3/runs/bend_recycle_v2_bend05/best.ckpt`, SHA256 `ec74e62f44ae58cc4d99ac977e88fb1795dabec8f3e2e342c01d8930885d1145`.

[Full gate decision, metric comparisons, intervals and checkpoint hashes](se3_clariden_evidence/bend_recycle_v2/results/quality.json). [Training/validation curves and stability reports](se3_clariden_evidence/bend_recycle_v2/results/gates_and_training.json).

Follow-on capacity/data preparation is documented in [the step-5 plan](se3_scaleup_v3.md).
