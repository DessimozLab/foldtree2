# Conditional step 5: data and capacity

For a new agent, start with the [SE(3) decoder handoff](se3_decoder_handoff.md).

**2026-10-20 update:** Both recovered four-epoch capacity trials completed,
including exact-resume/topology checks. Width 80 fails the aggregate quality
thresholds despite a small FAPE gain; keep the trained width-64 incumbent.
[Results, loss tradeoff and access diagnosis](se3_clariden_diagnosis_20261020.md).

**2026-10-19:** the original capacity chain stopped before width-80 training
because a stale CLI allowlist rejected the requested architecture. Width-64
exact recovery passed bitwise. No four-epoch capacity trial ran. The CLI fix,
bounded recovery and original failure evidence are documented in
[the recovery report](se3_scale_capacity_recovery_20261019.md).
The CPU export completed with **10,240 training IDs**, the retained 512
development IDs and 512 fresh confirmation IDs.

The completed bend/recycling quality gate passed with winner `bend05`, checkpoint
SHA256 `ec74e62f44ae58cc4d99ac977e88fb1795dabec8f3e2e342c01d8930885d1145`.
This meets the user's condition for bounded more-data/capacity experiments.
[Results and residual tradeoffs](se3_bend_recycle_v2_results.md) remain recorded;
no merge or production replacement is implied.

## Matched capacity probe

Compare hidden widths **64 and 80**, both depth 4, four heads, head dimension 16,
two shared passes, continuous decoder pairs, fixed normal lift and bend weight
0.5. Both train on the same existing 8,192/512 development split for four epochs,
global batch 16, seed/order 42, LR 5e-4, float32 forward/backward and float64
accumulation. Both start **fresh**, with zero coordinate-update heads and matching
initial seed predictions. They do not import shape-incompatible trained weights.
This isolates width under equal sample visits; it is not a warm continuation of
the successful trained checkpoint or a compute-matched comparison.

One-node exact recovery checks (20 minutes / 1.333 GPUh) and two-node agreement
checks (10 minutes / 1.333 GPUh) gate two sequential two-node/four-GPU-per-node
trials (45 minutes / 6 GPUh each). All use `normal`, account `a0117`, the existing
pinned extended container/ARM environment, and exclude `nid007494`. No automatic
retries. Maximum added reservation **14.667 GPUh**, bringing the ledger to
236.333 / 256 and leaving **19.667 GPUh** for data extraction and comparison.
Credit follows the user's prior confirmation, not an independently read portal
balance. Slurm `--test-only`, duplicate-output checks and the durable budget
ledger run before submission. The passed bend quality decision and exact winner
hash are verified again before queuing capacity work.

```bash
python3 -m scripts.submit_se3_scale_capacity --available-credit 15 --submit
```

## Data preparation completed

`prepare_se3_scaleup_inputs.py` targeted **16,384 training structures**, retains
all old 8,192 training IDs and 512 development IDs, and reserves **512 fresh
confirmation structures**. All selections exclude earlier training/holdout IDs,
respect the 256-residue cap, and use seed 42. Confirmation IDs are stored
separately from `bundle_template.val_ids`; they must not select checkpoints.

The script verifies the existing mk2 HDF5 SHA256, serialized production pair,
frozen extractor and passed quality decision before exporting existing graph
tensors. There is **no graph regeneration or production forward pass** locally.
Frozen feature extraction must run on Clariden, preserving the old cached entries
bit-for-bit and recording repeatability/parity checks before publishing a new
bundle. The current CPU preparation writes under
`/tmp/foldtree2-se3-recycling/runs/se3_scaleup_v3/inputs_v3`; its persistent log is
`runs/se3_scaleup_v3/prepare_v3.log`. The initial superseded preparer was stopped
before use. The fixed-16k attempt failed the available-structure check after
source verification: too few unused chains pass the existing length/geometry
filters. The corrected preparer selects the largest available expansion in
1,024-training-structure increments, between 9,216 and 16,384 total training
structures, while retaining all 512 fresh confirmation IDs. Requested and actual
counts and rejected sizes are logged. Only completed `inputs_v3` may be consumed.

Next, stage and validate those inputs, extract only added frozen entries on
Clariden, then compare the winning trained width-64 model continued on old versus
expanded training data. Match optimizer updates/sample visits: five epochs on
8,192 versus four on 10,240 gives 2,560 updates each at global batch 16. Preserve
the development split. Run
fresh confirmation only after selecting checkpoints. Reserve all extraction,
stability and training allocations within the remaining 18.333 GPUh after the
bounded capacity recovery, before
submitting; no expanded-data training job exists yet.

Local tests cover finite two-pass training at both widths, updated pair heads,
identical initial coordinate predictions, matched configuration/budget/dependency
rules, retained membership, disjoint fresh confirmation, and duplicate rejection.

## Capacity submissions

Pinned code `72e27f7d2dabd5c5bccc64593582f3f06f207948` passed 42 release-dependency
checks. Archive SHA256:
`88e916cfc0c42e73f7f2f9ca3cbcaa0a04f5b8d689594de082eaed0f4f29ef54`.
Eleven local capacity/cohort/bend/release tests passed. Jobs:

| Job | Stage |
|---:|---|
| 3675417 | One-node exact recovery checks |
| 3675418 | Two-node agreement checks |
| 3675419 | Fresh width-64 control, four epochs |
| 3675420 | Fresh width-80 candidate, four epochs |

At submission, the first check waited on priority and subsequent jobs on
successful dependencies. The ledger confirms **236.333333 / 256 GPUh** reserved.
[Receipts](se3_clariden_evidence/scale_capacity_v3/submission.json) retain all
commands and checksums. No new capacity result is inferred from submission.
The expanded-cohort CPU preparation subsequently completed: 2,561 exported
graphs (2,048 added training IDs, 512 confirmation IDs, one parity structure),
2.4 GiB of existing graph tensors, export identity
`2f45d9c5a8b35ef3adb6bf1dcd1b2ec69b89017adf9a5f1cf52c2b66b8d1a617`.
No expanded-data extraction/training is submitted yet. Confirmation IDs remain
separate from development/checkpoint selection.

The 16k shortage does not change the capacity configurations or residue cap.
The data transfer/extraction/train stages remain pending frozen GPU extraction
and bundle validation. The failed selection is preserved in `runs/se3_scaleup_v3/inputs_v2/`
and `prepare_v2.log`; it produced no published frozen bundle.

The CLI allowlist recovery is now queued as 3700111 → 3700112 → 3700114 →
3700115, with fresh pinned code `9b66111`. See the
[recovery report](se3_scale_capacity_recovery_20261019.md) for accounting and
receipts. Reserved total is now 237.666667 GPUh; no completed capacity training
result is available at submission.
