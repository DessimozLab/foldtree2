# Conditional step 5: data and capacity

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

## Data preparation in progress

`prepare_se3_scaleup_inputs.py` selects **16,384 training structures**, retains
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
`/tmp/foldtree2-se3-recycling/runs/se3_scaleup_v3/inputs_v2`; its persistent log is
`runs/se3_scaleup_v3/prepare_v2.log`. The initial superseded preparer was stopped
before use; only corrected `inputs_v2` may be consumed.

Next, stage and validate those inputs, extract only added frozen entries on
Clariden, then compare the winning trained width-64 model continued on old versus
expanded training data. Match optimizer updates/sample visits (e.g. old data four
epochs versus expanded data two epochs) and preserve the development split. Run
fresh confirmation only after selecting checkpoints. Reserve all extraction,
stability and training allocations within the remaining 19.667 GPUh before
submitting; no expanded-data training job exists yet.

Local tests cover finite two-pass training at both widths, updated pair heads,
identical initial coordinate predictions, matched configuration/budget/dependency
rules, retained membership, disjoint fresh confirmation, and duplicate rejection.
