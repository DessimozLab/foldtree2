# Capacity CLI failure and bounded recovery, 2026-10-19

The previous capacity experiment did not produce trained width-comparison
results. Job 3675417 completed the width-64 exact-resume check with bitwise
agreement (maximum parameter/buffer difference 0). It then failed before
initializing width 80 because the trainer's CLI allowlist still accepted only
width 32/depth 2 and width 64/depth 4. Its message incorrectly described every
unlisted model as having failed numerical gates. The model-building unit test
had bypassed that CLI check.

Width 80 therefore has no GPU numerical or quality result yet. Two-node check
3675418 and training jobs 3675419/3675420 were cancelled by dependency failure.
Slurm records zero elapsed time, no start timestamp and no child steps for all
three. Its `NNodes=2` field describes the requested allocation and must not be
misread as an executed allocation.

The fix adds only width 80/depth 4/four heads to the CLI's supported models.
That width requires tensor-FF input normalization and float64 gradient
accumulation. Float32 forward/backward, loss, data, initialization and clipping
are unchanged. Depth six and other unreviewed shapes remain blocked. A CLI
regression test now reaches cache validation from the real canonical width-80
configuration; finite two-pass learning tests cover both widths.

Recovery uses fresh `scale_capacity_v3r1_*` outputs and a new pinned code
release, because the trainer-source identity changed. It repeats the one-node
recovery checks and then the two-node agreement checks for both widths. Actual
four-epoch trials remain gated on successful checks. No old checkpoint, source,
cache, receipt or failure evidence is overwritten. There are no automatic retries.

## Resources and reservation accounting

All jobs use account a0117, partition normal, four GPUs per node, the existing
pinned container/environment and `nid007494` exclusion.

| Stage | Nodes / GPUs | Walltime | New reservation |
|---|---|---|---|
| One-node checks | 1 / 4 | 20 minutes | 1.333333 GPUh |
| Two-node checks | 2 / 8 | 10 minutes | 0; reuse unused 1.333333 GPUh |
| Fresh width 64, four epochs | 2 / 8 | 45 minutes | 0; reuse unused 6 GPUh |
| Fresh width 80, four epochs | 2 / 8 | 45 minutes | 0; reuse unused 6 GPUh |

The failed one-node reservation remains fully charged in the conservative
ledger. The other three original reservations remain charged too; each may
fund exactly one replacement with identical resources because its old job
never started. The recovery appends zero-charge rows linked to those parent
reservations, checks scheduler accounting and rejects repeat reuse. Submission
failures remain recorded and require diagnosis. No reservation is deleted or
silently reduced based on actual usage.

Maximum added reservation is 1.333333 GPUh. The total becomes
**237.666667 / 256 GPUh**, leaving **18.333333 GPUh** for data work. Project credit
uses the user's existing confirmation; it is not an independently read portal
balance. Live normal-partition maximum remains 12 hours; all requested limits
are shorter.

```bash
python3 -m scripts.recover_se3_scale_capacity --available-credit 15 --submit
```

This is a one-time recovery command. Do not rerun it after any reservation is
recorded; inspect the new receipts/checks instead. The old
[failure log, accounting and exact-resume gate](se3_clariden_evidence/scale_capacity_v3/failure_20261019.json)
remain available. Passing a two-update check will establish execution stability,
not a learning or width-quality conclusion.

## Data readiness

The separate CPU export is complete with 10,240 training structures (all old
8,192 retained plus 2,048 additions), the original 512 development structures,
and 512 fresh confirmation structures. Export identity and split disjointness
were checked. All stay within 256 residues. The 2,561 existing graph exports
occupy 2.4 GiB; no graph regeneration or frozen production forward pass occurred
locally. Frozen GPU extraction/cache publication and a matched data comparison
are still pending. Equal presentation counts would be five old-data epochs
versus four expanded-data epochs (2,560 updates each at global batch 16).

## Local validation

The focused capacity, cohort, submission and release tests pass: **12 passed**.
The Lightning trainer tests also pass: **12 passed, 1 expected failure** for the
known blocked depth-six configuration. The capacity tests exercise finite
two-pass learning and pair-head updates for both widths, the actual width-80
CLI, and single-use reservation recovery. CPU numerical checks do not replace
the pending one-/two-node CUDA checks.
