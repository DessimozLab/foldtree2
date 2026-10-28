# SE(3) gradient accumulation investigation, 2026-10-14

The normalized medium model amplifies ordinary float32 gradient-summation
roundoff. A local experiment reproduced divergence without DDP, networking or
checkpoint recovery: the only difference was grouping identical per-structure
gradients into four versus eight rank-like sums. At update 16 the maximum
parameter difference was **0.00269005**, comparable to the earlier Clariden
cross-topology discrepancy.

Replacing only the accumulator/reduction arithmetic with float64 kept the local
four/eight-rank emulations **bitwise identical through all 16 updates**. Model
forward/backward and AdamW remained float32. This establishes that accumulation
order is sufficient to produce the observed kind of divergence; it does not
claim that every possible distributed synchronization issue has been ruled out.

## Loss path and controlled trace

The production frozen bundle, seed 42, medium width 64/depth 4/four heads,
parameter-free tensor MLP normalization, global batch 16, AdamW 5e-4 and clipping
at 1 were held fixed. Each branch processed the identical structures in the
identical order. The script computes individual float32 gradients, groups them
as four ranks/four microsteps or eight ranks/two microsteps, then applies the
same global mean and optimizer. It is a summation-order emulation, not an
emulation of NCCL's exact reduction tree.

At update 1, the two float32 grouping orders differed by about `7.45e-8` in the
common-weight gradient vector. Maximum weight difference grew from `1.16e-10`
after update 1 to `1.24e-6` after update 2, `6.36e-6` after update 4, and
`0.00269005` after update 16. The float64 control produced zero model difference
throughout. These are local PyTorch 2.8/CUDA 12.8 results; the original cluster
uses PyTorch 2.11.0a0/CUDA 13.1, so they are not bitwise reconstructions of its
weights.

The FAPE gradient calculation was also checked on the 16 structures of the
third global batch. Coordinate gradients were larger than the global-distance
component: original float32 trace FAPE norms ranged approximately 0.114–1.039,
distance norms 0.0056–0.0096, and bond norms 0.103–0.162. Some predicted triples
were nearly collinear, with minimum valid bend sine around 0.00614. This is a
conditioning concern, not evidence of an incorrect derivative.

Float64 directional finite differences of the training FAPE matched autograd
across those samples, with maximum absolute error **4.72e-10** using a `1e-6`
coordinate perturbation. No FAPE formula, frame mask, loss weight, calibration
or evaluation metric was changed as part of the accumulation fix.

## Opt-in training implementation

`gradient_accumulation_dtype: float64` enables the following path in the
cache-only Lightning trainer:

- Backpropagate each unscaled structure loss in float32, with DDP gradient
  synchronization disabled during the per-structure backward.
- Add each gradient to a rank-local float64 buffer and clear the float32
  per-structure gradient before the next microstep.
- At the global-batch boundary, sum the float64 buffers across ranks and divide
  by the global structure count once. Verify that exactly 16 microbatches were
  consumed globally. Merge a parameter-use mask so globally unused parameters
  retain `grad=None` and their AdamW/weight-decay behavior.
- Cast the averaged gradients back to float32, apply the existing clipping at
  1, and run the unchanged float32 AdamW optimizer.

The temporary accumulator is reset at each completed optimizer update. The
checkpoint protocol saves only at those boundaries, so no partial accumulator
is silently resumed. Failure snapshots remain explicitly non-resumable.
The setting and helper source hash are bound into provenance. The default
float32/DDP accumulation path remains unchanged. There are no global dtype or
backend changes, gradient sanitization, silent loss skipping, or production
checkpoint modifications.

The helper requires float32 model parameters and dense gradients. Its float64
buffer costs approximately 13.5 MB for the 1.68-million-parameter medium model,
plus small parameter-use metadata. It doubles gradient payload bytes versus a
float32 reduction; performance should be measured before a larger rollout.

## Local verification and commands

The targeted suite passed **19 tests**, with the existing expected failure for
the blocked six-layer architecture. Tests cover the exact global mean, unused
parameters, missing microbatches, normalization provenance, and inclusion of the
new dependency in deployable archives.

On a real-cache 32-training/three-validation CPU fixture, serial and two-process
Gloo DDP model **and optimizer** states matched exactly. Interrupted/resumed
training also matched the uninterrupted serial state exactly. The first local
Gloo attempt failed because the restricted sandbox denied loopback sockets; the
successful check ran outside that sandbox and allocated no GPUs.

The GPU diagnostics used idle workstation GPU 0; GPU 1's active workload was
left untouched.

```bash
python -m scripts.trace_se3_accumulation \
  --cache runs/se3_clariden/bundle_4096 \
  --output runs/se3_clariden/gradient_trace_16_20261014 --updates 16
python -m scripts.trace_se3_accumulation \
  --cache runs/se3_clariden/bundle_4096 \
  --output runs/se3_clariden/gradient_trace_fp64_20261014 --updates 16 \
  --reduction-dtype float64
```

Use fresh output directories when reproducing these commands. Full models and
the sampled coordinate snapshots are retained under runs; tensors are not
committed. Numerical traces, finite differences, integration summary and tests
are in [gradient_precision](se3_clariden_evidence/gradient_precision/).

## Clariden verification

The pinned source revision is `8f300d6a48e45f39f6a2be66745e12e72574c4fb`.
The source-only archive includes all required Python/launch dependencies and
has SHA-256 `79fa67cb9b602ea5546388f38ade791233862c7181d48697f4870ec5e05ea56c`.
The existing pinned ARM container, dependency environment and frozen bundle were
reused. The three configs are `fp64_debug_one`, `fp64_debug_control`, and
`fp64_debug_two` under `configs/clariden_se3`.

The one-node/four-GPU job **3657662** completed in 291 seconds. After saving at
update 16 and resuming to 32, model parameters, buffers, optimizer state and
all validation metrics matched the uninterrupted control exactly. Each
evaluation counted all 512 holdouts once.

The subsequent two-node/eight-GPU job **3657694** completed in 209 seconds and
**passed the strict cross-topology gate**. After save/reload and 32 optimizer
updates, its model parameters and buffers were bitwise identical to the
one-node uninterrupted control, with maximum difference **0**. All 512-structure
validation means were also identical. An additional CPU comparison of the
downloaded checkpoints confirmed bitwise identical optimizer state.

Both checks retained `atol=rtol=1e-4` for cross-topology comparisons and zero
tolerance for the exact one-node resume comparison. No tolerance was relaxed.
This clears the previously observed 32-update agreement failure for the tested
normalized medium variant. It does not establish long-run learning quality or
four-node stability.

Identical validation values included geometry 4.657215, FAPE 0.939541, aligned
CA RMSD 16.54182 Å, TM-score 0.201666 and lDDT 0.205288. These are short
execution checks, not a capacity or learning gate. The next experiment should
be a bounded longer learning trial with normalization and high-precision
accumulation explicitly enabled; comparisons must use matching protocols.

Median observed update throughput was 13.27 global structures/second on four
GPUs and 24.17 on eight. Rank-zero peak allocated GPU memory was approximately
3.68 and 3.44 GiB respectively. These short-run medians are not a formal scaling
benchmark or an all-rank memory maximum. Actual allocation cost for both jobs
was **0.788 GPUh**, below the 4 GPUh ceiling.

Cluster gates, per-update events, per-structure validation, optimizer comparison
and original logs are preserved under
[gradient_precision/cluster](se3_clariden_evidence/gradient_precision/cluster/).
The durable cluster root remains `/capstor/store/cscs/swissai/a0117/foldtree2/se3`,
with runs `fp64_debug_one`, `fp64_debug_control` and `fp64_debug_two`.

The debug ceilings are 30 minutes on one node and 15 minutes on two nodes,
at most 4 additional GPUh. Conservative rollout reservations are now 52 of the
authorized 256 GPUh. The failed original checks remain preserved. No larger
learning trial, model expansion or production-checkpoint replacement was
started during this investigation.
