# Medium SE(3) numerical failure investigation, 2026-10-13

The medium Clariden trial failed because its unnormalized high-degree tensor
stream overflowed inside GotenNet. FAPE and the loss had not been evaluated on
the failing forward pass. The distributed timeout followed the numerical
exception; it was not the primary failure.

## Evidence

Job `3653794` completed six optimizer updates, then rank 0 raised
`GotenNet produced nonfinite valid invariant features` at epoch 0, cursor 6.
Other ranks waited in a broadcast until the default 30-minute NCCL timeout.
The allocation ended after 1,983 seconds, consuming about 2.20 GPUh on four
GPUs. The small control (`3653793`) completed all 16 epochs / 4,096 updates.
The medium-versus-small capacity gate could not be evaluated.

A local serial replay used seed 42, the identical frozen bundle, global batch
16, original medium initialization, AdamW 5e-4 and clipping at 1. It reproduced
the failure at the seventh update: structure `I3TCI4`, global-batch position 8,
226 stored residues / 225 valid residues. Position 8 belongs to rank 0 in the
four-rank stream. This is a reproduction on a different PyTorch/CUDA platform,
not a bitwise reproduction of the cluster weights.

Forward hooks located the first nonfinite module input at
`gotennet.layers.3.0.norm`, the fourth hierarchical tensor refinement block.
Immediately before it, the third equivariant feed-forward block increased the
degree-one tensor maximum from approximately `1.24e9` to `4.30e18`.
Its fourth-layer query/key products overflowed before LayerNorm could normalize
them. The previously added stable final high-degree norm runs too late to
protect these intermediate contractions.

A float64 audit of that same contraction found finite queries and keys with
maxima `3.01e19` and `2.49e19`. Their product reached `5.70e38`, exceeding the
float32 maximum `3.40e38`. The float64 product was finite; the float32 product
contained infinities. This confirms a representational overflow in the tensor
contraction, rather than an undefined geometry frame.

The feed-forward block computes a tensor projection, takes its norm, feeds
that unnormalized magnitude into an MLP, then multiplies the projection by an
MLP output. At large magnitudes this can grow approximately quadratically with
the tensor stream. More depth and channels amplify this path. Gradient clipping
cannot prevent a forward overflow before gradients exist.

## Controlled ablations

1. Rescaling the hierarchical query/key contractions before normalization
   passed the original failing operation on the preserved model/input. The
   fourth feed-forward block then failed while forming its projected-tensor
   norm (`gotennet.layers.3.3.mlps.0.0` received infinity). Fixing only the first
   overflowing operation therefore does not resolve the stream's growth.
   This contraction implementation remains a diagnostic ablation in the replay
   script, not a production backend change.
2. Fresh initialization with parameter-free LayerNorm on every equivariant
   feed-forward MLP input completed 16 global optimizer updates / 256 training
   structure presentations. All module outputs, losses and clipped gradients
   stayed finite, including the failing seventh-update group. Parameters,
   initialization seed, data order, optimizer, loss and clipping were otherwise
   held fixed. The code exposes this through an explicit experimental helper;
   canonical Lightning model construction does not enable it.

The normalization changes the model function. It is an architectural candidate,
not an algebraically equivalent repair for existing trained weights. It needs
a fresh run, explicit provenance, an entire-epoch stability check, and held-out
learning/structure metrics before a new capacity trial. A future comparison
must state this architectural difference, or normalize both capacity controls.
The six-layer rollout remains blocked. No replacement Clariden job was submitted.

## Verification and reproduction

The targeted refiner/Lightning suite passed **38 tests**, with one CUDA test
skipped by the restricted sandbox and one existing expected failure for the
blocked six-layer model. New CPU tests verify local opt-in behavior, reduced
quadratic growth, finite gradients and full-medium rotation/translation
equivariance at `atol=rtol=1e-4`. The 16-update experiment ran outside the
sandbox on idle workstation GPU 0; GPU 1's unrelated workload was preserved.

Local software: Python 3.9.23, PyTorch 2.8.0+cu128, CUDA 12.8.
Cluster software: Python 3.12.3, PyTorch 2.11.0a0+eb65b36914.nv26.02,
CUDA 13.1. Both used GotenNet 0.3.1. The frozen bundle SHA-256 is
`0250c8b94e10884553794470f2035355eb044e327feff3e66a72553f0e37e290`.

```bash
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
# Use fresh output directories. The original invocation is expected to fail.
python -m scripts.diagnose_se3_medium \
  --cache runs/se3_clariden/bundle_4096 \
  --output runs/se3_clariden/medium_diagnosis_original --updates 16
python -m scripts.diagnose_se3_medium \
  --cache runs/se3_clariden/bundle_4096 \
  --output runs/se3_clariden/medium_diagnosis_products --stable-products \
  --replay-failure runs/se3_clariden/medium_diagnosis_original/failure.pt
python -m scripts.diagnose_se3_medium \
  --cache runs/se3_clariden/bundle_4096 \
  --output runs/se3_clariden/medium_diagnosis_normalized \
  --normalize-ff --updates 16
```

Committed evidence is in
[medium_failure](se3_clariden_evidence/medium_failure/): cluster exception log,
original and rescaled-product traces, per-update results and targeted tests.
The full failing model/input snapshot is retained locally under
`runs/se3_clariden/medium_diagnosis_original/failure.pt`.

Future Lightning failures now record the active structure ID, length and
microbatch index, and attempt a rank-local diagnostic state snapshot without
collectives. These snapshots are explicitly marked non-resumable because an
accumulation update may be incomplete. A capture error is recorded without
replacing the original exception. This improves diagnosis; it does not change
the distributed watchdog timeout or automatically resubmit a job.

Canonical training mathematics remain unchanged. The diagnostic additions do
change source hashes, so strict resume provenance will reject old checkpoints
under this revision. The original pinned cluster checkout is preserved for
replaying or resuming its existing compatible checkpoints.
