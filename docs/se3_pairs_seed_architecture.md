# Continuous decoder pairs and learned initial geometry

Implemented on `se3-recycled-refiner`, independently switchable alongside
[shared recycling](se3_recycling_architecture.md). This is residue-level SE(3)
refinement. Production epoch-40 encoder/decoder weights stay frozen; the separate
staged transformer is unchanged. Architecture training and merging remain
conditional on a matched quality comparison. No architecture job was submitted.

## Continuous pair conditioning

The decoder's contact matrix contains useful continuous structure even when it
resembles a distance map. Its scores are **contact probabilities, not calibrated
Angstrom distances**. The production decoder computes

```
cosine_ij = normalize(z_i) dot normalize(z_j)
p_ij = clamp(sigmoid(2.4314587116241455 * cosine_ij - 1.3124585151672363), 1e-6, 1-1e-6)
```

These constants were read from the matched serialized production decoder, SHA256
`b9642afd20db829a5ce2dd1c908c86df0c351319347fccd2ad1d335b3aa9089d`.
The frozen cache already contains normalized `contact_z`, so no production
forward pass or graph/cache regeneration is needed. The trainer verifies that
the configured calibration identifies the cached decoder.

Each pair has 12 scalar channels: cosine, bounded logit, probability, three
sequence-separation channels and six radial basis functions of the current
predicted physical distance (centers 0/4/8/12/20/32 Angstrom, width 4 Angstrom).
A 12→32→hidden projection adds these features to GotenNet's initial learned
edge state, consumed by its tensor refinement and attention layers. Continuous
scores survive even when two inputs produce identical binary contact graphs.
Sequence indices preserve internal masked gaps; diagonal/padding pairs are zero.
Pairs are constructed independently per chain, before radius normalization.
Recycling recomputes geometry channels while frozen decoder channels remain fixed.

The projection ends in a zero-initialized layer. Enabling it preserves all
existing parameter draws and initial outputs, allowing a controlled warm start.
`PairGotenNet` is a local dense adaptation of pinned gotennet-pytorch 0.3.1;
it leaves installed backend globals untouched and compacts valid nodes before
attention. Sparse neighbor selection and atom-level inputs are explicitly unsupported.
The extra pair tensors remain quadratic in chain length: current candidate
configs retain the established 256-residue limit. Benchmark GPU memory before
increasing lengths or batch sizes.

## Learned initial geometry

The production three-channel bottleneck is essentially planar because of its
final LayerNorm; see the [original investigation](se3_investigation_20261009.md).
Simply increasing a scalar-conditioned equivariant model cannot reliably escape
that plane. The current fixed normal lift provides a starting point but uses
only one feature column with a fixed amplitude.

`LearnedGeometrySeed` preserves that lift and learns a bounded correction from
**all frozen latent/codebook features**. A small scalar MLP predicts coefficients
along local tangent, normal and binormal directions. The three-component
coefficient vector has a smooth norm bound; its maximum physical displacement
is 0.25 times each chain's RMS seed radius. It uses no target coordinates, global
spatial axes or random geometric perturbations. Undefined frames and incomplete
neighbor stencils receive zero learned correction, including coincident and
collinear chains. This construction is SO(3)/translation equivariant; oriented
normals deliberately break reflection symmetry.

The last MLP layer starts at zero, reproducing the original fixed lift exactly.
For this option, calibration performs only coordinate scaling, and the model
applies the fixed lift plus learned correction **once**, before all refinement
passes. Provenance records upstream lift fraction zero, avoiding a duplicate
lift during training or validation. Other modes retain upstream fraction 0.1.
The bound limits the correction; it does not impose a hard bound on total seed
coordinates or on subsequent refiner displacement.

Useful follow-up candidates, not yet implemented or validated:

* Condition seed coefficients on pooled decoder pair messages to place residues
  using nonlocal evidence as well as their individual frozen features.
* Train a pair-distance head from the frozen contact embeddings, then fit a
  seed using soft distance restraints plus chain continuity. Compare distance
  bins or distributions rather than treating contact probability as distance.
* Add a chirality-aware normal lift to any distance-fitted seed: distances alone
  cannot distinguish mirror structures. Mask uncertain frames consistently.

A learned distance reconstruction is the larger change: it needs its own
supervision, gradient/stability checks and ablation. The local learned lift is
implemented first because it can start exactly from the existing trained model.

## Matched training candidates

All five `configs/clariden_se3/pairs_seed_v1_*.yaml` use width 64, depth 4,
four heads, seed 42, float32 model computation, float64 accumulation/reduction,
global batch 16, 8,192 training / 512 validation structures and four epochs.

| Suffix | Continuous pairs | Learned seed | Passes |
|---|---|---|---:|
| `control` | No | No | 1 |
| `pairs` | Yes | No | 1 |
| `seed` | No | Yes | 1 |
| `pairs_seed` | Yes | Yes | 1 |
| `pairs_seed_two` | Yes | Yes | 2 |

They retain the same original loss and final evaluation-FAPE checkpoint selection.
If a separate loss trial wins, copy its exact objective and matching parent into
**every** arm before training. Use identical data/order, parent checkpoint and
optimizer reset; compare pair and seed effects before interpreting their
interaction with recycling. Follow the existing finite/resume/two-node gates.

Example trainer arguments inside an allocation (no submission implied):

```bash
python -m foldtree2.train_se3_lightning \
  --config configs/clariden_se3/pairs_seed_v1_pairs_seed.yaml \
  --warm-start /capstor/store/cscs/swissai/a0117/foldtree2/se3/runs/growth_fp64_8192/best.ckpt \
  --allow-refinement-change
```

The flag allows only reviewed architecture/config changes and the corresponding
movement of seed lift into the model. It retains model shape, loss, learning
rate, frozen identities, calibration scale, dataset and split checks. Only newly
enabled pair/seed parameters may be absent from the parent state; missing base
weights, unexpected keys or removal of a trained extension fail. Ordinary resume
remains strict, including hashes of all new modules.

## Verification and interpretation

**105 tests passed**, one GPU-only test was skipped, and the existing depth-six
instability remained an expected failure. CPU tests cover exact zero-init
compatibility, pair probability reconstruction,
continuous conditioning with identical contact graphs, active-extension rigid
equivariance at float32 atol/rtol 1e-4, mixed-length chains, masked NaNs, finite
backward paths, degenerate seeds, one-time seed/BatchNorm application, per-pass
pair geometry refresh, decoder identity and checkpoint migration guards.

Reproduce the production-cache audit from this checkout:

```bash
python -m scripts.audit_se3_pairs_seed \
  --cache /home/dmoi/projects/foldtree2/runs/se3_clariden/bundle_4096 \
  --checkpoint /home/dmoi/projects/foldtree2/runs/se3_clariden/growth-fp64-8192-best.ckpt \
  --output /tmp/se3-pairs-seed-audit --count 8
```

The audit restricts IDs to retained growth **training** structures, verifies
production/preprocessing identities, records paired metrics and parameter counts,
and checks nonzero finite gradients into new modules from the actual training
objective. It performs no optimizer steps. This verifies a usable learning path,
not improved structural quality. The retained [eight-structure audit](se3_clariden_evidence/pairs_seed_v1/audit.json)
confirmed exact first-pass agreement for every arm. Under the real loss, the
single-pass pair projection gradient norm was **0.130919** and the learned seed
norm was **4.010783**. Combined variants contain **1,688,389 parameters**, versus
**1,684,514** for the control; recycling shares those parameters. The untrained
second pass is still an inference audit, not a trained accuracy comparison.
CUDA and distributed gates remain required;
local GPU occupancy was checked outside the sandbox and both devices had
substantial allocations, so architecture GPU work was deferred.

Merge only after trained held-out FAPE improves without material structural
regression under the [existing merge gate](se3_recycling_architecture.md#merge-gate).
These development holdouts have been repeatedly monitored, and no independence
from production pretraining is claimed.
