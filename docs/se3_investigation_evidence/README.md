# Scale, residual and planarity investigation evidence

Recorded on 2026-10-09. See the [results report](../se3_investigation_20261009.md).
All training used CPU, width 32, depth 2, two heads, seed 42 and the existing
frozen epoch-40 production caches. No larger-data run or model promotion followed.

- `basic`: five controlled eight-structure, ten-epoch screens. Its preliminary
  full 32-structure gain-1 run was stopped after six epochs to investigate the
  measured update scale and scheduler behavior; it is not a completed overfit gate.
- `parameterization`: three controlled screens, followed by a reinitialized
  32-structure, 30-epoch run of the selected gain-32 residual. Includes 128
  disjoint evaluation pairs, gradient diagnostics, singular-value measurements,
  decoder LayerNorm/planarity checks, the untrained lift prototype and resource
  measurements. Complete event records verify the requested counts.
- `spatial_smoke`: one epoch on one structure for each of three seed/angle
  variants. This checks finite forwards/backwards and the new entrypoint; it is
  not evidence of successful learning from full-dimensional seeds.
- `metric_attempt`: interrupted screening before the distance-metric correction.
  Retained separately, without pooling its scores with the corrected protocol.
- `packaged_source_manifest.json`: hashes of current implementation and tools.
  The `sources` directories separately preserve the files executed for the
  recorded training runs. Later packaging added explicit CPU construction and
  the spatial-suite entrypoint; these did not change the recorded experiment.
- `checkpoint_inventory.json`: hashes and local paths of retained checkpoints.
  Dataset, latent cache and model weights remain local.

The new report uses centered pair distances and explicitly excludes lDDT
self-pairs. Calibration was fitted on the 32 original training IDs; those IDs
are disjoint from the 128 evaluation IDs. The selected model used the legacy
bounded angle head. A circular head was screened and tested, but its small
screen did not win by geometry loss.

The key outcomes are a 16.0% best training geometry reduction, worse held-out
FAPE than the calibrated seed, and an essentially planar input/output constraint.
The normal lift is a tested SO(3)-equivariant prototype; it has not passed a
32-structure training gate or a revised 512-structure pilot. It adds an oriented
normal term and therefore does not claim reflection equivariance.

CPU correctness: 69 passed and one CUDA-specific test skipped because the GPUs
were occupied. Check the retained final correctness log. Historical v2 results
and the staged transformer remain unchanged.
