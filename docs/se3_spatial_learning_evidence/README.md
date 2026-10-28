# Bounded 3D-seed learning evidence

See the [results and loss audit](../se3_spatial_learning_20261009.md).
The run completed 32 structures for 30 epochs, followed by 128 disjoint
held-out comparisons, on idle GPU 0. GPU 1's unrelated workload was left alone.
Production caches/checkpoints and the staged transformer were unchanged.

`analysis.json` records input identities, splits, configuration and executed
source hashes. `sources/` preserves the files used for training before the FAPE
masking correction. `audit_sources/` and `audit_source_manifest.json` preserve
the later correction, regression tests, audit and reporting tools separately.
The numerical audit used CPU float64 and the selected checkpoint; training
used CUDA float32. Both used the same structures and defined frame masks.

`screening.json`, `overfit.json`, `events.jsonl`, `heldout_pairs.json`,
`gate_summary.json`, `loss_audit.json` and the plot retain the measured results.
`checkpoint_inventory.json` binds local weights to their hashes; large weights,
datasets and frozen caches remain local. Software and GPU observations are
retained alongside console logs.

Correctness before the masking correction: 70 passed. After correction and
reference/gradient regression tests: 73 passed, including CUDA coverage. An
earlier test invocation omitted the existing temporary TM-align dependency
from PYTHONPATH; that failure is retained as `correctness-missing-dependency.log`.
It was corrected before final verification.

All three learning/quality gates failed. Do not infer readiness for a larger
run from the seed's full rank or finite gradients. The next diagnostic is a
versioned loss ablation, not a longer or wider run of this objective.

Reproduce the report without production model calls:

```bash
python scripts/se3/archive/report_se3_spatial_learning.py --evidence docs/se3_spatial_learning_evidence --output docs/se3_spatial_learning_20261009.md
```
