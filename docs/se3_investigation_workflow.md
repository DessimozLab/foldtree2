# Bounded SE(3) diagnostic workflow

See [measured findings](se3_investigation_20261009.md) and the
[evidence index](se3_investigation_evidence/README.md). This is a follow-up to
the completed [original validation](se3_validation.md), not a larger-data run.
Use a fresh output directory for each invocation; the diagnostic runner rejects
an existing event log. It defaults to CPU; `--device cuda:0` or `cuda:1`
explicitly selects an available GPU. Check availability before starting and
leave unrelated workloads alone.

```bash
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
export PYTHONPATH=/tmp/foldtree2-se3-deps:$PWD
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m scripts.investigate_se3_scale --help
```

The temporary `tmtools` installation and required production models/mk2 input
are documented in the original workflow. The scripts read the existing
`runs/se3_validation/overfit/provenance.json`,
`runs/se3_validation/pilot/provenance.json` and their frozen-output caches.
They verify dataset/checkpoint hashes and extractor identity before use, require
disjoint calibration/evaluation IDs, and do not invoke the production encoder
or decoder. Missing caches fail explicitly; no graph regeneration is performed.

## Completed controls

The basic screen used eight structures, 10 epochs per case. Its subsequent
32-structure gain-1 control was interrupted after six epochs when the low update
scale and scheduler behavior were identified. The parameterization screen used
the same eight structures, then reinitialized its selected case for the full
30-epoch, 32-structure test and evaluated 128 disjoint structures.

```bash
# Diagnostic-only replay of the basic screen, without starting a full overfit.
python -m scripts.investigate_se3_scale --output runs/se3_basic_replay --overfit-epochs 0
# Replays the completed parameterization experiment; about 25 minutes on this CPU.
python -m scripts.investigate_se3_scale --screen-suite parameterization --output runs/se3_parameterization_replay
```

Both use seed 42, width 32, depth 2, two heads, head dimension 16, batch 1,
effective batch 8, AdamW 5e-4/weight decay 0.01 and clipping 1. Structures are
the original 32 complete overfit chains, at most 256 residues; evaluation uses
the original 128 pilot holdouts, at most 384 residues. The screen selects the
eight shortest training chains. Calibration is a single scalar fitted on all
32 training chains, never the holdouts. Case selection uses training geometry.

The basic screen compares raw/calibrated inputs, absolute/residual output and
RMS-radius normalization. The parameterization screen isolates a geometry-based
scheduler, residual gain 32 and circular angle output. Checkpoints, per-epoch
losses and gradient norms, best-case summaries, paired metrics and identities
are saved. This small diagnostic runner has no resume option; the canonical
trainer's strict resume remains available at its original source revision.

## Experimental model options

`se3_denoiser` accepts the following optional settings; its historical defaults
are retained for comparison:

| Setting | Experimental effect |
|---|---|
| `coordinate_update='residual'` | Adds the vector update to the centered seed |
| `zero_init_coordinate_update=True` | Starts exactly at the seed with a trainable coordinate projection |
| `coordinate_normalization='rms_radius'` | Divides by masked RMS radius, with minimum scale 1, and restores original units |
| `coordinate_residual_gain=32` | Compensates for the small normalized degree-1 signal; experimental, not universally calibrated |
| `angle_output='sincos'` | Six paired logits represent three full-circle angles; zero pairs are rejected explicitly |
| `device='cpu'` | Avoids constructing a CUDA backend when the requested run is on CPU |

The normal-lift prototype is
`foldtree2.src.geometry_seeds.lift_seed_with_local_normals`. It uses a centered,
standardized invariant feature and oriented neighboring-bond normals, scaled
by a fixed fraction of seed RMS radius. Its construction transforms under
proper rotations and translations and masks incomplete/undefined stencils.
It does not consume target coordinates or claim reflection equivariance.

The spatial suite compares planar input, fraction-0.1 lifted input, and lifted
input with circular angles. **Only the following one-structure smoke was run:**

```bash
python -m scripts.investigate_se3_scale --screen-suite spatial --screen-count 1 --screen-epochs 1 --overfit-epochs 0 --output runs/se3_spatial_smoke_replay
```

A 32-structure lifted-seed learning test is recorded in the
[3D-seed learning report](se3_spatial_learning_20261009.md). The revised
512/128 pilot remains gated by those results. No full-dimensional seed is promoted merely because it escapes a
plane or passes a finite-gradient smoke check. Require ≥20% geometry learning
from the new model's own initialized baseline, improved paired held-out FAPE,
and no material structural regression before increasing the training scope.

The bounded test predeclares the circular lifted candidate rather than choosing
it from the control screens. It resets seed 42 for each model, trains 30 epochs
on the original 32 chains, and evaluates the 128 holdouts against the identical
lifted inputs. For an idle GPU 0:

```bash
python -m scripts.investigate_se3_scale --screen-suite spatial --device cuda:0 --train-case lifted_circular --output runs/se3_spatial_learning_replay
python scripts/report_se3_spatial_learning.py --evidence docs/se3_spatial_learning_evidence --output docs/se3_spatial_learning_20261009.md
```

## Report and source provenance

Regenerate the published report without datasets or model calls:

```bash
python scripts/report_se3_investigation.py --basic docs/se3_investigation_evidence/basic --parameterization docs/se3_investigation_evidence/parameterization --output docs/se3_investigation_20261009.md
python scripts/check_documentation.py
```

Executed source snapshots and packaged source hashes are separate in the
archive. Explicit CPU construction and the spatial-suite option were finalized
after the recorded 30-epoch run. Cached features remain unchanged. Old
checkpoints cannot bypass strict implementation provenance checks merely
because the new settings are optional; retain their original source revision.

The original v2 lDDT values remain archived. The investigation uses centered
distance calculations and an explicit self-pair mask; cross-protocol lDDT
comparisons are inappropriate. Production checkpoints, alphabet results and
the separate staged transformer were not modified.

## Loss ablation

The [loss-ablation report](se3_loss_ablation_20261009.md) compares the historical
objective with opt-in soft FAPE and explicit local bonds. Four eight-chain
screens isolate the two terms; original and combined objectives then receive
the same 32-chain, 30-epoch budget and 128 holdouts. The candidate and weights
are predeclared. The original clamped evaluation FAPE is retained.

```bash
python -m scripts.run_se3_loss_ablation --device cuda:0 --output runs/se3_loss_ablation_replay
python scripts/report_se3_loss_ablation.py --evidence docs/se3_loss_ablation_evidence --output docs/se3_loss_ablation_20261009.md
```

Only use the CUDA command when that GPU is available; `--device cpu` is also
supported. This diagnostic runner has no resume option. Frozen-cache identity
and source snapshots are retained, and production parameters are never loaded
into the optimizer. New loss modes are opt-in, and a successful bounded test
would authorize consideration of a revised pilot rather than larger training.
