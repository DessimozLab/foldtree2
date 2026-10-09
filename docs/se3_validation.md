# Production SE(3) residue refiner

This validation path is separate from `learn_production_staged_transformer_geometry.py` and its staged transformer. It refines the production geometry bottleneck with GotenNet. The 40-character epoch-40 encoder and decoder are resolved through `configs/production_models.yaml` and frozen throughout. No graph regeneration, alphabet benchmarking, production checkpoint replacement, distributed training or atom refinement is performed.

Activate the CUDA-capable `foldtree2` environment and select an idle GPU. On this workstation GPU 1 was made available for validation. Run GPU commands outside Codex's restricted sandbox as directed by `AGENTS.md`.

```bash
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
# TM-align bindings, if not installed; a temporary target keeps the environment unchanged.
python -m pip install --no-deps --target /tmp/foldtree2-se3-deps tmtools
export PYTHONPATH=/tmp/foldtree2-se3-deps:$PWD
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p runs/se3_validation
python -m pytest foldtree2/tests/test_se3_refiner.py foldtree2/tests/test_se3_validation.py foldtree2/tests/testing_fape_quaternion.py foldtree2/tests/testing_coarse_ca.py -q
python -u -m foldtree2.se3_validation --config configs/se3_validation_smoke.yaml > runs/se3_validation/smoke.log 2>&1
python -u -m foldtree2.se3_validation --config configs/se3_validation_overfit.yaml > runs/se3_validation/overfit.log 2>&1
python -u -m foldtree2.se3_validation --config configs/se3_validation_pilot.yaml > runs/se3_validation/pilot.log 2>&1
python scripts/report_se3_validation.py
```

Execute experiments sequentially after correctness and smoke checks pass. The overfit experiment trains and evaluates the same 32 structures for 30 epochs, bounded at 256 residues. The pilot uses 512 training and 128 disjoint validation structures bounded at 384 residues, up to 20 epochs, and stops after five validation FAPE epochs without an improvement of at least 1e-4. Structures outside the length bound are logged and excluded; chains are never cropped. The overfit split intentionally overlaps, while the pilot split is disjoint. These holdouts apply to the refiner, with no assertion about production pretraining overlap.

The bounded runner fixes float32, seed 42, hidden width 32, depth 2, two heads of dimension 16, residue geometry/FAPE plus circular angle loss, AdamW at 5e-4, clipping at 1, microbatch size 1 and effective batch size 8. A final short accumulation group is normalized by its actual size. It has no gradient sanitization, uncertainty weighting or silent empty-loss skipping. Geometry loss combines masked CA pair-distance smooth L1 divided by 10 with CA-frame FAPE clamped at 10 Å and divided by 10; the angle contribution has weight 0.1. Frames with undefined orientations and stencils across masked residues are excluded, including all three residues used by endpoint frame normals. Dense contacts combine geometry embedding top-k contacts, local sequence neighbors and bottleneck distance contacts.

GotenNet consumes projected latent/codebook features plus learned token embeddings. Its local spherical harmonic safeguard handles coincident points. Coordinates are centered over valid nodes and scaled by a rotation-invariant maximum radius, then returned in original units and translation. To prevent backend padding contributions, each graph's valid nodes are compacted for GotenNet and expanded afterward. Frames are computed independently per chain. Frame rotations are derived before restoring translation to avoid cancellation for tiny output traces. The staged transformer is unchanged.

## Cache and resume

Importing the refiner no longer changes PyTorch's default dtype or the global
GotenNet backend. Numerical safeguards apply only to the local subclass. Valid
nonfinite coordinates/features or outputs raise explicit errors. Coordinate
normalization subtracts the masked center and divides by
`max(1, maximum_valid_radius / 64)`, then restores that scale and center.
The rotation/translation tests use `atol=1e-4`, `rtol=1e-4` in float32; scalar
and angle outputs remain invariant and undefined orientations stay masked.

```bash
python -m foldtree2.se3_validation --config configs/se3_validation_pilot.yaml --cache-only
python -m foldtree2.se3_validation --config configs/se3_validation_pilot.yaml --resume-from runs/se3_validation/pilot/last.pt
```

Restarting a completed output directory requires `--resume-from`; use a fresh directory for a new experiment. Incompatible resumes are rejected before replacing provenance.

Existing token-only SE(3) checkpoints are incompatible with the continuous-feature architecture. The production encoder/decoder pair is unchanged.

Caches store latent/codebook features, tokens, bottleneck coordinates, contact embeddings and adjacency, targets and masks. Their identity includes dataset/checkpoint SHA-256 hashes and preprocessing settings/code. A full dataset hash is reused only if path, device, inode, size, mtime and ctime agree. Delete `runs/se3_validation/hashes` to force rehashing. Cached and fresh features and full refiner outputs are checked at startup. Resume restores optimizer, scheduler, epoch, step and CPU/CUDA RNG state and rejects a changed dataset, checkpoint pair, split, model/loss implementation or optimizer configuration. Checkpoint writes use temporary files and atomic replacement. The existing production Lightning trainer also accepts `--resume-from` and validates production provenance; use the bounded runner for the canonical experiments.

## Results and larger-run gate

See [the generated report](se3_validation_report.md). Each run retains `events.jsonl`, `provenance.json`, `baseline.json`, `paired_metrics.json`, `length_profile.json`, `last.pt` and `best.pt`. Events contain loss terms, effective structure counts, nonfinite failures, gradient norms, throughput and peak host/GPU memory. Paired metrics compare the same structures and masks. The best checkpoint is selected by validation FAPE.

Require at least a 20% overfit training geometry reduction and improvement over decoder coordinates. Larger training requires lower held-out FAPE with no material structural regression. The report uses operational thresholds of 5% for error metrics and 0.01 absolute for TM-score/lDDT. Preserve failures and diagnose them before increasing capacity or duration.

The committed [evidence archive](se3_validation_evidence/README.md) preserves
learning curves, epoch logs, per-structure metrics, split IDs, provenance,
failure attempts, software/hardware records and test/resume evidence. The
[dated results record](results/2026-10-09-se3-residue-validation-v2.md) summarizes
the completed protocol and remaining work. Large caches, HDF5 inputs and model
checkpoints remain under the local `runs/se3_validation` directory; checkpoint
hashes and paths are listed in the archive's `checkpoint_inventory.json`.

Regenerate the published report from the committed evidence:

```bash
python scripts/report_se3_validation.py --root docs/se3_validation_evidence --output docs/se3_validation_report.md
```

Residue-level bond evaluation covers CA–CA spacing; atom bond metrics and atom refinement remain deferred. Missing production angle heads are reported as unavailable. Length profiles measure dense adjacency and forward/backward costs at batch size 1. The length-bucket utility groups nearby lengths for future batching; batch sizes greater than 1 need measurement before use. Observed success up to 384 residues is not evidence that larger graphs or atom graphs are safe.

After the pilot, measure warmed forward/backward costs and mixed-length padding without optimizer updates:

```bash
python -m scripts.profile_se3_validation --config configs/se3_validation_pilot.yaml --checkpoint runs/se3_validation/pilot/best.pt
python scripts/report_se3_validation.py
```

The profiler uses per-measurement peak counters and includes both individual structures and cached mixed-length batches. `collate_cached` and `length_bucket_batches` provide batching support; canonical training stays at batch size 1. A profiling batch that fits does not establish training stability at that batch size.
