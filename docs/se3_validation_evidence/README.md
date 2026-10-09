# SE(3) validation evidence

Protocol: `se3-residue-validation-v2-endpoint-masks`. Recorded on 2026-10-09.
The [report](../se3_validation_report.md) is generated from this directory.
The [workflow](../se3_validation.md) documents commands, losses and limitations.

## Final experiments

| Directory | Training / evaluation | Epochs | Result |
|---|---:|---:|---|
| [smoke](smoke) | 1 / 1 | 1 | Finite outputs and gradients; frozen state unchanged |
| [overfit](overfit) | 32 / same 32 | 30 | 0.26% geometry loss reduction; 20% gate failed |
| [pilot](pilot) | 512 / disjoint 128 | 20 | Lower FAPE for 124/128 structures; larger-training gate remains blocked by overfit |

Each directory contains `events.jsonl`, `provenance.json`, `baseline.json`,
`paired_metrics.json`, `diagnostics.json`, `length_profile.json` and
`learning_curves.png`. Provenance records configuration, selected IDs, split
membership, exclusions, production checkpoint/dataset hashes and implementation
identity. Paired metrics use identical coordinates masks for the baseline and
refiner. Epoch events record loss terms, gradients, sample counts, throughput,
memory and completion checks.

## Verification and resources

- [CUDA correctness log](cuda-correctness-test.log): all 59 tests passed,
  including the original 29 geometry/frame tests.
- [Resume log](overfit-resume.log): restored epoch 30 and step 120 without
  additional updates.
- [Rejected resume](resume-rejection-check.json): incompatible provenance was
  rejected before replacing the audit record.
- [Warmed dense profile](dense_profile.json) and [profile log](dense_profile.log):
  eight measurements, finite backward gradients, no optimizer steps.
- [Hardware](hardware.json) and [software](software_versions.json): physical
  GPU 1, mapped to CUDA device 0; unrelated GPU 0 work was left running.
- [Source manifest](source_manifest.json): base revision and SHA-256 hashes of
  packaged implementation, configurations and tests. Training provenance binds
  the executed refiner and loss functions; rendering changes followed training.
- [Checkpoint inventory](checkpoint_inventory.json): hashes, sizes and local
  paths of experimental checkpoints. These weights, production checkpoints,
  HDF5 input and frozen-output caches are retained locally, outside this archive.

## Earlier attempts

`smoke_attempt_01` failed deterministic cache parity; `overfit_attempt_01`
failed target eligibility before training. `pilot_attempt_01` was interrupted
before an epoch completed to fix frame cancellation after restoring translation.
Their provenance, available events and top-level traceback logs are retained.

The completed `*_endpoint_mask_attempt` runs preceded the final endpoint mask
correction. Endpoint normals use three residues, and their masks previously
checked only two. Two overfit structures and nine pilot structures were
affected. Those results are separate from the final protocol and are not pooled
with it. The bounded experiments were repeated after the correction.

This archive contains about 1.7 MB of evidence. It contains no raw structure
coordinates, model weights, HDF5 data or cached latent tensors. Structure-level
refiner holdouts do not establish independence from production pretraining.
