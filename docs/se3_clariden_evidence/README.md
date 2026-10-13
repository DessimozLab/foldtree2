# Clariden rollout implementation evidence

Local numerical and integration verification on 2026-10-13. This archive contains
CPU/CUDA checks, actual training logs, per-structure validation pairs, source
hashes and restart comparisons. No Clariden allocations have been submitted.
The frozen input bundle is stored separately under runs and on project storage;
production checkpoint files and frozen tensors are not committed here.

`cuda_capacity_checks.json` records the original numerical failures;
`cuda_stable_norm_checks.json` records the medium model after the algebraically
equivalent final tensor-norm correction. The depth-six model remains blocked.
`medium-anomaly.log` identifies the original norm-backward failure.

`verification_final` contains serial/two-rank DDP and interrupted-run evidence.
The final release verification records source hashes for the staged code revision.
