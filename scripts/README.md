# Scripts — experiment drivers, benchmarks, HPC ops and release tooling

[Project](../README.md) | [Source map](../foldtree2/src/README.md) | [SE3 toolkit](./se3/README.md)

The `scripts/` tree holds experiment drivers, benchmarks, HPC operations and
release tooling, run with `python scripts/<folder>/<file>.py` from the repo
root. Each moved script bootstraps the repo root itself, so any working
directory works; everything imports against the editable `foldtree2` install.
SE(3) validation and investigation work lives in [se3/](./se3/README.md);
contact, benchmark, HPC, CATH, alphabet and ancestral work each have their own
package below.

## Folder index
| Package | Purpose |
|---|---|
| [se3/](./se3/README.md) | SE(3) validation, reconstruction and investigation toolkit |
| [contact/](./contact/README.md) | contact and pairwise-distance head training experiments |
| [benchmarks/](./benchmarks/README.md) | benchmark drivers and shared phylogenetic-information helpers |
| [curnagl/](./curnagl/README.md) | Curnagl HPC TCS operations (Slurm) |
| [cath/](./cath/README.md) | CATH family cohort preparation for the TCS benchmark |
| [alphabets/](./alphabets/README.md) | learned-alphabet experiments and substitution-matrix references |
| [ancestral/](./ancestral/README.md) | ancestral state reconstruction helpers |

## Shared root modules
- [benchmark_common.py](./benchmark_common.py): artifact and provenance primitives, imported across folders.
- [audit_cath_chains.py](./audit_cath_chains.py): shared CATH chain audit helper, imported by the CATH scripts.
- [rebuild_cath_families.py](./rebuild_cath_families.py): rebuild CATH family inputs, shared with `scripts/cath/`.
- [geometry_mlflow.py](./geometry_mlflow.py): MLflow tracking for geometry runs; test-covered by [foldtree2/tests/test_geometry_mlflow.py](../foldtree2/tests/test_geometry_mlflow.py).
- [check_documentation.py](./check_documentation.py): documentation gate, referenced by `.github/` workflows.
- [check_distribution.py](./check_distribution.py): distribution gate, referenced by `.github/` workflows.
- [stage_conda_source.py](./stage_conda_source.py): stage the conda source distribution for release/CI.

## Shell drivers
Queue and packaging glue at the scripts root; launch with `bash scripts/<name>.sh`:
- [clear_gpu_memory.sh](./clear_gpu_memory.sh): free residual GPU memory between queue items.
- [kill_zombie_processes.sh](./kill_zombie_processes.sh): reap zombie training processes.
- [relocate_workstation_data.sh](./relocate_workstation_data.sh): relocate workstation data stores.
- [queue_contact_large_batch32.sh](./queue_contact_large_batch32.sh): queue the large contact batch-32 screen.
- [queue_contact_levers.sh](./queue_contact_levers.sh): queue the contact-levers trainer.
- [queue_contact_muon_scale.sh](./queue_contact_muon_scale.sh): queue the Muon-scaled contact run.
- [queue_contact_xsa.sh](./queue_contact_xsa.sh): queue the XSA contact screen.
- [run_embedding_comparison.sh](./run_embedding_comparison.sh): run the embedding comparison.
- [run_monodecoder_configs.sh](./run_monodecoder_configs.sh): run monodecoder config sweeps.
- [smoke_test_conda_cli.sh](./smoke_test_conda_cli.sh): smoke-test the conda CLI.
- [test_conda_packaging.sh](./test_conda_packaging.sh): test conda packaging end-to-end.

## Reproducibility
Most drivers write under `runs/` (gitignored); evidence lands in `docs/` via the
report scripts. Pinned historical copies of these scripts live in
`docs/*_evidence/sources/` and `runs/*/sources` — never edit those, they record
what actually ran. No script here is currently obsolete: all are referenced by
docs, tests, CI or other scripts.