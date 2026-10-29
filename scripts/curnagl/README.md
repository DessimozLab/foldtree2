# Curnagl HPC TCS operations

Slurm-driven Curnagl TCS benchmark operations: the frozen-cohort entry point and the gated deployment callbacks that advance, supervise and check a run.

[Scripts home](../README.md) | [Project](../../README.md) | [Source modules](../../foldtree2/src/README.md) | [SE3 toolkit](../se3/README.md)

These are the Slurm callbacks for the Curnagl TCS benchmark, run on the UNIL HPC
cluster. The frozen-cohort entry point drives a single phylogenetic
reconstruction campaign end-to-end through a gated pipeline; the callbacks
advance, supervise and check each step before the next unlocks, so a failed or
stalled step never silently progresses the cohort.

## Modules
- [curnagl_tcs.py](./curnagl_tcs.py): main TCS benchmark entry point; offline frozen-cohort extension.
- [supervise_curnagl_tcs.py](./supervise_curnagl_tcs.py): bounded Slurm supervision of a running TCS campaign.
- [advance_curnagl_tcs.py](./advance_curnagl_tcs.py): advance a TCS run to its next gated step.
- [check_curnagl_tcs.py](./check_curnagl_tcs.py): check TCS run status and outputs.
- [check_curnagl_raxml_affinity.py](./check_curnagl_raxml_affinity.py): check RAxML affinity computations on Curnagl.
- [configure_curnagl_tcs.py](./configure_curnagl_tcs.py): configure a TCS run before staging.
- [install_curnagl_tcs_layer.py](./install_curnagl_tcs_layer.py): install the Curnagl TCS conda layer into the environment.
- [isolate_curnagl_tcs.py](./isolate_curnagl_tcs.py): isolate a TCS run from shared state.
- [curnagl_physical_step.py](./curnagl_physical_step.py): run a single physical TCS step on a compute node.
- [resume_curnagl_affinity.py](./resume_curnagl_affinity.py): resume an interrupted affinity run.
- [resume_curnagl_keepgoing.py](./resume_curnagl_keepgoing.py): resume a keepgoing run.
- [prepare_curnagl_step_recovery.py](./prepare_curnagl_step_recovery.py): prepare step-recovery inputs for a stalled run.
- [stage_curnagl_tcs.py](./stage_curnagl_tcs.py): stage TCS inputs onto the cluster scratch.

## Workflow
The full end-to-end workflow is documented in
[curnagl_tcs.md](../../docs/curnagl_tcs.md). These scripts talk to Slurm
(`sbatch`/`srun`) and require cluster credentials; most are one-step callbacks
in a gated deployment pipeline — the `advance`/`supervise`/`check` trio drives a
run forward, watches it, and verifies its outputs before the next step unlocks.
`curnagl_tcs.py` is the frozen-cohort entry point; the rest compose around it.
`configure_curnagl_tcs.py` and `stage_curnagl_tcs.py` prepare and land the
inputs, `install_curnagl_tcs_layer.py` provisions the conda layer, and the
`resume_*`/`prepare_curnagl_step_recovery.py` pair handles recovery from a
stalled or interrupted step without re-staging the whole cohort.

## Caveats
Do not run these on login nodes — `curnagl_physical_step.py` and the resume
helpers must execute on a compute node allocated for the cohort, otherwise the
affinity and keepgoing steps will be killed by the scheduler before they finish.
Pair each run with its Slurm job id and cohort manifest in provenance before
advancing, since the gate checks compare outputs against the manifest declared
at `configure` time. `isolate_curnagl_tcs.py` is the escape hatch for a run that
has drifted into shared scratch; use it before resuming.
`check_curnagl_raxml_affinity.py` validates the RAxML affinity step in
particular; run it after that step and before advancing, since a bad affinity
output is not caught by the generic TCS status check.