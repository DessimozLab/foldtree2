# Curnagl RAxML recovery, 2026-10-19

Deployment:
`/work/FAC/FBM/DBC/cdessim2/default/dmoi/tcs_runs/20261016-genus-tcs-v10`.
The full benchmark remains 3,853 families, 13 methods and 50,089 expected scores.

## Rejected multithreaded policy

Full job 66157528 failed on the AA inference for `CATH__3.20.20.330` with
RAxML's CPU oversubscription error. An exclusive eight-CPU Slurm step did not
make eight-thread RAxML reliable. Its reported physical-core count was also
misleading: Linux sysfs verified eight distinct physical cores in a mask that
RAxML reported as four.

We tested explicit physical-core affinity, 16 reserved CPUs per eight-thread
inference and two concurrent searches within a 32-CPU allocation. Jobs
66226303 (dna006, 94 seconds) and 66226375 (dna003, 96 seconds) passed 45/45
searches each. A real workflow check, 66226386, completed all 27 tasks and all
13 method-family scores in 194 seconds.

That evidence was insufficient. The resumed allocation 66226560 reran the
same 45 checks before launching TCS and failed one: repetition 1 of
`CATH__1.10.10.10-FT2_50`. Its Linux mask still exposed eight distinct physical
cores. The startup gate correctly stopped TCS before the full DAG launched.
**The multithreaded affinity policy is rejected.** The failures concern RAxML's
multithreaded timing detector; extra reservation and pinning did not eliminate
them. Neither the exact contribution of scheduler contention nor every timing
condition has been isolated.

An earlier probe, 66225928, was explicitly cancelled after 174 seconds because
repeating full pilot alignments would exceed the intended short verification
budget. A first workflow check, 66226347, stopped before inference because an
operational helper was absent from the older deployment. Both attempts and
the failed startup remain recorded; none is counted as passing.

## Reliable execution policy

Unfinished AA, 3Di and FT2 inference tasks use **one thread**, including RAxML.
Separate family tasks run concurrently; the 96-GB budget permits up to four
24-GB inference tasks. Other structural kernels retain their eight-thread
setting. The original RAxML binary, empirical matrices, seed 42, one search
worker and `rand{10},pars{10}` starts are retained. No safety check is disabled.
Single-thread inference trades within-family speed for reliable execution and
parallelism across families.

The operational workflow records the actual thread count through the frozen
driver's existing provenance. Completed methods retain their original thread
signature and output hashes; unfinished ML directories are moved to a
job-specific recovery directory before restarting with a one-thread signature.
The driver continues to validate all signatures and completed output hashes.
No frozen scientific source, production models or full-cohort plan is edited.

Exclusive, exact Slurm steps reserve two CPUs per tool thread. The affinity
wrapper reads permitted CPU socket/core IDs from Linux sysfs and selects one
hardware thread per physical core. It rejects insufficient topology and records
the before/after masks. OpenMP/BLAS limits follow the actual rule thread count.
The full allocation remains 32 CPUs, 96 GB and 24 hours.

## Verification and release

The bounded probe uses the first eight existing alignment rows from each of
seven ML models in both validated pilot families, plus the exact original
failed AA alignment. Each input is searched three times: 45 searches per node,
four concurrent single-thread searches. Source hashes and selected rows are
recorded. Sampled alignments produce no benchmark scores.

- Single-thread probes: **66226638** (dna003) and **66226639** (dna006),
  **45/45 searches passed on each node**.
- Fresh full-input family workflow: **66226705**. Its separate plan/output
  directory recomputes every method for `CATH__3.20.20.330` and requires all
  13 scores, valid aggregation and one-thread provenance for every ML method.
  This prevents already completed eight-thread results from satisfying the new
  execution test.
- Release controller: **66226706**. It requires successful Slurm accounting for
  all verification steps, matching binary/worker/workflow hashes, both-node
  checks, the fresh family gate, and the original 26-score pilot validation.

Both 45-search node checks and the fresh 13-method family aggregation passed.
The fresh gate confirms one-thread provenance for every ML method. Activation
job **66226755** submitted full TCS job **66226757**, with supervisor callback
**66226758**. Its own 45-search startup verification also passed on dna003;
the full allocation is running. Source/workflow/worker hashes and the submission receipt are
retained under `single_thread/` in the evidence directory.

The initial fresh subset check 66226640 and its dependent release 66226641 were
cancelled when review found that the subset retained both dataset names despite
containing only a CATH family. A fresh plan/output directory restricts its
aggregation to CATH; the full-cohort plan is unchanged. Both attempts remain
auditable. Local verification passes all **24 TCS tests**, including legacy
thread-signature reuse and preservation of unfinished ML attempts.

Each full allocation checks release hashes and reruns the bounded RAxML probe
on its assigned node before starting TCS. An error stops the allocation for
inspection. The existing quota check and infrastructure-only retry limits
remain in force. Launch verification is not a completed full benchmark.

Operational files are under `ops-20261019-single-thread/`; earlier rejected
policy files remain under `ops-20261019-affinity-v2/`. Reports and accounting
are retained in [the evidence directory](curnagl_affinity_evidence/20261019/).

## Keep-going continuation, 2026-10-20

Full job 66226757 stopped after 9h37m with 7,947/103,067 workflow steps
reported complete (8%). The stopping task was the TCS score for
`CATH__1.10.287.740/FoldTree_fident_exp`: its tree taxa differ from the frozen
taxonomy. The supervisor preserved the failure and did not retry it.

The user requested Snakemake `-k` so independent tasks continue despite failed
tasks; diagnosis of incomplete results is deferred. The operational launcher
now adds only `-k` to its existing Snakemake invocation. The 32-CPU, 96-GB,
24-hour CPU allocation, startup RAxML probes, provenance assertions, frozen
taxonomy checks and quota checks remain in force. Completed outputs are reused
under their existing signatures; no taxa are pruned and no failed score is
marked successful. Final aggregation remains blocked until its dependencies
complete, so an eventual failed Slurm state can coexist with useful progress.

[`scripts/curnagl/resume_curnagl_keepgoing.py`](../scripts/curnagl/resume_curnagl_keepgoing.py)
checks the identified failed allocation and launcher provenance, backs up the
launcher/supervision state, validates the changed batch script and activates the
existing supervisor. It rejects duplicate application. Local checks verified
that only `-k` is inserted and malformed/repeated changes are rejected.

Activation job **66276634** was submitted. The old operational files are saved
under `ops-20261020-keepgoing/`; its `receipt.json` records the activation and
new launcher SHA256
`06f3657a2596c291976c0f61a2878e2768d8e09f072aff5d956544fecb4df1d4`.
The supervisor keeps its existing bounded infrastructure retries; scientific
failures still require inspection after independent tasks finish.

Activation completed and submitted full job **66276636**, with supervisor
callback **66276638**. The full job is **PENDING (QOSGrpNodeLimit)** at the
status check; the callback waits on its completion. Quota validation passed
with approximately 339 GB of project headroom. This is a queued continuation,
not a completed benchmark.
