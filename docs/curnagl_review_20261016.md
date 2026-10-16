# Curnagl submission and code review: 2026-10-16

[TCS workflow](curnagl_tcs.md) | [Project](../README.md)

Read-only cluster inspection on 2026-10-16, approximately 14:12-14:15 CEST,
checked Slurm accounting, frozen plans, summaries, supervisor state and pilot
logs under `/work/FAC/FBM/DBC/cdessim2/default/dmoi/tcs_runs/`.
No jobs were submitted, cancelled or modified; no deployment was patched.
The shared VPN was left connected.

This review covers the Curnagl TCS deployment and CATH generation/recovery code.
It does not validate the separate SE(3)/Clariden or PhyloJEPA workloads.

## What actually ran

| Deployment / stage | Slurm job | Observed result |
| --- | --- | --- |
| `20261011-pilot-v2`, inventory | 65707749 | Failed, exit 1. Its supervisor is stopped with `needs_attention`. |
| Initial runtime check, then repaired check | 65707751 / 65707754 | Failed, then completed; this was not a completed inference pilot. |
| `20261011-cath-rebuild-v4`, rebuild | 65707944 | Completed: 266/488 CATH families passed, 222 failed. |
| Same deployment, inventory | 65707949 | Completed: frozen plan has 2,514 OMA + 266 CATH = 2,780 eligible families and 2,169 exclusions. |
| Same deployment, runtime check | 65707967 | Completed, but did not detect a missing native RAxML target. |
| Same deployment, TCS pilot | 65707973 | Failed, exit 1, after 1 min 22 s. No full sweep was released. |
| `20261011-cath-recovery-v5` | 65707979 | Completed: 355/488 CATH families passed. |
| `20261011-cath-recovery-v6` | 65708115 | Completed: 357/488 CATH families passed. |
| `20261012-cath-chain-audit-v7` | 65708242 | Slurm reports failed, exit 1. Do not label this whole job complete just because per-family audit files exist. |
| `20261012-cath-subsets-v8` | 65708277 | Completed: 485/488 families, 13,370 tips; 225 chain-validation exclusions plus five valid tips lost with undersized families. |
| `20261012-cath-genus-v9`, plan | 65710543 | Completed: 1,342 eligible families, below the requested cap of 2,000; maximum 100 genera per family. |
| Same deployment, generation pilot | 65710544 | Completed: one family, 100 tips. Generation pilot only, not a scored tree benchmark. |
| Same deployment, full generation | 65710545 | Completed, exit 0, in 1 h 7 min: 1,339/1,342 families and 18,545 tips in the generation summary. Batch peak RSS 4,191,020 K. |
| Same deployment, final independent validation | None found | No `generation-validated.json` or `validate-*.log`; no validation submission recorded. Not independently validated yet. |

The genus-generation exclusions are `1.20.5.290`, `3.30.46.10` and
`3.90.175.10`: each retained only three tips after rejecting reuse of an accession
already selected for another genus. Preserve these exclusions; do not silently
relax accession uniqueness.

The genus supervisor still records job 65710545 as `PENDING` with heartbeat
2026-10-12T21:26:07Z. The workstation service
`foldtree2-cath-genus-v9.service` is failed (exit 1). Its stale JSON is not live
scheduler evidence. This review did not establish the cause of that service
failure or restart it.

At inspection, `squeue -u dmoi` showed no active TCS/CATH jobs. The three pending
PhyloJEPA jobs were unrelated and left untouched.

## Verified pilot failure and local corrections

1. **Missing deployed RAxML binary.** The pilot's FT2-40 `ml.command.log` says
   `Executable raxml-ng not found` at the staged package-local native path.
   [Archive staging](../scripts/stage_curnagl_tcs.py) included the Python wrapper
   but omitted that native executable. It now includes the executable and its
   license. [Inventory](../scripts/curnagl_tcs.py) also freezes the binary hash.
2. **False-positive runtime gate.** The shared baseline's version collector
   records stdout/stderr without requiring exit zero. Thus wrapper error text
   could pass the previous runtime check. The deployment-specific
   [runtime check](../scripts/check_curnagl_tcs.py) now requires successful
   RAxML, Foldseek and (when configured) Foldmason version commands before
   writing a passing marker.
3. **Infrastructure errors could change the cohort.** Broad candidate/rebuild/
   audit catches could turn disk/memory/network failures into exclusions.
   [Generation](../scripts/generate_cath_families.py),
   [rebuild](../scripts/rebuild_cath_families.py) and
   [audit](../scripts/audit_cath_chains.py) now propagate infrastructure failures.
   Generation/rebuild still record HTTP 404/410 as missing candidates/references;
   server/rate-limit errors stop the job. Empty local alignments become explicit
   sequence-gate failures rather than uncaught indexing errors.
4. **Source snapshot could include unrelated project edits.**
   [Configuration](../scripts/configure_curnagl_tcs.py) now scopes Git status and
   the recorded patch to the legacy subdirectory. This matters because the
   observed cluster Git root encloses multiple projects.
5. **Chain aliasing touched a copied model's chain registry.** Detach the selected
   copied chain before renaming it to `A`; this removes sibling-ID warnings
   without changing selected residues or coordinates.

These corrections are local only. Existing archive manifests, frozen source,
plans and scientific results on Curnagl remain unchanged.

## Local validation

- Cluster/cohort regressions cover archive contents and binary hashes, failing
  executable wrappers, scoped snapshots, retry/coverage/memory/quota gates,
  chain identity, genus selection and infrastructure failures: **45 passed**,
  including the final exact 50 GiB cache-budget regression.
- Full CPU suite after the implementation changes: **224 passed, 1 skipped,
  1 xfailed**. GPU execution was deliberately disabled to preserve active work.
- Documentation checks: **51 documents, 65 CLI examples, zero errors**.
- Cluster/cohort scripts parse successfully and `git diff --check` passes.
- No new Slurm pilot or final-validation allocation was run. Local tests do not
  promote the old frozen deployments or establish completed full TCS results.

## Evidence identifiers

The following hashes identify the inspected files, not public data downloads:

- v9 generation plan SHA-256:
  `2feec0f7db38427d4277cf462b85bab61aba6f2db7149e066f28f539bfe2453f`
- v9 `data/summary.json` SHA-256:
  `bab828f3bb5ea1bff3d997b9c5a7b2e74fa2123a942ff12a02d4da27c988d552`
- v8 `cohort/summary.json` SHA-256:
  `87a77733b85f7fe25570228ff3fae14ab9cfbd9c4ef4b69908bffa19590f6fe9`
- Reused taxonomy SQLite SHA-256:
  `a0fba034811aec7be383f15df3f53ea99f48861b7f4ca6fac41bdd4ef0554596`

Historical inventory estimates (including 3,404 OMA families) are not the v4
frozen eligible count of 2,514. The repaired v8 cohort and newly generated v9
cohort are different protocols; never pool or silently substitute them.

## Follow-up gates

1. Inspect the failed workstation service log privately before reconnecting or
   restarting supervision. Authentication failures must not trigger blind retries.
2. After approval, submit only the existing v9 final validation job, using the
   original frozen generator/plan. Confirm all 1,339 successful families, 18,545
   tips, genus uniqueness, deposited-chain provenance and real inventory gates.
   Preserve failures and require the hash-bound validation marker.
3. Create a **new** TCS deployment from reviewed source with the intended CATH
   cohort; verify transfer hashes, real native tools and all five model bundles.
4. Create a new inventory/plan, run the 26-score OMA/CATH pilot, and inspect
   provenance, tool logs and accounted memory before authorizing the full sweep.
5. Compile paired per-family scores and uncertainty only after complete coverage.
   A successful dataset-generation job is not a topology benchmark result.

## Recovery started after approval, 2026-10-16

The observations above remain the pre-recovery record. After approval to get the
benchmark running, the following jobs were submitted:

| Stage | Job | Status at recovery check |
| --- | --- | --- |
| Original v9 independent generation validation | 66047632 | Running; original source and supervision checksums passed before submission. |
| New v10 runtime check | 66047940 | **Completed, exit 0**, 1 min 53 s, batch peak RSS 631,080 K; native tools, all five bundles, golden TCS and workflow parsing passed. |
| New v10 inventory | 66048014 | Queued with `afterok:66047632`; a separate marker/hash/count gate must pass before inventory starts. |
| New v10 26-score pilot | 66048060 | Queued with `afterok:66048014:66047940`; cannot start until both inventory and runtime succeed. |

The new deployment is
`/work/FAC/FBM/DBC/cdessim2/default/dmoi/tcs_runs/20261016-genus-tcs-v10`.
It uses the unchanged OMA source and the **new genus-deduplicated v9 CATH**
cohort, not the repaired v8 cohort. Its archive has 187 files, including all
five production bundles and the previously missing native RAxML executable.
The archive transfer was verified against SHA-256
`1e462625df451e5a11e9abfb1ef8dae4ef0b4f2d50eab5e63e843057923285cc`.
The immutable archive manifest records working-tree content hashes and base
commit `86e8583`; that commit alone is not the source snapshot.

The offline Gemmi 0.7.5 wheel was copied from the old deployment and installed
into a new deployment-local layer, with wheel/content hashes. No shared runtime
environment was changed. Operational scripts and job files are recorded
separately in `operational-source.json`, outside frozen scientific source.
Inventory requires the v9 marker to match the frozen plan/summary hashes and
the exact 1,342 planned / 1,339 validated / 18,545-tip / zero-duplicate scope.
It records the validated-cohort marker hash in `cohort-validation-input.json`.

The old supervisor log identifies its failure as **SSH unreachable while the
VPN remained connected**, not a failed generation job. It was not restarted.
New workstation monitoring uses user service `foldtree2-curnagl-v10.service`;
logs are under `runs/curnagl_tcs/20261016-genus-tcs-v10/`.
The first new monitor invocation also failed; a separate service-environment
access diagnostic then succeeded, and it was deliberately restarted once.
The Slurm dependency chain does not depend on workstation monitoring staying
alive. Any failed prerequisite leaves downstream work blocked for inspection.

The new controller state has `stop_after_pilot: true`. A complete pilot records
`pilot_complete`; **no full-sweep job has been submitted or authorized by this
recovery**. Final cohort validation, inventory coverage and pilot results remain
pending until their actual output/accounting gates pass.

Recovery-controller regressions passed (16 targeted tests); the final full CPU
suite passed with **226 passed, 1 skipped, 1 xfailed**. Documentation checks still
cover 51 documents / 65 examples with zero errors. The new monitor has produced
a verified remote heartbeat. Slurm statistics confirm active CPU work and reads
in the original validation allocation, not merely an accepted submission.
