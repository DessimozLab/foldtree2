# Curnagl TCS benchmark

This is the taxonomic consistency score benchmark, not SE3/TCS model training.
The isolated deployment extends the existing `ft2_benchmark` Snakemake workflow.
It imports native-tree inference from `src/topology_benchmark.py` and extracts the
four pure TCS functions from a snapshot of the cluster's `src/treescore.py`.
The live cluster checkouts, their edits, environments and other jobs stay intact.

Current submission/accounting review:
[2026-10-16 Curnagl status](curnagl_review_20261016.md). The genus dataset
generation completed, but final independent validation and a scored TCS sweep
have **not** completed. Do not infer job state from an old supervisor heartbeat.

## Cohorts and comparisons

Use the existing cluster OMA and CATH family roots in
[the configuration](../configs/curnagl_tcs.yaml). Inventory runs under Slurm, not
on the login node. Families are namespaced by dataset and relative path.
Every eligible family must have at least four unique taxa, existing selected
single-chain PDBs, matching metadata, and taxonomy IDs present in the frozen,
offline SQLite database. `finalset.csv`, when present, supplies the baseline's
retained cohort; otherwise `sequence_dataset.csv` supplies it. OMA retains the
baseline pLDDT minimum 10 / mean 50 filter; CATH does not use that filter.
Failures are listed explicitly in the inventory's exclusions. After the rebuild
stage, no downloads or later pruning are permitted. AA sequences are extracted from the selected PDB
residues and checked against metadata; a CATH domain may be a substring of the
metadata's full protein sequence.

CATH is rebuilt into a new deployment-local directory before inventory. Original
experimental PDBs are linked unchanged, not replaced with AlphaFold structures.
Metadata `Sequence` becomes the observed PDB sequence; the original reference is
retained as `UniProt_sequence`. Local reference alignment requires at least 95%
paired identity, 80% observed-residue coverage and 20 exact matches. This permits
expression tags and terminal truncations without accepting unrelated chains.
Missing, ambiguous or incompatible references trigger a cached, accession-specific
UniProt lookup. Every existing structure must pass; a failed member excludes the
whole family rather than silently reducing its cohort.

Inactive UniProt accessions are resolved through their explicit successor links,
accepting only one sequence-verified successor and retaining the original input
identifier. Deleted entries and incompatible current sequences can be recovered
from accession-specific UniSave historical records, but only when sequence and
taxonomy identify a unique verified reference. Successor chains and archived
records are cached and hash-bound. No identity/coverage threshold is lowered.
The historical reader extracts only accession, organism, taxonomy, version and
sequence fields, so legacy or newly introduced feature/reference annotation
syntax cannot break recovery. Complete record boundaries, valid residues and
the declared SQ sequence length remain mandatory; malformed identity/sequence
fields are reported as format failures rather than biological mismatches.
Recovery runs use a new output directory and can reuse the existing verified
taxonomy snapshot with `--taxonomy-db`; never modify a frozen benchmark cohort.

For a recovery rebuild, submit the following command through a CPU Slurm job
(8 CPUs, 32 GB), using a new `RECOVERY` directory and the cluster Python:

```bash
python -u scripts/rebuild_cath_families.py \
  --source ORIGINAL_CATH_ROOT --out RECOVERY/rebuilt/CATH_data \
  --taxonomy-dir RECOVERY/inputs/taxonomy \
  --taxonomy-db PREVIOUS_RUN/inputs/taxonomy/taxa.sqlite --workers 8
```

The reused SQLite database must match its sibling `snapshot.json` checksum.
An optional copy of the previous run's `rebuilt/uniprot-cache/` into the new
recovery directory saves requests; do not link a writable cache into a frozen
run. Recovery reports include the first failing accession for targeted follow-up.
An enlarged cohort requires a new inventory/plan and pilot, not an in-place edit
of previous benchmark manifests.

For a diagnostic audit of all chains in families excluded because no historical
reference passed, run `scripts/audit_cath_chains.py` under Slurm:

```bash
python -u scripts/audit_cath_chains.py \
  --summary RECOVERY/rebuilt/rebuild_summary.json \
  --cache AUDIT/cache --taxonomy-db PREVIOUS_RUN/inputs/taxonomy/taxa.sqlite \
  --out AUDIT/results --workers 8 --fetch-missing
```

Copy public reference caches into the new audit's cache first. The audit visits
every existing chain, including members after a failing chain; it never edits
PDBs, metadata, benchmark plans or inclusion decisions. `results/chains.csv`
reports best-reference identity, observed coverage and match counts;
`results/families/*.json` retains observed sequences, residue mappings, ranked
references, source hashes, taxonomy IDs and diagnostics. Historical PDB cross
references are unverified candidates, not proof of an original PDB-chain mapping.
Missing references may be downloaded only into the audit-local cache.

After explicit approval to prune, `scripts/retain_cath_subsets.py` builds a new
cohort from the recovery summary and completed audit JSON, without rerunning the
audited alignments. Fully validated families are linked unchanged; previously
excluded families retain only sequence-verified chains with resolvable original
ancestral taxonomy or a unique canonical reference taxon. At least four distinct
protein IDs are required. Remaining unaudited families are checked offline using
the existing reference cache. Taxonomic diversity is reported, not silently used
as an additional cutoff.

```bash
python scripts/retain_cath_subsets.py \
  --summary RECOVERY/rebuilt/rebuild_summary.json --audit AUDIT/results \
  --cache AUDIT/cache --taxonomy-db PREVIOUS_RUN/inputs/taxonomy/taxa.sqlite \
  --out NEW_SUBSET_RUN/cohort --workers 8
```

Each new family records the complete original/retained/excluded partition and
hash-bound audit evidence in `rebuild_provenance.json`. Inventory recognizes this
explicit subset schema; there is still no method-specific or downstream pruning.
`family_exclusions.csv` and `families/*.json` provide counts, fractions, reasons,
taxonomic diversity and excluded-chain metric summaries. `summary.json` reports
final cohort size and mean/population-SD/median/range of exclusions both across
all families and across previously failed families. Valid tips lost because a
whole family falls below four are counted separately from chain-validation
exclusions. A new benchmark inventory/plan is required before inference.

On 2026-10-12, subset deployment `20261012-cath-subsets-v8` (Slurm 65708277)
retained 485/488 CATH families and 13,370 protein tips. It preserved the 357
fully validated families and recovered 128 more. There were 225 chain-validation
removals: 213 sequence-gate failures, 11 missing/conflicting reference-taxonomy
cases and one missing reference. Five additional validated tips were omitted
with the undersized families `2.60.90.10` (3 tips), `3.20.20.70` (2 tips) and
`3.30.420.40` (no structures). Mean removals across the 131 previously failed
families were 1.72 with population SD 1.60 (median 1, range 0–10). The cohort is
at `/work/FAC/FBM/DBC/cdessim2/default/dmoi/tcs_runs/20261012-cath-subsets-v8/cohort/CATH_data`;
this is cohort validation, not a completed inference benchmark. Old plans and
the stopped pilot remain unchanged.

## Fresh genus-deduplicated CATH version

`scripts/generate_cath_families.py` implements the legacy CATH/SIFTS experimental
chain workflow in a separate output directory, with one representative per
canonical NCBI genus within each family and at least four genera. It joins the
current official CATH domain list, SIFTS chain-to-CATH/accession table and SIFTS
chain taxonomy table; all downloaded snapshots and the reused taxonomy DB are
hash-bound. Ambiguous chain mappings and taxa without a classified genus are
excluded, not assigned a guessed genus. This filter is within families only;
genera shared with the previous cohort are not globally excluded.

Run planning and generation under CPU Slurm allocations:

```bash
python scripts/generate_cath_families.py --out NEW_RUN/data \
  --taxonomy-db PREVIOUS_RUN/inputs/taxonomy/taxa.sqlite \
  --max-families 2000 --max-genera 100 --plan-only
python scripts/generate_cath_families.py --out NEW_RUN/data \
  --taxonomy-db PREVIOUS_RUN/inputs/taxonomy/taxa.sqlite --from-plan --workers 8
```

Candidates are ordered deterministically by CATH resolution, PDB ID, chain and
accession. Up to three alternatives per genus permit recovery from missing or
unusable structures, but only one successful tip is retained per genus. The
first deposited model and exact mapped author chain are selected from RCSB
mmCIF. Full observed chains are used, matching the legacy structural scope;
chains are not cropped to CATH domains. Only observed standard protein residues
with CA are exported. No missing residues or atoms are synthesized with PDBFixer.
PDB chain ID is aliased to A for format compatibility; observed residues and
coordinates are retained. The AA sequence is extracted from this same output.

Unlike the repaired legacy cohort's sequence-reference gate, this new dataset
uses independent curated SIFTS chain/accession/taxonomy membership and exact
deposited-chain provenance, not agreement with a current full UniProt sequence,
to establish identity. This evidence route is explicit in
`cath_generated_genus_v1` provenance and checked by benchmark inventory. It does
not relabel the previous sequence-gate failures as passing. New families record
their author chain, domain, taxon, genus, raw mmCIF, plan and output hashes.
`plan_summary.json`, per-family `reports/` and final `summary.json` expose all
selection exclusions, download failures and final counts. No old cohort is
overwritten and no old alignments/trees are reused for the new inputs.

For persistent dataset supervision, run `scripts/supervise_cath_generation.py`
with `--remote --deployment RUN_DIR` under a workstation user service. Its
operational copy and dependency live outside frozen scientific source.
Five-minute checks retain queue waits unchanged and record progress in
`generation-supervision.json`. Completed generation releases a CPU Slurm
validation job checking every successful family's curated inputs, deposited
chain sequence, genus uniqueness and actual benchmark input gate. Only then is
`generation-validated.json` written and supervision marked complete; all failed
families remain explicit exclusions.

Infrastructure interruptions (timeout/preemption/node/boot failure) permit at
most two safe resume attempts. Resume uses the original hashed generator and
plan, validates and reuses completed families, preserves incomplete family
directories under `recovery-preserved/`, and regenerates only unfinished work.
Conservative interrupted-download reservations are reconciled only after the
previous allocation is terminal. Scientific exclusions are not retried by this
mechanism. OOM, scientific validation failures, insufficient quota or rejected
authentication stop unattended work for inspection; no genus/provenance gate is
relaxed. Shared VPN access is preserved.

The generator, reference rebuild and chain audit propagate infrastructure errors
instead of reporting disk exhaustion, memory failures or transient network outages
as scientific exclusions. Missing deposited structures (HTTP 404/410) remain
explicit candidate exclusions in generation; rate limits/server errors stop the
job. The audit stops on reference HTTP failures for inspection. Completed frozen
runs retain their original code and counts; corrections apply to new deployments.

A fresh, checksum-verified NCBI taxonomy snapshot is built separately; the shared
taxonomy database remains untouched. Merged IDs and parent lineages are resolved
against this snapshot. Per-family rebuild provenance records residue mappings,
reference sources, thresholds and input hashes, which inventory validates again.
`rebuilt/rebuild_summary.json` and `rebuilt/reports/` record successes and failures.

All families use the same 13 methods:

- FT2 10, 20, 30, 40 and 50: mk2 graph conversion, paired production bundles,
  native MAFFT text alignment and matching multistate substitution matrix.
- AA: native MAFFT L-INS-i alignment and LG+G+I ML tree.
- 3Di: native Foldmason alignment and Q.3Di.AF+G+I ML tree.
- Original FoldTree: fident, alntmscore and lddt, each raw and exp-corrected,
  sharing one exhaustive structural all-v-all table per family and using QuickTree.

ML inference uses seed 42 and 10 random plus 10 parsimony starts. Each method
has its own tree. Every tree receives baseline branch cleanup (negative branches
set to zero, count recorded) and MAD rooting before the three baseline metrics
`score`, `root_score`, and `root_score_nr` are calculated. These raw scores depend
on family size, taxonomy, tree shape and rooting; compare methods within the
same family and report OMA and CATH separately.

The 50-state bundle retains its explicit, hash-bound nonconvergence acceptance.
Preflight validates this record; it never relabels that matrix as converged.

## Deployment and launch

Read the Curnagl access runbook and use the saved `unil-access` helper. Do not
record credentials in scripts or logs. Build a selective archive locally:

```bash
python scripts/stage_curnagl_tcs.py \
  --out /tmp/tcs-RUN_ID.tar.gz \
  --benchmark /path/to/ft2_benchmark \
  --original /path/to/fold_tree \
  --foldmason /path/to/conda/env/bin/foldmason
```

Transfer it and `configs/curnagl_tcs.yaml` into a **new** directory under
`/work/FAC/FBM/DBC/cdessim2/default/dmoi/tcs_runs/RUN_ID`, verify the archive
checksum, and extract there. Run the staged configuration script remotely:

```bash
python3 foldtree2/scripts/configure_curnagl_tcs.py \
  --deployment "$PWD" --template curnagl_tcs.yaml
```

This snapshots tracked legacy workflow/source worktree content, including edits,
and renders run configuration and five CPU Slurm job scripts. It
uses existing Snakemake 7.20 and the existing Python runtime, with staged source
selected through `PYTHONPATH`; Foldmason and its libraries form an isolated tool
layer. No global package installation is performed.

The generated `runtime-check.sbatch` runs `scripts/check_curnagl_tcs.py` under a
small Slurm allocation with the deployment's PATH/PYTHONPATH/library setup. It checks archive
hashes, all five bundles, the real baseline's golden TCS result, required tools,
and Snakemake dry-run parsing, then writes `runtime-check-passed.json` with package
versions. Supervision will not release the pilot without this marker.
The archive includes the package-local RAxML binary and its license, not just the
Python wrapper. Runtime checking requires successful native RAxML/Foldseek/
Foldmason version commands; executable presence or error-shaped version text is
not sufficient. The RAxML binary is also hash-bound in the frozen inventory.
If the existing environment lacks a required package, download a wheel for its
Python 3.12 and x86-64 platform, transfer it, and run the isolated installer with
the cluster interpreter **before inference begins**:

```bash
python scripts/install_curnagl_tcs_layer.py --deployment RUN_DIR package.whl
```

This writes a new `python-layer/` and a wheel/content hash manifest, updates
generated job PYTHONPATHs, and leaves the shared Conda environment unchanged.
Never install another version over an existing runtime layer. Re-run the runtime
check after installing it; retain `runtime-layer.json` with result provenance.
Supervision schedules the runtime check after inventory unless a successful
check already exists. A failed runtime check stops for inspection and repair.

Run supervision on the workstation (an existing authenticated VPN is required):

```bash
python scripts/supervise_curnagl_tcs.py \
  --remote --deployment /work/FAC/FBM/DBC/cdessim2/default/dmoi/tcs_runs/RUN_ID
```

`--once` performs one check/submission. The default interval is five minutes.
For persistent supervision, run this under a user service and retain its log.
The stages are:

1. CATH rebuild and isolated taxonomy refresh: 8 CPUs, 32 GB, eight hours;
   require at least one fully validated family before proceeding.
2. Inventory: 4 CPUs, 16 GB, four hours; freeze `plan.json` and `pilot.json`.
3. Runtime check: 4 CPUs, 16 GB, 30 minutes; verify tools, bundles and workflow.
4. Pilot: the first sorted eligible OMA and CATH families, 8 CPUs, 32 GB,
   four hours; require all **26** method-family scores and valid provenance.
5. Full sweep: only after pilot completion and peak RSS accounting <=24 GB;
   32 CPUs, 96 GB, 24 hours, at most four 8-thread inference tasks concurrently.

Only timeout, preemption, boot failure or node failure permit automatic
resubmission, at most twice per stage. Scientific failures, missing tools,
OOM, failed score gates and authentication/host-key failures stop supervision
for inspection. New submissions require at least 20 GiB available in both the
filesystem and the applicable project quota (`quotacheck`). The runner uses
`--rerun-incomplete`, never `--touch` or unconditional `--unlock`. Inspect any
stale lock and confirm the owning process is gone before manually unlocking.

## Inspecting and extending the run

The operational `supervisor.py` lives outside the frozen inference source; its
storage/accounting checks can be maintained without changing scientific assets.
`supervision.json` records heartbeat, current and previous job IDs, scheduler
state, attempts, accounting and pilot memory. Slurm logs live in `logs/`;
per-family logs, trees, native alignments, rooted trees and completion hashes
live in `results/{pilot,full}/families/`. `tcs_scores.csv` contains all paired
scores; `summary.csv` reports each dataset/method separately. `validated.json`
is written only after every expected score passes provenance validation.

Inspect with `squeue -j JOB_ID` and
`sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS`. Retain the input
archive manifest, baseline snapshot provenance, taxonomy checksum, frozen plans,
preflight tool hashes and all logs with the results. Family structures are linked,
not copied, but they must remain unchanged and available for the duration.

Change cohorts, tools, code, model bundles or settings only in a new deployment.
Do not reuse old trees based solely on their filenames: completion signatures and
output hashes must match. After completion, document the deployment ID, cohort
counts/exclusions, full method coverage, any accepted limitations, and links to
the summary tables. Adding a method requires expanding the registry, pilot gate,
coverage tests and this document before launching a new frozen run.
