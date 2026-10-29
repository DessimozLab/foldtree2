# Final benchmarking results to compile

[Project](../README.md) | [Experiment commands](experiments.md) |
[Benchmark notebooks](../foldtree2/notebooks/benchmarks/README.md) |
[Production artifacts](production_alphabet_readiness.md) |
[Release gates](releasing.md)

This is a compilation checklist, **not a claim that every benchmark passed**.
Prepared on 2026-10-16 from local Codex history and available manifests. Do not
launch expensive reruns just to populate this document; audit reusable artifacts
first and rerun only missing/invalid scopes.

## Evidence and scope

The local Codex session beginning 2026-09-30
(`01a0f149-3a81-7a50-b9bd-ab490cfb089c`, through October 12) records:
retaining 20/30/40 models, training 10/50, rebuilding all matrices after counting
fixes, extracting notebook experiments to scripts, adding AA/3Di comparisons,
native ancestral uncertainty, species-supermatrix modes and cluster TCS work.
The October 8 SE(3) validation session
(`01a11ba6-3302-7951-a7f2-26af6eebb30b`) and
[dated SE(3) records](se3_validation_report.md) cover a **separate** geometry
investigation. Session conclusions are leads; final numbers require the actual
result tables, coverage and provenance.

Local artifact paths below are audit locations, not distributed data or public
downloads. Keep large raw outputs out of the code package. Prepare compact,
versioned result records using the [maintenance template](documentation_maintenance.md#results-record-template).

## Results inventory

| ID | Result to compile | Required comparisons / final deliverable | Evidence and remaining check |
| --- | --- | --- | --- |
| B01 | Alphabet training / held-out reconstruction | FT2-10/20/30/40/50: selected matched checkpoints, architecture/features, train/validation proteins, seeds, AA/contact/geometry losses, occupancy/perplexity and curves. | [Production manifest](../configs/production_models.yaml), training/evaluation reports. Retained models differ in training history; disclose rather than implying a single retraining protocol. |
| B02 | Matrix estimation and convergence | All five sizes: references/pairs, full state coverage, sustained EMA histories, dimensions/state order and hashes; table of pass/exception status. | [Production readiness](production_alphabet_readiness.md). FT2-50 uses 25,325 alignment files and 26,709,596 pairs; final EMA 0.0864559 exceeds 0.025. Retain the explicit hash-bound acceptance exception; never mark it converged. |
| B03 | Discrete information content | AA, published-model 3Di, FT2-10/20/30/40/50: usage/effective states, entropy, held-out Markov rates, MDL proxies, AA/token and column-equivalent MI, positional entropy, k-mer discrimination and permutation controls. | [Information workflow](experiments.md#wider-information-benchmark). Audit requested analyses: core queue completion explicitly leaves extended notebook analyses and paired aggregation/plotting outstanding. Absent tables are not zero results. |
| B04 | Per-family phylogenetic gain | Same seven representations, same frozen 499-family OMA cohort. Separate controlled common-column/fixed-AA-topology and native own-alignment/own-tree gain; paired family differences and uncertainty. | `runs/local_benchmarks/oma_joint_comparison_20261001` records completed 499-family scope for FT2-20/30/40 plus AA/3Di. `accepted50_20261006/queue_status.json` records completed FT2-50 and joint-comparison jobs. Audit their artifact pointers to establish the final all-five result scope. |
| B05 | Fixed-species-topology supermatrix gain | Same seven representations, 499 requested families, fixed 21-species topology. Native and controlled summaries, family blocks, coverage, site/family metrics, paired bootstrap CIs; controlled cross-alphabet MI. | `species_supermatrix_native_20261003` and `species_supermatrix_controlled_20261003` have completed status for 499 families, but their tables contain AA/3Di plus FT2-10/20/30/40, not FT2-50. `all_alphabets_{native,controlled}_final_20261003` failed on a missing FT2-50 matrix. Verify any later successful replacement before reporting all five. |
| B06 | Native ancestral uncertainty | All seven representations' own alignments, fitted models and ML trees: posterior entropy/confidence/sharpness versus normalized depth, tip-entropy strata, per-node heatmaps, equal-family summaries and family-bootstrap intervals. | `full_native_ancestral_20261001` records 499 families and completed jobs for its scope; FT2-50 has a later queue. [Inspection notebook](../output/jupyter-notebook/ancestral-sharpness-depth.ipynb) defaults to only 50 common families. Freeze the final common available families, disclose omissions, and distinguish native token uncertainty from FT2-only AA decoding. |
| B07 | Topology / taxonomic consistency (TCS) | Cluster OMA and CATH; **13 methods**: FT2-10/20/30/40/50, AA, 3Di and original FoldTree fident/alntmscore/lddt, each raw/exponential. Matched-taxon per-family TCS and aggregate CIs, failures, runtime. | [Curnagl protocol](curnagl_tcs.md) and [October 16 accounting review](curnagl_review_20261016.md). The v4 frozen inventory has 2,514 OMA + 266 CATH families, distinct from the historical 3,404-OMA estimate. Its pilot failed because the deployed RAxML binary was missing; no full TCS sweep was released. A corrected new deployment, inventory and 26-score pilot are required. |
| B08 | CATH dataset audit and genus-controlled cohort | Publish family/tip counts, chain exclusions, identity/taxonomy rules, genus uniqueness and source provenance for each cohort version. Do not pool repaired and newly generated families. | Repaired subsets retained 485/488 families, 13,370 tips; 225 invalid chains removed plus five valid tips in undersized excluded families. The v9 full generation job 65710545 completed with 1,339/1,342 families and 18,545 tips. Final independent validation was never submitted and its marker is absent; the workstation supervisor service is failed. Generation counts are not completed scored inference. See the [verified status and hashes](curnagl_review_20261016.md). |
| B09 | Alignment/tree quality and AlphaFold-related comparison | Freeze dataset, reference truth/taxonomy, competing methods, matching residues/taxa, accuracy metrics and uncertainty before comparing. Include failures and sample selection. | [Setup notebook](../foldtree2/notebooks/benchmarks/setup_alphafold_benchmark.ipynb), [TCS notebook](../foldtree2/notebooks/benchmarks/mini_tcs_benchmark.ipynb). No final scored full comparison was established by this audit. |
| B10 | Scaling and deployability | Graph conversion, Foldcomp encoding, alignment and inference time; peak CPU/GPU memory, structures/residues, thread/device settings, cold/warm caches and package/platform versions. | [Foldcomp experiment](../foldtree2/notebooks/experiments/test_foldcompconversion.ipynb); package smoke tests are engineering evidence, not throughput benchmarks. Compile measured runs only; historical scratch scaling code is not a release CLI. |
| B11 | Geometry / SE(3) reconstruction (separate research result) | Baselines, overfit/held-out gates, FAPE/distance/local geometry, finite gradients, coordinate scale, seed/ablation protocol and failed runs. | [SE(3) report](se3_validation_report.md), [loss ablation](se3_loss_ablation_20261009.md), [Clariden](se3_clariden.md) and dated follow-ups. Preserve failed gates; do not present later pilots/continuations as validated production folding or alphabet improvements. |

## Rules for every final table/figure

- Identify the exact cohort, selected taxa/residues, requested/completed family
  counts, exclusions and partial/smoke limits. The workstation 499-family OMA
  marker set and cluster OMA TCS set are different.
- Record source revision **and dirty source snapshot where applicable**, command,
  configuration, random seeds/splits, software/hardware, artifact checksums and
  protocol identifiers. Keep FT2-50's exception in captions/provenance.
- Use a common contributing-family intersection for paired comparisons and
  report its size. State resampling unit and confidence interval method; do not
  treat dependent sites/nodes as independent families.
- Entropy/MI/rates are in bits; likelihood/gain is ordinarily in nats, with
  explicit converted bit metrics where provided. MDL is a proxy. Historical
  normalized gain in nats per bit is not a universal quality score.
- Native alignment widths/trees differ. Only controlled homologous columns permit
  column-paired MI. Do not rank different alphabets by raw likelihood totals.
- Inspect nonfinite/extreme normalized metrics near zero entropy before plotting;
  retain the denominator/eligible-site policy rather than hiding outliers.
- Reuse verified native alignment/tree/model artifacts across downstream analyses.
  A queue's "completed" status does not certify all notebook analyses or another
  cohort, and an old "running" JSON entry does not establish a live process.

## Compilation order and acceptance

1. Freeze B01/B02 bundles and provenance; confirm all five are distributable.
2. Audit B03/B04/B06 coverage and assemble one final seven-representation ledger.
3. Locate or complete B05 all-five reruns, preserving the failed pre-50 attempts.
4. Finish/audit B07/B08 on the cluster before interpreting TCS rankings.
5. Compile B09/B10 under explicit protocols; keep B11 separate unless its research
   claims are included in the release/manuscript.
6. Produce compact CSVs, figure sources, captions and a results record per
   protocol. Link each from this checklist with **validated**, **failed** or
   **pending** status after inspecting its evidence.

No new full benchmark, training, cluster submission or scientific result is
created by this documentation/packaging preparation.
