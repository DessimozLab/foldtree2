# Running experiments

[Back to README](../README.md)

Run scripts from the repository root in the installed environment. Datasets are
local inputs, not pip package data. See [maintenance](documentation_maintenance.md)
when adding a benchmark or reporting completion.

## Family inputs

The canonical cohort is the OMA marker collection used in
`treelikelihood_info_theory_benchmark.ipynb`, frozen in
`configs/oma_benchmark_cohort.json`: 500 requested OMA groups, 499 eligible
families. Group 768489 lacks its reference alignment/tree. The manifest records
per-family reference taxa, excluded extra PDBs, and hashes of marker FASTAs,
AA alignments, reference trees and selected PDBs. The standard family root
automatically uses this manifest. Changed inputs fail verification; regenerate
to a new manifest/output scope rather than mixing versions.

The runner expects a parent with family directories:

```text
families/
  family_001/
    structs/ACCESSION.pdb
    sequences.aligned.fa
    raxml_lg_tree.raxml.bestTree
```

Each family needs at least four PDBs. AA headers use `ACCESSION|SPECIES` labels
(optional trailing descriptions); PDB stems are accessions. Tree tips match AA
labels. Missing reference PDBs are errors; extra PDBs are excluded and recorded.
The AA alignment must represent the same residues encoded from the selected PDB
chain. Matching names/lengths alone does not prove scientific correspondence.

## Smoke test and full sweep

For the sequential local queue (CPU encoding, at most eight threads), use:

```bash
python scripts/run_local_benchmark_queue.py \
  --outdir runs/local_benchmarks/your_run \
  --training-pid YOUR_TRAINING_SUPERVISOR_PID \
  --training-dir runs/production_alphabets/notebook_heldout_20260930
```

The queue independently rebuilds retained 20/30/40 matrices, runs a per-model
pilot before full core information/gain and native ancestral reconstruction,
and runs AA/3Di native tree and ancestral baselines. It waits for validated
10/50 training outputs before building their matrices and benchmarks. Consult
`queue_status.json` and job logs: failures are recorded without stopping other
models. This is the implemented core suite, not completion of every planned
notebook analysis or the integrated seven-representation comparisons.

To benchmark ready bundles without rebuilding matrices or waiting for training:

```bash
python scripts/run_local_benchmark_queue.py --benchmark-only --sizes 30 40 \
  --max-families 5 --threads 8 --outdir runs/local_benchmarks/ready_models \
  --wait-for-service foldtree2-benchmark20-20261001.service
```

This runs the same first five eligible families for both FT2 models, plus AA/3Di
native ancestral baselines. Omit `--wait-for-service` when no other CPU benchmark
is running. Each FT2 representation retains its own alignment and inferred tree
for ancestral reconstruction. The limited scope is recorded and is not a full
499-family sweep. GPU encoding/training is not started by `--benchmark-only`.

For the full native-tree/ancestral sweep across the ready FT2 models and AA/3Di:

```bash
python scripts/run_local_benchmark_queue.py --benchmark-only \
  --trees-and-ancestral-only --sizes 20 30 40 --threads 8 \
  --alignment-root runs/local_benchmarks/notebook_frobenius_20260930/experiments \
  --reuse-native-root runs/local_benchmarks/complete_models_20261001 \
  --outdir runs/local_benchmarks/full_native_ancestral_20261001
```

This omits the family limit, reuses the completed 20-state input alignments,
prepares missing 30/40 alignments, and verifies cached five-family native
outputs before reusing them. It skips the separate wider information-analysis
stage. Alignment preparation still produces the existing core description/gain
tables as auxiliary outputs. Each strategy has its own alignment, fitted ML
tree, and ancestral probabilities; AA and 3Di failures are tracked independently.
Reused artifact locations are recorded in `queue_status.json`; a finished queue
with failures is not a complete sweep. Ten/50-state models are excluded until
their trained production bundles and converged matrices are ready.

New builds use sustained EMA convergence of absolute Frobenius changes:
`delta = ||M_new - M_previous||_F`, `EMA = (1/3)*delta + (2/3)*EMA_previous`.
The EMA span is five informative updates (with five-update warmup), followed by
five consecutive eligible updates with EMA below `0.025`. A threshold crossing
resets patience; a nonfinite or noninformative update resets both EMA and patience.
Use `--convergence-ema-span` and `--convergence-patience` to configure the monitor,
or the corresponding `--matrix-convergence-*` flags in the production builder,
with minimum 100 reference files, 10,000 pair counts, and full background-state
coverage. Matrices are evaluated after all available references, not promoted
at an early transient crossing. Previous production matrices are archived
before replacement. Cached staging encodings are reused only when their staged
encoder matches production. The manual notebook imports the same monitor.
Previously promoted matrices retain their original single-update convergence
reports; this code change does not retroactively certify them under the EMA rule.

For additional independent references, `scripts/extend_matrix_references.py`
excludes the original AFDB cluster IDs, deterministically samples five members
per new cluster, reuses/downloads their PDBs, stores mk2 graphs in a separate
HDF5 file, encodes them with the production checkpoint and runs Foldseek
all-vs-all. It adds 1,000-family batches (bounded at 5,000 by default), preserving
the original reference prefix and rebuilding counts over the combined unique
families/accessions. New inputs stay on the data disk. Promotion requires the
sustained EMA criterion; exhausted or failed builds remain staged. This does not
add AFDB families to the fixed OMA benchmarking cohort.

To rebuild using references already downloaded and encoded, without downloading,
converting graphs, or retraining, use a new staging directory:

```bash
python scripts/rebuild_downloaded_matrix_references.py --size 50 \
  --reference-root /mnt/data2/datasets/ft2_matrix_extension_20261003 \
  --outdir /mnt/data2/datasets/ft2_matrix_readerfix_20261004 --threads 8
```

The encoded FASTA reader removes only CR/LF line delimiters, preserves valid
control-character tokens, ignores blank lines, and retains the final record.
Concatenation avoids adding blank sequence lines. Matrix compilation verifies
the exact escaped encoder codebook before counting. The existing EMA gate is
unchanged; only converged, validated matrices can enter production. Add
`--benchmark-outdir runs/local_benchmarks/readerfix50_20261004` to run the
remaining family benchmarks and both all-alphabet supermatrix modes after
successful promotion. Failed rebuilds remain staged for inspection.

An explicitly approved exception can be recorded with
`scripts/promote_nonconverged_matrix.py --accept-nonconverged`, using a size,
staging directory, and nonempty reason. This validates all artifacts and full
state-pair coverage before recording a hash-bound acceptance sidecar. It never
changes `is_converged` or the convergence threshold. FT2-50 was accepted this way
on 2026-10-06; see the [readiness note](production_alphabet_readiness.md).
Its benchmark provenance must retain and disclose this exception.

Validate the selected bundle first. Use separate output directories:

```bash
python scripts/production_alphabet_experiments.py \
  --size 30 --families /path/to/families \
  --outdir results/smoke/30 --device cpu --threads 4 --max-families 1
```

For the full sweep, omit the limit:

```bash
python scripts/production_alphabet_experiments.py \
  --size 30 --families /path/to/families \
  --outdir results/full/30 --device cpu --threads 4
```

Repeat for each validated alphabet size; do not run incomplete bundles. Locations
are selected by the production manifest. The runner computes state usage, entropy,
AA/token MI, held-out Markov rates/MDL proxies, and fixed-AA-topology phylogenetic
gain. MAFFT/RAxML run on CPU; `--device` selects encoding.

Each finished family has `results.json`. Aggregates are `alphabet_description.csv`,
`entropy_rate_mdl.csv`, and `phylogenetic_gain.csv`. `completed.json` marks the
requested scope, including smoke runs. Inspect provenance, limits, and actual
counts before calling it a full sweep. The identical command reuses finished
families. Changed hashes/protocol/limits require a new directory. Inputs are not
fully dataset-hashed: use fresh directories and record dataset versions if
families change.

## Native ancestral character reconstruction

Inspect completed results with
[`ancestral-sharpness-depth.ipynb`](../output/jupyter-notebook/ancestral-sharpness-depth.ipynb).
It compares sharpness against normalized root-to-node distance, with equal-family
weighting, family-bootstrap intervals, tip-entropy strata and individual-node
posterior heatmaps. It reads existing artifact pointers without rerunning ASR.
The default is 50 common completed families; set `MAX_FAMILIES=None` for the
full available cohort. Missing models and partial baseline coverage are explicit.

`scripts/ancestral_state_uncertainty.py` uses the same shared RAxML-NG
`--ancestral` implementation and ancestral-table-to-FASTA conversion as the
FT2 treebuilder. Run it separately for AA, 3Di, and each FT2 alphabet, supplying
that strategy's **own alignment, ML tree, fitted model, and ML log**:

```bash
python scripts/ancestral_state_uncertainty.py \
  --alignment results/strategy/aligned.fasta \
  --tree results/strategy/ml.raxml.bestTree \
  --model results/strategy/ml.raxml.bestModel \
  --fitted-log results/strategy/ml.raxml.log \
  --states 'ARNDCQEGHILKMFPSTWYV' \
  --family family_001 --strategy 3Di_foldmason \
  --outdir results/strategy/ancestral --threads 8
```

`--states` must specify the exact matrix/engine state order, not the observed
characters: AA and the protein-encoded 3Di route use the order above; FT2 uses
its MULTI alphabet's order. The fitted model and branches are frozen during
reconstruction, so posterior differences do not reflect another optimization.
The treebuilder retains its historical optimization defaults unless its
`run_raxml_ng_ancestral_struct` method receives `fitted_model=...`.

Outputs include native ancestral states, `ancestral.fasta` consensus characters,
losslessly compressed native probabilities, and `node_site_uncertainty.csv.gz`
with full probability distributions and entropy/confidence measurements. FASTA
characters retain their original representation: 3Di letters are not amino
acids. The treebuilder's optional FT2-to-amino-acid decoder remains a separate,
FT2-only downstream operation, not a definition of ancestral uncertainty.

Use identical explicit outgroup taxa via `--outgroups` for each strategy when
available; otherwise each tree is MAD-rooted independently. A newly inserted
root is a distance origin, not an extra node with an invented posterior.
Nodes and columns from different alignments/trees are not automatically paired.
Use new output directories when the reconstruction protocol changes.

## Wider information benchmark

For the integrated comparison (3Di is required, not an optional baseline):

```bash
python scripts/compare_oma_representations.py --sizes 20 30 40 \
  --cohort configs/oma_benchmark_cohort.json \
  --experiment-root runs/local_benchmarks/notebook_frobenius_20260930/experiments \
  --native-root runs/local_benchmarks/full_native_ancestral_20261001 \
  --outdir runs/local_benchmarks/oma_joint_comparison \
  --wait-for-service foldtree2-full-native-ancestral-20261001.service
```

This includes AA, 3Di and all selected FT2 sizes in alphabet usage, entropy
rates, MDL proxies, residue/column-equivalent MI, position entropy and k-mer
discrimination, plus controlled and native phylogenetic gain. Incomplete family
or taxon coverage fails explicitly. Protein/family splits and seeds are shared;
k-mer folds use canonical family/accession ordering. Missing symbols are not
extra states and Markov/k-mer windows do not bridge missing positions.

Controlled gain projects all representations onto FoldMason's corresponding
AA/3Di columns, relabels tips to the OMA reference labels, and fits each model
on the same OMA AA topology. A shared occupancy mask selects reported columns.
Native gain uses each representation's own alignment and fitted tree, freezing
parameters for site likelihoods. It is not a column-paired comparison.
Ancestral reconstruction remains native and does not use controlled alignments.
Rates retain AA LG+G+I, 3Di published Q.3Di.AF+G+I, and FT2 custom+I.

The per-family controlled-gain output above is distinct from the concatenated
fixed-species-tree comparison below. Do not label it as a species-supermatrix result.

### Concatenated OMA alignment on a fixed species topology

`scripts/benchmark_species_supermatrix.py` extracts the notebook's supermatrix
padding, species mapping and IID-relative phylogenetic information workflow.
Its default fixed tree is `configs/oma_fixed_species_tree.nwk`, copied from the
notebook's historical `Information_benchmark/aa_astral_tree.nwk` (21 species).
This imported reference topology is not newly inferred from the current sweep.
All representations use that same topology; branches and model nuisance
parameters are fitted separately and frozen for site likelihoods. AA retains
LG+G+I, 3Di the published Q.3Di.AF+G+I, and FT2 its size-specific GTR+I model.

```bash
python scripts/benchmark_species_supermatrix.py --sizes 10 20 30 40 \
  --experiment-root runs/local_benchmarks/notebook_frobenius_20260930/experiments \
  --native-root runs/local_benchmarks/full_native_ancestral_20261001 \
  --species-tree configs/oma_fixed_species_tree.nwk \
  --outdir runs/local_benchmarks/species_supermatrix_ready --threads 8
```

AA and 3Di are mandatory. Add `50` only after its bundle, matrices and family
alignments are validated, using a fresh output directory for the expanded scope.
Missing family/model inputs fail rather than shrinking the cohort. `--prepare-only`
validates and concatenates without launching RAxML. `--max-families` is for pilots;
`--prune-tree-to-cohort` explicitly permits trimming unused species in such a pilot.
Full runs require an exact tree-tip/species match, without silent pruning.

Native mode reuses each strategy's aligned family sequences, pads missing species
with gaps, and concatenates in the same deterministic family order. It rejects
multiple copies per species rather than arbitrarily assigning orthologs across
families. Native columns have different widths and are not index-paired.
`--column-mode controlled` instead uses verified FoldMason residue correspondence
and projects FT2 tokens onto common columns. This mode shares an occupancy mask
and additionally writes column-paired cross-alphabet MI.

Outputs: per-model `supermatrix.fasta`, zero-based half-open `family_blocks.csv`,
`site_metrics.csv.gz`, `family_metrics.csv`, fitted RAxML tree/model and
`summary.json`; joint `summary.csv` and `paired_family_comparisons.csv` contain
paired family-bootstrap intervals. Site metrics include tip entropy, normalized
tip entropy, tree/IID log likelihoods, gain in nats and bits, and gain bits per
valid tip. Missingness defaults to <=30% with at least four valid species.
Family tables include zero-eligible-column families explicitly; comparisons
report their actual common contributing families. The historical normalized
gain is nats per bit of tip entropy, not a universal model-quality score.
Do not compare raw likelihood totals as if alphabet sizes, native widths and
coverage were identical. Use coverage reports and paired family differences;
use controlled mode for homologous-column comparisons. Completion markers and
input/output hashes guard against mixed model/tree/cohort results on resume.

Both supermatrix modes are queued on the workstation for all currently ready
alphabets. CPU execution is sequential: the current 10-character joint comparison,
then full native supermatrix, then full controlled supermatrix, then AFDB reference
expansion for 50. `--wait-for-service` can enforce this sequencing. After 50 passes
the EMA gate and finishes its native family benchmarks, the extension workflow
runs both supermatrix modes with all five FT2 sizes plus AA and 3Di. Native and
controlled results have separate output directories and must not be pooled as
one protocol.

Reuse the same stored native alignment, fitted model and tree for each
family/representation across information, native gain and ancestral analyses.
Use `--alignment-root` and `--reuse-native-root` when extending a sweep;
reuse follows artifact paths recorded in `queue_status.json`, including
outputs inherited from an earlier pilot, and verifies their provenance.
Controlled projections are distinct derived alignments required for the
common-column comparison; they do not require another native tree search.

The lower-level information script can also consume native 3Di directly:

```bash
python scripts/alphabet_information_benchmark.py \
  --experiment-dirs results/full/20 results/full/30 results/full/40 \
  --native-root results/native_sweep --cohort configs/oma_benchmark_cohort.json \
  --outdir results/information_comparison --permutation-controls
```

This uses finished family outputs with AA and 3Di arms: k-mer discrimination,
held-out entropy rates, cross-family MDL proxies, weighted positional entropy,
and AA/token MI. Orders default to 0–3; `--orders 0 1 2 3 4 5` extends the sweep.
Cross-family estimates need multiple families. Shared families must have identical
AA data; record overlap when comparing alphabets. Partial upstream sweeps can be
consumed, so this script's completion marker does not establish full upstream
completion.

AA/FT2-only legacy analyses require explicit `--legacy-ft2-only` and do not
constitute the complete comparison. The per-model preparation runner's old
description/gain tables are auxiliary inputs, not an integrated benchmark.

| Output | Contents |
| --- | --- |
| `sequences.csv`, `completed.json` | Input sequence table and analysis/provenance record. |
| `alphabet_usage.csv` | State usage, entropy, and effective alphabet size. |
| `entropy_rates.csv`, `mdl.csv` | Protein-held-out entropy rates and cross-family MDL proxies. |
| `aa_token_mi.csv`, `column_equivalent_mi.csv` | Residue-level MI and projection onto AA alignment columns. |
| `position_entropy.csv` | Weighted alignment-column entropy. |
| `kmer_discrimination.csv` | Weighted/unweighted discrimination estimates. |

Only tables with results are emitted; analyses that are disabled or lack
sufficient data can leave tables absent. Check messages, requested analyses,
and sample counts rather than interpreting an absent table as a zero result.

Entropy/rates/MI use bits; effective states are `2**entropy`. MDL is a model-cost
proxy, not an optimal compressor. Likelihood/gain is in nats. AA and structural
representations use the same family AA topology, with independently optimized
parameters/branch lengths. This is not a tree-search accuracy benchmark or the
species-supermatrix/ASTRAL notebook workflow. Independently aligned columns
must not be treated as corresponding sites for cross-alphabet MI.

## Other prepared alignments and trees

Copy the [gain specification](../scripts/phylogenetic_information_gain.spec.example.json)
and replace dataset paths/model details:

```bash
python scripts/phylogenetic_information_gain.py \
  --spec /path/to/experiment.json --outdir results/phylo_gain \
  --threads 4 --skip-cross-mi
```

Keep `--skip-cross-mi` for independently aligned representations. Enable column
comparisons only with established homologous-column correspondence.

## External machines

Transfer validated bundles, hashes, dataset versions, and configurations.
Alignment/tree-quality, scaling, and reconstruction experiments are not covered
automatically by the local sweep. Existing research entrypoints include
`mini_tcs_benchmark.ipynb`, `setup_alphafold_benchmark.ipynb`, and `foldtree2/workflow`.
Record commands, seeds, hardware/software, and actual coverage. The
[developer checklist](production_alphabet_readiness.md) keeps unrun requirements
separate from finished local results.
