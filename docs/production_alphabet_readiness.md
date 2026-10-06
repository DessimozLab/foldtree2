# Production alphabet preparation

## FT2-50 acceptance exception (2026-10-06)

The corrected 50-state matrix was explicitly accepted by the user for production
and benchmarking despite failing the sustained EMA gate. It uses 25,325 reference
alignment files and 26,709,596 counted pairs. Final EMA was 0.0864559 versus the
0.025 threshold; `is_converged` remains false. This is an acceptance exception,
not a convergence claim. The original failed build and history are preserved on
data2. A hash-bound `*_convergence_acceptance.json` accompanies the production
matrices and is recorded in benchmark provenance. Results using FT2-50 must
disclose this limitation; other matrix validity checks remain enforced.

For user-facing commands with portable input paths, see the
[experiment guide](experiments.md) and [training guide](training.md). The paths
and sample counts below describe this workstation's preparation run, not data
distributed with the package. Follow the
[documentation maintenance checklist](documentation_maintenance.md) as runs
complete or new experiments are added; keep pending external work explicit.

The artifact manifest is `configs/production_models.yaml`. It specifies one
encoder, its matching decoder, and its MAFFT and RAxML matrices for each of
10, 20, 30, 40, and 50 states. Existing 20/30/40 encoders are retained. The
40-state decoder is the epoch-40 checkpoint from `models/notebook`, paired
with the identical epoch-40 production encoder and its regenerated matrices.

The new 10/50 runs use `configs/production_alphabet.yaml`, based on the current
`test_monodecoders.ipynb` architecture and its main training file
`/mnt/data2/datasets/ft2_train_final.h5`: 20,089 stored graphs with 857 residue
features. Graphs live under `structs`, not the extra top-level accession groups.
The earlier 865-feature `structs_training_mk2.h5` training attempt is superseded
and preserved. The interrupted September run is preserved; fresh October runs
use `runs/production_alphabets/notebook_heldout_persistent_restart_20261001`.
Training uses a reproducible 10% held-out protein split, seed 7, and exports
full held-out metrics for the selected best encoder/decoder pair. Checkpoints
stay in staging until training has completed and metrics have been exported.

Activate the `foldtree2` conda environment, then run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/foldtree2-matplotlib \
python -u scripts/prepare_production_alphabets.py --sizes 10 50 --stages train \
  --outdir runs/production_alphabets/notebook_heldout_20260930
```

The stages are independently selectable with `--stages train matrices validate
experiments` and sizes with `--sizes 10 20 30 40 50`. Training skips existing
production pairs. New matrices use the existing structural alignment collection
at `/mnt/data2/datasets/struct_align`. Retained 20/30/40 encoders use the
857-feature graphs in `foldtree2/structalnfinal.h5`; approved new 10/50 encoders
use that same reference feature layout. The compatibility adapter for older
865-feature custom encoders remains available; unsupported layouts fail explicitly.
The matrix generator now reads
headerless Foldseek records correctly and counts every occurrence of repeated
state pairs. Original production matrices are preserved; the corrected counting
method is applied to all five alphabets before comparing them. Use
`--rebuild-matrices` to regenerate retained-model matrices in staging. Original
matrices and auxiliary files are preserved under each production folder's
`previous_matrices/<checksum>/`; results from their old method are preserved
under `runs/production_alphabets/<size>/previous_experiments/`.

On 2026-10-01, the corrected 30- and 40-state MAFFT/RAxML matrices were
revalidated and promoted to production under the approved 0.025 threshold.
Their final Frobenius changes were 0.011745 and 0.023167 respectively. Production
copies match staging, and their original matrices remain in `previous_matrices`.
The corrected 20-state matrices were already promoted. The 10/50 matrices
remain pending completion and validation of their fresh training runs.

The convergence gate uses the approved absolute Frobenius-change threshold
of 0.025 applied to an EMA (span five, alpha 1/3) of informative changes.
After five-update EMA warmup it requires five consecutive eligible updates
with EMA below the threshold. It still requires newly counted
pairs, at least 100 reference files, at least 10,000 pairs, finite scores, and
coverage of every state. Updates occur every 25 files in production. All 20,349
available reference files are eligible; there is no notebook-style 5,000-file
cap. A first snapshot or a run of empty/no-hit files cannot establish
convergence. The history, plots, and convergence criteria are saved with the
matrix. A nonconverged run exits nonzero and never promotes staged matrices.
Threshold, patience, and interval are configurable; inspect the history and
add reference data if the full collection does not establish convergence.

`test_makesubmat_manual.ipynb` uses the same production monitor and has the same
counting fixes; the older `makesubmat.ipynb` also has the counting fixes. Stale
outputs in both notebooks were cleared and must be regenerated.

Artifact validation checks checkpoint finiteness, codebook cardinality, MAFFT
symbols and dimensions, RAxML dimensions/rates/frequencies, and records SHA-256
checksums. Reports and logs are under `runs/production_alphabets/<size>/`.
Artifact validation does not establish scientific quality; inspect the held-out
reconstruction metrics and codebook occupancy before accepting trained models.

## Workstation experiments

The canonical common cohort is now `configs/oma_benchmark_cohort.json`, derived
from the OMA markers in the phylogenetic-gain notebook: 500 requested groups,
499 eligible groups; 768489 lacks its reference alignment/tree. Every approach
uses each family's frozen AA-reference taxa and selected PDBs. Extra structures
are excluded uniformly, and input hashes are verified.

`compare_oma_representations.py` is the integrated information/gain runner.
It requires AA and 3Di alongside all selected, validated FT2 alphabets. It also
verifies residue correspondence and emits separate common-column controlled
gain and own-alignment/own-tree native gain tables. The native ancestral sweep
remains strategy-specific. The older per-model tables below are preparatory
results; AA/FT2-only wider analyses must be explicitly marked legacy.

`scripts/production_alphabet_experiments.py` runs all local marker families with
at least four structures. It uses the numerical functions from
`alphabet_Information_content_benchmark.ipynb` for the discrete description:
state usage, Shannon entropy, effective state count, AA/token mutual information,
held-out Markov entropy rates (orders 0–3), and the notebook's MDL cost proxy.
Proteins are split before model fitting and contexts never cross protein
boundaries. All codebook states remain in the probability support.

Phylogenetic gain uses each family's existing AA tree topology for both
representations, with branch lengths and model parameters optimized separately.
The FT2 alignment uses that encoder's MAFFT matrix; likelihoods use its RAxML
matrix. AA uses LG+G+I, FT2 uses MULTI<K>_GTR{matrix}+I. This is a reproducible
per-family experiment, rather than the notebook's species-supermatrix/ASTRAL
experiment. No cross-alphabet column MI is computed between independently
aligned columns. AA/token MI instead uses direct residue correspondence.
Gain is reported in nats; tip entropy is in bits, following the existing script.
There are 499 eligible local families. Fifteen contain one extra PDB absent
from their reference AA alignment/tree; both analyses use the reference sample
and record those excluded PDBs. No reference AA structures are missing.

The reusable numerical methods are in `scripts/alphabet_information_metrics.py`;
scripts do not import or execute notebook files.

```bash
python scripts/prepare_production_alphabets.py \
  --sizes 20 30 40 --stages matrices validate experiments \
  --rebuild-matrices --device cpu --threads 8
```

Per-family `results.json` is written only after both analyses succeed and allows
restart. The aggregate outputs are `alphabet_description.csv`,
`entropy_rate_mdl.csv`, and `phylogenetic_gain.csv`. `completed.json` marks a
finished alphabet sweep. A different encoder/matrix checksum requires a new
output directory. `--max-families 1` is for smoke tests, not full results.

`scripts/alphabet_information_benchmark.py` runs the notebook's wider information
benchmarks from completed family outputs: AA baselines, optional within-sequence
shuffle controls, unweighted and weighted k-mer discrimination, protein-held-out
backoff entropy rates, cross-family MDL proxies, Henikoff-weighted positional
entropy, column-equivalent AA/FT2 MI, and unigram/bigram/conditional AA-token MI.
It writes CSV tables and a completion/provenance record. Orders default to 0–3;
use `--orders 0 1 2 3 4 5` for the notebook's larger MDL sweep.

```bash
python scripts/alphabet_information_benchmark.py \
  --experiment-dirs runs/production_alphabets/20/experiments \
                    runs/production_alphabets/30/experiments \
                    runs/production_alphabets/40/experiments \
  --outdir runs/production_alphabets/information_comparison \
  --permutation-controls
```

For already prepared alignments and trees, use the separate phylogenetic-gain
CLI with `scripts/phylogenetic_information_gain.spec.example.json` as a template:

```bash
python scripts/phylogenetic_information_gain.py \
  --spec experiment.json --outdir results/phylo_gain --threads 8 --skip-cross-mi
```

`--skip-cross-mi` is required when representations were aligned independently;
column MI is meaningful only with established homologous-column correspondence.
The production runner also invokes the standalone information script for each
finished alphabet and saves those larger results in `<size>/information/`.

## Experiments elsewhere and squash merge

Use the same artifact manifest and checksums on the external machine. Transfer
the selected four artifacts for each size, the associated training/provenance
reports, and the datasets needed by the target benchmark. Run the remaining
alignment/tree-quality, scaling, and reconstruction benchmarks there; the
existing entry points include `mini_tcs_benchmark.ipynb`,
`setup_alphafold_benchmark.ipynb`, and the workflows in `foldtree2/workflow`.
Record alphabet size, checkpoint checksums, dataset version, random seed, and
hardware with each result. These external experiments have not been run here.

Before squashing dev into main:

- Require validated bundles for all five sizes and inspect held-out model metrics.
- Require the full workstation experiment completion marker for every size.
- Collect the external experiment results.
- Run the package/CLI and broader repository checks, including the existing
  `scripts/test_conda_packaging.sh` and `scripts/smoke_test_conda_cli.sh` workflows.
- Review the pre-existing untracked `configs/learn_geometry/` separately.
- Decide how to distribute new binary models: `models/`, `*.pt`, and `*.json`
  are ignored by Git, although some older production binaries are tracked.
  A squash commit alone will not include newly created ignored artifacts.

No merge or commit is performed by these preparation scripts.
