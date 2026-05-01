# Brief for LLM Agent: Add Comparative Phylogenetic Information Analysis to Notebook

## Goal

Extend the notebook with analyses that compare how different aligned alphabets capture phylogenetically structured signal on a **fixed, trusted tree topology**, using **RAxML-based maximum-likelihood calculations**.

We have:

- a **fixed trusted tree topology**
- a **large concatenated alignment of single-copy orthologs**
- one row per species
- multiple aligned representations of the same taxa and columns, including:
  - **amino acid**
  - **FoldTree2**
  - **3Di**

The topology is fixed and trusted. We will use **RAxML** to compute, for each alphabet:

- **per-column likelihoods** under an ML model on the fixed tree
- **per-node character probabilities** for each column on that fixed tree

The main aim is to compare alphabets in terms of:

1. how much variability they contain
2. how much of that variability is organized by the fixed phylogeny
3. how efficiently they convert variability into phylogenetic signal
4. how redundant or complementary they are across aligned representations

---

## Important update to the implementation plan

This notebook should assume that the phylogenetic side is **not approximate** and **not placeholder-based**.

We can directly obtain from **RAxML**:

- **site-wise / column-wise log-likelihoods**
- **ancestral / node-wise character probability distributions**

These should be treated as core inputs for the analysis.

That means the notebook should be designed around:

- loading RAxML outputs
- aligning them to concatenated column indices
- deriving downstream information-theoretic and comparative summaries from them

Do **not** infer alternative topologies.  
Do **not** optimize topology in the notebook.  
Use the fixed tree only.

---

## Core scientific question

We want to compare alphabets such as AA, FT2, and 3Di on the same taxa and homologous concatenated positions, asking:

- which alphabet yields stronger phylogenetically structured site patterns?
- which alphabet yields more phylogenetic signal relative to its own entropy / complexity?
- which alphabet appears to over-partition states versus genuinely capturing more evolutionary structure?
- how much information is shared between aligned representations?

---

## Main analysis logic

For each alphabet and each alignment column, we want to compute and compare:

### Observed-tip quantities
- empirical tip-state distribution
- entropy at the leaves
- number of observed states
- coverage / missingness

### Tree-aware quantities from RAxML
- per-column log-likelihood on the fixed tree
- node-wise posterior / conditional character probabilities
- uncertainty across internal nodes
- optional expected ancestral-state entropy summaries

### Derived comparison quantities
- phylogenetic gain over a non-phylogenetic baseline
- normalized phylogenetic gain
- cross-representation mutual information between aligned columns
- relationships between tip entropy and tree fit
- relationships between ancestral uncertainty and tree fit

---

## High-level principles

### 1. Keep comparisons fair across alphabets

Raw quantities are not directly comparable across alphabets with different cardinalities.

In particular, avoid over-interpreting raw differences in:

- raw entropy
- raw log-likelihood
- raw number of states
- raw ancestral uncertainty

These are influenced by:

- alphabet size
- stationary/background frequencies
- sparsity
- missingness
- model parameterization

The main fair comparison should emphasize:

- **phylogenetic gain over an iid baseline**
- **normalized phylogenetic gain**
- **cross-representation mutual information**
- **distributional comparisons across matched columns / blocks**

---

## Inputs assumed to exist

The notebook should assume access to:

- a fixed trusted tree topology `T`
- species names matching the alignment row labels
- concatenated alignments for each alphabet:
  - `msa_aa`
  - `msa_ft2`
  - `msa_3di`
- optional mapping from concatenated columns to ortholog block / gene
- optional valid-state masks per alphabet
- **RAxML output files for each alphabet**, providing:
  - per-site / per-column likelihoods
  - per-node character probabilities

The code should be modular enough that RAxML outputs can be loaded from paths or pre-parsed tables.

---

## Required role of RAxML in this notebook

RAxML is the engine that supplies the fixed-tree ML quantities.

The notebook should assume that for each alphabet we can obtain:

### 1. Per-column likelihoods
For column `i` in alphabet `A`:

`ell_i^(A) = log P(x_i^(A) | T, theta_A)`

where:
- `T` is the trusted fixed tree
- `theta_A` is the alphabet-specific substitution model used in RAxML

### 2. Per-node character probabilities
For each column `i`, node `v`, and state `s`:

`P(X_{v,i} = s | x_i, T, theta_A)`

or the equivalent conditional / marginal probabilities exported or derived from RAxML outputs.

These node-wise probabilities should be used to compute uncertainty and information summaries over the tree.

---

## Main analysis questions

### Q1. How much variability does each alphabet contain at the tips?
Use entropy and number of observed states.

### Q2. How well does the fixed tree explain each alphabet?
Use RAxML per-column log-likelihoods.

### Q3. How much explanatory power comes specifically from phylogeny, beyond tip composition?
Use phylogenetic gain over an iid baseline.

### Q4. How efficiently does an alphabet convert variability into phylogenetic structure?
Use normalized gain.

### Q5. How certain or uncertain are ancestral reconstructions under each alphabet?
Use node-wise character probabilities to summarize internal uncertainty.

### Q6. How redundant or complementary are the alphabets?
Use aligned-column MI / NMI across representations.

---

## Core quantities to compute

For each alphabet `A` and each column `i`:

### A. Tip-level empirical quantities

#### 1. Empirical tip-state frequencies
Let `p_i^(A)(a)` be the empirical frequency of state `a` among valid tips in column `i`.

#### 2. Tip entropy
`H_tip_i^(A) = -sum_a p_i(a) log p_i(a)`

#### 3. Number of observed states
`K_obs_i^(A)`

#### 4. Valid taxa count
`n_valid_i^(A)`

#### 5. Gap or missing fraction
`gap_frac_i^(A)`

---

### B. Tree-aware quantities from RAxML

#### 6. Per-column phylogenetic log-likelihood
`ell_tree_i^(A) = log P(x_i^(A) | T, theta_A)`

This should come directly from RAxML site-likelihood output.

#### 7. Node-wise state probabilities
For each internal node `v`:
`q_{v,i}^(A)(s) = P(X_{v,i}=s | x_i, T, theta_A)`

#### 8. Node entropy
For node `v` and column `i`:
`H_node_{v,i}^(A) = -sum_s q_{v,i}(s) log q_{v,i}(s)`

#### 9. Mean ancestral uncertainty per column
For column `i`, summarize over internal nodes:
`H_anc_mean_i^(A) = mean_v H_node_{v,i}^(A)`

Optional alternatives:
- median ancestral entropy
- weighted mean by subtree size
- weighted mean by branch length

#### 10. Max ancestral uncertainty per column
`H_anc_max_i^(A)`

This can help identify columns with localized ambiguity.

#### 11. Root entropy
`H_root_i^(A)`

Useful as a compact summary if desired.

---

### C. Non-phylogenetic baseline quantities

#### 12. IID baseline log-likelihood
Define global background frequencies `pi_A` for alphabet `A`.

Then for observed tip states in column `i`:
`ell_iid_i^(A) = sum_{tips with valid states} log pi_A(x_tip)`

This is the key baseline controlling for:
- alphabet size
- background composition
- trivial symbol-frequency effects

---

### D. Derived quantities

#### 13. Phylogenetic gain
`Delta_i^(A) = ell_tree_i^(A) - ell_iid_i^(A)`

This is the main cross-alphabet comparison metric.

Interpretation:
- larger positive values mean the tree explains the column much better than iid composition alone

#### 14. Normalized phylogenetic gain
`Delta_norm_i^(A) = Delta_i^(A) / (H_tip_i^(A) + eps)`

This measures how efficiently the alphabet turns observed variability into tree-structured signal.

#### 15. Gain per valid taxon
Optional:
`Delta_per_taxon_i^(A) = Delta_i^(A) / n_valid_i^(A)`

Useful if coverage differs strongly.

#### 16. Gain relative to ancestral uncertainty
Optional diagnostic:
`Delta_i^(A) / (H_anc_mean_i^(A) + eps)`

This may help identify alphabets that produce strong fit with confident ancestral reconstructions.

---

## Main additional quantities from node probabilities

Because we can compute node-wise character probabilities, add analyses that explicitly use them.

### 1. Ancestral uncertainty profiles
For each alphabet, characterize the distribution of:
- mean internal-node entropy per column
- root entropy per column
- max internal-node entropy per column

These help answer:
- does this alphabet induce sharper ancestral reconstructions?
- does higher tip entropy necessarily imply high ancestral ambiguity?

### 2. Relationship between site likelihood and ancestral uncertainty
Plot and summarize relationships between:
- `ell_tree_i` or `Delta_i`
- `H_anc_mean_i`
- `H_root_i`

Interpretation examples:
- high gain + low ancestral entropy may indicate strong, coherent phylogenetic structure
- high tip entropy + low ancestral entropy may indicate a well-resolved changing signal
- low gain + high ancestral entropy may indicate noisy or weakly structured columns

### 3. Tree-distributed state diversity
If practical, derive expected state occupancy across nodes using node marginals.

For example:
- expected number of distinct states over internal nodes
- entropy of average node-state distribution across the tree

These are optional but potentially interesting.

---

## Recommended baselines

### 1. IID tip baseline
This is mandatory.

For each alphabet, use global background frequencies `pi_A` to compute:
`ell_iid_i^(A)`

This is the main baseline for fair cross-alphabet comparisons.

### 2. Permutation controls
Include negative controls such as:

- permuting taxa labels relative to the tree
- shuffling columns within taxa
- shuffling tip states within a column while preserving counts
- globally relabeling states by a bijection

Expected results:
- taxon/tree mismatch should strongly reduce tree-specific gain
- count-preserving within-column shuffles should reduce phylogenetic signal
- bijective relabeling should preserve structure up to consistent remapping

### 3. Optional null preserving tip entropy
If practical, generate null columns that preserve empirical symbol counts but destroy phylogenetic placement.

This is useful for checking whether gain is driven by phylogenetic clustering rather than composition alone.

---

## Cross-representation information analysis

For aligned columns across alphabets, compute pairwise dependence between representations:

- `I(AA_i ; FT2_i)`
- `I(AA_i ; 3Di_i)`
- `I(FT2_i ; 3Di_i)`

Prefer to report:

- raw MI
- normalized MI
- effective sample size
- permutation-baseline corrected scores if practical

Important constraints:
- compute MI only on taxa valid in both representations for that column
- exclude missing states unless explicitly modeled
- report coverage alongside MI

These comparisons help determine:
- how much FT2 and 3Di recode AA information
- whether structural alphabets capture complementary signal beyond AA

---

## Granularity of analysis

Implement at three levels.

### 1. Column level
Primary unit for the main metrics.

### 2. Ortholog block / gene level
Aggregate metrics within each single-copy ortholog block.

Recommended block-level summaries:
- mean tip entropy
- mean phylogenetic gain
- mean normalized gain
- mean ancestral uncertainty
- total gain
- mean MI with other alphabets

### 3. Whole-dataset level
Overall distributions, paired comparisons, and headline summary statistics.

---

## Handling gaps and missing data

This must be explicit and consistent.

Requirements:

- define valid states for each alphabet
- exclude gaps / missing states from entropy and MI calculations
- for likelihood calculations, treat gaps as missing/unknown according to RAxML conventions, not as ordinary alphabet states
- record `n_valid_taxa` for all column-level quantities

For every alphabet and every analysis, also report:
- usable column count
- mean valid taxa per column
- distribution of valid taxa counts

Do not interpret cross-alphabet differences without checking whether one representation has systematically different coverage.

---

## Main comparison metrics to emphasize

These should be the headline outputs.

### 1. Mean per-column tip entropy
Shows raw observed variability.

### 2. Mean per-column tree log-likelihood
Useful, but not sufficient alone.

### 3. Mean per-column phylogenetic gain
`mean(Delta_i^(A))`

This is the main fair comparison of tree-structured signal.

### 4. Mean normalized phylogenetic gain
`mean(Delta_norm_i^(A))`

This is the main efficiency-style comparison.

### 5. Mean ancestral uncertainty
`mean(H_anc_mean_i^(A))`

This compares how sharply each alphabet supports ancestral inference.

### 6. Cross-representation MI / NMI
These compare redundancy and complementarity across alphabets.

---

## Key visualizations to add

### Distribution plots
For each alphabet:
- histogram / density of tip entropy
- histogram / density of phylogenetic gain
- histogram / density of normalized gain
- histogram / density of mean ancestral entropy

### Scatter plots
Per column:
- tip entropy vs phylogenetic gain
- tip entropy vs normalized gain
- tip entropy vs ancestral uncertainty
- phylogenetic gain vs ancestral uncertainty
- valid taxa count vs gain

### Paired cross-alphabet comparisons
For homologous aligned columns:
- `Delta_i^(AA)` vs `Delta_i^(FT2)`
- `Delta_i^(AA)` vs `Delta_i^(3Di)`
- `Delta_i^(FT2)` vs `Delta_i^(3Di)`
- same comparisons for normalized gain
- same comparisons for ancestral uncertainty

### Block-level visualizations
Per ortholog block:
- mean gain by alphabet
- mean normalized gain by alphabet
- mean ancestral uncertainty by alphabet
- gain difference plots
- rank correlation plots across alphabets

### Cross-representation information plots
- MI / NMI distributions by alphabet pair
- block-level MI heatmaps
- MI vs phylogenetic gain
- MI vs ancestral uncertainty

### Control plots
Show collapse of signal under permutations or taxon shuffles.

---

## Statistical comparisons

Where feasible, include:

- paired tests across homologous aligned columns
  - paired permutation tests
  - Wilcoxon signed-rank tests
- bootstrap confidence intervals across:
  - columns
  - ortholog blocks
- effect sizes:
  - mean paired difference
  - median paired difference
  - bootstrap CI on differences

Prefer bootstrap confidence intervals over only reporting p-values.

---

## Recommended output tables

### 1. Per-column per-alphabet dataframe
One row per concatenated column and alphabet.

Suggested fields:

- `alphabet`
- `column_index`
- `block_id`
- `entropy_tip`
- `n_observed_states`
- `n_valid_taxa`
- `gap_fraction`
- `loglik_tree`
- `loglik_iid`
- `phylo_gain`
- `phylo_gain_norm`
- `anc_entropy_mean`
- `anc_entropy_max`
- `root_entropy`

### 2. Cross-representation per-column dataframe
One row per aligned column pair.

Suggested fields:

- `column_index`
- `block_id`
- `alphabet_x`
- `alphabet_y`
- `n_valid_joint`
- `mi`
- `nmi`
- optional permutation-based z-score

### 3. Block-level summary dataframe
Per block and alphabet:

- `block_id`
- `alphabet`
- `n_columns`
- `mean_entropy_tip`
- `mean_phylo_gain`
- `mean_phylo_gain_norm`
- `mean_anc_entropy`
- `median_phylo_gain`
- `mean_valid_taxa`

---

## Interpretation guidelines

The notebook should support interpretations like these.

### If an alphabet has higher entropy and higher phylogenetic gain
It may contain more structured evolutionary information.

### If an alphabet has higher entropy but only weak gain increase
It may be over-partitioning states or adding noise.

### If normalized gain is higher
That alphabet more efficiently turns variability into tree-structured signal.

### If ancestral uncertainty is lower at similar or higher gain
That alphabet may support sharper, more coherent ancestral inference.

### If MI with AA is moderate but phylogenetic gain is higher
A structural alphabet may be capturing complementary signal not reducible to amino-acid identity.

Avoid making claims based on entropy or raw likelihood alone.

---

## Practical implementation priorities

### Phase 1
Build robust loading and alignment of:
- concatenated MSAs
- per-column RAxML likelihood outputs
- node-wise character probabilities

Then compute:
- tip entropy
- iid baseline log-likelihood
- phylogenetic gain
- ancestral entropy summaries
- cross-representation MI/NMI

### Phase 2
Add:
- block-level aggregation
- paired cross-alphabet comparisons
- permutation controls
- bootstrap CIs

### Phase 3
Add:
- deeper uncertainty diagnostics
- null models preserving composition
- extended residual analyses

---

## Coding style requirements

- keep functions modular and notebook-friendly
- return tidy pandas dataframes
- make all assumptions explicit
- avoid hidden state
- keep missing-data handling explicit
- include docstrings for every function
- structure code so multiple alphabets can be handled via config dictionaries
- keep plotting functions reusable and publication-ready

---

## Suggested function groups

### Data loading and validation
- load fixed tree
- load concatenated alignments
- verify species order consistency
- load RAxML site-likelihood outputs
- load or parse node-wise character probability outputs

### Per-column tip statistics
- entropy
- observed state count
- valid taxa count
- gap fraction
- iid baseline likelihood

### Tree-aware summaries
- site likelihood joins
- node entropy computation
- mean / max / root ancestral entropy
- phylogenetic gain
- normalized gain

### Cross-representation analysis
- aligned-column MI / NMI
- permutation baselines
- block-level aggregation

### Visualization
- distributions
- paired comparisons
- uncertainty vs gain plots
- block-level comparisons
- control plots

---

## Important caveats

- raw quantities are not automatically comparable across alphabets
- larger alphabets may inflate entropy and sparsity
- MI estimates can be biased at low sample sizes
- coverage differences can distort comparisons
- likelihood comparisons depend on model specification
- ancestral probabilities are model-dependent and should be interpreted accordingly
- if different substitution models are used per alphabet, this should be stated clearly in markdown cells

---

## Deliverable

Add notebook sections that:

1. load and validate the fixed tree, alignments, and RAxML outputs
2. compute per-column tip statistics for each alphabet
3. compute iid baseline log-likelihoods
4. compute phylogenetic gain and normalized gain
5. derive ancestral uncertainty summaries from node-wise character probabilities
6. compute cross-representation MI / NMI
7. summarize results at column, block, and whole-dataset levels
8. add negative controls and uncertainty estimates where practical
9. produce clear comparative figures and tidy output tables

The final notebook should let us answer:

- Which alphabet carries more phylogenetically structured signal on the fixed tree?
- Which alphabet does so more efficiently relative to its own variability?
- Which alphabet yields sharper ancestral-state inference?
- How redundant or complementary are AA, FoldTree2, and 3Di?

---

## Final emphasis

The central quantities of interest are:

- **tip entropy**
- **RAxML per-column tree log-likelihood**
- **iid baseline log-likelihood**
- **phylogenetic gain**
- **normalized phylogenetic gain**
- **ancestral uncertainty from node-wise character probabilities**
- **cross-representation MI**

The primary fair cross-alphabet comparison should be:

`phylogenetic gain = tree log-likelihood - iid log-likelihood`

and the main efficiency comparison should be:

`normalized phylogenetic gain = phylogenetic gain / tip entropy`

The main added advantage of the current setup is that we can also compare alphabets in terms of:

`how much phylogenetic signal they provide while yielding confident ancestral character distributions across the tree`