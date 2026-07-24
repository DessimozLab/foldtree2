# FoldTree2 Methods Framework for Manuscript Preparation

## 1. Study Objective and Design Rationale
FoldTree2 is designed to infer phylogenetic relationships from protein three-dimensional structure, not only from primary sequence. The core methodological premise is that structural constraints can preserve deep evolutionary signal when sequence similarity is weak (#REF_STRUCTURE_VS_SEQUENCE_PHYLOGENY).

The implementation couples:
- graph-based structural featurization,
- neural discretization into a finite structural alphabet,
- structure-aware alignment scoring,
- maximum-likelihood tree inference with custom substitution matrices.

This modular decomposition was chosen to keep each stage interpretable and swappable for benchmarking (for example, changing alphabet size, alignment strategy, or substitution model while preserving the rest of the pipeline).

## 2. End-to-End Pipeline
The production FoldTree2 workflow follows:
1. Protein structures (PDB) to residue-level graphs.
2. Graphs to continuous embeddings via a graph encoder.
3. Continuous embeddings to discrete symbols via vector quantization.
4. Symbolic sequence alignment under a custom matrix.
5. Maximum-likelihood tree inference under a custom MULTI-state model.
6. Optional tree rooting and ancestral reconstruction.

Implemented in the treebuilder workflow in [foldtree2/ft2treebuilder.py](foldtree2/ft2treebuilder.py).

## 3. Structural Data Representation
### 3.1 Graph construction
Protein structures are transformed into graph objects using PDB2PyG in [foldtree2/src/pdbgraph.py](foldtree2/src/pdbgraph.py). Residues are represented as nodes; edge sets include at least backbone connectivity and contact relations, with additional geometric and secondary-structure-derived information depending on preprocessing mode.

The representation includes:
- residue-level physicochemical descriptors from aaindex-style features,
- backbone geometry descriptors (including angular context),
- contact topology and derived edge attributes,
- optional confidence-informed masking via pLDDT-derived thresholds.

This representation was selected to preserve both local stereochemistry and nonlocal fold constraints relevant for evolutionary comparison (#REF_PROTEIN_GRAPH_LEARNING).

### 3.2 Coordinate-frame consistency
Internal conversion workflows explicitly track transformations between coordinate-space, local frames, and quaternion/RT parameterizations (see [docs/representation_conversion_guide.md](docs/representation_conversion_guide.md)). This supports geometric consistency checks and robustness tests under controlled perturbations (#REF_FAPE, #REF_LDDT).

## 4. Neural Encoder and Discrete Structural Alphabet
### 4.1 Encoder architecture
The default encoder (mk1_Encoder) is implemented in [foldtree2/src/encoder.py](foldtree2/src/encoder.py). It is organized into:
- input module (feature normalization, projection),
- body module (graph message passing; SAGE/GAT/Transformer options),
- head module (projection to latent space and quantization interface).

This split is intentional to support optimizer partitioning and ablation of message-passing variants.

### 4.2 Vector quantization and codebook dynamics
FoldTree2 uses vector quantization to map latent residue embeddings to discrete tokens, enabling downstream alignment and likelihood modeling in a finite alphabet. The EMA quantizer implementation is in [foldtree2/src/quantizers.py](foldtree2/src/quantizers.py).

Key design choices include:
- configurable codebook size (num_embeddings) as a controllable phylogenetic granularity parameter,
- commitment-cost scheduling to stabilize early training,
- regularization terms promoting code usage diversity and reducing collapse.

The discretization stage is central: it converts geometric latent structure into alignable characters while retaining a probabilistic relation to the learned manifold (#REF_VQVAE, #REF_DISCRETE_REPRESENTATIONS_BIO).

## 5. Decoder Heads and Multi-Task Objectives
Training uses a multi-head decoder stack (notably MultiMonoDecoder) to supervise complementary outputs such as:
- amino-acid identity reconstruction,
- geometry-related outputs,
- optional secondary-structure and angular terms,
- optional edge/contact-related objectives.

Implementation details are in [foldtree2/src/mono_decoders.py](foldtree2/src/mono_decoders.py) and training orchestration in [foldtree2/learn_lightning.py](foldtree2/learn_lightning.py).

Multi-task supervision was used as an inductive bias to avoid a token space that only reconstructs one narrow signal, thereby encouraging biologically meaningful latent partitions (#REF_MULTITASK_LEARNING).

## 6. Model Training Protocol
### 6.1 Training framework
The recommended trainer uses PyTorch Lightning in [foldtree2/learn_lightning.py](foldtree2/learn_lightning.py), with support for:
- single- and multi-GPU training,
- distributed strategies (including DDP/FSDP/DeepSpeed options),
- mixed precision,
- checkpointing and TensorBoard logging.

### 6.2 Optimization and scheduling
Configurations support Muon-augmented optimization alongside AdamW components, learning-rate scheduling, gradient clipping, and gradient accumulation.

### 6.3 Reproducibility controls
Training scripts expose:
- explicit random seeds,
- deterministic CuDNN toggles,
- serialized config files for reproducible reruns.

Representative benchmark-ready hyperparameter sweeps are organized in [benchmark_configs/config_30_embeddings.yaml](benchmark_configs/config_30_embeddings.yaml) and sibling config files for multiple alphabet sizes.

## 7. Structural Substitution Matrix Estimation
Custom structure-derived substitution matrices are generated with [foldtree2/makesubmat.py](foldtree2/makesubmat.py).

The workflow includes:
1. selecting/downloading representative structures from AFDB cluster resources,
2. structural alignment using Foldseek,
3. encoding aligned residues with the trained structural tokenizer,
4. counting token substitutions,
5. exporting MAFFT-compatible and RAxML-compatible matrices.

This stage was designed to ensure scoring functions in alignment and likelihood inference are matched to the learned alphabet rather than borrowed from amino-acid substitution assumptions (#REF_FOLDSEEK, #REF_CUSTOM_SUBSTITUTION_MATRICES).

## 8. Tree Inference from Structural Tokens
The treebuilder routine in [foldtree2/ft2treebuilder.py](foldtree2/ft2treebuilder.py) performs:
- structure encoding to token sequences,
- symbol remapping for MAFFT/RAxML-safe alphabets,
- MAFFT text-mode alignment under the custom matrix,
- RAxML-NG maximum-likelihood inference using MULTI-state GTR-style parameterization,
- optional MAD rooting and ancestral-state reconstruction.

Character remapping is not cosmetic: it is required for compatibility with downstream phylogenetic software character constraints while preserving reversible token identity maps (#REF_MAFFT, #REF_RAXML_NG, #REF_MAD_ROOTING).

## 9. Amino-Acid Baseline Pipeline
For benchmark comparability, an amino-acid baseline is run with:
- MAFFT alignment of marker-gene FASTA sequences,
- RAxML-NG ML inference under LG+G+I in notebook analyses.

This provides a direct sequence-based comparator under widely used phylogenomic assumptions (#REF_LG_MODEL, #REF_ML_PHYLOGENY_STANDARD).

## 10. Benchmark Programs Implemented in the Repository
### 10.1 Information-theoretic benchmarking notebook
The notebook [foldtree2/notebooks/benchmarks/treelikelihood_info_theory_benchmark.ipynb](foldtree2/notebooks/benchmarks/treelikelihood_info_theory_benchmark.ipynb) compares amino-acid and FoldTree2 structural-character pipelines at family level, including site-likelihood extraction and cross-representation analyses.

### 10.2 Scripted phylogenetic information gain analysis
The command-line benchmark script [scripts/phylogenetic_information_gain.py](scripts/phylogenetic_information_gain.py) computes, per alignment column:
- tree-based site log-likelihood,
- IID baseline log-likelihood from global state frequencies,
- phylogenetic information gain:
  phylo_gain = loglik_tree - loglik_iid,
- entropy-normalized gain:
  phylo_gain_norm = phylo_gain / (entropy_tip + epsilon),
- optional cross-alphabet mutual information and normalized MI.

This benchmark was chosen to separate pure compositional effects from topology-aware phylogenetic signal (#REF_INFORMATION_THEORY_PHYLOGENY, #REF_MUTUAL_INFORMATION).

### 10.3 Alphabet-size scaling benchmarks
Config sweeps in [benchmark_configs/config_30_embeddings.yaml](benchmark_configs/config_30_embeddings.yaml) and related files (10 to 40 embeddings) evaluate how discrete alphabet cardinality affects model fit and downstream tree-inference performance.

## 11. Practical Design Choices and Their Rationale
### 11.1 Why discretize structure?
Discrete tokens allow the structural signal to be processed by mature alignment and likelihood engines. This bridges modern representation learning with established phylogenetic statistics.

### 11.2 Why custom substitution matrices?
A learned structural alphabet does not obey amino-acid exchange priors. Matrix estimation from encoded structural alignments aligns the inference model with the tokenizer's empirical transition landscape.

### 11.3 Why benchmark against amino-acid trees?
Amino-acid phylogenies are the standard baseline in most protein phylogenomics settings; direct comparison is essential to quantify incremental value and failure modes.

### 11.4 Why include information-theoretic metrics?
Likelihood differences alone can be difficult to interpret across alphabets. Entropy-normalized gain and cross-alphabet MI expose whether improvements reflect real phylogenetic structure rather than alphabet-size artifacts.

## 12. Reproducibility and Reporting Checklist for Manuscript
Recommended to report explicitly:
- software and commit version of FoldTree2,
- external tool versions (MAFFT, RAxML-NG, Foldseek, MAD),
- training dataset composition and filtering,
- encoder/decoder architecture and codebook size,
- exact training hyperparameters and schedules,
- matrix generation dataset and thresholds,
- benchmark family set definitions and inclusion/exclusion criteria,
- all evaluation metrics and their formulas,
- random seeds and hardware profile.

## 13. Suggested Figures and Tables for the Methods/Results Interface
- Figure: End-to-end structural phylogeny pipeline schematic.
- Figure: Encoder and quantization module overview.
- Figure: Information-gain distribution comparison (AA vs FT2).
- Figure: Alphabet-size sweep vs benchmark metrics.
- Table: Hyperparameters for all benchmark runs.
- Table: External tool settings and model assumptions.
- Table: Failure/edge-case handling criteria.

## 14. Citation Placeholder Tags
Use and replace these tags during manuscript preparation:
- #REF_STRUCTURE_VS_SEQUENCE_PHYLOGENY
- #REF_PROTEIN_GRAPH_LEARNING
- #REF_VQVAE
- #REF_DISCRETE_REPRESENTATIONS_BIO
- #REF_MULTITASK_LEARNING
- #REF_FOLDSEEK
- #REF_CUSTOM_SUBSTITUTION_MATRICES
- #REF_MAFFT
- #REF_RAXML_NG
- #REF_MAD_ROOTING
- #REF_LG_MODEL
- #REF_ML_PHYLOGENY_STANDARD
- #REF_INFORMATION_THEORY_PHYLOGENY
- #REF_MUTUAL_INFORMATION
- #REF_FAPE
- #REF_LDDT
