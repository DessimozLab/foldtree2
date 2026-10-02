# User Guide: Protein Structure Phylogenetics with FoldTree2

[Back to README](../README.md)

## Overview

FoldTree2 enables phylogenetic analysis using protein 3D structures instead of sequences. This approach can reveal evolutionary relationships that are not apparent from sequence alignments alone, especially for distantly related proteins with conserved structural folds.

### Why Structure-Based Phylogenetics?

Traditional sequence-based methods struggle with:
- **Distant homologs**: Structural features evolve slower than sequences
- **Convergent evolution**: Similar structures may arise independently
- **Structural conservation**: Functionally important folds remain stable

FoldTree2 addresses these by:
1. Encoding protein structures into discrete "structural alphabets"
2. Aligning structures using FoldSeek for sensitive detection of structural similarity
3. Generating custom substitution matrices from structural alignments
4. Inferring trees with RAxML-NG using structure-appropriate models

## Building a Phylogenetic Tree

### Prerequisites

- Trained encoder-decoder model (see :doc:`training` for training your own)
- Protein structures in PDB format (one file per protein)
- RAxML-NG and MAFFT installed (bundled with FoldTree2)

### Step-by-Step Workflow

#### 1. Prepare Your Structures

Ensure your PDB files:
- Have complete backbone atoms (N, CA, C, O) where possible
- Use unique, simple filenames (these become sequence identifiers)
- Contain one protein per file (or extract the chain of interest)

```bash
# Example: Extract a specific chain from a multi-chain PDB
# (Use Biopython, PyMOL, or other tools for this preprocessing step)
```

#### 2. Run Tree Inference

```bash
ft2treebuilder \
  --encoder path/to/encoder.pt \
  --decoder path/to/decoder.pt \
  --mafftmat path/to/mafft_matrix.mtx \
  --submat path/to/substitution_matrix.txt \
  --charmaps path/to/charmaps.pkl \
  --structures "/path/to/your_pdbs/*.pdb" \
  --outdir results/my_tree \
  --device cpu \
  --ncores 4
```

### Key Command-Line Options

| Option | Description |
|--------|-------------|
| `--encoder` | Path to trained encoder checkpoint (.pt) |
| `--decoder` | Path to trained decoder checkpoint (.pt) |
| `--mafftmat` | MAFFT substitution matrix for structural alignments |
| `--submat` | RAxML-compatible substitution matrix |
| `--charmaps` | Character mapping bundle for encoding |
| `--structures` | Glob pattern for PDB files (quoted!) |
| `--outdir` | Output directory for results |
| `--device` | Compute device: `cpu` or `cuda` |
| `--ncores` | Number of CPU cores for preprocessing |
| `--bs` | Enable bootstrapping (adds support values) |
| `--root` | Root tree using MAD algorithm |
| `--ancestral` | Reconstruct ancestral sequences (experimental) |

### Understanding the Output

| File | Description |
|------|-------------|
| `*_encoded.fasta` | Structural characters (not amino acids!) |
| `*_raxml_aln.fasta` | RAxML-compatible alignment |
| `*.raxml.bestTree` | Final phylogenetic tree (Newick format) |
| `*.raxml.bestModel` | Best-fit substitution model |
| `*.raxml.log` | RAxML inference log and diagnostics |

The tree is in **Newick format** and can be viewed with:
- FigTree, iTOL, or Dendroscope for visualization
- ETE Toolkit for programmatic analysis in Python

## Generating Custom Substitution Matrices

For novel protein families or research applications, you can generate structure-based substitution matrices:

### Workflow Overview

1. **Collect reference structures** - Curate a representative set
2. **Perform structural alignments** - Using FoldSeek or similar
3. **Encode alignments** - Using your trained encoder
4. **Compute substitution frequencies** - From aligned structural states

### Command

```bash
makesubmat \
  --modelname your_model_name \
  --modeldir path/to/models \
  --datadir path/to/reference_data \
  --dataset path/to/graphs.h5 \
  --device cpu \
  --encode_alns \
  --monitor-convergence \
  --require-convergence \
  --save-history \
  --plot
```

### Output Files

| File | Description |
|------|-------------|
| `*_mafftmat.mtx` | MAFFT-compatible substitution matrix |
| `*_submat.txt` | RAxML-compatible matrix |
| `*_pair_counts.pkl` | Character mapping and counts |
| Convergence plots | Training diagnostics |
| Metrics report | Quality assessment |

See :doc:`training` for details on matrix generation and validation.

## Training Your Own Models

For research applications requiring custom structural alphabets:

1. **Prepare training data**:
   ```bash
   pdbs-to-graphs /path/to/pdbs training_data.h5 --verbose
   ```

2. **Train the encoder-decoder**:
   ```bash
   python -m foldtree2.learn_lightning \
     --dataset training_data.h5 \
     --model-name my_model \
     --num-embeddings 30 \
     --epochs 1000 \
     --batch-size 16 \
     --device cuda
   ```

3. **Generate substitution matrix** from trained model

See :doc:`training` for complete details on model training and evaluation.

## Practical Considerations

### Choosing Alphabet Size

The number of structural states (alphabet size) affects resolution:
- **Smaller (10-20)**: Coarse structural classification, robust to noise
- **Medium (30-40)**: Balance of resolution and robustness (recommended)
- **Larger (50+)**: Fine-grained discrimination, requires more data

### Quality Control

- Verify reconstruction accuracy on held-out test set
- Check state occupancy (avoid highly imbalanced alphabets)
- Compare tree topology with sequence-based methods
- Assess bootstrap support for key clades

### Common Pitfalls

- **Insufficient structural diversity**: Results in uninformative matrices
- **Overfitting**: Too many states for available data
- **Alignment errors**: Garbage in, garbage out - verify alignments
- **Model mismatch**: Encoder/decoder must be from same training run

## Further Resources

- :doc:`overview` - Core concepts and architecture
- :doc:`training` - Model training and matrix generation
- :doc:`experiments` - Running phylogenetic-gain experiments
- :doc:`manuscript_methods_framework` - Methodological details

Use a fresh output directory for changed structures/artifacts. The tree CLI may
reuse files without artifact provenance checks. `--redo` applies to RAxML, not
whole-pipeline invalidation. The current CLI does not consistently forward its
`--overwrite` flag; do not rely on it to regenerate everything.
`--ncores` controls alignment/inference parallelism; GPU selection controls
encoding, while MAFFT/RAxML remain CPU programs. See
[troubleshooting](installation.md#troubleshooting).

## Future model hosting

Recommended, not implemented: host all five alphabet bundles in a versioned
Hugging Face model repository, pin downloads to a revision, and keep code on
GitHub. The [Hub accepts custom models and associated files](https://huggingface.co/docs/hub/models-uploading).
Include a model card, license, configuration, checksums, compatible code revision,
and evaluation evidence. Confirm model/data redistribution rights separately;
the code's MIT license does not automatically settle those rights.
GitHub Releases are an alternative for frozen bundles.
Do not publish unfinished models or guess public URLs. Longer-term portability
would benefit from weights plus explicit architecture configuration rather than
full pickled Python objects; that requires a separate loader migration.
