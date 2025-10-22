<p align="center">
  <img src="logo.png" alt="FoldTree2 Logo" width="300"/>
</p>

# FoldTree2: Maximum Likelihood Phylogenetic Tree Inference from Protein Structures

FoldTree2 is a Python package and toolkit for inferring phylogenetic trees from protein structures using maximum likelihood methods. It provides tools for converting protein structure files (PDBs) into graph representations, deriving structural alignments, and building phylogenetic trees based on structural data.

## Features
- **PDB to Graph Conversion:** Convert protein structures into graph-based representations suitable for machine learning and phylogenetic analysis.
- **Custom Substitution Matrices:** Generate and use structure-based substitution matrices for alignments.
- **Maximum Likelihood Tree Inference:** Build phylogenetic trees from structural alignments using maximum likelihood approaches.
- **Flexible Pipeline:** Modular scripts for each step: graph creation, encoding, alignment, and tree inference.

## Installation

### Using pip and conda

First create the environment 

```bash
conda env create --name foldtree2 --file=foldtree2.yml
conda activate foldtree2
```
and then install the project with pip

```bash
pip install .
```
This will install all required dependencies as specified in `pyproject.toml` and `setup.py`.


## Command Line Tools

FoldTree2 provides several command-line tools that are automatically installed and available system-wide:

- **`foldtree2`** / **`ft2treebuilder`**: Main phylogenetic tree inference pipeline
- **`pdbs-to-graphs`**: Convert PDB files to graph representations
- **`makesubmat`**: Generate structure-based substitution matrices
- **`raxml-ng`**: Maximum likelihood phylogenetic inference (bundled RAxML-NG)
- **`mad`**: Minimal Ancestor Deviation tree rooting
- **`hex2maffttext`** / **`maffttext2hex`**: MAFFT format conversion utilities

All tools include help documentation accessible with the `--help` flag.

## Quick Start: Using Pretrained Models

For most users, FoldTree2 provides pretrained models that can be used directly to infer phylogenetic trees from protein structures.

### Basic Workflow
Build phylogenetic trees from a folder of PDB structures using pretrained models:

```bash
foldtree2 --model mergeddecoder_foldtree2_test \
  --structures <YOURSTRUCTUREFOLDER> \
  --outdir <RESULTSFOLDER>
```

This single command will:
1. Convert PDB files to graph representations
2. Use pretrained models to encode structural features
3. Generate structure-based substitution matrices
4. Create structural alignments
5. Infer a maximum likelihood phylogenetic tree

### Available Pretrained Models
- `mergeddecoder_foldtree2_test`: General-purpose model for diverse protein structures
- `small`: Lightweight model for smaller datasets
- Additional models may be available in the `models/` directory

### Output Files
The pipeline generates several output files in your results directory:
- **Phylogenetic tree**: `.tre` files in Newick format
- **Alignments**: `.aln` files showing structural alignments
- **Substitution matrices**: Custom matrices based on structural similarity
- **Log files**: Detailed information about the inference process

## Advanced Usage: Training Custom Models

For advanced users who want to train their own models or work with specialized datasets, FoldTree2 provides a complete training pipeline.

### 1. Prepare Training Data
Convert your PDB files to a graph HDF5 dataset suitable for training:
```bash
pdbs-to-graphs <input_pdb_dir> <training_graphs.h5> --aapropcsv config/aaindex1.csv
```

### 2. Train Custom Models
FoldTree2 provides several training scripts with different features:

#### Standard Training
```bash
python learn_monodecoder.py \
  --dataset <training_graphs.h5> \
  --modelname <my_custom_model> \
  --epochs 100 \
  --batch-size 20 \
  --hidden-size 256 \
  --embedding-dim 128 \
  --outdir ./models/

```
See the complete list of options with `--help`.

#### Lightning-based Training (Recommended)
For advanced features like distributed training, automatic checkpointing, and logging:
```bash
python learn_lightning.py \
  --dataset <training_graphs.h5> \
  --modelname <my_lightning_model> \
  --epochs 100 \
  --batch-size 20 \
  --learning-rate 1e-4 \
  --outdir ./models/ \
  --clip-grad
```
See the complete list of options with `--help`.

#### Key Training Parameters
- `--dataset`: Path to your HDF5 graph dataset
- `--modelname`: Name for your trained model
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Training batch size (default: 20)
- `--hidden-size`: Hidden layer dimensions (default: 256)
- `--embedding-dim`: Embedding dimensions (default: 128)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--clip-grad`: Enable gradient clipping for stability

### 3. Generate Custom Substitution Matrices
Create structure-based substitution matrices using your trained model:
```bash
makesubmat \
  --modelname <my_custom_model> \
  --modeldir ./models/ \
  --datadir <data_dir> \
  --outdir_base <results_dir> \
  --dataset <input_graphs.h5> \
  --encode_alns
```

This script has utilities to download structures from the AFDB cluster database, align clusters as reference alignments using Foldseek, encode structures and derive substitution matrices.

See the complete list of options with `--help`.

### 4. Use Your Custom Model
Once trained, use your custom model in the main pipeline:
```bash
foldtree2 --model <my_custom_model> \
  --structures <YOURSTRUCTUREFOLDER> \
  --outdir <RESULTSFOLDER>
```

### Training Tips
- **GPU Acceleration**: Training is significantly faster with CUDA-enabled GPUs
- **Dataset Size**: Larger, more diverse datasets generally produce better models
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and architectures
- **Monitoring**: Use TensorBoard logs to monitor training progress
- **Checkpointing**: Save model checkpoints regularly to resume training if interrupted

## Requirements
- Python 3.7+
- See `pyproject.toml` or `setup.py` for a full list of dependencies.

## License
MIT License (see LICENSE.txt)

## Author
Dave Moi (<dmoi@unil.ch>)

---
For more details, see the source code and scripts in the repository.
