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

## Usage

### 1. Convert PDBs to Graphs
Convert a directory of PDB files into a graph HDF5 dataset:
```bash
pdbs-to-graphs <input_pdb_dir> <output_graphs.h5> --aapropcsv config/aaindex1.csv
```

### 2. Generate Substitution Matrices and Alignments
Generate structure-based substitution matrices and alignments:
```bash
makesubmat --modelname <model_name> --modeldir models/ --datadir <data_dir> --outdir_base <results_dir> --dataset <input_graphs.h5> --encode_alns
```

- `--modelname`: Name of the model to use (e.g., `small`)
- `--modeldir`: Directory containing model `.pkl` files
- `--datadir`: Directory with datasets and alignment files
- `--outdir_base`: Output directory for results
- `--dataset`: HDF5 file with graph data (from step 1)
- `--encode_alns`: Encode alignments using the model

### 3. Build Phylogenetic Trees
Run the tree builder to infer a phylogenetic tree from encoded alignments:
```bash
foldtree2 --model mergeddecoder_foldtree2_test \
  --structures <YOURSTRUCTUREFOLDER> \
  --outdir <RESULTSFOLDER> \
```

## Training Custom Models

FoldTree2 supports training custom graph neural network models using datasets generated with `pdbgraph`.

### 1. Prepare Your Dataset
First, convert your PDB files to a graph HDF5 dataset as described above:
```bash
pdbs-to-graphs <input_pdb_dir> <output_graphs.h5> --aapropcsv config/aaindex1.csv
```

### 2. Train a Model
You can train a model using the provided training scripts (e.g., `learn.py` or `learn_lightning.py`). These scripts expect a graph HDF5 file as input.

Example command:
```bash
python learn.py --dataset <output_graphs.h5> --modelname <my_model> --epochs 50 --batch_size 8 --outdir ./models/
```

- `--dataset`: Path to your HDF5 graph dataset (from pdbgraph)
- `--modelname`: Name for your trained model
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--outdir`: Directory to save the trained model and logs

You can also use `learn_lightning.py` for PyTorch Lightning-based training, which supports advanced features like logging, checkpoints, and multi-GPU training.

### 3. Use Your Trained Model
After training, your model will be saved as a `.pkl` file in the specified output directory. You can use this model in the FoldTree2 pipeline for encoding alignments and building trees, as shown in the workflow above.

## Requirements
- Python 3.7+
- See `pyproject.toml` or `setup.py` for a full list of dependencies.

## License
MIT License (see LICENSE.txt)

## Author
Dave Moi (<dmoi@unil.ch>)

---
For more details, see the source code and scripts in the repository.
