# FoldTree2 AI Agent Instructions

## Project Overview
FoldTree2 performs phylogenetic tree inference from protein 3D structures using neural network-based structural encoding. The core workflow: PDB → Graph → Encoder → Discrete Alphabet → Substitution Matrix → Tree Inference.

## Architecture Components

### Core Neural Network Stack
- **Encoder** ([foldtree2/src/encoder.py](../foldtree2/src/encoder.py)): `mk1_Encoder` converts protein graphs to discrete embeddings via VQ-VAE
  - Modular architecture: `input` (MLPs), `body` (graph convolutions), `head` (quantization)
  - Supports multiple GNN flavors: `sage`, `gat`, `transformer`
  - Vector quantization with optional EMA codebook updates
- **Decoders** ([foldtree2/src/mono_decoders.py](../foldtree2/src/mono_decoders.py)): 
  - `MultiMonoDecoder`: Combined AA sequence, geometry, angles, secondary structure prediction
  - `HeteroGAE_geo_Decoder`: Heterogeneous graph autoencoder for pairwise distances
  - `Transformer_AA_Decoder`: Sequence reconstruction with attention
- **Graph Conversion** ([foldtree2/src/pdbgraph.py](../foldtree2/src/pdbgraph.py)): `PDB2PyG` converts PDB files to PyTorch Geometric graphs
  - Uses ProDy for interaction matrices and DSSP for secondary structure
  - Node features from the packaged amino-acid properties CSV resolved at runtime
  - Edge features: distances, angles, contact maps (default 15Å threshold)

### Training Systems
Two parallel training scripts (same model architecture, different frameworks):
- **[learn_monodecoder.py](../foldtree2/learn_monodecoder.py)**: Original single-GPU implementation
- **[learn_lightning.py](../foldtree2/learn_lightning.py)**: Multi-GPU with PyTorch Lightning (recommended)
  - Supports DDP strategy for distributed training
  - Mixed precision training with `torch.cuda.amp`
  - Muon optimizer for modular architectures (`--use-muon`)
  - See [LEARN_LIGHTNING_UPDATE.md](../LEARN_LIGHTNING_UPDATE.md) for migration details

### Pipeline Components
- **[ft2treebuilder.py](../foldtree2/ft2treebuilder.py)**: `treebuilder` class orchestrates end-to-end phylogenetic inference
  - Encodes structures → discrete sequences using trained encoder
  - Generates custom character alphabets for RAxML
  - Calls MAFFT (with custom substitution matrices) → RAxML-NG → MAD rooting
- **[makesubmat.py](../foldtree2/makesubmat.py)**: Creates structure-based substitution matrices
  - Downloads AFDB cluster representatives → Foldseek alignment → encode → count transitions
  - Outputs MAFFT `.mat` files and RAxML-compatible matrices

## Critical Workflows

### Training a New Model
```bash
# Prepare data
pdbs-to-graphs <pdb_dir> training.h5

# Train (Lightning version with multi-GPU)
python foldtree2/learn_lightning.py --config config_multi_gpu_training.yaml

# Or use command-line args
python foldtree2/learn_lightning.py \
  --dataset training.h5 \
  --model-name my_model \
  --epochs 1000 \
  --batch-size 15 \
  --use-muon \
  --mixed-precision \
  --gpus 4
```

### Building Trees from Structures
```bash
# Complete pipeline (requires pretrained model in models/)
foldtree2 --model mergeddecoder_foldtree2_test \
  --structures <pdb_folder> \
  --outdir results/

# Generate custom substitution matrix first
makesubmat --modelname my_model \
  --download_structs --convert_to_pyg --align_structs --encode_alns
```

## Project-Specific Conventions

### Configuration Files
- YAML configs in root (e.g., `config_multi_gpu_training.yaml`) set all hyperparameters
- Command-line args override config file values
- All training scripts support `--config` flag for reproducible experiments

### Loss Weights (from learn_lightning.py)
```python
edgeweight = 0.25      # Graph reconstruction
logitweight = 0.25     # Sequence reconstruction (CE)
xweight = 1.0          # Node coordinate MSE
vqweight = 0.1         # VQ commitment loss
angles_weight = 0.05   # Backbone angles
ss_weight = 0.25       # Secondary structure (3-class)
```

### HDF5 Dataset Structure
Training data stored as HDF5 with keys:
- `data_X`: Serialized PyTorch Geometric `Data` objects
- Loaded via `StructureDataset` class in pdbgraph.py
- Example datasets: `structs_train_final.h5`, `structs_training.h5`

### Model Checkpointing
- Models saved as `.pt` files containing full state dict + metadata
- Use `save_checkpoint()` and `load_checkpoint()` in encoder.py/mono_decoders.py
- Include `model_class`, `model_args`, `model_kwargs` for reconstruction

### Character Encoding for Phylogenetics
- Discrete embeddings mapped to Unicode characters (chr(1) - chr(num_embeddings))
- Special character replacements for RAxML compatibility (see `treebuilder.replace_dict`)
- Alphabets sorted and persisted in pickle files for consistency

## Key Files Reference
- Entry points: [encode_pdbs.py](../foldtree2/encode_pdbs.py), [ft2treebuilder.py](../foldtree2/ft2treebuilder.py)
- Loss functions: [foldtree2/src/losses/](../foldtree2/src/losses/)
- Quantizers: [foldtree2/src/quantizers.py](../foldtree2/src/quantizers.py) (VQ-VAE implementation)
- External tools bundled: `raxml-ng/`, `madroot/`, MAFFT wrappers in `mafft_tools/`

## Development Notes
- Environment setup: `conda env create -f foldtree2.yml && pip install .`
- GPU strongly recommended (CUDA required for training)
- TensorBoard logs in `./runs/` by default
- Multiprocessing for PDB conversion: use `--multiprocessing --ncpu 25` in encode_pdbs.py
- pLDDT masking: Use `--mask-plddt --plddt-threshold 0.3` to exclude low-confidence regions during training
