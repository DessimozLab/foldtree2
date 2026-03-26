# coding: utf-8
# learn_monodecoder.py - Training script for MultiMonoDecoder

import os

from foldtree2.src.losses.fape import rotation_matrix_to_quaternion
# Set CUDA memory allocator to use expandable segments to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from torch_geometric.data import DataLoader
import numpy as np
from foldtree2.src import pdbgraph
from foldtree2.src import encoder as ecdr
from foldtree2.src.losses.losses import recon_loss_diag, aa_reconstruction_loss, angles_reconstruction_loss, UncertaintyWeighting , batch_fape_loss, batch_lddt_loss, batch_delta_loss
from foldtree2.src.mono_decoders import MultiMonoDecoder
from foldtree2.src.visualization import create_reconstruction_figure

import os
import tqdm
import random
import torch.nn.functional as F
import pickle
import argparse
import sys
import time
import gc
import math
import warnings
import yaml
from matplotlib import pyplot as plt
import json
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings("ignore", category=UserWarning)


def build_notebook_mono_configs(args, converter, hidden_size, ndim_godnode, ndim_fft2r, ndim_fft2i):
	"""Build MultiMonoDecoder configs matching the notebook defaults as closely as possible."""
	geometry_output_rt = args.output_rt if args.geometry_output_rt is None else args.geometry_output_rt
	geometry_cnn_output_fft = args.output_fft if args.geometry_cnn_output_fft is None else args.geometry_cnn_output_fft

	mono_configs = {
		'sequence_transformer': {
			'in_channels': {'res': args.embedding_dim},
			'xdim': 20,
			'concat_positions': False,
			'hidden_channels': {('res', 'backbone', 'res'): [hidden_size], ('res', 'backbonerev', 'res'): [hidden_size]},
			'layers': 2,
			'AAdecoder_hidden': [hidden_size, hidden_size, hidden_size],
			'amino_mapper': converter.aaindex,
			'nheads': 5,
			'dropout': 0.001,
			'normalize': False,
			'residual': False,
			'use_cnn_decoder': args.sequence_use_cnn_decoder,
			'output_ss': args.sequence_output_ss,
			'learn_positions': True,
		},
	}

	if args.use_geometry_transformer:
		mono_configs['geometry_transformer'] = {
			'in_channels': {'res': args.embedding_dim},
			'concat_positions': False,
			'hidden_channels': {('res', 'backbone', 'res'): [hidden_size], ('res', 'backbonerev', 'res'): [hidden_size]},
			'layers': 2,
			'nheads': 5,
			'RTdecoder_hidden': [hidden_size, hidden_size, hidden_size],
			'ssdecoder_hidden': [hidden_size, hidden_size, hidden_size],
			'anglesdecoder_hidden': [hidden_size, hidden_size, hidden_size],
			'dropout': 0.001,
			'normalize': False,
			'residual': False,
			'learn_positions': True,
			'output_rt': geometry_output_rt,
			'output_ss': args.geometry_output_ss,
			'output_angles': args.geometry_output_angles,
		}

	if args.use_geometry_cnn:
		mono_configs['geometry_cnn'] = {
			'in_channels': {'res': args.embedding_dim, 'godnode4decoder': ndim_godnode, 'foldx': 23, 'fft2r': ndim_fft2r, 'fft2i': ndim_fft2i},
			'concat_positions': False,
			'conv_channels': [2 * hidden_size, hidden_size, hidden_size],
			'kernel_sizes': [3] * args.nconv_layers,
			'FFT2decoder_hidden': [hidden_size // 2, hidden_size // 2],
			'contactdecoder_hidden': [hidden_size // 2, hidden_size // 4],
			'ssdecoder_hidden': [hidden_size // 2, hidden_size // 2],
			'Xdecoder_hidden': [hidden_size, hidden_size],
			'anglesdecoder_hidden': [hidden_size, hidden_size, hidden_size // 2],
			'RTdecoder_hidden': [hidden_size // 2, hidden_size // 4],
			'metadata': converter.metadata,
			'dropout': 0.001,
			'output_fft': geometry_cnn_output_fft,
			'output_rt': args.geometry_cnn_output_rt,
			'output_angles': args.geometry_cnn_output_angles,
			'output_ss': args.geometry_cnn_output_ss,
			'normalize': True,
			'residual': False,
			'output_edge_logits': args.geometry_cnn_output_edge_logits,
			'ncat': 8,
			'contact_mlp': False,
			'pool_type': 'global_mean',
			'learn_positions': True,
		}

	if args.output_foldx:
		mono_configs['foldx'] = {
			'in_channels': {'res': args.embedding_dim, 'godnode4decoder': ndim_godnode, 'foldx': 23},
			'concat_positions': False,
			'hidden_channels': {('res', 'backbone', 'res'): [hidden_size] * 3, ('res', 'backbonerev', 'res'): [hidden_size] * 3,
							   ('res', 'informs', 'godnode4decoder'): [hidden_size] * 3,
							   ('godnode4decoder', 'informs', 'res'): [hidden_size] * 3},
			'layers': 3,
			'foldx_hidden': [hidden_size, hidden_size // 2],
			'nheads': 2,
			'metadata': converter.metadata,
			'flavor': 'sage',
			'dropout': 0.005,
			'normalize': True,
			'residual': False,
		}

	return mono_configs

# Try to import Muon optimizer
try:
	from muon import MuonWithAuxAdam
	MUON_AVAILABLE = True
except ImportError:
	print("Warning: Muon optimizer not available. Install with: pip install git+https://github.com/KellerJordan/Muon")
	MUON_AVAILABLE = False

# Try to import Lion optimizer
try:
	from foldtree2.src.lion_optimizer import Lion
	LION_AVAILABLE = True
except ImportError:
	print("Warning: Lion optimizer not available. Lion option will fall back to AdamW.")
	LION_AVAILABLE = False

# Try to import transformers schedulers, fall back to PyTorch schedulers
try:
	from transformers import (
		get_linear_schedule_with_warmup,
		get_cosine_schedule_with_warmup,
		get_cosine_with_hard_restarts_schedule_with_warmup,
		get_polynomial_decay_schedule_with_warmup
	)
	TRANSFORMERS_AVAILABLE = True
except ImportError:
	print("Warning: transformers library not available. Using PyTorch schedulers only.")
	TRANSFORMERS_AVAILABLE = False

def print_about():
	ascii_art = r'''

+-----------------------------------------------------------+
|                         foldtree2                          |
|                 VQ-VAE + Multi-Task Training               |
|     Discrete Structural Tokens → Sequence • Geometry • CM   |
|        Mixed Precision • Accumulation/Clipping • TB         |
|                      🧬   🧠   🌳                          |
+-----------------------------------------------------------+


	'''
	print(ascii_art)
	print("FoldTree2 Training Script")
	print("-" * 50)
	print("Train FoldTree2 encoder-decoder models for structural phylogenetics.\n")
	print("Features:")
	print("  • VQ-VAE encoder for discrete structural tokens")
	print("  • Multi-task decoder (sequence, geometry, contacts)")
	print("  • Mixed precision training")
	print("  • Gradient accumulation & clipping")
	print("  • Optional Muon optimizer for modular architectures")
	print("  • TensorBoard logging\n")
	print("Project: https://github.com/DessimozLab/foldtree2")
	print("Contact: dmoi@unil.ch\n")
	print("Run with --help for usage instructions.")

# Add argparse for CLI configuration
parser = argparse.ArgumentParser(description='Train model with MultiMonoDecoders for sequence and geometry prediction')
parser.add_argument('--about', action='store_true',
					help='Show information about this tool and exit')
parser.add_argument('--config', '-c', type=str, default=None,
					help='Path to config file (YAML or JSON). Command-line args override config file values.')
parser.add_argument('--dataset', '-d', type=str, default='structs_train_final.h5',
					help='Path to the dataset file (default: structs_train_final.h5)')
parser.add_argument('--hidden-size', '-hs', type=int, default=150,
					help='Hidden layer size (default: 150)')
parser.add_argument('--epochs', '-e', type=int, default=100,
					help='Number of epochs for training (default: 100)')
parser.add_argument('--device', type=str, default=None,
					help='Device to run on (e.g., cuda:0, cuda:1, cpu) (default: auto-select)')
parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4,
					help='Learning rate (default: 1e-4)')
parser.add_argument('--batch-size', '-bs', type=int, default=10,
					help='Batch size (default: 10)')
parser.add_argument('--max-residues', type=int, default=None,
					help='Maximum residue count per structure; larger entries are skipped (default: None)')
parser.add_argument('--output-dir', '-o', type=str, default='./models/',
					help='Directory to save models/results (default: ./models/)')
parser.add_argument('--model-name', type=str, default='monodecoder_model',
					help='Model name for saving (default: monodecoder_model)')
parser.add_argument('--num-embeddings', type=int, default=30,
					help='Number of embeddings for the encoder (default: 30)')
parser.add_argument('--embedding-dim', type=int, default=128,
					help='Embedding dimension for the encoder (default: 128)')
parser.add_argument('--se3-transformer', action='store_true',
					help='Use SE3Transformer instead of GNN')
parser.add_argument('--overwrite', action='store_true',
					help='Overwrite saved model if exists, otherwise continue training')
parser.add_argument('--output-fft', action='store_true',
					help='Train the model with FFT output')
parser.add_argument('--output-rt', action='store_true',
					help='Train the model with rotation and translation output')
parser.add_argument('--output-foldx' , action='store_true',
					help='Train the model with Foldx energy prediction output')
parser.add_argument('--use-geometry-transformer', action=argparse.BooleanOptionalAction, default=True,
					help='Enable the notebook-style geometry_transformer decoder (default: True)')
parser.add_argument('--use-geometry-cnn', action=argparse.BooleanOptionalAction, default=True,
					help='Enable the notebook-style geometry_cnn decoder (default: True)')
parser.add_argument('--sequence-use-cnn-decoder', action=argparse.BooleanOptionalAction, default=True,
					help='Use the CNN AA head inside sequence_transformer, matching the notebook (default: True)')
parser.add_argument('--sequence-output-ss', action=argparse.BooleanOptionalAction, default=False,
					help='Allow sequence_transformer to emit secondary structure predictions (default: False)')
parser.add_argument('--geometry-output-rt', action=argparse.BooleanOptionalAction, default=None,
					help='Allow geometry_transformer to emit rotation/translation outputs (default: follows --output-rt)')
parser.add_argument('--geometry-output-ss', action=argparse.BooleanOptionalAction, default=True,
					help='Allow geometry_transformer to emit secondary structure predictions (default: True)')
parser.add_argument('--geometry-output-angles', action=argparse.BooleanOptionalAction, default=True,
					help='Allow geometry_transformer to emit bond angle predictions (default: True)')
parser.add_argument('--geometry-cnn-output-fft', action=argparse.BooleanOptionalAction, default=None,
					help='Allow geometry_cnn to emit FFT predictions (default: follows --output-fft)')
parser.add_argument('--geometry-cnn-output-rt', action=argparse.BooleanOptionalAction, default=False,
					help='Allow geometry_cnn to emit rotation/translation outputs (default: False)')
parser.add_argument('--geometry-cnn-output-angles', action=argparse.BooleanOptionalAction, default=False,
					help='Allow geometry_cnn to emit bond angle predictions (default: False)')
parser.add_argument('--geometry-cnn-output-ss', action=argparse.BooleanOptionalAction, default=False,
					help='Allow geometry_cnn to emit secondary structure predictions (default: False)')
parser.add_argument('--geometry-cnn-output-edge-logits', action=argparse.BooleanOptionalAction, default=True,
					help='Allow geometry_cnn to emit contact edge logits/probabilities (default: True)')
parser.add_argument('--seed', type=int, default=0,
					help='Random seed for reproducibility')
parser.add_argument('--hetero-gae', action='store_true',
					help='Use HeteroGAE_Decoder instead of MultiMonoDecoder')
parser.add_argument('--clip-grad', action='store_true',
					help='Enable gradient clipping during training')
parser.add_argument('--burn-in', type=int, default=0,
					help='Burn-in period for training (default: 0, no burn-in)')
parser.add_argument('--EMA', action='store_true', help='Use Exponential Moving Average for encoder codebook')
parser.add_argument('--tensorboard-dir', type=str, default='./runs/',
					help='Directory for TensorBoard logs (default: ./runs/)')
parser.add_argument('--run-name', type=str, default=None,
					help='Name for this training run (default: auto-generated from timestamp)')
parser.add_argument('--save-config', type=str, default=None,
					help='Save current configuration to file (YAML format)')
parser.add_argument('--lr-warmup-steps', type=int, default=0,
					help='Number of steps for learning rate warmup (default: 0, no warmup)')
parser.add_argument('--lr-warmup-ratio', type=float, default=0.0,
					help='Warmup ratio (fraction of total steps). Overrides --lr-warmup-steps if > 0 (default: 0.0)')
parser.add_argument('--lr-schedule', type=str, default='plateau', 
					choices=['plateau', 'cosine', 'linear', 'cosine_restarts', 'polynomial', 'none'],
					help='Learning rate schedule (default: plateau)')
parser.add_argument('--lr-min', type=float, default=1e-6,
					help='Minimum learning rate for cosine/linear schedules (default: 1e-6)')
parser.add_argument('--gradient-accumulation-steps', '--grad-accum', type=int, default=2,
					help='Number of gradient accumulation steps (default: 2)')
parser.add_argument('--num-cycles', type=int, default=3,
					help='Number of cycles for cosine_restarts scheduler (default: 3)')
parser.add_argument('--use-muon', action='store_true',
					help='Use Muon optimizer for modular encoder/decoder (requires Muon package)')
parser.add_argument('--optimizer-type', type=str, default='adamw', choices=['adamw', 'lion', 'muon'],
					help='Optimizer type (default: adamw). If --use-muon is set, this is overridden to muon.')
parser.add_argument('--use-muon-encoder', action='store_true',
					help='Use Muon-compatible encoder (mk1_MuonEncoder)')
parser.add_argument('--use-muon-decoders', action='store_true',
					help='Use Muon-compatible decoders in MultiMonoDecoder')
parser.add_argument('--muon-lr', type=float, default=0.02,
					help='Learning rate for Muon optimizer (default: 0.02)')
parser.add_argument('--adamw-lr', type=float, default=1e-4,
					help='Learning rate for AdamW when using Muon (default: 1e-4)')
parser.add_argument('--lion-lr', type=float, default=1e-5,
					help='Learning rate for Lion optimizer (default: 1e-5)')
parser.add_argument('--lion-beta1', type=float, default=0.9,
					help='Beta1 for Lion optimizer (default: 0.9)')
parser.add_argument('--lion-beta2', type=float, default=0.99,
					help='Beta2 for Lion optimizer (default: 0.99)')
parser.add_argument('--lion-weight-decay', type=float, default=0.1,
					help='Weight decay for Lion optimizer (default: 0.1)')
parser.add_argument('--mixed-precision', action='store_true', default=True,
					help='Use mixed precision training (default: True)')
parser.add_argument('--mask-plddt', action='store_true',
					help='Mask low pLDDT residues in loss calculations')
parser.add_argument('--plddt-threshold', type=float, default=0.3,
					help='pLDDT threshold for masking (default: 0.3)')
# lDDT and FAPE loss arguments
parser.add_argument('--lddt-loss', action='store_true', default=False,
					help='Enable lDDT loss during training')
parser.add_argument('--lddt-weight', type=float, default=0.0,
					help='Weight for lDDT loss (default: 0.0)')
parser.add_argument('--fape-loss', action='store_true', default=False,
					help='Enable FAPE loss during training')
parser.add_argument('--fape-weight', type=float, default=0.0,
					help='Weight for FAPE loss (default: 0.0)')
parser.add_argument('--delta-loss', action='store_true', default=False,
					help='Enable delta loss during training')
parser.add_argument('--delta-weight', type=float, default=0.0,
					help='Weight for delta loss (default: 0.0)')

# Early stopping parameters
parser.add_argument('--early-stopping', action='store_true',
					help='Enable early stopping based on a monitored metric')
parser.add_argument('--early-stopping-metric', type=str, default='val/loss',
					help='Metric key to monitor for early stopping (default: val/loss)')
parser.add_argument('--early-stopping-mode', type=str, default='min',
					choices=['min', 'max'],
					help='Whether to minimize or maximize the monitored metric (default: min)')
parser.add_argument('--early-stopping-patience', type=int, default=10,
					help='Number of epochs with no improvement before stopping (default: 10)')
parser.add_argument('--early-stopping-min-delta', type=float, default=0.0,
					help='Minimum change to qualify as an improvement (default: 0.0)')
parser.add_argument('--early-stopping-warmup-epochs', type=int, default=0,
					help='Number of initial epochs to skip early stopping checks (default: 0)')

# Validation parameters
parser.add_argument('--val-split', type=float, default=0.1,
					help='Fraction of data to use for validation (default: 0.1)')
parser.add_argument('--val-seed', type=int, default=42,
					help='Random seed for train/val split (default: 42)')

# Commitment cost scheduling
parser.add_argument('--commitment-cost', type=float, default=0.9,
					help='Final commitment cost for VQ-VAE (default: 0.9)')
parser.add_argument('--use-commitment-scheduling', action='store_true',
					help='Enable commitment cost scheduling (warmup from low to final value)')
parser.add_argument('--commitment-schedule', type=str, default='cosine',
					choices=['cosine', 'linear', 'none'],
					help='Commitment cost schedule type (default: cosine)')
parser.add_argument('--commitment-warmup-steps', type=int, default=1000,
					help='Number of steps to warmup commitment cost (default: 1000)')
parser.add_argument('--commitment-start', type=float, default=0.5,
					help='Starting commitment cost when using scheduling (default: 0.5)')

# Loss weight arguments
parser.add_argument('--edgeweight', type=float, default=0.1,
					help='Weight for edge reconstruction loss (default: 0.1)')
parser.add_argument('--logitweight', type=float, default=0.1,
					help='Weight for logit loss (default: 0.1)')
parser.add_argument('--xweight', type=float, default=0.1,
					help='Weight for AA reconstruction loss (default: 0.1)')
parser.add_argument('--fft2weight', type=float, default=0.01,
					help='Weight for FFT2 loss (default: 0.01)')
parser.add_argument('--vqweight', type=float, default=0.005,
					help='Weight for VQ-VAE loss (default: 0.005)')
parser.add_argument('--angles-weight', type=float, default=0.1,
					help='Weight for angles reconstruction loss (default: 0.1)')
parser.add_argument('--ss-weight', type=float, default=0.1,
					help='Weight for secondary structure loss (default: 0.1)')

parser.add_argument('--jump-aa-loss', type=int, default=None,
					help='Jump amino acid reconstruction loss to .5 after n epochs (burn-in) to stabilize training (default: None)')
parser.add_argument('--jump-ss-loss', type=int, default=None,
					help='Jump secondary structure loss to .5 after n epochs (burn-in) to stabilize training (default: None)')


# Uncertainty weighting
parser.add_argument('--use-uncertainty-weighting', action='store_true',
					help='Use uncertainty-based loss weighting (Kendall & Gal method, learns per-task weights)')

parser.add_argument('--nconv-layers', type=int, default=3,
					help='Number of convolutional layers in the geometry decoder (default: 3)')

# Loss normalization
parser.add_argument('--normalize-loss-weights', action='store_true',
					help='Normalize all loss weights to 1.0 (overrides individual weight settings)')
parser.add_argument('--use-weight-scheduler', action='store_true',
					help='Enable per-loss weight scheduling during training')
parser.add_argument('--loss-warmup-steps', type=int, default=0,
					help='Warmup steps for per-loss weight scheduling (default: 0)')
parser.add_argument('--loss-schedule-x', type=str, default='linear',
					choices=['constant', 'linear', 'cosine', 'cosine_restarts', 'polynomial'],
					help='Schedule for amino-acid reconstruction weight (default: linear)')
parser.add_argument('--loss-schedule-logit', type=str, default='linear',
					choices=['constant', 'linear', 'cosine', 'cosine_restarts', 'polynomial'],
					help='Schedule for edge logit weight (default: linear)')
parser.add_argument('--loss-schedule-edge', type=str, default='cosine_restarts',
					choices=['constant', 'linear', 'cosine', 'cosine_restarts', 'polynomial'],
					help='Schedule for edge reconstruction weight (default: cosine_restarts)')
parser.add_argument('--loss-schedule-vq', type=str, default='cosine_restarts',
					choices=['constant', 'linear', 'cosine', 'cosine_restarts', 'polynomial'],
					help='Schedule for VQ loss weight (default: cosine_restarts)')
parser.add_argument('--loss-schedule-fft2', type=str, default='cosine_restarts',
					choices=['constant', 'linear', 'cosine', 'cosine_restarts', 'polynomial'],
					help='Schedule for FFT2 loss weight (default: cosine_restarts)')
parser.add_argument('--loss-schedule-angles', type=str, default='linear',
					choices=['constant', 'linear', 'cosine', 'cosine_restarts', 'polynomial'],
					help='Schedule for angles loss weight (default: linear)')
parser.add_argument('--loss-schedule-ss', type=str, default='linear',
					choices=['constant', 'linear', 'cosine', 'cosine_restarts', 'polynomial'],
					help='Schedule for SS loss weight (default: linear)')
parser.add_argument('--loss-power-ss', type=float, default=2.0,
					help='Polynomial power parameter used for SS loss schedule (default: 2.0)')
parser.add_argument('--loss-cycles-edge', type=int, default=3,
					help='Cycles for edge cosine_restarts schedule (default: 3)')
parser.add_argument('--loss-cycles-vq', type=int, default=10,
					help='Cycles for VQ cosine_restarts schedule (default: 10)')
parser.add_argument('--loss-cycles-fft2', type=int, default=5,
					help='Cycles for FFT2 cosine_restarts schedule (default: 5)')

# Tensor Core precision
parser.add_argument('--tensor-core-precision', type=str, default='high',
					choices=['highest', 'high', 'medium'],
					help='Float32 matrix multiplication precision for Tensor Cores (default: high)')

# Handle --about flag before argument parsing
if '--about' in sys.argv:
	print_about()
	sys.exit(0)

# Print an overview of the arguments and example command if no arguments provided
if len(sys.argv) == 1:
	print('No arguments provided. Use -h for help.')
	print('Example command: python learn_monodecoder.py -d structs_traininffttest.h5 -o ./models/ -lr 0.0001 -e 20 -bs 5')
	print('Example with config: python learn_monodecoder.py --config my_config.yaml')
	print('Available arguments:')
	parser.print_help()
	sys.exit(0)

# Parse arguments first to check for config file
args = parser.parse_args()

# Load config file if provided
if args.config:
	print(f"Loading configuration from {args.config}")
	config_path = args.config
	if config_path.endswith('.yaml') or config_path.endswith('.yml'):
		with open(config_path, 'r') as f:
			config = yaml.safe_load(f)
	elif config_path.endswith('.json'):
		with open(config_path, 'r') as f:
			config = json.load(f)
	else:
		raise ValueError("Config file must be YAML (.yaml/.yml) or JSON (.json)")
	
	# Legacy aliases accepted from older notebook/config naming
	config_to_arg_map = {
		'dataset_path': 'dataset',
		'num_epochs': 'epochs',
		'random_seed': 'seed',
		'scheduler_type': 'lr_schedule',
		'warmup_steps': 'lr_warmup_steps',
		'warmup_ratio': 'lr_warmup_ratio',
		'use_mixed_precision': 'mixed_precision',
		'use_weight_scheduler': 'use_weight_scheduler',
	}
	
	# Set defaults from config file, but allow CLI args to override
	for key, value in config.items():
		# Map config key to argument name if needed
		normalized_key = key.replace('-', '_')
		arg_key = config_to_arg_map.get(normalized_key, normalized_key)
		
		# Only set from config if not explicitly provided via CLI
		if hasattr(args, arg_key):
			# Check if the argument was provided via CLI (not default)
			cli_value = getattr(args, arg_key)
			default_value = parser.get_default(arg_key)
			if cli_value == default_value:
				# Use config value since CLI didn't override
				setattr(args, arg_key, value)
				print(f"  {arg_key}: {value} (from config key '{key}')")
			else:
				print(f"  {arg_key}: {cli_value} (from CLI, overriding config)")
		else:
			print(f"  Warning: Unknown config key '{key}' - ignoring")
else:
	print("No config file provided, using command-line arguments and defaults")

if args.use_muon:
	args.optimizer_type = 'muon'

# Set seeds for reproducibility
torch.manual_seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set tensor core precision for better performance on modern GPUs
if torch.cuda.is_available():
	torch.set_float32_matmul_precision(args.tensor_core_precision)
	print(f"Tensor Core precision set to: {args.tensor_core_precision}")

if args.EMA:
	print("Using Exponential Moving Average for encoder codebook")
else:
	print("Not using Exponential Moving Average for encoder codebook")
	args.EMA = False

if args.overwrite:
	print("Overwrite mode enabled. Existing models will be overwritten.")
else:
	print("Overwrite mode disabled. Existing models will not be overwritten.")
	args.overwrite = False
if args.clip_grad:
	print("Gradient clipping enabled.")
else:
	print("Gradient clipping disabled.")
	args.clip_grad = False



# Batched reconstruction helper function
def decode_batch_reconstruction(encoder, decoder, z_batch, device, converter, verbose=False):
	"""
	Batch decode discrete embeddings back to predictions.
	
	Args:
		encoder: Trained encoder model
		decoder: Trained decoder model  
				# lDDT loss
				lddt_loss = torch.tensor(0.0, device=device)
				if getattr(args, 'lddt_loss', False):
					from foldtree2.src.losses.losses import lddt_reconstruction_loss
					if out.get('coords') is not None and hasattr(batch, 'coords') and hasattr(batch['coords'], 'x'):
						lddt_loss = lddt_reconstruction_loss(
							out['coords'], batch['coords'].x,
							plddt=batch['plddt'].x if args.mask_plddt else None,
							plddt_thresh=args.plddt_threshold if args.mask_plddt else 0.0
						)
				# FAPE loss
				fape_loss = torch.tensor(0.0, device=device)
				if getattr(args, 'fape_loss', False):
					from foldtree2.src.losses.losses import quaternion_fape_loss
					if all([out.get('quat') is not None, out.get('trans') is not None, hasattr(batch, 'quat'), hasattr(batch['quat'], 'x'), hasattr(batch, 'trans'), hasattr(batch['trans'], 'x')]):
						fape_loss = quaternion_fape_loss(
							batch['quat'].x, batch['trans'].x,
							out['quat'], out['trans']
						)
		z_batch: List of discrete embedding index tensors
		device: PyTorch device
		converter: PDB2PyG converter
						args.lddt_weight * lddt_loss + args.fape_weight * fape_loss)
		verbose: Print debug information
		
	Returns:
		List of dictionaries containing predictions for each sequence
	"""
	encoder.eval()
	decoder.eval()
	
	with torch.no_grad():
		results = decoder.decode_batch_with_contacts(z_batch, device, converter, encoder)
	
	if verbose:
		print(f"Decoded batch of {len(z_batch)} sequences")
		for i, result in enumerate(results):
			print(f"\nSequence {i}:")
			for key, value in result.items():
				if value is not None:
					print(f"  {key}: {value.shape}")
	
	return results

# Print configuration
print(f"Configuration:")
print(f"  Dataset: {args.dataset}")
print(f"  Hidden Size: {args.hidden_size}")
print(f"  Epochs: {args.epochs}")
print(f"  Device: {args.device if args.device else 'auto-select'}")
print(f"  Learning Rate: {args.learning_rate}")
print(f"  Batch Size: {args.batch_size}")
print(f"  Output Directory: {args.output_dir}")
print(f"  Model Name: {args.model_name}")
print(f"  Number of Embeddings: {args.num_embeddings}")
print(f"  Embedding Dimension: {args.embedding_dim}")
print(f"  SE3 Transformer: {'Enabled' if args.se3_transformer else 'Disabled'}")
print(f"  Overwrite Existing Models: {'Yes' if args.overwrite else 'No'}")
print(f"  Output FFT: {'Enabled' if args.output_fft else 'Disabled'}")
print(f"  Output RT: {'Enabled' if args.output_rt else 'Disabled'}")
print(f"  Output Foldx: {'Enabled' if args.output_foldx else 'Disabled'}")
print(f"  Sequence CNN Head: {'Enabled' if args.sequence_use_cnn_decoder else 'Disabled'}")
print(f"  Geometry Transformer: {'Enabled' if args.use_geometry_transformer else 'Disabled'}")
print(f"  Geometry CNN: {'Enabled' if args.use_geometry_cnn else 'Disabled'}")
print(f"  Sequence Output SS: {'Enabled' if args.sequence_output_ss else 'Disabled'}")
print(f"  Geometry Output RT: {'Enabled' if (args.output_rt if args.geometry_output_rt is None else args.geometry_output_rt) else 'Disabled'}")
print(f"  Geometry Output SS: {'Enabled' if args.geometry_output_ss else 'Disabled'}")
print(f"  Geometry Output Angles: {'Enabled' if args.geometry_output_angles else 'Disabled'}")
print(f"  Geometry CNN Output FFT: {'Enabled' if (args.output_fft if args.geometry_cnn_output_fft is None else args.geometry_cnn_output_fft) else 'Disabled'}")
print(f"  Geometry CNN Output RT: {'Enabled' if args.geometry_cnn_output_rt else 'Disabled'}")
print(f"  Geometry CNN Output Angles: {'Enabled' if args.geometry_cnn_output_angles else 'Disabled'}")
print(f"  Geometry CNN Output SS: {'Enabled' if args.geometry_cnn_output_ss else 'Disabled'}")
print(f"  Geometry CNN Edge Logits: {'Enabled' if args.geometry_cnn_output_edge_logits else 'Disabled'}")
print(f"  Hetero GAE Decoder: {'Enabled' if args.hetero_gae else 'Disabled'}")
print(f"  Gradient Clipping: {'Enabled' if args.clip_grad else 'Disabled'}")
print(f"  Burn-in Period: {args.burn_in} epochs")
print(f"  Random Seed: {args.seed}")
print(f"  Exponential Moving Average: {'Enabled' if args.EMA else 'Disabled'}")
print(f"  TensorBoard Directory: {args.tensorboard_dir}")
print(f"  Run Name: {args.run_name if args.run_name else 'auto-generated'}")
print(f"  LR Schedule: {args.lr_schedule}")
print(f"  Optimizer Type: {args.optimizer_type}")
print(f"  LR Warmup Steps: {args.lr_warmup_steps}")
print(f"  LR Warmup Ratio: {args.lr_warmup_ratio}")
print(f"  Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
if args.lr_schedule in ['cosine', 'linear', 'cosine_restarts', 'polynomial']:
	print(f"  LR Min: {args.lr_min}")
if args.lr_schedule == 'cosine_restarts':
	print(f"  Num Cycles: {args.num_cycles}")
print(f"  Commitment Cost: {args.commitment_cost}")
print(f"  Use Commitment Scheduling: {args.use_commitment_scheduling}")
if args.use_commitment_scheduling:
	print(f"  Commitment Schedule: {args.commitment_schedule}")
	print(f"  Commitment Warmup Steps: {args.commitment_warmup_steps}")
	print(f"  Commitment Start: {args.commitment_start}")
print(f"  Use Weight Scheduler: {args.use_weight_scheduler}")
if args.use_weight_scheduler:
	print(f"  Loss Warmup Steps: {args.loss_warmup_steps}")
	print(f"  Loss Schedules: x={args.loss_schedule_x}, logit={args.loss_schedule_logit}, edge={args.loss_schedule_edge}, vq={args.loss_schedule_vq}, fft2={args.loss_schedule_fft2}, angles={args.loss_schedule_angles}, ss={args.loss_schedule_ss}")




# Loss weights (from args, with defaults matching notebook)
if args.normalize_loss_weights:
	print("Loss weight normalization enabled - setting all weights to imporance of task")
	edgeweight = 2.0
	logitweight = 1.0
	xweight = 2.0
	fft2weight = 1.0
	vqweight = 1.0
	angles_weight = 0.1
	ss_weight = 0.1
else:
	edgeweight = args.edgeweight
	logitweight = args.logitweight
	xweight = args.xweight
	fft2weight = args.fft2weight
	vqweight = args.vqweight
	angles_weight = args.angles_weight
	ss_weight = args.ss_weight

	
print(f"Loss Weights:")
print(f"  Normalize Loss Weights: {args.normalize_loss_weights}")
print(f"  Edge Weight: {edgeweight}")
print(f"  Logit Weight: {logitweight}")
print(f"  X Weight: {xweight}")
print(f"  FFT2 Weight: {fft2weight}")
print(f"  VQ Weight: {vqweight}")
print(f"  Angles Weight: {angles_weight}")
print(f"  SS Weight: {ss_weight}")

# Per-epoch weight values (for optional scheduling/visualization)
xweight_epoch = xweight
ss_weight_epoch = ss_weight

# Save configuration if requested
if args.save_config:
	config_dict = vars(args).copy()
	# Remove non-serializable or irrelevant fields
	config_dict.pop('save_config', None)
	config_dict.pop('config', None)
	with open(args.save_config, 'w') as f:
		yaml.dump(config_dict, f, default_flow_style=False)
	print(f"Configuration saved to {args.save_config}")

if os.path.exists(args.output_dir) and args.overwrite:
	#remove existing model
	if os.path.exists(os.path.join(args.output_dir, args.model_name + '_best_encoder.pt')):
		os.remove(os.path.join(args.output_dir, args.model_name + '_best_encoder.pt'))
	if os.path.exists(os.path.join(args.output_dir, args.model_name + '_best_decoder.pt')):
		os.remove(os.path.join(args.output_dir, args.model_name + '_best_decoder.pt'))

# Data setup
datadir = '../../datasets/foldtree2/'
dataset_path = args.dataset
converter = pdbgraph.PDB2PyG(aapropcsv='./foldtree2/config/aaindex1.csv')
struct_dat = pdbgraph.StructureDataset(dataset_path)

# Filter by maximum protein size (residue count) if requested
if args.max_residues is not None:
	max_residues = args.max_residues
	print(f"Filtering dataset to max_residues={max_residues}")
	keep_indices = []
	for i in tqdm.tqdm(range(len(struct_dat)), desc='Filtering structs by residue count', leave=False):
		sample = struct_dat[i]
		n_res = sample['res'].x.shape[0] if 'res' in sample else 0
		if n_res <= max_residues:
			keep_indices.append(i)
	print(f"Kept {len(keep_indices)} / {len(struct_dat)} structures after max_residues filter")
	struct_dat = torch.utils.data.Subset(struct_dat, keep_indices)

# Create train/validation split
torch.manual_seed(args.val_seed)
val_size = int(len(struct_dat) * args.val_split)
train_size = len(struct_dat) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(struct_dat, [train_size, val_size])

print(f"Dataset split: {train_size} training samples, {val_size} validation samples")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
data_sample = next(iter(train_loader))

# Set device
if args.device:
	device = torch.device(args.device)
else:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Get dimensions from data sample
ndim = data_sample['res'].x.shape[1]
ndim_godnode = data_sample['godnode'].x.shape[1]
ndim_fft2i = data_sample['fourier2di'].x.shape[1]
ndim_fft2r = data_sample['fourier2dr'].x.shape[1]


# Create output directory
modeldir = args.output_dir
os.makedirs(modeldir, exist_ok=True)
modelname = args.model_name

# Setup TensorBoard
if args.run_name:
	run_name = args.run_name
else:
	# Auto-generate run name with timestamp
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	run_name = f"{modelname}_{timestamp}"

tensorboard_log_dir = os.path.join(args.tensorboard_dir, run_name)
os.makedirs(tensorboard_log_dir, exist_ok=True)
print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
print(f"To view: tensorboard --logdir={args.tensorboard_dir}")

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=tensorboard_log_dir)

# Initialize or load model
encoder_path = os.path.join(modeldir, modelname + '_best_encoder.pt')
decoder_path = os.path.join(modeldir, modelname + '_best_decoder.pt')
if os.path.exists(encoder_path) and os.path.exists(decoder_path) and args.overwrite == False:
	print(f"Loading existing model from {encoder_path} and {decoder_path}")
	if os.path.exists(os.path.join(modeldir, modelname + '_info.txt')):
		with open(os.path.join(modeldir, modelname + '_info.txt'), 'r') as f:
			model_info = f.read()
		print("Model info:", model_info)
	# Load encoder and decoder from saved model
	encoder = torch.load(encoder_path, map_location=device, weights_only=False)
	decoder = torch.load(decoder_path, map_location=device, weights_only=False)
else:
	print("Creating new model...")
	# Model setup
	hidden_size = args.hidden_size
	encoder = ecdr.mk1_Encoder(
		in_channels=ndim,
		hidden_channels=[hidden_size, hidden_size],#, hidden_size],
		out_channels=args.embedding_dim,
		metadata={'edge_types': [('res','contactPoints','res')]},
		num_embeddings=args.num_embeddings,
		commitment_cost=args.commitment_cost,
		edge_dim=12,
		encoder_hidden=hidden_size,
		EMA=args.EMA,
		nheads=16,
		dropout_p=0.005,
		reset_codes=False,
		flavor='transformer',
		fftin=True,
		use_commitment_scheduling=args.use_commitment_scheduling,
		commitment_warmup_steps=args.commitment_warmup_steps,
		commitment_schedule='linear',
		commitment_start=args.commitment_start,
		concat_positions=True,
		learn_positions=True
	)
	if args.hetero_gae:
		# HeteroGAE_Decoder config (example, adjust as needed)
		decoder = ecdr.HeteroGAE_Decoder(
			in_channels={'res': args.embedding_dim, 'godnode4decoder': ndim_godnode, 'foldx': 23},
			concat_positions=False,
			hidden_channels={('res','backbone','res'): [hidden_size]*5, ('res','backbonerev','res'): [hidden_size]*5},
			layers=3,
			AAdecoder_hidden=[hidden_size, hidden_size, hidden_size//2],
			Xdecoder_hidden=[hidden_size, hidden_size, hidden_size],
			contactdecoder_hidden=[hidden_size//2, hidden_size//2],
			nheads=5,
			amino_mapper=converter.aaindex,
			flavor='mfconv',
			dropout=0.005,
			normalize=True,
			residual=False,
			contact_mlp=False,
		)
	else:
		# MultiMonoDecoder for sequence and geometry
		print("Using notebook-style decoder configuration")
		mono_configs = build_notebook_mono_configs(args, converter, hidden_size, ndim_godnode, ndim_fft2r, ndim_fft2i)
		# Initialize decoder
		decoder = MultiMonoDecoder( configs=mono_configs)

# Move models to device
encoder = encoder.to(device)
decoder = decoder.to(device)
print("Encoder:", encoder)
print("Decoder:", decoder)

# Initialize uncertainty weighting if requested
if args.use_uncertainty_weighting:
	print("Initializing uncertainty weighting...")
	uncertainy_weighting = UncertaintyWeighting(
		task_names=[
			'aa_loss',
			'edge_loss',
			'vq_loss',
			'fft2_loss',
			'angles_loss',
			'ss_loss',
			'logit_loss',
			'lddt_loss',
			'fape_loss',
			'delta_loss',
		],
		device=device
	)
	uncertainy_weighting = uncertainy_weighting.to(device)
	print("Uncertainty weighting initialized")
else:
	uncertainy_weighting = None

# Training setup - Optimizer
if args.optimizer_type == 'muon' and MUON_AVAILABLE:
	print("Using Muon optimizer")
	hidden_weights = []
	hidden_gains_biases = []
	nonhidden_params = []
	
	# Helper function to check if a model has modular structure
	def has_modular_structure(model):
		return hasattr(model, 'input') and hasattr(model, 'body') and hasattr(model, 'head')
	
	# Process encoder
	if has_modular_structure(encoder):
		print("Using modular encoder structure")
		hidden_weights += [p for p in encoder.body.parameters() if p.ndim >= 2]
		hidden_gains_biases += [p for p in encoder.body.parameters() if p.ndim < 2]
		nonhidden_params += [*encoder.head.parameters(), *encoder.input.parameters()]
	else:
		print("Encoder is not modular - using AdamW for all encoder parameters")
		nonhidden_params += list(encoder.parameters())
	
	# Process decoder
	if hasattr(decoder, 'decoders'):
		print(f"Using MultiMonoDecoder with {len(decoder.decoders)} sub-decoders")
		for name, subdecoder in decoder.decoders.items():
			if has_modular_structure(subdecoder):
				print(f"  - {name}: modular structure detected")
				hidden_weights += [p for p in subdecoder.body.parameters() if p.ndim >= 2]
				hidden_gains_biases += [p for p in subdecoder.body.parameters() if p.ndim < 2]
				nonhidden_params += [*subdecoder.head.parameters(), *subdecoder.input.parameters()]
			else:
				print(f"  - {name}: non-modular, using AdamW")
				nonhidden_params += list(subdecoder.parameters())
	elif has_modular_structure(decoder):
		print("Using modular single decoder structure")
		hidden_weights += [p for p in decoder.body.parameters() if p.ndim >= 2]
		hidden_gains_biases += [p for p in decoder.body.parameters() if p.ndim < 2]
		nonhidden_params += [*decoder.head.parameters(), *decoder.input.parameters()]
	else:
		print("Decoder is not modular - using AdamW for all decoder parameters")
		nonhidden_params += list(decoder.parameters())
	
	print(f"\nParameter groups for Muon optimizer:")
	print(f"  Hidden weights (Muon):           {len(hidden_weights)} tensors")
	print(f"  Hidden gains/biases (AdamW):     {len(hidden_gains_biases)} tensors")
	print(f"  Non-hidden params (AdamW):       {len(nonhidden_params)} tensors")
	
	param_groups = [
		dict(params=hidden_weights, use_muon=True,
			lr=args.muon_lr, weight_decay=0.01),
		dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
			lr=args.adamw_lr, betas=(0.9, 0.95), weight_decay=0.01),
	]
	
	# Add uncertainty weighting parameters if enabled
	if args.use_uncertainty_weighting:
		param_groups.append(dict(params=uncertainy_weighting.parameters(), use_muon=False,
								lr=args.adamw_lr * 0.1, betas=(0.9, 0.95), weight_decay=0.01))
	
	# Initialize process group for Muon optimizer (required even for single-GPU)
	import torch.distributed as dist
	if not dist.is_available() or not dist.is_initialized():
		try:
			import os as dist_os
			dist_os.environ.setdefault('MASTER_ADDR', 'localhost')
			dist_os.environ.setdefault('MASTER_PORT', '12355')
			dist_os.environ.setdefault('RANK', '0')
			dist_os.environ.setdefault('WORLD_SIZE', '1')
			dist.init_process_group(backend='gloo', init_method='env://')
			print("Initialized single-process group for Muon optimizer")
			optimizer = MuonWithAuxAdam(param_groups)
		except Exception as e:
			print(f"Warning: Could not initialize process group for Muon: {e}")
			print("Falling back to AdamW optimizer")
			optimizer = torch.optim.AdamW(
				list(encoder.parameters()) + list(decoder.parameters()), 
				lr=args.learning_rate, 
				weight_decay=0.000001
			)

	else:
		optimizer = MuonWithAuxAdam(param_groups)
elif args.optimizer_type == 'lion':
	if not LION_AVAILABLE:
		print("Lion optimizer not available, falling back to AdamW")
		params = list(encoder.parameters()) + list(decoder.parameters())
		if args.use_uncertainty_weighting:
			param_groups = [
				{'params': params, 'lr': args.learning_rate},
				{'params': uncertainy_weighting.parameters(), 'lr': args.learning_rate * 0.1}
			]
			optimizer = torch.optim.AdamW(param_groups, weight_decay=0.000001)
		else:
			optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.000001)
	else:
		print("Using Lion optimizer")
		lion_betas = (args.lion_beta1, args.lion_beta2)
		params = list(encoder.parameters()) + list(decoder.parameters())
		if args.use_uncertainty_weighting:
			optimizer = Lion([
				{'params': params, 'lr': args.lion_lr, 'betas': lion_betas, 'weight_decay': args.lion_weight_decay},
				{'params': uncertainy_weighting.parameters(), 'lr': args.lion_lr / 10.0, 'betas': lion_betas, 'weight_decay': 0.0},
			])
		else:
			optimizer = Lion(params, lr=args.lion_lr, betas=lion_betas, weight_decay=args.lion_weight_decay)
else:
	print("Using AdamW optimizer")
	params = list(encoder.parameters()) + list(decoder.parameters())
	if args.use_uncertainty_weighting:
		# Use separate parameter groups with different learning rates
		param_groups = [
			{'params': params, 'lr': args.learning_rate},
			{'params': uncertainy_weighting.parameters(), 'lr': args.learning_rate * 0.1}
		]
		optimizer = torch.optim.AdamW(param_groups, weight_decay=0.000001)
	else:
		optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.000001)

# Define scheduler function with process group initialization
def get_scheduler(optimizer, scheduler_type, num_warmup_steps, num_training_steps, **kwargs):
	if scheduler_type == 'linear' and TRANSFORMERS_AVAILABLE:
		return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps), 'step'
	elif scheduler_type == 'cosine' and TRANSFORMERS_AVAILABLE:
		return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps), 'step'
	elif scheduler_type == 'cosine_restarts' and TRANSFORMERS_AVAILABLE:
		return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=kwargs.get('num_cycles', 3)), 'step'
	elif scheduler_type == 'polynomial' and TRANSFORMERS_AVAILABLE:
		return get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, lr_end=0.0, power=1.0), 'step'
	elif scheduler_type == 'plateau':
		# ReduceLROnPlateau doesn't require distributed process groups - it only monitors loss values
		return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2), 'epoch'
	elif scheduler_type == 'none':
		return None, None
	else:
		raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def loss_weight_scheduler(step, total_steps, schedule_type='linear', warmup_steps=0, power=1.0, num_cycles=1):
	"""Return scale in [0,1] for per-loss weight scheduling."""
	if total_steps <= 0:
		return 1.0
	if warmup_steps > 0 and step < warmup_steps:
		return step / max(warmup_steps, 1)

	post_warmup_total = max(total_steps - warmup_steps, 1)
	progress = (step - warmup_steps) / post_warmup_total
	progress = min(max(progress, 0.0), 1.0)

	if schedule_type == 'constant':
		return 1.0
	if schedule_type == 'linear':
		return 1.0 - progress
	if schedule_type == 'cosine':
		return 0.5 * (1.0 + math.cos(math.pi * progress))
	if schedule_type == 'cosine_restarts':
		cycle_progress = (progress * num_cycles) % 1.0
		return 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
	if schedule_type == 'polynomial':
		return (1.0 - progress) ** power
	raise ValueError(f"Unknown loss schedule type: {schedule_type}")

# Learning rate scheduler setup
total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps  # Adjust for gradient accumulation

# Calculate warmup steps
if args.lr_warmup_ratio > 0:
	warmup_steps = int(total_steps * args.lr_warmup_ratio)
	print(f"Using warmup ratio {args.lr_warmup_ratio:.2%}, calculated warmup_steps: {warmup_steps}")
else:
	warmup_steps = args.lr_warmup_steps

loss_warmup_steps = args.loss_warmup_steps if args.loss_warmup_steps > 0 else warmup_steps

# Initialize scheduler using the new function
scheduler, scheduler_step_mode = get_scheduler(
	optimizer,
	scheduler_type=args.lr_schedule,
	num_warmup_steps=warmup_steps,
	num_training_steps=total_steps,
	num_cycles=args.num_cycles
)


print(f"\nScheduler Configuration:")
print(f"  Schedule type: {args.lr_schedule}")
print(f"  Scheduler step mode: {scheduler_step_mode}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Total training steps: {total_steps}")
print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
if args.lr_schedule in ['cosine', 'linear', 'cosine_restarts', 'polynomial']:
	print(f"  Min learning rate: {args.lr_min}")

# Function to analyze gradient norms
def analyze_gradient_norms(model, top_k=3):
	"""
	Analyzes gradients in the given model and returns the top_k layers with
	highest and lowest gradient norms.
	"""
	grad_norms = []
	for name, param in model.named_parameters():
		if param.grad is not None:
			grad_norm = float(param.grad.norm().item())
			grad_norms.append((name, grad_norm))
	# Sort by gradient norms
	grad_norms.sort(key=lambda x: x[1])
	# Get lowest and highest
	lowest = grad_norms[:top_k]
	highest = grad_norms[-top_k:][::-1]
	return {'highest': highest, 'lowest': lowest}

# Write model parameters to file
with open(os.path.join(modeldir, modelname + '_info.txt'), 'w') as f:
	f.write(f'Date: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
	f.write(f'Run name: {run_name}\n')
	f.write(f'TensorBoard log dir: {tensorboard_log_dir}\n')
	f.write(f'Encoder: {encoder}\n')
	f.write(f'Decoder: {decoder}\n')
	f.write(f'Learning rate: {args.learning_rate}\n')
	f.write(f'Batch size: {args.batch_size}\n')
	f.write(f'Hidden size: {args.hidden_size}\n')
	f.write(f'Embedding dimension: {args.embedding_dim}\n')
	f.write(f'Number of embeddings: {args.num_embeddings}\n')
	f.write(f'Normalize Loss Weights: {args.normalize_loss_weights}\n')
	f.write(f'Loss weights - Edge: {edgeweight}, X: {xweight}, FFT2: {fft2weight}, VQ: {vqweight}\n')
	f.write(f'Loss weights - Angles: {angles_weight}, SS: {ss_weight}, Logit: {logitweight}\n')
	f.write(f'LR Schedule: {args.lr_schedule}\n')
	f.write(f'LR Warmup Steps: {warmup_steps}\n')
	f.write(f'Gradient Accumulation Steps: {args.gradient_accumulation_steps}\n')
	f.write(f'Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}\n')
	f.write(f'Commitment Cost: {args.commitment_cost}\n')
	f.write(f'Use Commitment Scheduling: {args.use_commitment_scheduling}\n')
	if args.use_commitment_scheduling:
		f.write(f'Commitment Schedule: {args.commitment_schedule}\n')
		f.write(f'Commitment Warmup Steps: {args.commitment_warmup_steps}\n')
		f.write(f'Commitment Start: {args.commitment_start}\n')
	f.write(f'Optimizer Type: {args.optimizer_type}\n')
	f.write(f'Use Weight Scheduler: {args.use_weight_scheduler}\n')

# Save configuration to TensorBoard
config_text = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
writer.add_text('Configuration', config_text, 0)

# Log hyperparameters
hparams_dict = {
	'learning_rate': args.learning_rate,
	'batch_size': args.batch_size,
	'hidden_size': args.hidden_size,
	'embedding_dim': args.embedding_dim,
	'num_embeddings': args.num_embeddings,
	'epochs': args.epochs,
	'seed': args.seed,
}
metrics_dict = {}
writer.add_hparams(hparams_dict, metrics_dict)


def quick_validate(encoder, decoder, val_dataset, device, args, n_samples=10):
	"""Run a quick validation on a small random subset of the validation set."""
	encoder.eval()
	decoder.eval()

	indices = torch.randperm(len(val_dataset))[:min(n_samples, len(val_dataset))].tolist()
	samples = [val_dataset[i] for i in indices]
	from torch_geometric.data import Batch
	data = Batch.from_data_list(samples).to(device)

	total_loss_x = total_loss_edge = total_vq = 0.0
	total_angles_loss = total_loss_fft2 = total_logit_loss = total_ss_loss = 0.0

	with torch.no_grad():
		z, vqloss = encoder(data)
		data['res'].x = z
		out = decoder(data, None)
		edge_index = data.edge_index_dict.get(('res', 'contactPoints', 'res')) if hasattr(data, 'edge_index_dict') else None

		logitloss = torch.tensor(0.0, device=device)
		edgeloss = torch.tensor(0.0, device=device)
		if edge_index is not None:
			edgeloss, logitloss = recon_loss_diag(data, edge_index, decoder, plddt=args.mask_plddt, key='edge_probs', normalize=args.normalize_loss_weights)

		xloss = aa_reconstruction_loss(data['AA'].x, out['aa'], normalize=args.normalize_loss_weights)

		fft2loss = torch.tensor(0.0, device=device)
		if 'fft2pred' in out and out['fft2pred'] is not None:
			fft2loss = F.smooth_l1_loss(torch.cat([data['fourier2dr'].x, data['fourier2di'].x], axis=1), out['fft2pred'])

		angles_loss = torch.tensor(0.0, device=device)
		if out.get('angles') is not None:
			angles_loss = angles_reconstruction_loss(out['angles'], data['bondangles'].x, plddt_mask=data['plddt'].x if args.mask_plddt else None, normalize=args.normalize_loss_weights)

		ss_loss = torch.tensor(0.0, device=device)
		if out.get('ss_pred') is not None:
			if args.mask_plddt:
				mask = (data['plddt'].x >= args.plddt_threshold).squeeze()
				if mask.sum() > 0:
					ss_loss = F.cross_entropy(out['ss_pred'][mask], data['ss'].x[mask])
			else:
				ss_loss = F.cross_entropy(out['ss_pred'], data['ss'].x)

		# lDDT loss
		lddt_loss = torch.tensor(0.0, device=device)
		if (args.lddt_weight > 0 or args.lddt_loss) and out.get('coords') is not None and hasattr(data, 'coords') and hasattr(data['coords'], 'x'):
			from foldtree2.src.losses.losses import batch_lddt_loss
			lddt_loss = batch_lddt_loss(
				pred_q=out.get('quat', None),
				pred_t=out.get('trans', None),
				true_coords=data['coords'].x,
				batch=getattr(data['res'], 'batch', None),
				plddt=data['plddt'].x if args.mask_plddt else None,
				plddt_thresh=args.plddt_threshold if args.mask_plddt else 0.0,
			)

		# Delta loss
		delta_loss_val = torch.tensor(0.0, device=device)
		if (args.delta_weight > 0 or args.delta_loss) and out.get('coords') is not None and hasattr(data, 'coords') and hasattr(data['coords'], 'x'):
			from foldtree2.src.losses.losses import batch_delta_loss
			try:
				if out.get('quat') is not None and out.get('trans') is not None:
					delta_loss_val = batch_delta_loss(
						true_ca=data['coords'].x,
						pred_q=out['quat'],
						pred_t=out['trans'],
						batch=getattr(data['res'], 'batch', None),
						plddt=data['plddt'].x if args.mask_plddt else None,
						plddt_thresh=args.plddt_threshold if args.mask_plddt else 0.0,
					)
				else:
					from foldtree2.src.losses.fape import delta_loss
					delta_loss_val = delta_loss(
						data['coords'].x,
						out['coords'],
						plddt=data['plddt'].x if args.mask_plddt else None,
						plddt_thresh=args.plddt_threshold if args.mask_plddt else 0.0,
					)
			except Exception as exc:
				print(f"Warning: delta_loss calculation failed: {exc}")
				delta_loss_val = torch.tensor(0.0, device=device)

		# FAPE loss
		fape_loss = torch.tensor(0.0, device=device)
		if (args.fape_weight > 0 or args.fape_loss) and out.get('quat') is not None and out.get('trans') is not None and hasattr(data, 'quat') and hasattr(data['quat'], 'x') and hasattr(data, 'trans') and hasattr(data['trans'], 'x'):
			from foldtree2.src.losses.losses import batch_fape_loss
			fape_loss = batch_fape_loss(
				true_q=data['quat'].x,
				true_t=data['trans'].x,
				pred_q=out['quat'],
				pred_t=out['trans'],
				batch=getattr(data['res'], 'batch', None),
			)

	vq_val = float(vqloss.item()) if isinstance(vqloss, torch.Tensor) else float(vqloss)
	avg_loss_x      = float(xloss.item())
	avg_logit_loss  = float(logitloss.item())
	avg_loss_edge   = float(edgeloss.item())
	avg_loss_fft2   = float(fft2loss.item())
	avg_angles_loss = float(angles_loss.item())
	avg_loss_vq     = vq_val
	avg_ss_loss     = float(ss_loss.item())
	avg_lddt_loss   = float(lddt_loss.item())
	avg_fape_loss   = float(fape_loss.item())
	avg_delta_loss  = float(delta_loss_val.item())
	avg_total_loss  = (avg_loss_x + avg_loss_edge + avg_loss_vq + avg_loss_fft2 + avg_angles_loss + avg_logit_loss + avg_ss_loss +
					   args.lddt_weight * avg_lddt_loss + args.fape_weight * avg_fape_loss + args.delta_weight * avg_delta_loss)

	encoder.train()
	decoder.train()

	return {
		'val/loss':        avg_total_loss,
		'val/aa_loss':     avg_loss_x,
		'val/edge_loss':   avg_loss_edge,
		'val/vq_loss':     avg_loss_vq,
		'val/fft2_loss':   avg_loss_fft2,
		'val/angles_loss': avg_angles_loss,
		'val/ss_loss':     avg_ss_loss,
		'val/logit_loss':  avg_logit_loss,
		'val/lddt_loss':   avg_lddt_loss,
		'val/fape_loss':   avg_fape_loss,
		'val/delta_loss':  avg_delta_loss,
	}


def validate(encoder, decoder, val_loader, device, args):
	"""Run validation and compute metrics."""
	encoder.eval()
	decoder.eval()
	
	total_loss_x = 0
	total_loss_edge = 0
	total_vq = 0
	total_angles_loss = 0
	total_loss_fft2 = 0
	total_logit_loss = 0
	total_ss_loss = 0
	total_lddt_loss = 0
	total_fape_loss = 0
	total_delta_loss = 0
	num_batches = 0
	
	with torch.no_grad():
		for data in tqdm.tqdm(val_loader, desc="Validation", leave=False):
			data = data.to(device)
			
			# Forward pass
			z, vqloss = encoder(data)
			data['res'].x = z
			if args.normalize_loss_weights:
				vqloss = vqloss / data['AA'].x.shape[0]  # Normalize by number of residues
			# Forward pass through decoder
			out = decoder(data, None)
			edge_index = data.edge_index_dict.get(('res', 'contactPoints', 'res')) if hasattr(data, 'edge_index_dict') else None

			# Edge reconstruction loss
			logitloss = torch.tensor(0.0, device=device)
			edgeloss = torch.tensor(0.0, device=device)
			if edge_index is not None:
				edgeloss, logitloss = recon_loss_diag(data, edge_index, decoder, plddt=args.mask_plddt, key='edge_probs' , normalize = args.normalize_loss_weights)
			
			# Amino acid reconstruction loss
			xloss = aa_reconstruction_loss(data['AA'].x, out['aa'] , normalize=args.normalize_loss_weights)
			
			# FFT2 loss
			fft2loss = torch.tensor(0.0, device=device)
			if 'fft2pred' in out and out['fft2pred'] is not None:
				fft2loss = F.smooth_l1_loss(torch.cat([data['fourier2dr'].x, data['fourier2di'].x], axis=1), out['fft2pred'])

			# Angles loss
			angles_loss = torch.tensor(0.0, device=device)
			if out.get('angles') is not None:
				angles_loss = angles_reconstruction_loss(out['angles'], data['bondangles'].x, plddt_mask=data['plddt'].x if args.mask_plddt else None, normalize=args.normalize_loss_weights)
				 
			# Secondary structure loss
			ss_loss = torch.tensor(0.0, device=device)
			if out.get('ss_pred') is not None:
				if args.mask_plddt:
					mask = (data['plddt'].x >= args.plddt_threshold).squeeze()
					if mask.sum() > 0:
						ss_loss = F.cross_entropy(out['ss_pred'][mask], data['ss'].x[mask])
				else:
					ss_loss = F.cross_entropy(out['ss_pred'], data['ss'].x)

			fape_loss = torch.tensor(0.0, device=device)
			lddt_loss = torch.tensor(0.0, device=device)
			delta_loss = torch.tensor(0.0, device=device)

			if out.get('quat_pred') is not None and out.get('trans_pred') is not None and 'R_true' in data.node_types and 't_true' in data.node_types:
				batch_idx = data['res'].batch if 'res' in data.node_types else None
				true_R = data['R_true'].x.float()
				true_t = data['t_true'].x.float()
				pred_q = out['quat_pred'].float()
				pred_t = out['trans_pred'].float()
				true_ca = data['coords'].x.float() if 'coords' in data.node_types else None
				true_q = rotation_matrix_to_quaternion(true_R)

				if args.fape_weight > 0:
					fape_loss = batch_fape_loss(
						true_q=true_q,
						true_t=true_t,
						pred_q=pred_q,
						pred_t=pred_t,
						batch=batch_idx,
					)
				if args.lddt_weight > 0:
					lddt_loss = batch_lddt_loss(
						pred_q=pred_q,
						pred_t=pred_t,
						true_coords=true_ca,
						batch=batch_idx,
						plddt=(data['plddt'].x if args.mask_plddt else None),
						plddt_thresh=args.plddt_threshold,
					)
				if args.delta_weight > 0:
					delta_loss = batch_delta_loss(
						true_ca=true_ca,
						pred_q=pred_q,
						pred_t=pred_t,
						batch=batch_idx,
						plddt=(data['plddt'].x if args.mask_plddt else None),
						plddt_thresh=args.plddt_threshold,
					)

			# Accumulate losses
			total_loss_x += float(xloss.item())
			total_logit_loss += float(logitloss.item())
			total_loss_edge += float(edgeloss.item())
			total_loss_fft2 += float(fft2loss.item())
			total_angles_loss += float(angles_loss.item())
			total_vq += float(vqloss.item()) if isinstance(vqloss, torch.Tensor) else float(vqloss)
			total_ss_loss += float(ss_loss.item())
			total_lddt_loss += float(lddt_loss.item())
			total_fape_loss += float(fape_loss.item())
			total_delta_loss += float(delta_loss.item())
			num_batches += 1
	
	denominator = num_batches if args.normalize_loss_weights == False else 1 
	# Calculate average losses
	avg_loss_x = total_loss_x / denominator
	avg_loss_edge = total_loss_edge / denominator
	avg_loss_vq = total_vq / denominator
	avg_loss_fft2 = total_loss_fft2 / denominator
	avg_angles_loss = total_angles_loss / denominator
	avg_logit_loss = total_logit_loss / denominator
	avg_ss_loss = total_ss_loss / denominator
	avg_lddt_loss = total_lddt_loss / denominator
	avg_fape_loss = total_fape_loss / denominator
	avg_delta_loss = total_delta_loss / denominator
	avg_total_loss = (avg_loss_x + avg_loss_edge + avg_loss_vq + 
					  avg_loss_fft2 + avg_angles_loss + avg_logit_loss + avg_ss_loss +
					  args.lddt_weight * avg_lddt_loss + args.fape_weight * avg_fape_loss + args.delta_weight * avg_delta_loss)
	
	encoder.train()
	decoder.train()
	
	return {
		'val/loss': avg_total_loss,
		'val/aa_loss': avg_loss_x,
		'val/edge_loss': avg_loss_edge,
		'val/vq_loss': avg_loss_vq,
		'val/fft2_loss': avg_loss_fft2,
		'val/angles_loss': avg_angles_loss,
		'val/ss_loss': avg_ss_loss,
		'val/logit_loss': avg_logit_loss,
		'val/lddt_loss': avg_lddt_loss,
		'val/fape_loss': avg_fape_loss,
		'val/delta_loss': avg_delta_loss
	}

# Training loop
encoder.train()
decoder.train()
clip_grad = args.clip_grad  # Enable gradient clipping
best_loss = float('inf')
global_step = 0  # Track global training steps for warmup and scheduling


amp_dtype = torch.float16
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
	amp_dtype = torch.bfloat16

use_scaler = bool(args.mixed_precision and amp_dtype == torch.float16)

# Initialize GradScaler for mixed precision training
if args.mixed_precision:
	scaler = GradScaler(enabled=use_scaler, init_scale=2**8, growth_interval=2000)
	print(f"Mixed Precision Training Enabled (autocast dtype={amp_dtype}, scaler={'on' if use_scaler else 'off'})")
else:
	scaler = None
	print("Mixed Precision Training Disabled")

print(f"\nTraining Configuration:")
print(f"  Total epochs: {args.epochs}")
print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
print(f"  Steps per epoch: {len(train_loader)}")
print(f"  Effective steps per epoch: {len(train_loader) // args.gradient_accumulation_steps}")
print(f"  Mask pLDDT: {args.mask_plddt}")
if args.mask_plddt:
	print(f"  pLDDT threshold: {args.plddt_threshold}")
print(f"  Validation split: {args.val_split}")
print(f"  Validation seed: {args.val_seed}")
print(f"  Using device: {device}")
print(f"  Early stopping: {'Enabled' if args.early_stopping else 'Disabled'}")
if args.early_stopping:
	print(f"    Monitor: {args.early_stopping_metric}")
	print(f"    Mode: {args.early_stopping_mode}")
	print(f"    Patience: {args.early_stopping_patience}")
	print(f"    Min delta: {args.early_stopping_min_delta}")
	print(f"    Warmup epochs: {args.early_stopping_warmup_epochs}")
print()

# Early stopping state
early_stop_best = None
early_stop_wait = 0

for epoch in range(args.epochs):
	# Keep per-epoch weights accessible for scheduling and TensorBoard logging
	xweight_epoch = xweight
	ss_weight_epoch = ss_weight

	total_loss_x = 0
	total_loss_edge = 0
	total_vq = 0
	total_angles_loss = 0
	total_loss_fft2 = 0
	total_logit_loss = 0
	total_ss_loss = 0
	total_lddt_loss = 0
	total_fape_loss = 0
	total_delta_loss = 0
	
	for batch_idx, data in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
		# Periodically clear CUDA cache to avoid OOM errors
		if torch.cuda.is_available() and batch_idx % 10 == 0 and batch_idx > 0:
			import gc
			torch.cuda.empty_cache()
			gc.collect()
		data = data.to(device)

		# Skip unstable batches early if any input node features contain NaN/Inf.
		bad_batch = False
		for node_type in getattr(data, 'node_types', []):
			node_x = getattr(data[node_type], 'x', None)
			if node_x is not None and (torch.isnan(node_x).any() or torch.isinf(node_x).any()):
				bad_batch = True
				break
		if bad_batch:
			print(f"Skipping batch {batch_idx}: input contains NaN/Inf")
			optimizer.zero_grad(set_to_none=True)
			continue
		
		# Forward pass with autocast for mixed precision
		if args.mixed_precision:
			with autocast(enabled=True, dtype=amp_dtype):
				z, vqloss = encoder(data)
				data['res'].x = z
				
				# Forward pass through decoder
				out = decoder(data, None)
				edge_index = data.edge_index_dict.get(('res', 'contactPoints', 'res')) if hasattr(data, 'edge_index_dict') else None

				# Edge reconstruction loss
				logitloss = torch.tensor(0.0, device=device)
				edgeloss = torch.tensor(0.0, device=device)
				if edge_index is not None:
					edgeloss, logitloss = recon_loss_diag(data, edge_index, decoder, plddt=args.mask_plddt, key='edge_probs' , normalize=args.normalize_loss_weights)
				
				# Amino acid reconstruction loss
				xloss = aa_reconstruction_loss(data['AA'].x, out['aa'], normalize=args.normalize_loss_weights)
				
				# FFT2 loss
				fft2loss = torch.tensor(0.0, device=device)
				if 'fft2pred' in out and out['fft2pred'] is not None:
					fft2loss = F.smooth_l1_loss(torch.cat([data['fourier2dr'].x, data['fourier2di'].x], axis=1), out['fft2pred'])

				# Angles loss
				angles_loss = torch.tensor(0.0, device=device)
				if out.get('angles') is not None:
					angles_loss = angles_reconstruction_loss(out['angles'], data['bondangles'].x, plddt_mask=data['plddt'].x if args.mask_plddt else None, normalize=args.normalize_loss_weights)
				
				# Secondary structure loss
				ss_loss = torch.tensor(0.0, device=device)
				if out.get('ss_pred') is not None:
					if args.mask_plddt:
						mask = (data['plddt'].x >= args.plddt_threshold).squeeze()
						if mask.sum() > 0:
							ss_loss = F.cross_entropy(out['ss_pred'][mask], data['ss'].x[mask])
					else:
						ss_loss = F.cross_entropy(out['ss_pred'], data['ss'].x)

				# FAPE and lDDT losses computed in float32 outside autocast to prevent
				fape_loss = torch.tensor(0.0, device=device)
				lddt_loss = torch.tensor(0.0, device=device)
				delta_loss = torch.tensor(0.0, device=device)

				if out.get('quat_pred') is not None and out.get('trans_pred') is not None and 'R_true' in data.node_types and 't_true' in data.node_types:
					batch_idx = data['res'].batch if 'res' in data.node_types else None

					true_R = data['R_true'].x.float()
					true_t = data['t_true'].x.float()
					pred_q = out['quat_pred'].float()
					pred_t = out['trans_pred'].float()
					true_ca = data['coords'].x.float() if 'coords' in data.node_types else None
					true_q = rotation_matrix_to_quaternion(true_R)

					if args.fape_weight > 0:
						fape_loss = batch_fape_loss(
							true_q=true_q,
							true_t=true_t,
							pred_q=pred_q,
							pred_t=pred_t,
							batch=batch_idx,
						)
					if args.lddt_weight > 0:
						lddt_loss = batch_lddt_loss(
							pred_q=pred_q,
							pred_t=pred_t,
							true_coords=true_ca,
							batch=batch_idx,
							plddt=(data['plddt'].x if mask_plddt else None),
							plddt_thresh=plddt_threshold,
						)
					if args.delta_weight > 0:
						delta_loss = batch_delta_loss(
							true_ca=true_ca,
							pred_q=pred_q,
							pred_t=pred_t,
							batch=batch_idx,
							plddt=(data['plddt'].x if mask_plddt else None),
							plddt_thresh=plddt_threshold,
						)

				if args.use_weight_scheduler:
					x_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_x, warmup_steps=loss_warmup_steps)
					logit_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_logit, warmup_steps=loss_warmup_steps)
					edge_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_edge, warmup_steps=loss_warmup_steps, num_cycles=args.loss_cycles_edge)
					vq_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_vq, warmup_steps=loss_warmup_steps, num_cycles=args.loss_cycles_vq)
					fft2_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_fft2, warmup_steps=loss_warmup_steps, num_cycles=args.loss_cycles_fft2)
					angles_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_angles, warmup_steps=loss_warmup_steps)
					ss_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_ss, warmup_steps=loss_warmup_steps, power=args.loss_power_ss)
				else:
					x_scale = logit_scale = edge_scale = vq_scale = fft2_scale = angles_scale = ss_scale = 1.0

				current_xweight = xweight_epoch * x_scale
				current_logitweight = logitweight * logit_scale
				current_edgeweight = edgeweight * edge_scale
				current_vqweight = vqweight * vq_scale
				current_fft2weight = fft2weight * fft2_scale
				current_angles_weight = angles_weight * angles_scale
				current_ss_weight = ss_weight_epoch * ss_scale
					
				
				if args.use_uncertainty_weighting:
					loss = uncertainy_weighting.forward(
						torch.stack([
							current_xweight * xloss,
							current_edgeweight * edgeloss,
							current_vqweight * vqloss,
							current_fft2weight * fft2loss,
							current_angles_weight * angles_loss,
							current_ss_weight * ss_loss,
							current_logitweight * logitloss,
							args.lddt_weight * lddt_loss,
							args.fape_weight * fape_loss,
							args.delta_weight * delta_loss,
						])
					)
				else:
					loss = (
						current_xweight * xloss + current_edgeweight * edgeloss + current_vqweight * vqloss +
						current_fft2weight * fft2loss + current_angles_weight * angles_loss +
						current_ss_weight * ss_loss + current_logitweight * logitloss +
						args.lddt_weight * lddt_loss + args.fape_weight * fape_loss + args.delta_weight * delta_loss
					)
				
				# Scale loss by gradient accumulation steps
				loss = loss / args.gradient_accumulation_steps
		else:
			# Non-mixed precision path
			z, vqloss = encoder(data)
			data['res'].x = z
			
			if args.normalize_loss_weights:
				vqloss = vqloss / data['AA'].x.shape[0]  # Normalize by number of residues
			out = decoder(data, None)
			edge_index = data.edge_index_dict.get(('res', 'contactPoints', 'res')) if hasattr(data, 'edge_index_dict') else None

			logitloss = torch.tensor(0.0, device=device)
			edgeloss = torch.tensor(0.0, device=device)
			if edge_index is not None:
				edgeloss, logitloss = recon_loss_diag(data, edge_index, decoder, plddt=args.mask_plddt, key='edge_probs', normalize=args.normalize_loss_weights)
			
			xloss = aa_reconstruction_loss(data['AA'].x, out['aa'], normalize=args.normalize_loss_weights)
			
			fft2loss = torch.tensor(0.0, device=device)
			if 'fft2pred' in out and out['fft2pred'] is not None:
				fft2loss = F.smooth_l1_loss(torch.cat([data['fourier2dr'].x, data['fourier2di'].x], axis=1), out['fft2pred'])

			angles_loss = torch.tensor(0.0, device=device)
			if out.get('angles') is not None:
				angles_loss = angles_reconstruction_loss(out['angles'], data['bondangles'].x, plddt_mask=data['plddt'].x if args.mask_plddt else None, normalize=args.normalize_loss_weights)

			# Secondary structure loss
			ss_loss = torch.tensor(0.0, device=device)
			if out.get('ss_pred') is not None:
				if args.mask_plddt:
					mask = (data['plddt'].x >= args.plddt_threshold).squeeze()
					if mask.sum() > 0:
						ss_loss = F.cross_entropy(out['ss_pred'][mask], data['ss'].x[mask])
				else:
					ss_loss = F.cross_entropy(out['ss_pred'], data['ss'].x)

			if args.use_weight_scheduler:
				x_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_x, warmup_steps=loss_warmup_steps)
				logit_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_logit, warmup_steps=loss_warmup_steps)
				edge_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_edge, warmup_steps=loss_warmup_steps, num_cycles=args.loss_cycles_edge)
				vq_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_vq, warmup_steps=loss_warmup_steps, num_cycles=args.loss_cycles_vq)
				fft2_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_fft2, warmup_steps=loss_warmup_steps, num_cycles=args.loss_cycles_fft2)
				angles_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_angles, warmup_steps=loss_warmup_steps)
				ss_scale = loss_weight_scheduler(global_step, total_steps, schedule_type=args.loss_schedule_ss, warmup_steps=loss_warmup_steps, power=args.loss_power_ss)
			else:
				x_scale = logit_scale = edge_scale = vq_scale = fft2_scale = angles_scale = ss_scale = 1.0

			current_xweight = xweight_epoch * x_scale
			current_logitweight = logitweight * logit_scale
			current_edgeweight = edgeweight * edge_scale
			current_vqweight = vqweight * vq_scale
			current_fft2weight = fft2weight * fft2_scale
			current_angles_weight = angles_weight * angles_scale
			current_ss_weight = ss_weight_epoch * ss_scale
				
			if args.use_uncertainty_weighting:
				loss = uncertainy_weighting.forward(
					torch.stack([
						current_xweight * xloss,
						current_edgeweight * edgeloss,
						current_vqweight * vqloss,
						current_fft2weight * fft2loss,
						current_angles_weight * angles_loss,
						current_ss_weight * ss_loss,
						current_logitweight * logitloss,
					])
				)
			else:
				loss = (
					current_xweight * xloss + current_edgeweight * edgeloss + current_vqweight * vqloss +
					current_fft2weight * fft2loss + current_angles_weight * angles_loss +
					current_ss_weight * ss_loss + current_logitweight * logitloss
				)
			
			loss = loss / args.gradient_accumulation_steps

		if not torch.isfinite(loss):
			print(f"Skipping batch {batch_idx}: loss is non-finite")
			optimizer.zero_grad(set_to_none=True)
			continue
		
		# Backward pass with gradient scaling
		if args.mixed_precision and use_scaler:
			scaler.scale(loss).backward()
		else:
			loss.backward()
		
		# Only update weights every gradient_accumulation_steps
		if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
			if clip_grad:
				if args.mixed_precision and use_scaler:
					scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
				torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
			
			# Step optimizer with scaler
			if args.mixed_precision and use_scaler:
				scaler.step(optimizer)
				scaler.update()
			else:
				optimizer.step()
			optimizer.zero_grad(set_to_none=True)  # Better than zero_grad()

			# Step scheduler if it's a step-based scheduler
			if scheduler is not None and scheduler_step_mode == 'step':
				scheduler.step()
			
			torch.cuda.empty_cache()  # Clear cache after each update to reduce fragmentation
			gc.collect()  # Run garbage collection to free memory
			global_step += 1

		

		# Accumulate losses (unscaled for reporting)
		total_loss_x += float(xloss.item())
		total_logit_loss += float(logitloss.item())
		total_loss_edge += float(edgeloss.item())
		total_loss_fft2 += float(fft2loss.item())
		total_angles_loss += float(angles_loss.item())
		total_vq += float(vqloss.item()) if isinstance(vqloss, torch.Tensor) else float(vqloss)
		total_ss_loss += float(ss_loss.item())
		total_lddt_loss += float(lddt_loss.item())
		total_fape_loss += float(fape_loss.item())
		total_delta_loss += float(delta_loss.item())
	
	# Clean up any remaining gradients at epoch end
	if len(train_loader) % args.gradient_accumulation_steps != 0:
		if clip_grad:
			if args.mixed_precision and use_scaler:
				scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
			torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
		if args.mixed_precision and use_scaler:
			scaler.step(optimizer)
			scaler.update()
		else:
			optimizer.step()
		optimizer.zero_grad(set_to_none=True)  # Better than zero_grad()

	denominator = len(train_loader)  # report total loss if not normalizing, otherwise report average loss

	# Calculate average losses
	avg_loss_x = total_loss_x / denominator
	avg_loss_edge = total_loss_edge / denominator
	avg_loss_vq = total_vq / denominator
	avg_loss_fft2 = total_loss_fft2 / denominator
	avg_angles_loss = total_angles_loss / denominator
	avg_logit_loss = total_logit_loss / denominator
	avg_ss_loss = total_ss_loss / denominator
	avg_lddt_loss = total_lddt_loss / denominator
	avg_fape_loss = total_fape_loss / denominator
	avg_delta_loss = total_delta_loss / denominator

	avg_total_loss = (avg_loss_x + avg_loss_edge + avg_loss_vq + 
					  avg_loss_fft2 + avg_angles_loss + avg_logit_loss + avg_ss_loss +
					  args.lddt_weight * avg_lddt_loss + args.fape_weight * avg_fape_loss + args.delta_weight * avg_delta_loss)
	
	# Clear CUDA cache
	torch.cuda.empty_cache()
	gc.collect()

	# Run quick validation on 10 random proteins
	val_metrics = quick_validate(encoder, decoder, val_dataset, device, args, n_samples=5)
	
	# Clear CUDA cache
	torch.cuda.empty_cache()
	gc.collect()


	# Update learning rate scheduler (for epoch-based schedulers)
	if scheduler is not None and scheduler_step_mode == 'epoch':
		if args.lr_schedule == 'plateau':
			scheduler.step(val_metrics['val/loss'])  # Use validation loss for plateau scheduler
		else:
			scheduler.step()
	
	# Print metrics
	print(f"Epoch {epoch+1}:")
	print(f"  Train - AA Loss: {avg_loss_x:.4f}, Edge Loss: {avg_loss_edge:.4f}, "
		  f"VQ Loss: {avg_loss_vq:.4f}, FFT2 Loss: {avg_loss_fft2:.4f}")
	print(f"  Train - Angles Loss: {avg_angles_loss:.4f}, SS Loss: {avg_ss_loss:.4f}, "
		  f"Logit Loss: {avg_logit_loss:.4f}, lDDT Loss: {avg_lddt_loss:.4f}, FAPE Loss: {avg_fape_loss:.4f}, Delta Loss: {avg_delta_loss:.4f}")
	print(f"  Val   - AA Loss: {val_metrics['val/aa_loss']:.4f}, Edge Loss: {val_metrics['val/edge_loss']:.4f}, "
		  f"VQ Loss: {val_metrics['val/vq_loss']:.4f}, FFT2 Loss: {val_metrics['val/fft2_loss']:.4f}")
	print(f"  Val   - Angles Loss: {val_metrics['val/angles_loss']:.4f}, SS Loss: {val_metrics['val/ss_loss']:.4f}, "
		  f"Logit Loss: {val_metrics['val/logit_loss']:.4f}, lDDT Loss: {val_metrics.get('val/lddt_loss', 0.0):.4f}, "
		  f"FAPE Loss: {val_metrics.get('val/fape_loss', 0.0):.4f}, Delta Loss: {val_metrics.get('val/delta_loss', 0.0):.4f}")
	current_lr = optimizer.param_groups[0]['lr']
	print(f"  Train Total Loss: {avg_total_loss:.4f}, Val Total Loss: {val_metrics['val/loss']:.4f}, LR: {current_lr:.6f}")
	
	# Print commitment cost if using scheduling
	if args.use_commitment_scheduling and hasattr(encoder, 'vector_quantizer'):
		current_commitment = encoder.vector_quantizer.get_commitment_cost()
		print(f"Commitment Cost: {current_commitment:.4f}")
	
	# Gradient analysis
	print("Gradient norms (encoder):", analyze_gradient_norms(encoder))
	print("Gradient norms (decoder):", analyze_gradient_norms(decoder))
	
	# Log to tensorboard
	writer.add_scalar('Train/AA_Loss', avg_loss_x, epoch)
	writer.add_scalar('Train/Edge_Loss', avg_loss_edge, epoch)
	writer.add_scalar('Train/VQ_Loss', avg_loss_vq, epoch)
	writer.add_scalar('Train/FFT2_Loss', avg_loss_fft2, epoch)
	writer.add_scalar('Train/Angles_Loss', avg_angles_loss, epoch)
	writer.add_scalar('Train/SS_Loss', avg_ss_loss, epoch)
	writer.add_scalar('Train/Logit_Loss', avg_logit_loss, epoch)
	writer.add_scalar('Train/Total_Loss', avg_total_loss, epoch)
	
	# Log validation metrics
	for metric_name, metric_value in val_metrics.items():
		writer.add_scalar(metric_name.replace('val/', 'Val/'), metric_value, epoch)
	
	writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
	
	# Log commitment cost if using scheduling
	if args.use_commitment_scheduling and hasattr(encoder, 'vector_quantizer'):
		current_commitment = float(encoder.vector_quantizer.get_commitment_cost())
		writer.add_scalar('Training/Commitment_Cost', current_commitment, epoch)
	
	# Log loss weights
	writer.add_scalar('Weights/Edge', edgeweight, epoch)
	writer.add_scalar('Weights/X', xweight, epoch)
	writer.add_scalar('Weights/FFT2', fft2weight, epoch)
	writer.add_scalar('Weights/VQ', vqweight, epoch)
	writer.add_scalar('Weights/SS', ss_weight_epoch, epoch)
	writer.add_scalar('Weights/Logit', logitweight, epoch)

	# Early stopping check
	if args.early_stopping and epoch >= args.early_stopping_warmup_epochs:
		metric_key = args.early_stopping_metric
		if metric_key not in val_metrics:
			print(f"Warning: Early stopping metric '{metric_key}' not found in validation metrics. Available keys: {list(val_metrics.keys())}")
		else:
			current_metric = val_metrics[metric_key]
			if early_stop_best is None:
				early_stop_best = current_metric
				early_stop_wait = 0
			else:
				if args.early_stopping_mode == 'min':
					improved = current_metric < (early_stop_best - args.early_stopping_min_delta)
				else:
					improved = current_metric > (early_stop_best + args.early_stopping_min_delta)

				if improved:
					early_stop_best = current_metric
					early_stop_wait = 0
				else:
					early_stop_wait += 1
					print(f"Early stopping patience: {early_stop_wait}/{args.early_stopping_patience}")
					if early_stop_wait >= args.early_stopping_patience:
						print("Early stopping triggered. Stopping training.")
						break
	
	# Log gradient norms
	#encoder_grad_norms = analyze_gradient_norms(encoder, top_k=1)
	#decoder_grad_norms = analyze_gradient_norms(decoder, top_k=1)
	#if encoder_grad_norms['highest']:
	#    writer.add_scalar('Gradients/Encoder_Max', float(encoder_grad_norms['highest'][0][1]), epoch)
	#if encoder_grad_norms['lowest']:
	#    writer.add_scalar('Gradients/Encoder_Min', float(encoder_grad_norms['lowest'][0][1]), epoch)
	#if decoder_grad_norms['highest']:
	#    writer.add_scalar('Gradients/Decoder_Max', float(decoder_grad_norms['highest'][0][1]), epoch)
	#if decoder_grad_norms['lowest']:
	#    writer.add_scalar('Gradients/Decoder_Min', float(decoder_grad_norms['lowest'][0][1]), epoch)
	
	# Update metrics for hparams logging
	metrics_dict['best_val_loss'] = best_loss
	metrics_dict['final_train_aa_loss'] = avg_loss_x
	metrics_dict['final_train_edge_loss'] = avg_loss_edge
	metrics_dict['final_val_loss'] = val_metrics['val/loss']
	metrics_dict['final_val_aa_loss'] = val_metrics['val/aa_loss']
	metrics_dict['final_val_edge_loss'] = val_metrics['val/edge_loss']
	
	# Save best model based on validation loss
	if val_metrics['val/loss'] < best_loss:
		best_loss = val_metrics['val/loss']
		print(f"Saving best model with validation loss: {best_loss:.4f}")
		#save as pth
		encoder_path = os.path.join(modeldir, f"{modelname}_best_encoder.pt")
		decoder_path = os.path.join(modeldir, f"{modelname}_best_decoder.pt")
		torch.save(encoder, encoder_path)
		torch.save(decoder, decoder_path)

		#save state dict
		encoder_path = os.path.join(modeldir, f"{modelname}_best_encoder_state.pth")
		decoder_path = os.path.join(modeldir, f"{modelname}_best_decoder_state.pth")
		torch.save(encoder.state_dict(), encoder_path)
		torch.save(decoder.state_dict(), decoder_path)

	# Save checkpoint and visualize every 10 epochs
	if (epoch + 1) % 10 == 0:
		print(f"\n{'─'*80}")
		print(f"Saving checkpoint at epoch {epoch+1}...")
		print(f"{'─'*80}")
		
		# Save models
		torch.save(encoder, os.path.join(modeldir, f"{modelname}_encoder_epoch{epoch+1}.pt"))
		torch.save(decoder, os.path.join(modeldir, f"{modelname}_decoder_epoch{epoch+1}.pt"))
		
		# Optional: Generate batched visualization (can be toggled)
		USE_BATCHED_VIZ = True  # Set to False to disable visualization
		
		if USE_BATCHED_VIZ:
			try:
				# Collect a few samples for batched visualization
				viz_samples = []
				for i in range(min(4, len(struct_dat))):  # Visualize up to 4 samples
					sample_idx = random.randint(0, len(struct_dat) - 1)
					viz_samples.append(struct_dat[sample_idx])
				
				print(f"Generating batched reconstruction visualization for {len(viz_samples)} samples...")
				os.makedirs('figures', exist_ok=True)
				
				from foldtree2.src.visualization import visualize_batch_reconstructions
				figs_and_metrics = visualize_batch_reconstructions(
					encoder, decoder, viz_samples, device, args.num_embeddings,
					converter, epoch=epoch+1, save_dir=f'figures/epoch_{epoch+1}'
				)
				
				for i, (fig, metrics) in enumerate(figs_and_metrics):
					print(f"\nSample {i+1} metrics:")
					for key, value in metrics.items():
						print(f"  {key}: {value}")
					
					# Log to tensorboard if available
					if writer is not None:
						writer.add_figure(f'Reconstruction/Sample_{i+1}', fig, epoch+1)
					
					# Close figure to free memory
					plt.close(fig)
				
				print(f"Visualizations saved to figures/epoch_{epoch+1}/")
			except Exception as e:
				print(f"Warning: Visualization failed with error: {e}")
				import traceback
				traceback.print_exc()

	

# Save final model
#with open(os.path.join(modeldir, modelname + '.pkl'), 'wb') as f:
#    pickle.dump((encoder, decoder), f)
torch.save(encoder, os.path.join(modeldir, f"{modelname}_encoder_final.pt"))
torch.save(decoder, os.path.join(modeldir, f"{modelname}_decoder_final.pt"))

# Log final hyperparameters and metrics
writer.add_hparams(hparams_dict, metrics_dict)

# Close TensorBoard writer
writer.close()

print(f"Training complete! Final model saved to {os.path.join(modeldir, modelname + '_encoder_final.pt')} and {os.path.join(modeldir, modelname + '_decoder_final.pt')}")
print(f"TensorBoard logs saved to: {tensorboard_log_dir}")
print(f"View with: tensorboard --logdir={args.tensorboard_dir}")
