# learn_lightning.py - PyTorch Lightning training with multi-GPU support
import torch
from torch_geometric.data import DataLoader
import numpy as np
from foldtree2.src import pdbgraph
from foldtree2.src import encoder as ecdr
from foldtree2.src.losses.losses import recon_loss_diag, aa_reconstruction_loss, angles_reconstruction_loss
from foldtree2.src.mono_decoders import MultiMonoDecoder
import os
import tqdm
import random
import torch.nn.functional as F
import pickle
import argparse
import sys
import time
import warnings
import yaml
import json
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
warnings.filterwarnings("ignore", category=UserWarning)

# Try to import Muon optimizer
try:
    from muon import MuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    print("Warning: Muon optimizer not available. Install with: pip install git+https://github.com/KellerJordan/Muon")
    MUON_AVAILABLE = False

# Try to import transformers schedulers
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
|                Lightning Trainer (multi-GPU)               |
|          DDP • FSDP • DeepSpeed  |  AMP  |  Checkpoints    |
|             Scale training across GPUs for FoldTree2       |
|                      🧬   🧠   🌳                          |
+-----------------------------------------------------------+


    '''
    print(ascii_art)
    print("FoldTree2 Lightning Training Script")
    print("-" * 50)
    print("Train FoldTree2 models with PyTorch Lightning multi-GPU support.\n")
    print("Features:")
    print("  • Multi-GPU training (DDP, FSDP, DeepSpeed)")
    print("  • Mixed precision (16-bit) training")
    print("  • Automatic checkpointing & logging")
    print("  • Learning rate scheduling")
    print("  • Gradient accumulation")
    print("  • Distributed training strategies\n")
    print("Project: https://github.com/DessimozLab/foldtree2")
    print("Contact: dmoi@unil.ch\n")
    print("Run with --help for usage instructions.")

# Argument parsing
parser = argparse.ArgumentParser(description='Train model with MultiMonoDecoder for sequence and geometry prediction (Lightning version)')
parser.add_argument('--about', action='store_true',
                    help='Show information about this tool and exit')
parser.add_argument('--config', '-c', type=str, default=None,
                    help='Path to config file (YAML or JSON). Command-line args override config file values.')
parser.add_argument('--dataset', '-d', type=str, default='structs_traininffttest.h5',
                    help='Path to the dataset file (default: structs_traininffttest.h5)')
parser.add_argument('--hidden-size', '-hs', type=int, default=256,
                    help='Hidden layer size (default: 256)')
parser.add_argument('--epochs', '-e', type=int, default=100,
                    help='Number of epochs for training (default: 100)')
parser.add_argument('--device', type=str, default=None,
                    help='Device to run on (e.g., cuda:0, cuda:1, cpu) (default: auto-select)')
parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4)')
parser.add_argument('--batch-size', '-bs', type=int, default=20,
                    help='Batch size (default: 20)')
parser.add_argument('--output-dir', '-o', type=str, default='./models/',
                    help='Directory to save models/results (default: ./models/)')
parser.add_argument('--model-name', type=str, default='lightning_monodecoder',
                    help='Model name for saving (default: lightning_monodecoder)')
parser.add_argument('--num-embeddings', type=int, default=40,
                    help='Number of embeddings for the encoder (default: 40)')
parser.add_argument('--embedding-dim', type=int, default=128,
                    help='Embedding dimension for the encoder (default: 128)')
parser.add_argument('--se3-transformer', action='store_true',
                    help='Use SE3Transformer instead of GNN')
parser.add_argument('--overwrite', action='store_true', help='Overwrite saved model if exists, otherwise continue training')
parser.add_argument('--output-fft', action='store_true', help='Train the model with FFT output')
parser.add_argument('--output-rt', action='store_true', help='Train the model with rotation and translation output')
parser.add_argument('--output-foldx', action='store_true', help='Train the model with Foldx energy prediction output')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument('--clip-grad', action='store_true', help='Enable gradient clipping during training')
parser.add_argument('--burn-in', type=int, default=0, help='Burn-in period for training (default: 0, no burn-in)')
parser.add_argument('--hetero-gae', action='store_true', help='Use HeteroGAE_Decoder instead of MultiMonoDecoder')
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
parser.add_argument('--gradient-accumulation-steps', '--grad-accum', type=int, default=1,
                    help='Number of gradient accumulation steps (default: 1, no accumulation)')
parser.add_argument('--num-cycles', type=int, default=3,
                    help='Number of cycles for cosine_restarts scheduler (default: 3)')

# Muon optimizer arguments
parser.add_argument('--use-muon', action='store_true',
                    help='Use Muon optimizer for modular encoder/decoder (requires Muon package)')
parser.add_argument('--use-muon-encoder', action='store_true',
                    help='Use Muon-compatible encoder (mk1_MuonEncoder)')
parser.add_argument('--use-muon-decoders', action='store_true',
                    help='Use Muon-compatible decoders in MultiMonoDecoder')
parser.add_argument('--muon-lr', type=float, default=0.02,
                    help='Learning rate for Muon optimizer (default: 0.02)')
parser.add_argument('--adamw-lr', type=float, default=3e-4,
                    help='Learning rate for AdamW when using Muon (default: 3e-4)')

# Mixed precision and pLDDT masking
parser.add_argument('--mixed-precision', action='store_true', default=True,
                    help='Use mixed precision training (default: True)')
parser.add_argument('--mask-plddt', action='store_true',
                    help='Mask low pLDDT residues in loss calculations')
parser.add_argument('--plddt-threshold', type=float, default=0.3,
                    help='pLDDT threshold for masking (default: 0.3)')

# Multi-GPU settings
parser.add_argument('--gpus', type=int, default=-1,
                    help='Number of GPUs to use (default: -1, use all available GPUs; set to specific number to limit)')
parser.add_argument('--strategy', type=str, default='auto',
                    choices=['auto', 'ddp', 'ddp_spawn', 'dp', 'fsdp', 'deepspeed', 'ddp_sharded'],
                    help='Distributed training strategy (default: auto). Use fsdp or deepspeed for very large models that exceed GPU memory')
parser.add_argument('--fsdp-cpu-offload', action='store_true',
                    help='Offload FSDP parameters to CPU to save GPU memory (slower but enables larger models)')
parser.add_argument('--deepspeed-stage', type=int, default=2, choices=[1, 2, 3],
                    help='DeepSpeed ZeRO optimization stage (1=optimizer, 2=+gradients, 3=+parameters) (default: 2)')

# Commitment cost scheduling
parser.add_argument('--commitment-cost', type=float, default=0.9,
                    help='Final commitment cost for VQ-VAE (default: 0.9)')
parser.add_argument('--use-commitment-scheduling', action='store_true',
                    help='Enable commitment cost scheduling (warmup from low to final value)')
parser.add_argument('--commitment-schedule', type=str, default='cosine',
                    choices=['cosine', 'linear', 'none'],
                    help='Commitment cost schedule type (default: cosine)')
parser.add_argument('--commitment-warmup-steps', type=int, default=5000,
                    help='Number of steps to warmup commitment cost (default: 5000)')
parser.add_argument('--commitment-start', type=float, default=0.1,
                    help='Starting commitment cost when using scheduling (default: 0.1)')

#data directory
parser.add_argument('--data-dir', type=str, default='../../datasets/foldtree2/',
                    help='Directory containing the dataset (default: ../../datasets/foldtree2/)')

#prop csv
parser.add_argument('--aapropcsv', type=str, default='config/aaindex1.csv',
                    help='Amino acid property CSV file (default: config/aaindex1.csv)')

# Loss weight arguments
parser.add_argument('--edge-weight', type=float, default=0.25,
                    help='Weight for edge reconstruction loss (default: 0.25)')
parser.add_argument('--logit-weight', type=float, default=0.25,
                    help='Weight for logit loss (default: 0.25)')
parser.add_argument('--x-weight', type=float, default=5.0,
                    help='Weight for coordinate reconstruction loss (default: 5.0)')
parser.add_argument('--fft2-weight', type=float, default=0.01,
                    help='Weight for FFT2 loss (default: 0.01)')
parser.add_argument('--vq-weight', type=float, default=0.1,
                    help='Weight for VQ-VAE loss (default: 0.1)')
parser.add_argument('--angles-weight', type=float, default=0.05,
                    help='Weight for angles reconstruction loss (default: 0.05)')
parser.add_argument('--ss-weight', type=float, default=0.25,
                    help='Weight for secondary structure loss (default: 0.25)')

# Tensor Core precision
parser.add_argument('--tensor-core-precision', type=str, default='high',
                    choices=['highest', 'high', 'medium'],
                    help='Float32 matrix multiplication precision for Tensor Cores (default: high)')

# Handle --about flag before argument parsing
if '--about' in sys.argv:
    print_about()
    sys.exit(0)

if len(sys.argv) == 1:
    print('No arguments provided. Use -h for help.')
    print('Example command (data parallelism): python learn_lightning.py -d structs_training_godnodemk5.h5 -o ./models/ -lr 0.0001 -e 20 -bs 5 --gpus 4')
    print('Example command (model parallelism for large models): python learn_lightning.py -d structs.h5 -o ./models/ --strategy fsdp --gpus 4')
    print('Example with DeepSpeed: python learn_lightning.py -d structs.h5 -o ./models/ --strategy deepspeed --deepspeed-stage 3 --gpus 8')
    print('Example with config: python learn_lightning.py --config my_config.yaml')
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
    
    # Set defaults from config file, but allow CLI args to override
    for key, value in config.items():
        if hasattr(args, key):
            cli_value = getattr(args, key)
            default_value = parser.get_default(key)
            if cli_value == default_value:
                setattr(args, key, value)
                print(f"  {key}: {value} (from config)")
            else:
                print(f"  {key}: {cli_value} (from CLI, overriding config)")
else:
    print("No config file provided, using command-line arguments and defaults")

# Set seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set tensor core precision for better performance on modern GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision(args.tensor_core_precision)
    print(f"Tensor Core precision set to: {args.tensor_core_precision}")

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
print(f"  Hetero GAE Decoder: {'Enabled' if args.hetero_gae else 'Disabled'}")
print(f"  Gradient Clipping: {'Enabled' if args.clip_grad else 'Disabled'}")
print(f"  Burn-in Period: {args.burn_in} epochs")
print(f"  Random Seed: {args.seed}")
print(f"  Exponential Moving Average: {'Enabled' if args.EMA else 'Disabled'}")
print(f"  TensorBoard Directory: {args.tensorboard_dir}")
print(f"  Run Name: {args.run_name if args.run_name else 'auto-generated'}")
print(f"  LR Schedule: {args.lr_schedule}")
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
print(f"Loss Weights:")
print(f"  Edge Weight: {args.edge_weight}")
print(f"  Logit Weight: {args.logit_weight}")
print(f"  X Weight: {args.x_weight}")
print(f"  FFT2 Weight: {args.fft2_weight}")
print(f"  VQ Weight: {args.vq_weight}")
print(f"  Angles Weight: {args.angles_weight}")
print(f"  SS Weight: {args.ss_weight}")

# Save configuration if requested
if args.save_config:
    config_dict = vars(args).copy()
    config_dict.pop('save_config', None)
    config_dict.pop('config', None)
    with open(args.save_config, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"Configuration saved to {args.save_config}")

# Define PyTorch Lightning Module
class FoldTree2Model(pl.LightningModule):
    def __init__(self, encoder, decoder, args, converter):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args
        self.converter = converter
        
        # Loss weights
        self.edgeweight = args.edge_weight
        self.logitweight = args.logit_weight
        self.xweight = args.x_weight
        self.fft2weight = args.fft2_weight
        self.vqweight = args.vq_weight
        self.angles_weight = args.angles_weight
        self.ss_weight = args.ss_weight
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['encoder', 'decoder', 'converter'])
    
    def forward(self, data):
        z, vqloss = self.encoder(data)
        data['res'].x = z
        out = self.decoder(data, None)
        return out, vqloss
    
    def training_step(self, batch, batch_idx):
        data = batch
        
        # Forward pass
        out, vqloss = self(data)
        
        # Get edge index
        edge_index = data.edge_index_dict.get(('res', 'contactPoints', 'res')) if hasattr(data, 'edge_index_dict') else None
        
        # Edge reconstruction loss
        logitloss = torch.tensor(0.0, device=self.device)
        edgeloss = torch.tensor(0.0, device=self.device)
        if edge_index is not None:
            edgeloss, logitloss = recon_loss_diag(data, edge_index, self.decoder, plddt=self.args.mask_plddt, key='edge_probs')
        
        # Amino acid reconstruction loss
        xloss = aa_reconstruction_loss(data['AA'].x, out['aa'])
        
        # FFT2 loss
        fft2loss = torch.tensor(0.0, device=self.device)
        if 'fft2pred' in out and out['fft2pred'] is not None:
            fft2loss = F.smooth_l1_loss(torch.cat([data['fourier2dr'].x, data['fourier2di'].x], axis=1), out['fft2pred'])
        
        # Angles loss
        angles_loss = torch.tensor(0.0, device=self.device)
        if out.get('angles') is not None:
            angles_loss = angles_reconstruction_loss(out['angles'], data['bondangles'].x, 
                                                     plddt_mask=data['plddt'].x if self.args.mask_plddt else None)
        
        # Secondary structure loss
        ss_loss = torch.tensor(0.0, device=self.device)
        if out.get('ss_pred') is not None:
            if self.args.mask_plddt:
                mask = (data['plddt'].x >= self.args.plddt_threshold).squeeze()
                if mask.sum() > 0:
                    ss_loss = F.cross_entropy(out['ss_pred'][mask], data['ss'].x[mask])
            else:
                ss_loss = F.cross_entropy(out['ss_pred'], data['ss'].x)
        
        # Total loss
        loss = (self.xweight * xloss + self.edgeweight * edgeloss + self.vqweight * vqloss +
                self.fft2weight * fft2loss + self.angles_weight * angles_loss +
                self.ss_weight * ss_loss + self.logitweight * logitloss)
        
        # Get batch size from PyG batch (number of graphs in batch)
        # For PyTorch Geometric, we need to count unique batch indices
        batch_size = data['res'].batch.max().item() + 1 if hasattr(data['res'], 'batch') and data['res'].batch is not None else 1
        
        # Log metrics with explicit batch_size to avoid iteration over FeatureStore
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train/aa_loss', xloss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train/edge_loss', edgeloss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train/vq_loss', vqloss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train/fft2_loss', fft2loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train/angles_loss', angles_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train/ss_loss', ss_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train/logit_loss', logitloss, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Log commitment cost if using scheduling
        if self.args.use_commitment_scheduling and hasattr(self.encoder, 'vector_quantizer'):
            current_commitment = self.encoder.vector_quantizer.get_commitment_cost()
            self.log('train/commitment_cost', current_commitment, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return loss
    
    def configure_optimizers(self):
        # When using DeepSpeed, let it handle the optimizer via the config
        # This avoids ZeRO-Offload incompatibility with PyTorch optimizers
        if self.args.strategy == 'deepspeed':
            return None  # DeepSpeed will configure optimizer from deepspeed_config
        
        # Optimizer setup for other strategies
        if self.args.use_muon and MUON_AVAILABLE:
            hidden_weights = []
            hidden_gains_biases = []
            nonhidden_params = []
            
            def has_modular_structure(model):
                return hasattr(model, 'input') and hasattr(model, 'body') and hasattr(model, 'head')
            
            # Process encoder
            if has_modular_structure(self.encoder):
                hidden_weights += [p for p in self.encoder.body.parameters() if p.ndim >= 2]
                hidden_gains_biases += [p for p in self.encoder.body.parameters() if p.ndim < 2]
                nonhidden_params += [*self.encoder.head.parameters(), *self.encoder.input.parameters()]
            else:
                nonhidden_params += list(self.encoder.parameters())
            
            # Process decoder
            if hasattr(self.decoder, 'decoders'):
                for name, subdecoder in self.decoder.decoders.items():
                    if has_modular_structure(subdecoder):
                        hidden_weights += [p for p in subdecoder.body.parameters() if p.ndim >= 2]
                        hidden_gains_biases += [p for p in subdecoder.body.parameters() if p.ndim < 2]
                        nonhidden_params += [*subdecoder.head.parameters(), *subdecoder.input.parameters()]
                    else:
                        nonhidden_params += list(subdecoder.parameters())
            elif has_modular_structure(self.decoder):
                hidden_weights += [p for p in self.decoder.body.parameters() if p.ndim >= 2]
                hidden_gains_biases += [p for p in self.decoder.body.parameters() if p.ndim < 2]
                nonhidden_params += [*self.decoder.head.parameters(), *self.decoder.input.parameters()]
            else:
                nonhidden_params += list(self.decoder.parameters())
            
            param_groups = [
                dict(params=hidden_weights, use_muon=True, lr=self.args.muon_lr, weight_decay=0.01),
                dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
                     lr=self.args.adamw_lr, betas=(0.9, 0.95), weight_decay=0.01),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            optimizer = torch.optim.AdamW(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=self.args.learning_rate,
                weight_decay=0.000001
            )
        
        # Scheduler setup
        total_steps = self.trainer.estimated_stepping_batches
        
        if self.args.lr_warmup_ratio > 0:
            warmup_steps = int(total_steps * self.args.lr_warmup_ratio)
        else:
            warmup_steps = self.args.lr_warmup_steps
        
        if self.args.lr_schedule == 'linear' and TRANSFORMERS_AVAILABLE:
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        elif self.args.lr_schedule == 'cosine' and TRANSFORMERS_AVAILABLE:
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        elif self.args.lr_schedule == 'cosine_restarts' and TRANSFORMERS_AVAILABLE:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, warmup_steps, total_steps, num_cycles=self.args.num_cycles)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        elif self.args.lr_schedule == 'polynomial' and TRANSFORMERS_AVAILABLE:
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer, warmup_steps, total_steps, lr_end=0.0, power=1.0)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        elif self.args.lr_schedule == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'train/loss_epoch'}}
        else:
            return optimizer

if os.path.exists(args.output_dir) and args.overwrite:
    if os.path.exists(os.path.join(args.output_dir, args.model_name + '_best_encoder.pt')):
        os.remove(os.path.join(args.output_dir, args.model_name + '_best_encoder.pt'))
    if os.path.exists(os.path.join(args.output_dir, args.model_name + '_best_decoder.pt')):
        os.remove(os.path.join(args.output_dir, args.model_name + '_best_decoder.pt'))

# Data setup
dataset_path = args.dataset
converter = pdbgraph.PDB2PyG(aapropcsv='./foldtree2/config/aaindex1.csv')
struct_dat = pdbgraph.StructureDataset(dataset_path)
train_loader = DataLoader(struct_dat, batch_size=args.batch_size, shuffle=True, num_workers=4)
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

# Loss weights (from args, with defaults matching notebook)
edgeweight = args.edge_weight
logitweight = args.logit_weight
xweight = args.x_weight
fft2weight = args.fft2_weight
vqweight = args.vq_weight
angles_weight = args.angles_weight
ss_weight = args.ss_weight

# Create output directory
modeldir = args.output_dir
os.makedirs(modeldir, exist_ok=True)
modelname = args.model_name

# Setup TensorBoard
if args.run_name:
    run_name = args.run_name
else:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{modelname}_{timestamp}"

tensorboard_log_dir = os.path.join(args.tensorboard_dir, run_name)
os.makedirs(tensorboard_log_dir, exist_ok=True)
print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
print(f"To view: tensorboard --logdir={args.tensorboard_dir}")

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=tensorboard_log_dir)

# Log full configuration to TensorBoard
config_text = "Training Configuration:\n" + "="*50 + "\n"
for key, value in sorted(vars(args).items()):
    config_text += f"{key}: {value}\n"
config_text += "="*50
writer.add_text('Configuration', config_text, 0)

# Initialize or load model
encoder_path = os.path.join(modeldir, modelname + '_best_encoder.pt')
decoder_path = os.path.join(modeldir, modelname + '_best_decoder.pt')
if os.path.exists(encoder_path) and os.path.exists(decoder_path) and not args.overwrite:
    print(f"Loading existing model from {encoder_path} and {decoder_path}")
    if os.path.exists(os.path.join(modeldir, modelname + '_info.txt')):
        with open(os.path.join(modeldir, modelname + '_info.txt'), 'r') as f:
            model_info = f.read()
        print("Model info:", model_info)
    encoder = torch.load(encoder_path, map_location=device, weights_only=False)
    decoder = torch.load(decoder_path, map_location=device, weights_only=False)
else:
    print("Creating new model...")
    hidden_size = args.hidden_size
    
    # Encoder
    if args.se3_transformer:
        encoder = se3e.se3_Encoder(
            in_channels=ndim,
            hidden_channels=[hidden_size//2, hidden_size//2],
            out_channels=args.embedding_dim,
            metadata={'edge_types': [('res','contactPoints','res'), ('res','hbond','res')]},
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
            edge_dim=1,
            encoder_hidden=hidden_size,
            EMA=args.EMA,
            nheads=5,
            dropout_p=0.005,
            reset_codes=False,
            flavor='transformer',
            fftin=True,
            use_commitment_scheduling=args.use_commitment_scheduling,
            commitment_warmup_steps=args.commitment_warmup_steps,
            commitment_schedule=args.commitment_schedule,
            commitment_start=args.commitment_start
        )
    elif args.use_muon_encoder and MUON_AVAILABLE:
        print("Using Muon-compatible mk1_MuonEncoder")
        encoder = ecdr.mk1_MuonEncoder(
            in_channels=ndim,
            hidden_channels=[hidden_size, hidden_size, hidden_size],
            out_channels=args.embedding_dim,
            metadata={'edge_types': [('res','contactPoints','res')]},
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
            edge_dim=1,
            encoder_hidden=hidden_size,
            EMA=args.EMA,
            nheads=16,
            dropout_p=0.01,
            reset_codes=False,
            flavor='transformer',
            fftin=True,
            use_commitment_scheduling=args.use_commitment_scheduling,
            commitment_warmup_steps=args.commitment_warmup_steps,
            commitment_schedule='cosine_with_restart',
            commitment_start=args.commitment_start,
            concat_positions=True
        )
    else:
        print("Using standard mk1_Encoder with dd.ipynb configuration")
        encoder = ecdr.mk1_Encoder(
            in_channels=ndim,
            hidden_channels=[hidden_size, hidden_size, hidden_size],
            out_channels=args.embedding_dim,
            metadata={'edge_types': [('res', 'contactPoints', 'res')]},
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
            edge_dim=1,
            encoder_hidden=hidden_size,
            EMA=args.EMA,
            nheads=10,
            dropout_p=0.01,
            reset_codes=False,
            flavor='transformer',
            fftin=True,
            use_commitment_scheduling=args.use_commitment_scheduling,
            commitment_warmup_steps=args.commitment_warmup_steps,
            commitment_schedule='cosine',
            commitment_start=args.commitment_start,
            concat_positions=False,
            learn_positions=True
        )
    
    if args.hetero_gae:
        # HeteroGAE_Decoder config
        decoder = ft2.HeteroGAE_Decoder(
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
        if args.use_muon_decoders and MUON_AVAILABLE:
            print("Using Muon-compatible decoders")
            mono_configs = {
                'sequence_transformer': {
                    'decoder_type': 'Transformer_AA_MuonDecoder',
                    'in_channels': {'res': args.embedding_dim},
                    'xdim': 20,
                    'concat_positions': True,
                    'hidden_channels': {('res','backbone','res'): [hidden_size], ('res','backbonerev','res'): [hidden_size]},
                    'layers': 2,
                    'AAdecoder_hidden': [hidden_size, hidden_size, hidden_size//2],
                    'amino_mapper': converter.aaindex,
                    'flavor': 'sage',
                    'nheads': 5,
                    'dropout': 0.001,
                    'normalize': False,
                    'residual': False,
                    'use_cnn_decoder': True,
                    'output_ss': True
                },
                
                'geometry_cnn': {
                    'decoder_type': 'CNN_geo_MuonDecoder',
                    'in_channels': {'res': args.embedding_dim, 'godnode4decoder': ndim_godnode, 'foldx': 23, 'fft2r': ndim_fft2r, 'fft2i': ndim_fft2i},
                    'concat_positions': True,
                    'conv_channels': [hidden_size, hidden_size//2, hidden_size//2],
                    'kernel_sizes': [3, 3, 3],
                    'FFT2decoder_hidden': [hidden_size//2, hidden_size//2],
                    'contactdecoder_hidden': [hidden_size//2, hidden_size//4],
                    'ssdecoder_hidden': [hidden_size//2, hidden_size//2],
                    'Xdecoder_hidden': [hidden_size//2, hidden_size//4],
                    'anglesdecoder_hidden': [hidden_size//2, hidden_size//4],
                    'RTdecoder_hidden': [hidden_size//2, hidden_size//4],
                    'metadata': converter.metadata,
                    'dropout': 0.001,
                    'output_fft': False,
                    'output_rt': False,
                    'output_angles': True,
                    'output_ss': False,
                    'normalize': True,
                    'residual': False,
                    'output_edge_logits': True,
                    'ncat': 8,
                    'contact_mlp': False,
                    'pool_type': 'global_mean'
                },
            }
        else:
            print("Using standard decoders with dd.ipynb configuration")
            mono_configs = {
                'sequence_transformer': {
                    'in_channels': {'res': args.embedding_dim},
                    'xdim': 20,
                    'concat_positions': False,
                    'hidden_channels': {('res','backbone','res'): [hidden_size], ('res','backbonerev','res'): [hidden_size]},
                    'layers': 2,
                    'AAdecoder_hidden': [hidden_size, hidden_size, hidden_size//2],
                    'amino_mapper': converter.aaindex,
                    'flavor': 'sage',
                    'nheads': 8,
                    'dropout': 0.001,
                    'normalize': False,
                    'residual': False,
                    'use_cnn_decoder': False,
                    'output_ss': False,
                    'learn_positions': True
                },
                
                'geometry_cnn': {
                    'in_channels': {'res': args.embedding_dim, 'godnode4decoder': ndim_godnode, 'foldx': 23, 'fft2r': ndim_fft2r, 'fft2i': ndim_fft2i},
                    'concat_positions': False,
                    'conv_channels': [hidden_size, hidden_size//2, hidden_size//2],
                    'kernel_sizes': [3, 3, 3],
                    'FFT2decoder_hidden': [hidden_size//2, hidden_size//2],
                    'contactdecoder_hidden': [hidden_size//2, hidden_size//4],
                    'ssdecoder_hidden': [hidden_size//2, hidden_size//2],
                    'Xdecoder_hidden': [hidden_size//2, hidden_size//4],
                    'anglesdecoder_hidden': [hidden_size//2, hidden_size//4],
                    'RTdecoder_hidden': [hidden_size//2, hidden_size//4],
                    'metadata': converter.metadata,
                    'dropout': 0.001,
                    'output_fft': False,
                    'output_rt': False,
                    'output_angles': True,
                    'output_ss': True,
                    'normalize': True,
                    'residual': False,
                    'output_edge_logits': True,
                    'ncat': 8,
                    'contact_mlp': False,
                    'pool_type': 'global_mean',
                    'learn_positions': True
                },
            }
        
        if args.output_foldx:
            mono_configs['foldx'] = {
                'in_channels': {'res': args.embedding_dim, 'godnode4decoder': ndim_godnode, 'foldx': 23},
                'concat_positions': True,
                'hidden_channels': {('res','backbone','res'): [hidden_size]*3, ('res','backbonerev','res'): [hidden_size]*3,
                                   ('res','informs','godnode4decoder'): [hidden_size]*3,
                                   ('godnode4decoder','informs','res'): [hidden_size]*3},
                'layers': 3,
                'foldx_hidden': [hidden_size, hidden_size//2],
                'nheads': 2,
                'metadata': converter.metadata,
                'flavor': 'sage',
                'dropout': 0.005,
                'normalize': True,
                'residual': False
            }
        
        decoder = MultiMonoDecoder(configs=mono_configs)

print("Encoder:", encoder)
print("Decoder:", decoder)

# Initialize Lightning model
model = FoldTree2Model(encoder, decoder, args, converter)

# Setup Lightning Trainer
# Setup callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=modeldir,
    filename=f'{modelname}' + '-{epoch:02d}-{train/loss:.4f}',
    save_top_k=3,
    monitor='train/loss_epoch',
    mode='min',
    save_last=True,
)

lr_monitor = LearningRateMonitor(logging_interval='step')

# Setup logger
logger = TensorBoardLogger(args.tensorboard_dir, name=run_name)

# Write model info
with open(os.path.join(modeldir, modelname + '_info.txt'), 'w') as f:
    f.write(f'Date: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f'Encoder: {encoder}\n')
    f.write(f'Decoder: {decoder}\n')
    f.write(f'Learning rate: {args.learning_rate}\n')
    f.write(f'Batch size: {args.batch_size}\n')
    f.write(f'Hidden size: {args.hidden_size}\n')
    f.write(f'Embedding dimension: {args.embedding_dim}\n')
    f.write(f'Number of embeddings: {args.num_embeddings}\n')
    f.write(f'Loss weights - Edge: {edgeweight}, X: {xweight}, FFT2: {fft2weight}, VQ: {vqweight}\n')

# Determine strategy
if args.gpus == -1:
    # Use all available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using all {num_gpus} available GPUs")
    devices = num_gpus if num_gpus > 0 else 'auto'
else:
    devices = args.gpus
    print(f"Using {devices} GPU(s)")



# Lightning Trainer
# Determine strategy - use find_unused_parameters for all strategies to handle unused model parameters
if devices > 1 or devices == -1:
    # For multi-GPU, configure strategy based on user selection
    if args.strategy == 'auto' or args.strategy == 'ddp':
        strategy = 'ddp_find_unused_parameters_true'
    elif args.strategy == 'ddp_spawn':
        from pytorch_lightning.strategies import DDPSpawnStrategy
        strategy = DDPSpawnStrategy(find_unused_parameters=True)
        print("Using DDPSpawn strategy with find_unused_parameters=True")
    elif args.strategy == 'fsdp':
        # FSDP for large models that don't fit in single GPU
        from pytorch_lightning.strategies import FSDPStrategy
        from torch.distributed.fsdp import CPUOffload, BackwardPrefetch
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools
        
        # Configure FSDP wrapping policy (wrap layers > 1e8 params)
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1e8
        )
        
        # FSDP doesn't have find_unused_parameters, but handles unused params automatically
        # through its sharding mechanism
        strategy = FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=args.fsdp_cpu_offload) if args.fsdp_cpu_offload else None,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            state_dict_type="full",  # Save full model state
        )
        print(f"Using FSDP strategy with CPU offload: {args.fsdp_cpu_offload}")
        print("Note: FSDP automatically handles unused parameters through its sharding mechanism")
    elif args.strategy == 'deepspeed':
        # DeepSpeed for extreme model sizes
        from pytorch_lightning.strategies import DeepSpeedStrategy
        
        deepspeed_config = {
            "zero_optimization": {
                "stage": args.deepspeed_stage,
                "offload_optimizer": {"device": "cpu", "pin_memory": True} if args.deepspeed_stage >= 2 else None,
                "offload_param": {"device": "cpu", "pin_memory": True} if args.deepspeed_stage == 3 else None,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
            },
            # Optimizer configuration - required when using ZeRO-Offload
            # DeepSpeed will use its own CPU-optimized AdamW when offload_optimizer is enabled
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.000001
                }
            },
            "fp16": {"enabled": args.mixed_precision},
            "bf16": {"enabled": False},
            "train_micro_batch_size_per_gpu": args.batch_size,
        }
        # Note: gradient_accumulation_steps is NOT set here - Lightning passes it via accumulate_grad_batches
        
        # DeepSpeed handles unused parameters automatically through ZeRO optimization
        strategy = DeepSpeedStrategy(config=deepspeed_config)
        print(f"Using DeepSpeed ZeRO Stage {args.deepspeed_stage}")
        print("Note: DeepSpeed automatically handles unused parameters through ZeRO optimization")
        print(f"Note: Using DeepSpeed's built-in AdamW optimizer (required for ZeRO-Offload)")
    elif args.strategy == 'ddp_sharded':
        # FairScale's sharded DDP (alternative to FSDP)
        from pytorch_lightning.strategies import DDPShardedStrategy
        strategy = DDPShardedStrategy(find_unused_parameters=True)
        print("Using DDPSharded strategy (FairScale) with find_unused_parameters=True")
    elif args.strategy == 'dp':
        # DataParallel - legacy single-node strategy
        strategy = 'dp'
        print("Using DataParallel (legacy) - unused parameters handled automatically")
    else:
        strategy = args.strategy
else:
    strategy = 'auto'

trainer = pl.Trainer(
    max_epochs=args.epochs,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=devices,
    strategy=strategy,
    precision='16-mixed' if args.mixed_precision else 32,
    gradient_clip_val=1.0 if args.clip_grad else 0,
    # Lightning automatically passes this to DeepSpeed config as gradient_accumulation_steps
    accumulate_grad_batches=args.gradient_accumulation_steps,
    callbacks=[checkpoint_callback, lr_monitor],
    logger=logger,
    log_every_n_steps=10,
    deterministic=True,
    enable_progress_bar=True,
)

print(f"\nTraining Configuration:")
print(f"  Total epochs: {args.epochs}")
print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * (devices if isinstance(devices, int) else 1)}")
print(f"  Mask pLDDT: {args.mask_plddt}")
if args.mask_plddt:
    print(f"  pLDDT threshold: {args.plddt_threshold}")
print(f"  Mixed precision: {args.mixed_precision}")
print(f"  Gradient clipping: {args.clip_grad}")
print()

# Train the model
trainer.fit(model, train_loader)

# Save final model in PyTorch format
print("\nSaving final models...")
torch.save(model.encoder, os.path.join(modeldir, f"{modelname}_encoder_final.pt"))
torch.save(model.decoder, os.path.join(modeldir, f"{modelname}_decoder_final.pt"))

print("Training complete! Models saved:")
print(f"  Lightning checkpoint: {checkpoint_callback.best_model_path}")
print(f"  Encoder: {os.path.join(modeldir, modelname + '_encoder_final.pt')}")
print(f"  Decoder: {os.path.join(modeldir, modelname + '_decoder_final.pt')}")
print(f"  Best loss: {checkpoint_callback.best_model_score:.4f}")
print(f"  TensorBoard logs: {logger.log_dir}")
