# coding: utf-8
# learn_monodecoder.py - Training script for MultiMonoDecoder

import torch
from torch_geometric.data import DataLoader
import numpy as np
from foldtree2.src import pdbgraph
from foldtree2.src import foldtree2_ecddcd as ft2
from foldtree2.src.losses.losses import recon_loss_diag, aa_reconstruction_loss
from foldtree2.src.mono_decoders import MultiMonoDecoder
import os
import tqdm
import random
import torch.nn.functional as F
import pickle
import argparse
import sys
import time
import foldtree2.src.se3_strcut_decoder as se3e
import warnings
import yaml
import json
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning)

# Add argparse for CLI configuration
parser = argparse.ArgumentParser(description='Train model with MultiMonoDecoders for sequence and geometry prediction')
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
parser.add_argument('--model-name', type=str, default='monodecoder_model',
                    help='Model name for saving (default: monodecoder_model)')
parser.add_argument('--num-embeddings', type=int, default=40,
                    help='Number of embeddings for the encoder (default: 40)')
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
parser.add_argument('--lr-schedule', type=str, default='plateau', choices=['plateau', 'cosine', 'linear', 'none'],
                    help='Learning rate schedule: plateau (ReduceLROnPlateau), cosine, linear decay, or none (default: plateau)')
parser.add_argument('--lr-min', type=float, default=1e-6,
                    help='Minimum learning rate for cosine/linear schedules (default: 1e-6)')

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
    
    # Set defaults from config file, but allow CLI args to override
    for key, value in config.items():
        # Only set from config if not explicitly provided via CLI
        if hasattr(args, key):
            # Check if the argument was provided via CLI (not default)
            cli_value = getattr(args, key)
            default_value = parser.get_default(key)
            if cli_value == default_value:
                # Use config value since CLI didn't override
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
if args.lr_schedule in ['cosine', 'linear']:
    print(f"  LR Min: {args.lr_min}")

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
    if os.path.exists(os.path.join(args.output_dir, args.model_name + '_best.pkl')):
        os.remove(os.path.join(args.output_dir, args.model_name + '_best.pkl'))

# Data setup
datadir = '../../datasets/foldtree2/'
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

# Loss weights
edgeweight = 0.05
logitweight = 0.08
xweight = 0.1
fft2weight = 0.01
vqweight = 0.001
angles_weight = 0.001

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
if os.path.exists(os.path.join(modeldir, modelname + '_best.pkl')) and args.overwrite == False:
    print(f"Loading existing model from {os.path.join(modeldir, modelname + '_best.pkl')}")
    if os.path.exists(os.path.join(modeldir, modelname + '_info.txt')):
        with open(os.path.join(modeldir, modelname + '_info.txt'), 'r') as f:
            model_info = f.read()
        print("Model info:", model_info)
    # Load encoder and decoder from saved model
    with open(os.path.join(modeldir, modelname + '_best.pkl'), 'rb') as f:
        encoder, decoder = pickle.load(f)
else:
    print("Creating new model...")
    # Model setup
    hidden_size = args.hidden_size
    
    # Encoder
    if args.se3_transformer:
        encoder = se3e.se3_Encoder(
            in_channels=ndim,
            hidden_channels=[hidden_size//2, hidden_size//2],
            out_channels=args.embedding_dim,
            metadata={'edge_types': [('res','contactPoints','res'), ('res','hbond','res')]},
            num_embeddings=args.num_embeddings,
            commitment_cost=0.8,
            edge_dim=1,
            encoder_hidden=hidden_size,
            EMA= args.EMA,
            nheads=5,
            dropout_p=0.005,
            reset_codes=False,
            flavor='transformer',
            fftin=True
        )
    else:
        encoder = ft2.mk1_Encoder(
            in_channels=ndim,
            hidden_channels=[hidden_size, hidden_size],
            out_channels=args.embedding_dim,
            metadata={'edge_types': [('res','contactPoints','res')]},
            num_embeddings=args.num_embeddings,
            commitment_cost=0.9,
            edge_dim=1,
            encoder_hidden=hidden_size,
            EMA=args.EMA,
            nheads=8,
            dropout_p=0.01,
            reset_codes=False,
            flavor='transformer',
            fftin=True
        )

    if args.hetero_gae:
        # HeteroGAE_Decoder config (example, adjust as needed)
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
        mono_configs = {
            'sequence_transformer': {
                'in_channels': {'res': args.embedding_dim},
                'xdim': 20,
                'concat_positions': True,
                'hidden_channels': {
                    ('res','backbone','res'): [hidden_size*2], 
                    ('res','backbonerev','res'): [hidden_size*2]
                },
                'layers': 2,
                'AAdecoder_hidden': [hidden_size, hidden_size, hidden_size//2],
                'amino_mapper': converter.aaindex,
                'flavor': 'sage',
                'nheads': 4,
                'dropout': 0.005,
                'normalize': False,
                'residual': False
            },
            
            'contacts': {
                'in_channels': {
                    'res': args.embedding_dim, 
                    'godnode4decoder': ndim_godnode, 
                    'foldx': 23,
                    'fft2r': ndim_fft2r, 
                    'fft2i': ndim_fft2i
                },
                'concat_positions': True,
                'hidden_channels': {
                    ('res','backbone','res'): [hidden_size]*8, 
                    ('res','backbonerev','res'): [hidden_size]*8, 
                    ('res','informs','godnode4decoder'): [hidden_size]*8, 
                    ('godnode4decoder','informs','res'): [hidden_size]*8
                },
                'layers': 4,
                'FFT2decoder_hidden': [hidden_size, hidden_size, hidden_size],
                'contactdecoder_hidden': [hidden_size//4, hidden_size//8],
                'anglesdecoder_hidden': [hidden_size//2, hidden_size//2],
                'nheads': 1,
                'Xdecoder_hidden': [hidden_size, hidden_size, hidden_size],
                'metadata': converter.metadata,
                'flavor': 'sage',
                'dropout': 0.005,
                'output_fft': False,
                'output_rt': False,
                'output_angles': False,
                'normalize': True,
                'residual': False,
                'contact_mlp': False,
                'ncat': 16,
                'output_edge_logits': True
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



        '''

        'sequence_transformer': {
                'in_channels': {'res': args.embedding_dim},
                'xdim': 20,
                'concat_positions': True,
                'hidden_channels': {('res','backbone','res'): [hidden_size]*3 , ('res','backbonerev','res'): [hidden_size]*3},
                'layers': 1,
                'AAdecoder_hidden': [hidden_size, hidden_size, hidden_size//2],
                'amino_mapper': converter.aaindex,
                'flavor': 'sage',
                'nheads': 2,
                'dropout': 0.005,
                'normalize': False,
                'residual': False
            },
            
        'sequence': {
            'in_channels': {'res': args.embedding_dim},
            'xdim': 20,  # 20 amino acids
            'concat_positions': True,
            'hidden_channels': {('res','backbone','res'): [hidden_size]*5, ('res','backbonerev','res'): [hidden_size]*5},
            'layers': 5,
            'AAdecoder_hidden': [hidden_size, hidden_size//2, hidden_size//2],
            'amino_mapper': converter.aaindex,
            'flavor': 'sage',
            'nheads' : 1,
            'dropout': 0.005,
            'normalize': True,
            'residual': False
        },
        '''
        

        # Initialize decoder
        #tasks = ['sequence_transformer', 'contacts']
        tasks = ['sequence', 'contacts']
        decoder = MultiMonoDecoder( configs=mono_configs)

# Move models to device
encoder = encoder.to(device)
decoder = decoder.to(device)
print("Encoder:", encoder)
print("Decoder:", decoder)

# Training setup
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate , weight_decay=0.000001)

# Learning rate scheduler setup
total_steps = len(train_loader) * args.epochs
warmup_steps = args.lr_warmup_steps

if args.lr_schedule == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
elif args.lr_schedule == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr_min)
elif args.lr_schedule == 'linear':
    # Linear decay from learning_rate to lr_min
    lambda_lr = lambda step: max(args.lr_min / args.learning_rate, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
else:  # 'none'
    scheduler = None

# Learning rate warmup function
def get_warmup_lr(current_step, warmup_steps, base_lr):
    """Linear warmup from 0 to base_lr over warmup_steps"""
    if warmup_steps == 0 or current_step >= warmup_steps:
        return base_lr
    return base_lr * (current_step / warmup_steps)

def set_lr(optimizer, lr):
    """Set learning rate for all parameter groups"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

print(f"Learning rate schedule: {args.lr_schedule}")
print(f"Warmup steps: {warmup_steps}")
print(f"Total training steps: {total_steps}")
if args.lr_schedule in ['cosine', 'linear']:
    print(f"Min learning rate: {args.lr_min}")

# Function to analyze gradient norms
def analyze_gradient_norms(model, top_k=3):
    """
    Analyzes gradients in the given model and returns the top_k layers with
    highest and lowest gradient norms.
    """
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
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
    f.write(f'Loss weights - Edge: {edgeweight}, X: {xweight}, FFT2: {fft2weight}, VQ: {vqweight}\n')

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

# Training loop
encoder.train()
decoder.train()
clip_grad = args.clip_grad  # Enable gradient clipping
burn_in = args.burn_in  # Use burn-in period if specified

best_loss = float('inf')
done_burn = False
after_burn_in = args.epochs - burn_in if burn_in else args.epochs
global_step = 0  # Track global training steps for warmup

print(f"Total epochs: {args.epochs}, Burn-in epochs: {burn_in}, After burn-in epochs: {after_burn_in}")
for epoch in range(args.epochs):
    if burn_in and epoch < burn_in:
        print(f"Burn-in epoch {epoch+1}/{args.epochs}: Adjusting loss weights")
        edgeweight = 0
        xweight = 1
        fft2weight = 0
        vqweight = 0.001
        #change learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] =   args.learning_rate * 10 
            print(f"Using learning rate {param_group['lr']:.6f} during burn-in")
        done_burn = True
    if done_burn and epoch >= burn_in:
        done_burn = False
        # After burn-in, use normal weights
        print(f"Training epoch {epoch+1}/{args.epochs}: Using adjusted loss weights")
        xweight = 1  # Normal weight for amino acid reconstruction
        edgeweight = 0.001  # Normal weight for edge loss
        fft2weight = 0.001  # Normal weight for FFT2 loss
        vqweight = 0.001  # Normal weight for VQ loss
        #change learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
            print(f"Using learning rate {param_group['lr']:.6f} after burn-in")

    total_loss_x = 0
    total_loss_edge = 0
    total_vq = 0
    total_angles_loss = 0
    total_loss_fft2 = 0
    total_logit_loss = 0
    
    for data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        data = data.to(device)
        
        # Learning rate warmup
        if warmup_steps > 0 and global_step < warmup_steps:
            warmup_lr = get_warmup_lr(global_step, warmup_steps, args.learning_rate)
            set_lr(optimizer, warmup_lr)
        
        optimizer.zero_grad()
        
        # Forward through encoder
        z, vqloss = encoder(data)
        data['res'].x = z
        
        # Forward through decoder
        out = decoder(data, None)
        
        # Get outputs
        recon_x = out['aa'] if 'aa' in out else None
        fft2_x = out['fft2pred'] if 'fft2pred' in out else None
        
        # Edge loss: use contactPoints if available
        edge_index = data.edge_index_dict.get(('res', 'contactPoints', 'res'))
        logitloss = torch.tensor(0.0, device=device)
        if edge_index is not None:
            edgeloss, logitloss = recon_loss_diag(
                data, edge_index, decoder, 
                plddt=False, offdiag=False, key='edge_probs'
            )
        else:
            edgeloss = torch.tensor(0.0, device=device)
        
        # Amino acid reconstruction loss
        xloss = aa_reconstruction_loss(data['AA'].x, recon_x)
        
        # FFT2 loss if available
        if fft2_x is not None:
            fft2loss = F.smooth_l1_loss(
                torch.cat([data['fourier2dr'].x, data['fourier2di'].x], axis=1), 
                fft2_x
            )
        else:
            fft2loss = torch.tensor(0.0, device=device)
        
        # Angles loss
        angles_loss = torch.tensor(0.0, device=device)
        if 'angles' in out and out['angles'] is not None:
            angles = out['angles']
            angles_loss = F.smooth_l1_loss(angles, data['bondangles'].x)
    
        # Total loss
        loss = (xweight * xloss + edgeweight * edgeloss + vqweight * vqloss + 
                fft2loss * fft2weight + angles_loss * angles_weight + 
                logitloss * logitweight)
        
        # Backward and optimize
        loss.backward()
        
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            
        optimizer.step()
        
        # Step the scheduler (for step-based schedulers)
        if scheduler is not None and args.lr_schedule in ['cosine', 'linear']:
            if global_step >= warmup_steps:
                scheduler.step()
        
        global_step += 1
        
        # Accumulate metrics
        total_loss_x += xloss.item()
        total_logit_loss += logitloss.item()
        total_loss_edge += edgeloss.item()
        total_loss_fft2 += fft2loss.item()
        total_angles_loss += angles_loss.item()
        total_vq += vqloss.item() if isinstance(vqloss, torch.Tensor) else float(vqloss)
    
    # Calculate average losses
    avg_loss_x = total_loss_x / len(train_loader)
    avg_loss_edge = total_loss_edge / len(train_loader)
    avg_loss_vq = total_vq / len(train_loader)
    avg_loss_fft2 = total_loss_fft2 / len(train_loader)
    avg_angles_loss = total_angles_loss / len(train_loader)
    avg_logit_loss = total_logit_loss / len(train_loader)
    avg_total_loss = (avg_loss_x + avg_loss_edge + avg_loss_vq + 
                      avg_loss_fft2 + avg_angles_loss + avg_logit_loss)
    
    # Update learning rate scheduler (for epoch-based schedulers)
    if scheduler is not None and args.lr_schedule == 'plateau':
        scheduler.step(avg_loss_x)
    
    # Print metrics
    print(f"Epoch {epoch+1}: AA Loss: {avg_loss_x:.4f}, "
          f"Edge Loss: {avg_loss_edge:.4f}, VQ Loss: {avg_loss_vq:.4f}, "
          f"FFT2 Loss: {avg_loss_fft2:.4f}, Angles Loss: {avg_angles_loss:.4f}, "
          f"Logit Loss: {avg_logit_loss:.4f}")
    print(f"Total Loss: {avg_total_loss:.4f}, "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    #if avg_loss_edge > avg_loss_x:
    #    edgeweight *= 1.5
    #    print(f"Increasing xweight to {edgeweight:.4f} due to higher edge loss")        
    #if avg_loss_x > avg_loss_edge:
    #    xweight *= 1.5
    #    print(f"Increasing AA weight to {xweight:.4f} due to higher AA loss")

    # Gradient analysis
    print("Gradient norms (encoder):", analyze_gradient_norms(encoder))
    print("Gradient norms (decoder):", analyze_gradient_norms(decoder))
    
    # Log to tensorboard
    writer.add_scalar('Loss/AA', avg_loss_x, epoch)
    writer.add_scalar('Loss/Edge', avg_loss_edge, epoch)
    writer.add_scalar('Loss/VQ', avg_loss_vq, epoch)
    writer.add_scalar('Loss/FFT2', avg_loss_fft2, epoch)
    writer.add_scalar('Loss/Angles', avg_angles_loss, epoch)
    writer.add_scalar('Loss/Logit', avg_logit_loss, epoch)
    writer.add_scalar('Loss/Total', avg_total_loss, epoch)
    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
    
    # Log loss weights
    writer.add_scalar('Weights/Edge', edgeweight, epoch)
    writer.add_scalar('Weights/X', xweight, epoch)
    writer.add_scalar('Weights/FFT2', fft2weight, epoch)
    writer.add_scalar('Weights/VQ', vqweight, epoch)
    
    # Log gradient norms
    encoder_grad_norms = analyze_gradient_norms(encoder, top_k=1)
    decoder_grad_norms = analyze_gradient_norms(decoder, top_k=1)
    if encoder_grad_norms['highest']:
        writer.add_scalar('Gradients/Encoder_Max', encoder_grad_norms['highest'][0][1], epoch)
    if encoder_grad_norms['lowest']:
        writer.add_scalar('Gradients/Encoder_Min', encoder_grad_norms['lowest'][0][1], epoch)
    if decoder_grad_norms['highest']:
        writer.add_scalar('Gradients/Decoder_Max', decoder_grad_norms['highest'][0][1], epoch)
    if decoder_grad_norms['lowest']:
        writer.add_scalar('Gradients/Decoder_Min', decoder_grad_norms['lowest'][0][1], epoch)
    
    # Update metrics for hparams logging
    metrics_dict['best_loss'] = best_loss
    metrics_dict['final_aa_loss'] = avg_loss_x
    metrics_dict['final_edge_loss'] = avg_loss_edge
    
    # Save best model
    if avg_total_loss < best_loss:
        best_loss = avg_total_loss
        print(f"Saving best model with loss: {best_loss:.4f}")
        with open(os.path.join(modeldir, modelname + '_best.pkl'), 'wb') as f:
            pickle.dump((encoder, decoder), f)
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        #with open(os.path.join(modeldir, f"{modelname}_epoch{epoch+1}.pkl"), 'wb') as f:
        #    pickle.dump((encoder, decoder), f)
        #save pth 
        torch.save(encoder, os.path.join(modeldir, f"{modelname}_encoder_epoch{epoch+1}.pt"))
        torch.save(decoder, os.path.join(modeldir, f"{modelname}_decoder_epoch{epoch+1}.pt"))

# Save final model
#with open(os.path.join(modeldir, modelname + '.pkl'), 'wb') as f:
#    pickle.dump((encoder, decoder), f)
torch.save(encoder, os.path.join(modeldir, f"{modelname}_encoder_final.pt"))
torch.save(decoder, os.path.join(modeldir, f"{modelname}_decoder_final.pt"))

# Log final hyperparameters and metrics
writer.add_hparams(hparams_dict, metrics_dict)

# Close TensorBoard writer
writer.close()

print(f"Training complete! Final model saved to {os.path.join(modeldir, modelname + '.pkl')}")
print(f"TensorBoard logs saved to: {tensorboard_log_dir}")
print(f"View with: tensorboard --logdir={args.tensorboard_dir}")
