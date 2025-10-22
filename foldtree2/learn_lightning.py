# learn_lightning.py - Refactored to follow learn_monodecoder.py structure
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Argument parsing
parser = argparse.ArgumentParser(description='Train model with MultiMonoDecoder for sequence and geometry prediction (Lightning version)')
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
parser.add_argument('--overwrite', action='store_true', help='Overwrite saved model if exists, otherwise continue training')
parser.add_argument('--output-fft', action='store_true', help='Train the model with FFT output')
parser.add_argument('--output-rt', action='store_true', help='Train the model with rotation and translation output')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument('--clip-grad', action='store_true', help='Enable gradient clipping during training')
parser.add_argument('--burn-in', type=int, default=0, help='Burn-in period for training (default: 0, no burn-in)')
parser.add_argument('--hetero-gae', action='store_true', help='Use HeteroGAE_Decoder instead of MultiMonoDecoder')

#data directory
parser.add_argument('--data-dir', type=str, default='../../datasets/foldtree2/',
                    help='Directory containing the dataset (default: ../../datasets/foldtree2/)')

#prop csv
parser.add_argument('--aapropcsv', type=str, default='config/aaindex1.csv',
                    help='Amino acid property CSV file (default: config/aaindex1.csv)')

if len(sys.argv) == 1:
    print('No arguments provided. Use -h for help.')
    print('Example command: python learn_lightning.py -d structs_training_godnodemk5.h5 -o ./models/ -lr 0.0001 -e 20 -bs 5')
    parser.print_help()
    sys.exit(0)


args = parser.parse_args()

# Set seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# Initialize or load model
model_path = os.path.join(modeldir, modelname + '.pkl')
if os.path.exists(model_path) and not args.overwrite:
    print(f"Loading existing model from {model_path}")
    with open(model_path, 'rb') as f:
        encoder, decoder = pickle.load(f)
else:
    print("Creating new model...")
    hidden_size = args.hidden_size
    encoder = ft2.mk1_Encoder(
        in_channels=ndim,
        hidden_channels=[hidden_size, hidden_size],
        out_channels=args.embedding_dim,
        metadata={'edge_types': [('res', 'contactPoints', 'res')]},
        num_embeddings=args.num_embeddings,
        commitment_cost=0.9,
        edge_dim=1,
        encoder_hidden=hidden_size,
        EMA=True,
        nheads=8,
        dropout_p=0.01,
        reset_codes=False,
        flavor='transformer',
        fftin=True
    )
    # MultiMonoDecoder for sequence and geometry
    mono_configs = {
        'sequence_transformer': {
            'in_channels': {'res': args.embedding_dim},
            'xdim': 20,
            'concat_positions': True,
            'hidden_channels': {
                ('res', 'backbone', 'res'): [hidden_size*2],
                ('res', 'backbonerev', 'res'): [hidden_size*2]
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
                ('res', 'backbone', 'res'): [hidden_size]*8,
                ('res', 'backbonerev', 'res'): [hidden_size]*8,
                ('res', 'informs', 'godnode4decoder'): [hidden_size]*8,
                ('godnode4decoder', 'informs', 'res'): [hidden_size]*8
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
    decoder = MultiMonoDecoder(configs=mono_configs)

# Move models to device
encoder = encoder.to(device)
decoder = decoder.to(device)
print("Encoder:", encoder)
print("Decoder:", decoder)

# Training setup
optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()), 
    lr=args.learning_rate
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    factor=0.5, 
    patience=2
)

# Function to analyze gradient norms
def analyze_gradient_norms(model, top_k=3):
    norms = [(n, p.grad.norm().item()) for n, p in model.named_parameters() if p.grad is not None]
    norms = sorted(norms, key=lambda x: x[1], reverse=True)
    return norms[:top_k]

# Write model parameters to file
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

# Training loop
encoder.train()
decoder.train()
clip_grad = args.clip_grad  # Enable gradient clipping
burn_in = args.burn_in  # Use burn-in period if specified

best_loss = float('inf')
done_burn = False

after_burn_in = args.epochs - burn_in if burn_in else args.epochs
print(f"Total epochs: {args.epochs}, Burn-in epochs: {burn_in}, After burn-in epochs: {after_burn_in}")
for epoch in range(args.epochs):
    if burn_in and epoch < burn_in:
        print(f"Burn-in epoch {epoch+1}/{args.epochs}: Adjusting loss weights")
        xweight = 1
        edgeweight = 0.0001  # Lower initial weight for edge loss
        fft2weight = 0.001  # Lower initial weight for FFT2 loss
        vqweight = 0.01  # Lower initial weight for VQ loss
        #change learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] =   args.learning_rate * 10 
            print(f"Using learning rate {param_group['lr']:.6f} during burn-in")
        done_burn = True
    if done_burn and epoch >= burn_in:
        done_burn = False
        # After burn-in, use normal weights
        print(f"Training epoch {epoch+1}/{args.epochs}: Using adjusted loss weights")
        xweight = 0.1  # Normal weight for amino acid reconstruction
        edgeweight = 0.001  # Normal weight for edge loss
        fft2weight = 0.001  # Normal weight for FFT2 loss
        vqweight = 0.01  # Normal weight for VQ loss
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
    total_loss = 0
    for data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        data = data.to(device)
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
        # Accumulate metrics
        total_loss_x += xloss.item()
        total_logit_loss += logitloss.item()
        total_loss_edge += edgeloss.item()
        total_loss_fft2 += fft2loss.item()
        total_angles_loss += angles_loss.item()
        total_vq += (vqloss.item() if isinstance(vqloss, torch.Tensor) 
                     else float(vqloss))
        total_loss += loss.item()
    # Calculate average losses
    avg_loss_x = total_loss_x / len(train_loader)
    avg_loss_edge = total_loss_edge / len(train_loader)
    avg_loss_vq = total_vq / len(train_loader)
    avg_loss_fft2 = total_loss_fft2 / len(train_loader)
    avg_angles_loss = total_angles_loss / len(train_loader)
    avg_logit_loss = total_logit_loss / len(train_loader)
    avg_total_loss = total_loss / len(train_loader)
    # Update learning rate
    scheduler.step(avg_loss_x)
    # Print metrics
    print(f"Epoch {epoch+1}: AA Loss: {avg_loss_x:.4f}, "
          f"Edge Loss: {avg_loss_edge:.4f}, VQ Loss: {avg_loss_vq:.4f}, "
          f"FFT2 Loss: {avg_loss_fft2:.4f}, Angles Loss: {avg_angles_loss:.4f}, "
          f"Logit Loss: {avg_logit_loss:.4f}")
    print(f"Total Loss: {avg_total_loss:.4f}, "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
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
    # Save best model
    if avg_total_loss < best_loss:
        best_loss = avg_total_loss
        print(f"Saving best model with loss: {best_loss:.4f}")
        with open(os.path.join(modeldir, modelname + '_best.pkl'), 'wb') as f:
            pickle.dump((encoder, decoder), f)
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        with open(os.path.join(modeldir, f"{modelname}_epoch{epoch+1}.pkl"), 'wb') as f:
            pickle.dump((encoder, decoder), f)
# Save final model
with open(os.path.join(modeldir, modelname + '.pkl'), 'wb') as f:
    pickle.dump((encoder, decoder), f)

print(f"Training complete! Final model saved to {os.path.join(modeldir, modelname + '.pkl')}")
