import foldtree2.src.encoder as ecdr
from converter import pdbgraph
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import pickle
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import random
import tqdm
import sys

# Experiment setup
experiment_dir = './scaling_experiment/'
os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(f"{experiment_dir}/checkpoints", exist_ok=True)

# Model configuration
hidden_size = 500
num_epochs = 20
modeldir = './models/'
datadir = '../../datasets/'
batch_size = 30
num_embeddings = 40
embedding_dim = 20
learning_rate = 0.0001
edgeweight = 0.01
xweight = 0.05
vqweight = 0.001
foldxweight = 0.001
fapeweight = 0.01
angleweight = 0.01
lddt_weight = 0.1
dist_weight = 0.01
clip_grad = True
ema = True
fapeloss = True
lddtloss = False
denoise = True
seed = 0
fft2weight = 0.001  # Add this line to define fft2weight for the loss calculation

# Set seeds for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data loading
print("Loading dataset...")
full_dataset = pdbgraph.StructureDataset('structs_training_godnodemk5.h5')
converter = pdbgraph.PDB2PyG()
sample_loader = DataLoader(full_dataset, batch_size=1)
data_sample = next(iter(sample_loader))
ndim = data_sample['res'].x.shape[1]
ndim_godnode = data_sample['godnode'].x.shape[1]
print(f"Dataset loaded with {len(full_dataset)} samples")
print(f"Using ndim={ndim}, ndim_godnode={ndim_godnode}")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model setup
encoder = ecdr.mk1_Encoder(
    in_channels=ndim,
    hidden_channels=[hidden_size, hidden_size, hidden_size],
    out_channels=embedding_dim,
    metadata={'edge_types': [('res','contactPoints','res'), ('res','hbond','res')]},
    num_embeddings=num_embeddings,
    commitment_cost=0.9,
    edge_dim=1,
    encoder_hidden=hidden_size,
    EMA=ema,
    nheads=5,
    dropout_p=0.005,
    reset_codes=False,
    flavor='transformer',
    fftin=True
)
mono_configs = {
    'sequence_transformer': {
        'in_channels': {'res': embedding_dim},
        'xdim': 20,
        'concat_positions': True,
        'hidden_channels': {('res','backbone','res'): [hidden_size]*3, ('res','backbonerev','res'): [hidden_size]*3},
        'layers': 3,
        'AAdecoder_hidden': [hidden_size, hidden_size, hidden_size//2],
        'Xdecoder_hidden': [hidden_size, hidden_size, hidden_size],
        'contactdecoder_hidden': [hidden_size//2, hidden_size//2],
        'nheads': 5,
        'amino_mapper': converter.aaindex,
        'flavor': 'sage',
        'dropout': 0.005,
        'normalize': True,
        'residual': False,
        'contact_mlp': True
    },
    'contacts': {
        'in_channels': {'res': embedding_dim, 'godnode4decoder': ndim_godnode, 'foldx': 23},
        'concat_positions': True,
        'hidden_channels': {('res','backbone','res'): [hidden_size]*3, ('res','backbonerev','res'): [hidden_size]*3, 
                           ('res','informs','godnode4decoder'): [hidden_size]*3, 
                           ('godnode4decoder','informs','res'): [hidden_size]*3},
        'layers': 3,
        'FFT2decoder_hidden': [hidden_size*2, hidden_size*2],
        'contactdecoder_hidden': [hidden_size, hidden_size//2],
        'nheads': 2,
        'Xdecoder_hidden': [hidden_size, hidden_size, hidden_size],
        'metadata': converter.metadata,
        'flavor': 'sage',
        'dropout': 0.005,
        'output_fft': False,
        'output_rt': False,
        'normalize': True,
        'residual': False,
        'contact_mlp': True
    }
}
from src.mono_decoders import MultiMonoDecoder
decoder = MultiMonoDecoder(configs=mono_configs)

encoder = encoder.to(device)
decoder = decoder.to(device)
print("Encoder:", encoder)
print("Decoder:", decoder)

optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Training loop
best_loss = float('inf')
for epoch in range(num_epochs):
    total_loss_x = 0
    total_loss_edge = 0
    total_vq = 0
    total_fft2_loss = 0
    total_loss = 0
    for data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        data = data.to(device)
        optimizer.zero_grad()
        z, vqloss = encoder(data)
        data['res'].x = z
        out = decoder(data, None)
        recon_x = out['aa'] if 'aa' in out else None
        fft2_x = out['fft2pred'] if 'fft2pred' in out else None
        edge_index = data.edge_index_dict.get(('res', 'contactPoints', 'res'))
        if edge_index is not None:
            edgeloss, _ = ecdr.recon_loss_diag(data, edge_index, decoder, plddt=True, offdiag=True, key='edge_probs')
        else:
            edgeloss = torch.tensor(0.0, device=device)
        xloss = ecdr.aa_reconstruction_loss(data['AA'].x, recon_x)
        if fft2_x is not None:
            F_hat = torch.complex(fft2_x[:, :1], fft2_x[:, 1:])
            F = torch.complex(data['fourier2dr'].x, data['fourier2di'].x)
            mag_loss = torch.mean(torch.abs((torch.abs(F_hat) - torch.abs(F))))
            phase_loss = torch.mean(torch.abs((torch.angle(F_hat) - torch.angle(F))))
            fft2loss = mag_loss + phase_loss
        else:
            fft2loss = torch.tensor(0.0, device=device)
        loss = xweight * xloss + edgeweight * edgeloss + vqweight * vqloss + fft2weight * fft2loss
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss_x += xloss.item()
        total_loss_edge += edgeloss.item()
        total_vq += vqloss.item() if isinstance(vqloss, torch.Tensor) else float(vqloss)
        total_fft2_loss += fft2loss.item() if isinstance(fft2loss, torch.Tensor) else float(fft2loss)
        total_loss += loss.item()
    avg_loss_x = total_loss_x / len(train_loader)
    avg_loss_edge = total_loss_edge / len(train_loader)
    avg_loss_vq = total_vq / len(train_loader)
    avg_loss_fft2 = total_fft2_loss / len(train_loader)
    avg_total_loss = total_loss / len(train_loader)
    scheduler.step(avg_total_loss)
    print(f"Epoch {epoch+1}: AA Loss: {avg_loss_x:.4f}, Edge Loss: {avg_loss_edge:.4f}, "
          f"VQ Loss: {avg_loss_vq:.4f}, FFT2 Loss: {avg_loss_fft2:.4f}")
    print(f"Total Loss: {avg_total_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    if avg_total_loss < best_loss:
        best_loss = avg_total_loss
        print(f"Saving best model with loss: {best_loss:.4f}")
        with open(os.path.join(experiment_dir, 'best_model.pkl'), 'wb') as f:
            pickle.dump((encoder, decoder), f)
    if (epoch + 1) % 10 == 0:
        with open(os.path.join(experiment_dir, f"model_epoch{epoch+1}.pkl"), 'wb') as f:
            pickle.dump((encoder, decoder), f)
with open(os.path.join(experiment_dir, 'final_model.pkl'), 'wb') as f:
    pickle.dump((encoder, decoder), f)
print(f"Training complete! Final model saved to {os.path.join(experiment_dir, 'final_model.pkl')}")
