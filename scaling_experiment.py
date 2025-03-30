import foldtree2_ecddcd as ft2
from converter import pdbgraph
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
import pickle
import os
import time
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# Create directory for experiment results
experiment_dir = './scaling_experiment/'
os.makedirs(experiment_dir, exist_ok=True)

# Experiment configuration
hidden_sizes = [10, 20, 50, 100]  # Different hidden layer sizes to test
dataset_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]  # Dataset size fractions to test
num_epochs = 50  # Fixed number of epochs for each experiment
modeldir = './models/'
datadir = '../../datasets/'

# Fixed hyperparameters (based on original script)
batch_size = 10
num_embeddings = 40
embedding_dim = 20
learning_rate = 0.001  # Added learning rate hyperparameter
edgeweight = 0.1
xweight = 0.1
vqweight = 0.001
foldxweight = 0.001
fapeweight = 0.01
angleweight = 0.01
lddt_weight = 0.1
dist_weight = 0.01
clip_grad = True
ema = True

# Setting the seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load the full dataset once
full_dataset = pdbgraph.StructureDataset('structs_training_godnodemk5.h5')
converter = pdbgraph.PDB2PyG()

# Get a data sample to determine dimensions
sample_loader = DataLoader(full_dataset, batch_size=1)
data_sample = next(iter(sample_loader))
ndim = data_sample['res'].x.shape[1]
ndim_godnode = data_sample['godnode'].x.shape[1]

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Results storage
results = {
    'hidden_size': [],
    'dataset_fraction': [],
    'dataset_size': [],
    'epoch': [],
    'aa_loss': [],
    'edge_loss': [],
    'vq_loss': [],
    'total_loss': [],
    'training_time': []
}

# Function to initialize model weights
def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)

def run_experiment(hidden_size, dataset_fraction):
    print(f"\n=== Starting experiment with hidden_size={hidden_size}, dataset_fraction={dataset_fraction} ===")
    
    # Create dataset subset
    subset_size = int(len(full_dataset) * dataset_fraction)
    indices = torch.randperm(len(full_dataset))[:subset_size]
    subset_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    train_loader = DataLoader(
        subset_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        worker_init_fn=np.random.seed(0),
        num_workers=6
    )
    
    # Create writer for this experiment
    exp_name = f"hidden{hidden_size}_data{dataset_fraction}"
    writer = SummaryWriter(log_dir=f"{experiment_dir}/runs/{exp_name}")
    
    # Create models with specified hidden size
    encoder_layers = 2
    encoder = ft2.mk1_Encoder(
        in_channels=ndim, 
        hidden_channels=[hidden_size] * encoder_layers,
        out_channels=embedding_dim, 
        metadata={'edge_types': [
            ('res','contactPoints', 'res'), 
            ('res','backbone', 'res'),
            ('res','backbonerev', 'res')
        ]},
        num_embeddings=num_embeddings, 
        commitment_cost=0.9, 
        edge_dim=1,
        encoder_hidden=hidden_size * 4, 
        EMA=ema, 
        nheads=10, 
        dropout_p=0.005,
        reset_codes=False, 
        flavor='transformer'
    )
    
    decoder_layers = 3
    decoder = ft2.HeteroGAE_Decoder(
        in_channels={
            'res': encoder.out_channels, 
            'godnode4decoder': ndim_godnode,
            'foldx': 23
        },
        hidden_channels={
            ('res', 'window', 'res'): [hidden_size] * decoder_layers,
            ('res', 'backbone', 'res'): [hidden_size] * decoder_layers,
            ('res', 'backbonerev', 'res'): [hidden_size] * decoder_layers,
            ('res', 'informs', 'godnode4decoder'): [hidden_size] * decoder_layers,
        },
        layers=decoder_layers,
        metadata=converter.metadata,
        amino_mapper=converter.aaindex,
        concat_positions=False,
        flavor='transformer',
        output_foldx=True,
        geometry=False,
        denoise=False,
        Xdecoder_hidden=[hidden_size, hidden_size//2, hidden_size//5],
        PINNdecoder_hidden=[hidden_size//2, hidden_size//4, hidden_size//5],
        geodecoder_hidden=[hidden_size//3, hidden_size//3, hidden_size//3],
        AAdecoder_hidden=[hidden_size, hidden_size//5, hidden_size//10],
        contactdecoder_hidden=[hidden_size//2, hidden_size//4],
        nheads=10,
        dropout=0.005,
        residual=False,
        normalize=True,
        contact_mlp=False
    )
    
    # Apply weight initialization
    encoder.apply(init_weights)
    decoder.apply(init_weights)
    
    # Move models to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=learning_rate  # Use the learning rate from fixed hyperparameters
    )
    
    # Training preparation
    encoder.train()
    decoder.train()
    
    # Log parameters
    writer.add_text('Parameters', f'Model hidden size: {hidden_size}', 0)
    writer.add_text('Parameters', f'Dataset fraction: {dataset_fraction}', 0)
    writer.add_text('Parameters', f'Dataset size: {subset_size}', 0)
    writer.add_text('Parameters', f'Number of parameters: {sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())}', 0)
    writer.add_text('Parameters', f'Learning rate: {learning_rate}', 0)
    
    # Initialize models with one forward pass
    init_done = False
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss_x = 0
        total_loss_edge = 0
        total_vq = 0
        
        for data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data = data.to(device)
            
            # Initialize lazy modules if not done yet
            if not init_done:
                with torch.no_grad():
                    z, vqloss = encoder.forward(data)
                    data['res'].x = z
                    recon_x, edge_probs, zgodnode, foldxout, r, t, angles, r2, t2, angles2 = decoder(data, None)
                    init_done = True
                    continue
            
            optimizer.zero_grad()
            
            # Forward pass
            z, vqloss = encoder.forward(data)
            data['res'].x = z
            edgeloss, distloss = ft2.recon_loss(data, data.edge_index_dict[('res', 'contactPoints', 'res')], decoder, plddt=False, offdiag=False)
            recon_x, edge_probs, zgodnode, foldxout, r, t, angles, r2, t2, angles2 = decoder(data, None)
            
            # Compute amino acid reconstruction loss
            xloss = ft2.aa_reconstruction_loss(data['AA'].x, recon_x)
            
            # Compute total loss
            loss = xweight * xloss + edgeweight * edgeloss + vqweight * vqloss
            
            # Backpropagation
            loss.backward()
            
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            total_loss_x += xloss.item()
            total_loss_edge += edgeloss.item()
            total_vq += vqloss.item()
        
        # Log results
        avg_xloss = total_loss_x / len(train_loader)
        avg_edgeloss = total_loss_edge / len(train_loader)
        avg_vqloss = total_vq / len(train_loader)
        total_loss = avg_xloss * xweight + avg_edgeloss * edgeweight + avg_vqloss * vqweight
        
        writer.add_scalar('Loss/AA', avg_xloss, epoch)
        writer.add_scalar('Loss/Edge', avg_edgeloss, epoch)
        writer.add_scalar('Loss/VQ', avg_vqloss, epoch)
        writer.add_scalar('Loss/Total', total_loss, epoch)
        
        # Store results
        results['hidden_size'].append(hidden_size)
        results['dataset_fraction'].append(dataset_fraction)
        results['dataset_size'].append(subset_size)
        results['epoch'].append(epoch)
        results['aa_loss'].append(avg_xloss)
        results['edge_loss'].append(avg_edgeloss)
        results['vq_loss'].append(avg_vqloss)
        results['total_loss'].append(total_loss)
        results['training_time'].append(time.time() - start_time)
        
        print(f"Epoch {epoch+1}/{num_epochs}, AA Loss: {avg_xloss:.4f}, Edge Loss: {avg_edgeloss:.4f}, VQ Loss: {avg_vqloss:.4f}, Total Loss: {total_loss:.4f}")
    
    # Save model for this configuration
    model_path = f"{experiment_dir}/model_hidden{hidden_size}_data{dataset_fraction}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump((encoder, decoder), f)
    
    # Close writer
    writer.close()
    
    return avg_xloss, avg_edgeloss, avg_vqloss, total_loss

# Run all experiments
for hidden_size in hidden_sizes:
    for dataset_fraction in dataset_fractions:
        try:
            run_experiment(hidden_size, dataset_fraction)
        except Exception as e:
            print(f"Error in experiment with hidden_size={hidden_size}, dataset_fraction={dataset_fraction}: {e}")
            continue

# Create results dataframe
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv(f"{experiment_dir}/scaling_experiment_results.csv", index=False)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Scaling Experiment Results', fontsize=16)

# Plot 1: Loss vs Hidden Size for different dataset sizes (final epoch)
final_epochs = results_df.groupby(['hidden_size', 'dataset_fraction']).max('epoch')
for dataset_fraction in dataset_fractions:
    subset = final_epochs[final_epochs.index.get_level_values('dataset_fraction') == dataset_fraction]
    axes[0, 0].plot(
        subset.index.get_level_values('hidden_size'), 
        subset['total_loss'], 
        marker='o', 
        label=f'Dataset {int(dataset_fraction*100)}%'
    )
axes[0, 0].set_xlabel('Hidden Size')
axes[0, 0].set_ylabel('Final Total Loss')
axes[0, 0].legend()
axes[0, 0].set_title('Model Size vs Final Loss')

# Plot 2: Loss vs Dataset Size for different hidden sizes (final epoch)
for hidden_size in hidden_sizes:
    subset = final_epochs[final_epochs.index.get_level_values('hidden_size') == hidden_size]
    axes[0, 1].plot(
        subset.index.get_level_values('dataset_fraction'), 
        subset['total_loss'], 
        marker='o', 
        label=f'Hidden Size {hidden_size}'
    )
axes[0, 1].set_xlabel('Dataset Fraction')
axes[0, 1].set_ylabel('Final Total Loss')
axes[0, 1].legend()
axes[0, 1].set_title('Dataset Size vs Final Loss')

# Plot 3: Training time vs Model Size
time_data = results_df.groupby('hidden_size')['training_time'].max()
axes[1, 0].plot(time_data.index, time_data.values, marker='o')
axes[1, 0].set_xlabel('Hidden Size')
axes[1, 0].set_ylabel('Training Time (s)')
axes[1, 0].set_title('Training Time vs Model Size')

# Plot 4: AA Loss vs Dataset Size
for hidden_size in hidden_sizes:
    subset = final_epochs[final_epochs.index.get_level_values('hidden_size') == hidden_size]
    axes[1, 1].plot(
        subset.index.get_level_values('dataset_fraction'), 
        subset['aa_loss'], 
        marker='o', 
        label=f'Hidden Size {hidden_size}'
    )
axes[1, 1].set_xlabel('Dataset Fraction')
axes[1, 1].set_ylabel('Final AA Loss')
axes[1, 1].legend()
axes[1, 1].set_title('Dataset Size vs AA Reconstruction Loss')

plt.tight_layout()
plt.savefig(f"{experiment_dir}/scaling_experiment_results.png")
plt.close()

print(f"Scaling experiment completed. Results saved to {experiment_dir}")