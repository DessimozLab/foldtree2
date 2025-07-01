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
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run scaling experiment with configurable parameters')
parser.add_argument('--dataset', '-d', type=str, default='structs_training_godnodemk5.h5',
					help='Path to the dataset file (default: structs_training_godnodemk5.h5)')
parser.add_argument('--hidden-size', '-hs', type=int, default=50,
					help='Hidden layer size (default: 50)')
parser.add_argument('--dataset-fractions', '-df', type=float, nargs='+', default=[ 0.25, 0.5, 0.75, 1.0],
					help='Dataset size fractions to test (default: 0.25 0.5 0.75 1.0)')
parser.add_argument('--epochs', '-e', type=int, default=50,
					help='Number of epochs for training (default: 50)')
parser.add_argument('--device', type=str, default=None,
					help='Device to run on (e.g., cuda:0, cuda:1, cpu) (default: auto-select)')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.00005,
					help='Learning rate (default: 0.001)')
parser.add_argument('--batch-size', '-bs', type=int, default=10,
					help='Batch size (default: 10)')
parser.add_argument('--output-dir', '-o', type=str, default='./scaling_experiment/',
					help='Directory to save experiment results (default: ./scaling_experiment/)')
parser.add_argument('--model-name', type=str, default='scaling_experiment_model',
                    help='Model name for saving (default: scaling_experiment_model)')
parser.add_argument('--num-embeddings', type=int, default=40,
                    help='Number of embeddings for the encoder (default: 40)')
parser.add_argument('--embedding-dim', type=int, default=20,
                    help='Embedding dimension for the encoder (default: 20)')
parser.add_argument('--overwrite', action='store_true', help='Overwrite saved model if exists, otherwise continue training')
parser.add_argument('--geometry', action='store_true', help='Train the model with geometry')
parser.add_argument('--fapeloss', action='store_true', help='Train the model with FAPE loss')
parser.add_argument('--lddtloss', action='store_true', help='Train the model with LDDT loss')
parser.add_argument('--concat-positions', action='store_true', help='Train the model with positional encoding')
parser.add_argument('--transformer', action='store_true', help='Train the model with transformer decoder')
parser.add_argument('--output-foldx', action='store_true', help='Train the model with foldx output')
args = parser.parse_args()

# Set defaults for these options if not provided
overwrite = args.overwrite
geometry = args.geometry
fapeloss = args.fapeloss
lddtloss = args.lddtloss
concat_positions = args.concat_positions
transformer = args.transformer
output_foldx = args.output_foldx
if transformer:
    concat_positions = True

# Use args for configuration
experiment_dir = args.output_dir
os.makedirs(experiment_dir, exist_ok=True)
modelname = args.model_name
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.epochs
num_embeddings = args.num_embeddings
embedding_dim = args.embedding_dim

# Fixed hyperparameters (based on original script)
edgeweight = .01
xweight = .1
vqweight = .0001
foldxweight = .001
fapeweight = .001
angleweight = .001
lddt_weight = .1
dist_weight = .01
err_eps = 1e-2
clip_grad = True
ema = True
denoise = True

# Setting the seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load the dataset
dataset_path = args.dataset
print(f"Loading dataset: {dataset_path}")
full_dataset = pdbgraph.StructureDataset(dataset_path)
converter = pdbgraph.PDB2PyG()

# Get a data sample to determine dimensions
sample_loader = DataLoader(full_dataset, batch_size=1)
data_sample = next(iter(sample_loader))
ndim = data_sample['res'].x.shape[1]
ndim_godnode = data_sample['godnode'].x.shape[1]

# Set device
if args.device:
    device = torch.device(args.device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
	'angle_loss': [],    # Added
	'fape_loss': [],     # Added
	'dist_loss': [],     # Added
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
	
	# Model loading logic (match learn.py)
	model_path = os.path.join(experiment_dir, f"{modelname}_hidden{hidden_size}_data{dataset_fraction}.pkl")
	if os.path.exists(model_path) and not overwrite:
		with open(model_path, 'rb') as f:
			encoder, decoder = pickle.load(f)
	else:
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
			concat_positions=concat_positions,
			flavor='sage',
			output_foldx=output_foldx,
			 geometry=geometry,      # Enable geometry prediction
			 denoise=denoise,        # Enable denoiser
			Xdecoder_hidden=[hidden_size, max(1, hidden_size//2), max(1, hidden_size//5)],
			PINNdecoder_hidden=[max(1, hidden_size//2), max(1, hidden_size//4), max(1, hidden_size//5)],
			geodecoder_hidden=[max(1, hidden_size//3), max(1, hidden_size//3), max(1, hidden_size//3)],
			AAdecoder_hidden=[hidden_size, max(1, hidden_size//2), max(1, hidden_size//2)],
			contactdecoder_hidden=[max(1, hidden_size//2), max(1, hidden_size//4)],
			nheads=10,
			dropout=0.005,
			residual=False,
			normalize=True,
			contact_mlp=True
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
		lr=learning_rate
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
	
	# Training preparation
	encoder.train()
	decoder.train()
	
	# Log parameters
	writer.add_text('Parameters', f'Model hidden size: {hidden_size}', 0)
	writer.add_text('Parameters', f'Dataset fraction: {dataset_fraction}', 0)
	writer.add_text('Parameters', f'Dataset size: {subset_size}', 0)
	writer.add_text('Parameters', f'Learning rate: {learning_rate}', 0)
	
	# Initialize lazy modules with a first batch before training
	print("Initializing lazy modules with first batch...")
	init_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
	data = next(iter(init_loader)).to(device)
	
	with torch.no_grad():
		#encoder.eval()
		#decoder.eval()
		z,vqloss = encoder.forward(data )
		data['res'].x = z
		recon_x , edge_probs , zgodnode , foldxout, r , t , angles , r2,t2, angles2 = decoder(  data , None  ) 
		writer.add_text('Parameters', f'Number of parameters: {sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())}', 0)

	encoder.train()
	decoder.train()
	print("Initialization complete.")
	
	# Training loop (learn_monodecoder.py style)
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
				edgeloss, _ = ft2.recon_loss_diag(data, edge_index, decoder, plddt=True, offdiag=True, key='edge_probs')
			else:
				edgeloss = torch.tensor(0.0, device=device)
			xloss = ft2.aa_reconstruction_loss(data['AA'].x, recon_x)
			if fft2_x is not None:
				# FFT2 loss: real and imaginary parts
				F_hat = torch.complex(fft2_x[:, :fft2_x.shape[1]//2], fft2_x[:, fft2_x.shape[1]//2:])
				F = torch.complex(data['fourier2dr'].x, data['fourier2di'].x)
				mag_loss = torch.mean(torch.abs((torch.abs(F_hat) - torch.abs(F))))
				phase_loss = torch.mean(torch.abs((torch.angle(F_hat) - torch.angle(F))))
				ff