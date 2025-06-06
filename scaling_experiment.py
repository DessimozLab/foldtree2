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
	
	# Training loop
	start_time = time.time()
	for epoch in range(num_epochs):
		total_loss_x = 0
		total_loss_edge = 0
		total_vq = 0
		total_angle = 0
		total_fape = 0
		total_dist = 0
		total_foldx = 0
		total_lddt = 0
		for data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
			data = data.to(device)
			
			optimizer.zero_grad()
			
			# Forward pass (no need for initialization check now)
			z, vqloss = encoder.forward(data)
			data['res'].x = z
			edgeloss, distloss = ft2.recon_loss(data, data.edge_index_dict[('res', 'contactPoints', 'res')], decoder, plddt=False, offdiag=False)
			recon_x, edge_probs, zgodnode, foldxout, r, t, angles, r2, t2, angles2 = decoder(data, None)
			
			# Compute amino acid reconstruction loss
			xloss = ft2.aa_reconstruction_loss(data['AA'].x, recon_x)
			
			# Foldx loss
			if output_foldx and 'Foldx' in data:
				data['Foldx'].x = data['Foldx'].x.view(-1, 23)
				if hasattr(decoder, 'bn_foldx'):
					data['Foldx'].x = decoder.bn_foldx(data['Foldx'].x)
				foldxout = foldxout.view(data['Foldx'].x.shape)
				foldxloss = F.smooth_l1_loss(foldxout, data['Foldx'].x)
			else:
				foldxloss = torch.tensor(0.0, device=device)
			
			# Geometry losses
			if geometry:
				# Angle loss with reduction='none' first
				angleloss = F.smooth_l1_loss(angles, data.x_dict['bondangles'], reduction='none')
				fploss = torch.tensor(0.0, device=device)
				lddt_loss = torch.tensor(0.0, device=device)
				
				# Additional calculations if denoise is enabled
				if denoise:
					# Add second angle loss from denoised prediction
					angleloss += F.smooth_l1_loss(angles2, data.x_dict['bondangles'], reduction='none')
					
					# FAPE loss using the denoised predictions (r2, t2)
					if fapeloss and 't_true' in data and 'R_true' in data:
						batch_data = data['t_true'].batch if hasattr(data['t_true'], 'batch') else None
						t_true = data['t_true'].x
						R_true = data['R_true'].x
						
						# Use r2, t2 for the denoised predictions
						fploss = ft2.fape_loss(
							true_R=R_true,
							true_t=t_true,
							pred_R=r2,
							pred_t=t2,
							batch=batch_data,
							d_clamp=10.0,
							eps=1e-8,
							plddt=None,
							soft=False
						)
					if lddtloss and 'coords' in data:
						lddt_loss = ft2.lddt_loss(
							coord_true=data['coords'].x,
							pred_R=r,
							pred_t=t,
							batch=data['res'].batch
						)
				
				# Take the mean of angle loss at the end
				angleloss = angleloss.mean()
			else:
				angleloss = torch.tensor(0.0, device=device)
				fploss = torch.tensor(0.0, device=device)
				lddt_loss = torch.tensor(0.0, device=device)
			
			# Compute total loss including new components
			loss = (xweight * xloss + 
					edgeweight * edgeloss + 
					vqweight * vqloss + 
					foldxweight * foldxloss +
					fapeweight * fploss + 
					angleweight * angleloss + 
					lddt_weight * lddt_loss +
					dist_weight * distloss)
			
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
			total_angle += angleloss.item()
			total_fape += fploss.item()
			total_dist += distloss.item()
			total_foldx += foldxloss.item()
			total_lddt += lddt_loss.item()
		
		# Scheduler step
		scheduler.step(total_loss_x)
		
		# Log results
		avg_xloss = total_loss_x / len(train_loader)
		avg_edgeloss = total_loss_edge / len(train_loader)
		avg_vqloss = total_vq / len(train_loader)
		avg_angleloss = total_angle / len(train_loader)
		avg_fapeloss = total_fape / len(train_loader)
		avg_distloss = total_dist / len(train_loader)
		avg_foldxloss = total_foldx / len(train_loader)
		avg_lddtloss = total_lddt / len(train_loader)
		
		total_loss = (avg_xloss * xweight +
					  avg_edgeloss * edgeweight +
					  avg_vqloss * vqweight +
					  avg_foldxloss * foldxweight +
					  avg_angleloss * angleweight +
					  avg_fapeloss * fapeweight +
					  avg_lddtloss * lddt_weight +
					  avg_distloss * dist_weight)
		
		# Add new metrics to tensorboard
		writer.add_scalar('Loss/AA', avg_xloss, epoch)
		writer.add_scalar('Loss/Edge', avg_edgeloss, epoch)
		writer.add_scalar('Loss/VQ', avg_vqloss, epoch)
		writer.add_scalar('Loss/Foldx', avg_foldxloss, epoch)
		writer.add_scalar('Loss/FAPE', avg_fapeloss, epoch)
		writer.add_scalar('Loss/Angle', avg_angleloss, epoch)
		writer.add_scalar('Loss/LDDT', avg_lddtloss, epoch)
		writer.add_scalar('Loss/Dist', avg_distloss, epoch)
		writer.add_scalar('Loss/Total', total_loss, epoch)
		
		# Store results
		results['hidden_size'].append(hidden_size)
		results['dataset_fraction'].append(dataset_fraction)
		results['dataset_size'].append(subset_size)
		results['epoch'].append(epoch)
		results['aa_loss'].append(avg_xloss)
		results['edge_loss'].append(avg_edgeloss)
		results['vq_loss'].append(avg_vqloss)
		results['angle_loss'].append(avg_angleloss)
		results['fape_loss'].append(avg_fapeloss)
		results['dist_loss'].append(avg_distloss)
		results['total_loss'].append(total_loss)
		results['training_time'].append(time.time() - start_time)
		
		print(f"Epoch {epoch+1}/{num_epochs}, AA Loss: {avg_xloss:.4f}, Edge Loss: {avg_edgeloss:.4f}, VQ Loss: {avg_vqloss:.4f}, Foldx Loss: {avg_foldxloss:.4f}")
		print(f"Angle Loss: {avg_angleloss:.4f}, FAPE Loss: {avg_fapeloss:.4f}, LDDT Loss: {avg_lddtloss:.4f}, Dist Loss: {avg_distloss:.4f}, Total Loss: {total_loss:.4f}")
	
	# Save model for this configuration
	with open(model_path, 'wb') as f:
		pickle.dump((encoder, decoder), f)
	
	# Close writer
	writer.close()
	
	return avg_xloss, avg_edgeloss, avg_vqloss, total_loss

if __name__ == "__main__":
	# Run all experiments
	for hidden_size in hidden_sizes:
		for dataset_fraction in dataset_fractions:
			try:
				run_experiment(hidden_size, dataset_fraction)
			except Exception as e:
				import traceback
				#print traceback
				print( 		traceback.format_exc()		)
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