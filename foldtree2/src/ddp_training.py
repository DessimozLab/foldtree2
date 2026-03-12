"""
Distributed Data Parallel training functions for FoldTree2.
This module contains the train_distributed function that can be properly
pickled and used with torch.multiprocessing.spawn()
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
import tqdm

# Import FoldTree2 modules
from foldtree2.src.pdbgraph import StructureDataset
from foldtree2.src.losses.losses import recon_loss_diag, aa_reconstruction_loss, angles_reconstruction_loss

def setup_ddp(rank, world_size, backend='nccl'):
	"""
	Initialize the distributed environment.
	
	Args:
		rank (int): Unique identifier of each process
		world_size (int): Total number of processes
		backend (str): Backend to use for distributed training ('nccl', 'gloo', 'mpi')
	"""
	import os
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12356'  # Use different port than Muon optimizer (12355)
	dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_ddp():
	"""
	Clean up the distributed environment.
	"""
	dist.destroy_process_group()


def train_distributed(rank, world_size, encoder, decoder, optimizer, scheduler,
					  dataset_path, num_epochs, batch_size, 
					  edgeweight, logitweight, xweight, fft2weight, vqweight, 
					  angles_weight, ss_weight,
					  mask_plddt=False, plddt_threshold=0.3,
					  mixed_precision=True, gradient_accumulation_steps=1,
					  clip_grad=True, scheduler_step_mode='epoch',
					  save_path='./models/checkpoint.pt'):
	"""
	Distributed training function that runs on each GPU.
	
	Args:
		rank (int): Unique identifier of each process (GPU)
		world_size (int): Total number of GPUs
		encoder: The encoder model
		decoder: The decoder model
		optimizer: The optimizer
		scheduler: Learning rate scheduler
		dataset_path (str): Path to HDF5 dataset file
		num_epochs (int): Number of training epochs
		batch_size (int): Batch size per GPU
		Loss weights for different components
		mask_plddt (bool): Whether to mask low pLDDT residues
		plddt_threshold (float): pLDDT threshold for masking
		mixed_precision (bool): Whether to use mixed precision training
		gradient_accumulation_steps (int): Number of gradient accumulation steps
		clip_grad (bool): Whether to clip gradients
		scheduler_step_mode (str): When to step the scheduler ('epoch' or 'step')
		save_path (str): Path to save checkpoints
	"""

	try:
		torch.autograd.set_detect_anomaly(True)

		# Setup distributed training
		setup_ddp(rank, world_size)
		
		# Set device for this process
		device = torch.device(f'cuda:{rank}')
		torch.cuda.set_device(device)
		
		# Move models to device
		encoder = encoder.to(device)
		decoder = decoder.to(device)
		
		# Wrap models with DDP
		# Set find_unused_parameters=True because encoder/decoder have conditional modules
		encoder = DDP(encoder, device_ids=[rank], find_unused_parameters=True)
		decoder = DDP(decoder, device_ids=[rank], find_unused_parameters=True)

	except:
		import traceback
		traceback.print_exc()
		raise

	# Create dataset in each process (avoids HDF5 pickling issues)
	train_dataset = StructureDataset(dataset_path)
	
	# Create distributed sampler
	train_sampler = DistributedSampler(
		train_dataset,
		num_replicas=world_size,
		rank=rank,
		shuffle=True
	)
	
	# Create data loader
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		sampler=train_sampler,
		num_workers=0,  # Keep 0 to avoid nested multiprocessing with DDP
		pin_memory=True
	)
	
	# Initialize gradient scaler for mixed precision
	scaler = GradScaler() if mixed_precision else None
	
	# Training loop
	for epoch in range(num_epochs):
		# Set epoch for sampler (ensures different shuffling each epoch)
		train_sampler.set_epoch(epoch)
		
		# Training metrics
		total_loss_x = 0
		total_loss_edge = 0
		total_vq = 0
		total_angles_loss = 0
		total_loss_fft2 = 0
		total_logit_loss = 0
		total_ss_loss = 0
		
		# Progress bar only on rank 0
		if rank == 0:
			pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
		else:
			pbar = train_loader
		
		for batch_idx, data in enumerate(pbar):
			data = data.to(device)
			


			try: 
				# Forward pass with autocast for mixed precision
				if mixed_precision:
					with autocast():
						z, vqloss = encoder(data)
						data['res'].x = z
						
						# Forward pass through decoder
						out = decoder(data, None)
						edge_index = data.edge_index_dict.get(('res', 'contactPoints', 'res'))
						
						# Edge reconstruction loss
						logitloss = torch.tensor(0.0, device=device)
						edgeloss = torch.tensor(0.0, device=device)
						if edge_index is not None:
							edgeloss, logitloss = recon_loss_diag(data, edge_index, decoder, plddt=mask_plddt, key='edge_probs')
						
						# Amino acid reconstruction loss
						xloss = aa_reconstruction_loss(data['AA'].x, out['aa'])
						
						# FFT2 loss
						fft2loss = torch.tensor(0.0, device=device)
						if 'fft2pred' in out and out['fft2pred'] is not None:
							fft2loss = F.smooth_l1_loss(
								torch.cat([data['fourier2dr'].x, data['fourier2di'].x], axis=1),
								out['fft2pred']
							)
						
						# Angles loss
						angles_loss = torch.tensor(0.0, device=device)
						if out.get('angles') is not None:
							angles_loss = angles_reconstruction_loss(
								out['angles'],
								data['bondangles'].x,
								plddt_mask=data['plddt'].x if mask_plddt else None
							)
						
						# Secondary structure loss
						ss_loss = torch.tensor(0.0, device=device)
						if out.get('ss_pred') is not None:
							if mask_plddt:
								mask = (data['plddt'].x >= plddt_threshold).squeeze()
								ss_loss = F.cross_entropy(out['ss_pred'][mask], data['ss'].x[mask])
							else:
								ss_loss = F.cross_entropy(out['ss_pred'], data['ss'].x)
						
						# Total loss
						loss = (xweight * xloss + edgeweight * edgeloss + vqweight * vqloss +
							fft2weight * fft2loss + angles_weight * angles_loss +
							ss_weight * ss_loss + logitweight * logitloss)
						
						# Scale loss by gradient accumulation steps
						loss = loss / gradient_accumulation_steps
				else:
					# Non-mixed precision path
					z, vqloss = encoder(data)
					data['res'].x = z
					
					out = decoder(data, None)
					edge_index = data.edge_index_dict.get(('res', 'contactPoints', 'res'))
					
					logitloss = torch.tensor(0.0, device=device)
					edgeloss = torch.tensor(0.0, device=device)
					if edge_index is not None:
						edgeloss, logitloss = recon_loss_diag(data, edge_index, decoder, plddt=mask_plddt, key='edge_probs')
					
					xloss = aa_reconstruction_loss(data['AA'].x, out['aa'])
					
					fft2loss = torch.tensor(0.0, device=device)
					if 'fft2pred' in out and out['fft2pred'] is not None:
						fft2loss = F.smooth_l1_loss(
							torch.cat([data['fourier2dr'].x, data['fourier2di'].x], axis=1),
							out['fft2pred']
						)
					
					angles_loss = torch.tensor(0.0, device=device)
					if out.get('angles') is not None:
						angles_loss = angles_reconstruction_loss(
							out['angles'],
							data['bondangles'].x,
							plddt_mask=data['plddt'].x if mask_plddt else None
						)
					
					ss_loss = torch.tensor(0.0, device=device)
					if out.get('ss_pred') is not None:
						if mask_plddt:
							mask = (data['plddt'].x >= plddt_threshold).squeeze()
							ss_loss = F.cross_entropy(out['ss_pred'][mask], data['ss'].x[mask])
						else:
							ss_loss = F.cross_entropy(out['ss_pred'], data['ss'].x)
					
					loss = (xweight * xloss + edgeweight * edgeloss + vqweight * vqloss +
						fft2weight * fft2loss + angles_weight * angles_loss +
						ss_weight * ss_loss + logitweight * logitloss)
					
					loss = loss / gradient_accumulation_steps
				
				# Backward pass with gradient scaling
				if mixed_precision:
					scaler.scale(loss).backward()
				else:
					loss.backward()
				
				# Only update weights every gradient_accumulation_steps
				if (batch_idx + 1) % gradient_accumulation_steps == 0:
					if clip_grad:
						if mixed_precision:
							scaler.unscale_(optimizer)
						torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
						torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
					
					# Step optimizer with scaler
					if mixed_precision:
						scaler.step(optimizer)
						scaler.update()
					else:
						optimizer.step()
					optimizer.zero_grad()
					
					# Step scheduler if it's a step-based scheduler
					if scheduler is not None and scheduler_step_mode == 'step':
						scheduler.step()
				
				# Accumulate losses (unscaled for reporting)
				total_loss_x += xloss.item()
				total_logit_loss += logitloss.item()
				total_loss_edge += edgeloss.item()
				total_loss_fft2 += fft2loss.item()
				total_angles_loss += angles_loss.item()
				total_vq += vqloss.item() if isinstance(vqloss, torch.Tensor) else float(vqloss)
				total_ss_loss += ss_loss.item()
			except:
				import traceback
				traceback.print_exc()
				raise
	

		# Clean up any remaining gradients at epoch end
		if len(train_loader) % gradient_accumulation_steps != 0:
			if clip_grad:
				if mixed_precision:
					scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
				torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
			if mixed_precision:
				scaler.step(optimizer)
				scaler.update()
			else:
				optimizer.step()
			optimizer.zero_grad()
		
		# Calculate average losses
		avg_loss_x = total_loss_x / len(train_loader)
		avg_loss_edge = total_loss_edge / len(train_loader)
		avg_loss_vq = total_vq / len(train_loader)
		avg_loss_fft2 = total_loss_fft2 / len(train_loader)
		avg_angles_loss = total_angles_loss / len(train_loader)
		avg_logit_loss = total_logit_loss / len(train_loader)
		avg_ss_loss = total_ss_loss / len(train_loader)
		avg_total_loss = (avg_loss_x + avg_loss_edge + avg_loss_vq +
						 avg_loss_fft2 + avg_angles_loss + avg_logit_loss + avg_ss_loss)
		
		# Synchronize losses across all GPUs for logging
		loss_tensor = torch.tensor([avg_total_loss], device=device)
		dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
		
		# Print metrics (only on rank 0)
		if rank == 0:
			print(f"Epoch {epoch+1}: AA Loss: {avg_loss_x:.4f}, "
				 f"Edge Loss: {avg_loss_edge:.4f}, VQ Loss: {avg_loss_vq:.4f}, "
				 f"FFT2 Loss: {avg_loss_fft2:.4f}, Angles Loss: {avg_angles_loss:.4f}, "
				 f"SS Loss: {avg_ss_loss:.4f}, Logit Loss: {avg_logit_loss:.4f}")
			current_lr = optimizer.param_groups[0]['lr']
			print(f"Total Loss: {avg_total_loss:.4f}, LR: {current_lr:.6f}")
		
		# Update learning rate scheduler (for epoch-based schedulers, only on rank 0)
		if rank == 0 and scheduler is not None and scheduler_step_mode == 'epoch':
			if hasattr(scheduler, 'step'):
				scheduler.step(avg_loss_x)
		
		# Save checkpoint (only on rank 0)
		if rank == 0 and (epoch + 1) % 10 == 0:
			checkpoint_path = save_path.replace('.pt', f'_epoch{epoch+1}.pt')
			torch.save({
				'epoch': epoch,
				'encoder_state_dict': encoder.module.state_dict(),
				'decoder_state_dict': decoder.module.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': avg_total_loss,
			}, checkpoint_path)
			print(f"Saved checkpoint to {checkpoint_path}")
	

	# Always cleanup distributed resources
	cleanup_ddp()
	if rank == 0:
		print("DDP cleanup completed")

	# Clean up
	cleanup_ddp()
