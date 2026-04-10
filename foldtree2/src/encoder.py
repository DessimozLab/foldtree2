#!/usr/bin/env python
# coding: utf-8

import importlib
import os
import queue
import threading
from foldtree2.src.pdbgraph import StructureDataset
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch_geometric.nn import (
    GATv2Conv,
    GraphNorm,
    JumpingKnowledge,
    SAGEConv,
    TransformerConv,
)
from foldtree2.src.layers import *
from foldtree2.src.losses import *
from foldtree2.src.quantizers import *
import copy
from torch_geometric.loader import DataLoader

# Note: datadir is defined but may not be used throughout the module
datadir = '../../datasets/foldtree2/'
#encoder super class

class mk1_Encoder(torch.nn.Module):
	"""
	Muon-compatible encoder with modular architecture.
	Separates input, body, and head modules for compatibility with Muon optimizer.
	
	- input: Preprocessing and initial MLPs (optimized with AdamW)
	- body: Graph convolution layers (weights optimized with Muon, gains/biases with AdamW)
	- head: Output projection and quantization (optimized with AdamW)
	
	Args:
		in_channels (int): Number of input node features (e.g., from aaindex properties).
		hidden_channels (list[int]): Hidden layer sizes for graph convolutions.
		out_channels (int): Dimensionality of the output embedding / codebook vectors.
		num_embeddings (int): Number of discrete codes in the VQ codebook (alphabet size).
		commitment_cost (float): VQ-VAE commitment loss weight (beta).
		metadata (dict): Graph metadata with 'edge_types' for heterogeneous graphs.
		edge_dim (int): Edge feature dimension for TransformerConv (default: 1).
		encoder_hidden (int): Hidden size for final MLP before quantization (default: 100).
		dropout_p (float): Dropout probability throughout the encoder (default: 0.05).
		EMA (bool): Use EMA codebook updates instead of gradient descent (default: False).
		reset_codes (bool): Reset unused codebook entries during training (default: True).
		nheads (int): Number of attention heads for GAT/Transformer convs (default: 3).
		flavor (str): GNN type - 'sage', 'gat', or 'transformer' (default: 'sage').
		fftin (bool): Use Fourier feature input branch (default: False).
		use_commitment_scheduling (bool): Enable commitment cost warmup (default: False).
		commitment_warmup_steps (int): Steps for commitment schedule warmup (default: 5000).
		commitment_schedule (str): Schedule type - 'cosine' or 'linear' (default: 'cosine').
		commitment_start (float): Initial commitment cost during warmup (default: 0.1).
		concat_positions (bool): Concatenate raw positional encodings to input (default: True).
		learn_positions (bool): Use learned MLP for positional encoding (default: True).
		**kwargs: Additional arguments (stored but not used).
	"""
	
	def __init__(
		self,
		# === Architecture ===
		in_channels: int,
		hidden_channels: list,
		out_channels: int,
		# === Vector Quantization ===
		num_embeddings: int,
		commitment_cost: float,
		# === Graph Structure ===
		metadata: dict = {},
		edge_dim: int = 1,
		# === Network Config ===
		encoder_hidden: int = 100,
		dropout_p: float = 0.05,
		nheads: int = 3,
		flavor: str = 'sage',
		# === Codebook Options ===
		EMA: bool = False,
		reset_codes: bool = True,
		# === Commitment Scheduling ===
		use_commitment_scheduling: bool = False,
		commitment_warmup_steps: int = 5000,
		commitment_schedule: str = 'cosine',
		commitment_start: float = 0.1,
		# === Position Encoding ===
		concat_positions: bool = True,
		learn_positions: bool = True,
		# === Optional Features ===
		fftin: bool = False,
		**kwargs
	):
		super(mk1_Encoder, self).__init__()

		# Save all arguments to constructor
		self.args = locals()
		self.args.pop('self')

		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		
		self.num_embeddings = num_embeddings
		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.encoder_hidden = encoder_hidden
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.concat_positions = concat_positions
		self.learn_positions = learn_positions
		self.edge_dim = edge_dim
		self.fftin = fftin
		


		# Determine input channels
		self.in_channels = in_channels
		
		self.in_with_positions = copy.deepcopy(self.in_channels)
		if self.concat_positions and self.learn_positions==False:
			self.in_with_positions += 256
		if self.learn_positions == True:
			self.concat_positions = False
			self.position_mlp = Position_MLP(in_channels=256, hidden_channels=[256,256, 256], out_channels=64, dropout=0.05)
			self.in_with_positions += 64
		else:
			self.position_mlp = None

		# ===================== INPUT MODULE =====================
		# Preprocessing layers: LayerNorm, input MLPs, dropout
		self.input = nn.ModuleDict()
		

		self.input['dropout'] = nn.Dropout(p=dropout_p)
		self.input['ln'] = nn.LayerNorm(self.in_channels, eps=1e-6)
		print(self.input['ln'])

		self.input['inmlp'] = nn.Sequential(
			nn.Dropout(dropout_p),
			nn.Linear(self.in_with_positions, hidden_channels[0] * 2),
			nn.GELU(),
			nn.Linear(hidden_channels[0] * 2, hidden_channels[0]),
			nn.GELU(),
		)
		
		if self.fftin:
			self.input['ffin'] = nn.Sequential(
				nn.Dropout(dropout_p),
				nn.LayerNorm(2 * 80, eps=1e-6),
				nn.Linear(2 * 80, hidden_channels[0] * 2),
				nn.GELU(),
				nn.Linear(hidden_channels[0] * 2, hidden_channels[0]),
				nn.GELU()
			)

		# ===================== BODY MODULE =====================
		# Graph convolution layers and normalization
		self.body = nn.ModuleDict()
		
		self.body['convs'] = nn.ModuleList()
		self.body['norms'] = nn.ModuleList()
		
		for i in range(1, len(hidden_channels)):
			if flavor == 'gat':
				conv_dict = nn.ModuleDict({
					'_'.join(edge_type): GATv2Conv(
						hidden_channels[i-1], 
						hidden_channels[i], 
						heads=nheads, 
						dropout=dropout_p,
						concat=False
					)
					for edge_type in metadata['edge_types']
				})
			elif flavor == 'transformer':
				conv_dict = nn.ModuleDict({
					'_'.join(edge_type): TransformerConv(
						hidden_channels[i-1], 
						hidden_channels[i],
						edge_dim=self.edge_dim,  
						heads=nheads, 
						dropout=dropout_p,
						concat=False
					)
					for edge_type in metadata['edge_types']
				})
			elif flavor == 'sage':
				conv_dict = nn.ModuleDict({
					'_'.join(edge_type): SAGEConv(
						hidden_channels[i-1], 
						hidden_channels[i]
					)
					for edge_type in metadata['edge_types']
				})
			else:
				raise ValueError(f"Unknown flavor: {flavor}")
			
			self.body['convs'].append(conv_dict)
			self.body['norms'].append(GraphNorm(hidden_channels[i]))
		
		self.body['jk'] = JumpingKnowledge(mode='cat')

		# ===================== HEAD MODULE =====================
		# Output projection and quantization
		self.head = nn.ModuleDict()
		
		self.head['lin'] = nn.Sequential(
			nn.Linear(hidden_channels[-1] * (len(hidden_channels) - 1), self.encoder_hidden),
			nn.GELU(),
			nn.Linear(self.encoder_hidden, self.encoder_hidden),
			nn.GELU(),
		)
		
		self.head['out_dense'] = nn.Sequential(
			nn.Linear(self.encoder_hidden + 20, self.encoder_hidden),
			nn.GELU(),
			nn.Linear(self.encoder_hidden, self.encoder_hidden),
			nn.GELU(),
			nn.Linear(self.encoder_hidden, self.out_channels),
			nn.Tanh()
		)
		
		# Vector quantizer
		if EMA:
			self.vector_quantizer = VectorQuantizerEMA(
				num_embeddings, out_channels, commitment_cost, 
				reset=reset_codes,
				use_commitment_scheduling=use_commitment_scheduling,
				commitment_warmup_steps=commitment_warmup_steps,
				commitment_schedule=commitment_schedule,
				commitment_start=commitment_start,
				commitment_end=commitment_cost
			)
		else:
			self.vector_quantizer = VectorQuantizer(num_embeddings, out_channels, commitment_cost)
	
	def forward(self, data, edge_attr_dict=None, **kwargs):
		x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
		
		# ===================== INPUT PROCESSING =====================
		# Ensure input is contiguous and has the correct dtype for LayerNorm
		# This fixes DeepSpeed FP16 compatibility issues
		x_dict['res'] = x_dict['res'].contiguous()
		
		if 'debug' in kwargs and kwargs['debug']:
			print(f"Input shape: {x_dict['res'].shape}, dtype: {x_dict['res'].dtype}")
			print(f"LayerNorm normalized_shape: {self.input['ln'].normalized_shape}")
		
		# Cast to float32 for LayerNorm (required for numerical stability with FP16)
		# Also ensure LayerNorm weights are in FP32
		original_dtype = x_dict['res'].dtype
		if original_dtype != torch.float32:
			# Temporarily move LayerNorm to float32 if needed
			ln = self.input['ln'].float()
			x_dict['res'] = ln(x_dict['res'].float()).to(original_dtype)
		else:
			x_dict['res'] = self.input['ln'](x_dict['res'])
	

		if self.concat_positions and not self.learn_positions:
			x_dict['res'] = torch.cat([x_dict['res'], data['positions'].x], dim=1)
		if self.learn_positions == True and self.position_mlp is not None:
			pos_enc = self.position_mlp(data['positions'].x)
			
			x_dict['res'] = torch.cat([x_dict['res'], pos_enc], dim=1)
		
		if 'debug' in kwargs and kwargs['debug']:
			print(x_dict['res'].shape, 'x_dict[res] shape')
		
		x = self.input['inmlp'](x_dict['res'])
		# Add fourier features if present
		if self.fftin and 'ffin' in self.input:
			x += self.input['ffin'](torch.cat([data['fourier1dr'].x, data['fourier1di'].x], dim=1))
		
		# ===================== BODY PROCESSING =====================
		x_save = []
		for i, convs in enumerate(self.body['convs']):
			# Apply graph convolutions and average over all edge types
			if edge_attr_dict is not None:
				x_list = []
				for edge_type, conv in convs.items():
					edge_key = tuple(edge_type.split('_'))
					edge_attr = edge_attr_dict[edge_key]

					# Normalize edge attributes to [num_edges, edge_dim] for TransformerConv.
					if edge_attr is not None and edge_attr.dim() == 1:
						edge_attr = edge_attr.unsqueeze(-1)
					if edge_attr is not None and edge_attr.size(-1) != self.edge_dim:
						if edge_attr.size(-1) > self.edge_dim:
							edge_attr = edge_attr[:, :self.edge_dim]
						else:
							pad_cols = self.edge_dim - edge_attr.size(-1)
							edge_attr = F.pad(edge_attr, (0, pad_cols))

					x_list.append(conv(x, edge_index=edge_index_dict[edge_key], edge_attr=edge_attr))
			else:
				x_list = [conv(x, edge_index=edge_index_dict[tuple(edge_type.split('_'))]) 
						 for edge_type, conv in convs.items()]
			
			x = torch.stack(x_list, dim=0).mean(dim=0)
			x = F.gelu(x)
			x = self.body['norms'][i](x)
			x_save.append(x)
		
		x = self.body['jk'](x_save)
		
		# ===================== HEAD PROCESSING =====================
		x = self.head['lin'](x)
		x = self.head['out_dense'](torch.cat([x, x_dict['AA']], dim=1))
		batch = data['res'].batch if hasattr(data['res'], 'batch') and data['res'].batch is not None else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
		
		# Quantization
		z_quantized, vq_loss = self.vector_quantizer(x , batch=batch)
		
		if 'debug' in kwargs and kwargs['debug']:
			print('z_quantized shape:', z_quantized.shape)
			print('vq_loss:', vq_loss)
			print('x shape:', x.shape)
			print('x_dict keys:', x_dict.keys())
			print('edge_index_dict keys:', edge_index_dict.keys())
		
		return z_quantized, vq_loss


	def encode_structures(self, device, dataset_path):
		"""Encode protein structures using trained model."""
		if os.path.exists(dataset_path):
			print(f"Loading dataset from {dataset_path}")
			struct_dat = StructureDataset(dataset_path)
		else:
			print(f"Dataset {dataset_path} not found!")
			return None
		
		print(f"Loaded {len(struct_dat)} structures")
		encoder_loader = DataLoader(struct_dat, batch_size=1, shuffle=False)
		
		def databatch2list(loader):
			for data in loader:
				data = data.to_data_list()
				for d in data:
					d = d.to(device)
					yield d
		
		encoder_loader = databatch2list(encoder_loader)
		
		# Encode structures
		output_path = os.path.join(output_dir, modelname + '_aln_encoded.fasta')
		self.encode_structures_fasta(encoder_loader, output_path, replace=True)
		print(f"Encoded structures saved to {output_path}")
		return output_path


	def encode_structures_fasta(self, dataloader, filename=None, 
							   verbose=False, alphabet=None, replace=True, **kwargs):
		"""
		Write an encoded fasta for use with mafft and raxmlng. 
		Only doable with alphabet size of less than 248.
		"""
		# Replace characters with special characters to avoid issues with fasta format
		replace_dict = {
			chr(0): chr(246), '"': chr(248), '#': chr(247), '>': chr(249), 
			'=': chr(250), '<': chr(251), '-': chr(252), ' ': chr(253), 
			'\r': chr(254), '\n': chr(255)
		}
		
		if filename is None:
			raise ValueError('Filename must be provided for fasta encoding')

		# Check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		
		if alphabet is not None:
			print('using alphabet')
			print(alphabet)
		
		with open(filename, 'w') as f:
			for i, data in tqdm.tqdm(enumerate(dataloader), desc='Encoding structures to FASTA'):
				if 'debug' in kwargs and kwargs['debug']:
					print('res shape', data['res'].x.shape[0])
				
				data = data.to(self.device)
				z, qloss = self.forward(data)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = data.identifier
				f.write(f'>{identifier}\n')
				outstr = ''
				
				for char in strdata[0]:
					# Start at 0x01
					if alphabet is not None:
						char = alphabet[char]
					else:
						char = chr(char + 1)
					if replace and char in replace_dict:
						char = replace_dict[char]
					outstr += char
				
				if 'debug' in kwargs and kwargs['debug']:
					print('len outstring', len(outstr))
					assert len(outstr) == data['res'].x.shape[0], \
						f"Output string length {len(outstr)} does not match AA length {data['res'].x.shape[0]} for identifier {identifier}"
					assert len(outstr) == data['AA'].x.shape[0], \
						f"Output string length {len(outstr)} does not match AA length {data['AA'].x.shape[0]} for identifier {identifier}"
					assert len(outstr) == z.shape[0], \
						f"Output string length {len(outstr)} does not match z shape {z.shape[0]} for identifier {identifier}"
				
				f.write(outstr)
				f.write('\n')
		
		return filename
	
	def encode_structures_batched(self, device, dataset_path, batch_size=8):
		"""Encode protein structures using trained model.
		
		Args:
			device: Device to use for encoding
			dataset_path: Path to the HDF5 dataset file
			batch_size: Number of structures to process in parallel (default: 8)
		"""
		
		if os.path.exists(dataset_path):
			print(f"Loading dataset from {dataset_path}")
			struct_dat = StructureDataset(dataset_path)
		else:
			print(f"Dataset {dataset_path} not found!")
			return None
		
		print(f"Loaded {len(struct_dat)} structures")
		encoder_loader = DataLoader(struct_dat, batch_size=batch_size, shuffle=False)
		
		# Encode structures
		output_path = os.path.join(output_dir, modelname + '_aln_encoded.fasta')
		self.encode_structures_fasta_batched(encoder_loader, output_path, replace=True)
		print(f"Encoded structures saved to {output_path}")
		return output_path
	
	def encode_structures_fasta_batched(self, dataloader, filename=None, 
								verbose=False, alphabet=None, replace=True, **kwargs):
		"""
		Write an encoded fasta for use with mafft and raxmlng. 
		Only doable with alphabet size of less than 248.
		
		Args:
			dataloader: DataLoader yielding batches of protein structures
			filename: Output FASTA filename
			verbose: Print debug information
			alphabet: Optional custom alphabet mapping
			replace: Whether to replace special characters
		"""
		# Replace characters with special characters to avoid issues with fasta format
		replace_dict = {
			chr(0): chr(246), '"': chr(248), '#': chr(247), '>': chr(249), 
			'=': chr(250), '<': chr(251), '-': chr(252), ' ': chr(253), 
			'\r': chr(254), '\n': chr(255)
		}
		
		if filename is None:
			raise ValueError('Filename must be provided for fasta encoding')

		# Check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		
		if alphabet is not None:
			print('using alphabet')
			print(alphabet)
		
		with open(filename, 'w') as f:
			for batch in tqdm.tqdm(dataloader, desc='Encoding structures to FASTA'):
				# Move batch to device
				batch = batch.to(self.device)
				
				# Forward pass on entire batch
				z, qloss = self.forward(batch)
				
				# Discretize all structures in batch
				strdata_indices, strdata_chars = self.vector_quantizer.discretize_z(z)
				
				# Get batch information
				if hasattr(batch['res'], 'batch') and batch['res'].batch is not None:
					batch_indices = batch['res'].batch
					num_graphs = batch_indices.max().item() + 1
				else:
					# Single graph case
					num_graphs = 1
					batch_indices = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
				
				# Process each structure in the batch
				for graph_idx in range(num_graphs):
					# Get mask for this graph
					mask = (batch_indices == graph_idx)
					graph_strdata = strdata_indices[mask]
					
					# Get identifier - handle both batched and single structure cases
					if hasattr(batch, 'identifier'):
						if isinstance(batch.identifier, list):
							identifier = batch.identifier[graph_idx]
						else:
							identifier = batch.identifier
					else:
						identifier = f"structure_{graph_idx}"
					
					# Write FASTA header
					f.write(f'>{identifier}\n')
					outstr = ''
					
					# Convert indices to characters
					for char_idx in graph_strdata:
						# Start at 0x01
						if alphabet is not None:
							char = alphabet[char_idx.item()]
						else:
							char = chr(char_idx.item() + 1)
						if replace and char in replace_dict:
							char = replace_dict[char]
						outstr += char
					
					# Debug assertions
					if 'debug' in kwargs and kwargs['debug']:
						print('len outstring', len(outstr))
						graph_res_x = batch['res'].x[mask]
						graph_aa_x = batch['AA'].x[mask]
						print(f"Graph {graph_idx}: res={graph_res_x.shape[0]}, AA={graph_aa_x.shape[0]}, z={mask.sum().item()}")
						assert len(outstr) == graph_res_x.shape[0], \
							f"Output string length {len(outstr)} does not match AA length {graph_res_x.shape[0]} for identifier {identifier}"
						assert len(outstr) == graph_aa_x.shape[0], \
							f"Output string length {len(outstr)} does not match AA length {graph_aa_x.shape[0]} for identifier {identifier}"
						assert len(outstr) == mask.sum().item(), \
							f"Output string length {len(outstr)} does not match z shape {mask.sum().item()} for identifier {identifier}"
					
					# Write sequence
					f.write(outstr)
					f.write('\n')
		
		return filename

	def encode_foldcomp_fasta(
		self,
		foldcomp_db,
		filename,
		ids=None,
		max_structures=None,
		chunk_size=1024,
		queue_size=4,
		batch_size=16,
		cache_size=0,
		replace=True,
		alphabet=None,
		verbose=False,
	):
		"""Encode a Foldcomp DB directly to token FASTA using chunked prefetch.

		This is intended for very large databases (e.g., AlphaFold DB). It avoids
		materializing an HDF5 graph dataset and streams chunks into encoder batches.
		"""
		from foldtree2.src.pdbgraphmk2 import FoldcompStructureDataset, _load_foldcomp_ids

		if filename is None:
			raise ValueError('Filename must be provided for fasta encoding')
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')

		replace_dict = {
			chr(0): chr(246), '"': chr(248), '#': chr(247), '>': chr(249),
			'=': chr(250), '<': chr(251), '-': chr(252), ' ': chr(253),
			'\r': chr(254), '\n': chr(255)
		}

		if ids is None:
			ids = _load_foldcomp_ids(foldcomp_db)
		else:
			ids = list(ids)

		if max_structures is not None:
			ids = ids[:int(max_structures)]

		if len(ids) == 0:
			raise ValueError('No Foldcomp IDs provided/found to encode')

		dataset = FoldcompStructureDataset(
			foldcomp_db,
			ids=ids,
			converter=None,
			cache_size=int(cache_size),
			persistent_db=False,
		)

		q = queue.Queue(maxsize=max(1, int(queue_size)))
		sentinel = object()

		def _producer():
			try:
				for i in range(0, len(ids), int(chunk_size)):
					chunk_ids = ids[i:i + int(chunk_size)]
					graphs = dataset.get_many(chunk_ids)
					q.put(graphs)
			except Exception as e:
				q.put(e)
			finally:
				q.put(sentinel)

		producer = threading.Thread(target=_producer, daemon=True)
		producer.start()

		encoded_count = 0
		with open(filename, 'w') as f:
			pbar = tqdm.tqdm(total=len(ids), desc='Encoding Foldcomp DB to FASTA', disable=not verbose)
			while True:
				item = q.get()
				if item is sentinel:
					break
				if isinstance(item, Exception):
					raise item

				chunk_graphs = item
				batch_loader = DataLoader(chunk_graphs, batch_size=max(1, int(batch_size)), shuffle=False)

				for batch in batch_loader:
					batch = batch.to(self.device)
					z, _ = self.forward(batch)
					strdata_indices, _ = self.vector_quantizer.discretize_z(z)

					if hasattr(batch['res'], 'batch') and batch['res'].batch is not None:
						batch_indices = batch['res'].batch
						num_graphs = int(batch_indices.max().item()) + 1
					else:
						num_graphs = 1
						batch_indices = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)

					for graph_idx in range(num_graphs):
						mask = (batch_indices == graph_idx)
						graph_tokens = strdata_indices[mask]

						if hasattr(batch, 'identifier'):
							if isinstance(batch.identifier, list):
								identifier = batch.identifier[graph_idx]
							else:
								identifier = batch.identifier
						else:
							identifier = f'structure_{encoded_count}'

						f.write(f'>{identifier}\n')
						outstr = ''
						for char_idx in graph_tokens:
							if alphabet is not None:
								char = alphabet[char_idx.item()]
							else:
								char = chr(char_idx.item() + 1)
							if replace and char in replace_dict:
								char = replace_dict[char]
							outstr += char
						f.write(outstr + '\n')
						encoded_count += 1
						pbar.update(1)

			pbar.close()

		producer.join()
		return filename


class AttentionPooling(nn.Module):
	def __init__(self, embedding_dim, hidden_dim):
		super(AttentionPooling, self).__init__()
		self.fc = nn.Linear(embedding_dim, hidden_dim)
		self.attention = nn.Linear(hidden_dim, 1)

	def forward(self, token_embeddings, mask=None):
		scores = torch.tanh(self.fc(token_embeddings))
		scores = self.attention(scores).squeeze(-1)
		if mask is not None:
			scores = scores.masked_fill(mask == 0, float('-inf'))
		attn_weights = F.softmax(scores, dim=-1)
		pooled_embedding = torch.sum(token_embeddings * attn_weights.unsqueeze(-1), dim=0)
		
		return pooled_embedding


def save_model(model, optimizer, epoch, file_path):
	"""
	Save the model's state dictionary, optimizer's state dictionary, and other metadata to a file.

	Args:
		model (torch.nn.Module): The model to save.
		optimizer (torch.optim.Optimizer): The optimizer used for training.
		epoch (int): The current epoch number.
		file_path (str): The file path to save the model to.
	"""
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'model_class': model.__class__.__name__,
		'model_args': model.args,
		'model_kwargs': model.kwargs,
	}, file_path)


def load_model(file_path):
	"""
	Load the model's state dictionary, optimizer's state dictionary, and other metadata from a file.

	Args:
		file_path (str): The file path to load the model from.

	Returns:
		model (torch.nn.Module): The loaded model.
		optimizer (torch.optim.Optimizer): The loaded optimizer.
		epoch (int): The epoch number to resume training from.
	"""
	checkpoint = torch.load(file_path)

	# Dynamically import the module containing the model class
	model_module = importlib.import_module(__name__)

	# Instantiate the model with the saved arguments
	model_class = getattr(model_module, checkpoint['model_class'])
	model = model_class(*checkpoint['model_args'], **checkpoint['model_kwargs'])

	# Load the saved state dictionary into the model
	model.load_state_dict(checkpoint['model_state_dict'])

	# Assuming the optimizer is Adam, you can modify this to match your optimizer
	optimizer = torch.optim.Adam(model.parameters())
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	epoch = checkpoint['epoch']
	return model, optimizer, epoch

def load_encoded_fasta(filename, alphabet=None, replace=None):
	seqstr = ''
	seqdict = {}

	with open(filename, 'r') as f:

		#read all chars of file into a string
		for i,line in enumerate(tqdm.tqdm(f)):

			if line[0] == '>' and i > 0:
				seqdict[ID] = seqstr[:-1]
				ID = line[1:].strip()
				seqstr = ''
			elif line[0] == '>' and i == 0:
				ID = line[1:].strip()
				seqstr = ''
			else:
				seqstr += line
		if '' in seqdict:
			del seqdict['']
		
	encoded_df = pd.DataFrame( seqdict.items() , columns=['protid', 'seq'] )
	encoded_df['seqlen'] = encoded_df.seq.map( lambda x: len(x) )
	#change index to protid
	encoded_df.index = encoded_df.protid
	encoded_df = encoded_df.drop( 'protid', axis=1 )
	encoded_df['ord'] = encoded_df.seq.map( lambda x: [ ord(c) for c in x] )
	#hex starts at 1
	encoded_df['hex2'] = encoded_df.ord.map( lambda x: [ hex(c) for c in x] )
	return encoded_df

	