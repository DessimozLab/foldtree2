#import libraries
#!/usr/bin/env python
# coding: utf-8

import copy
import importlib
import warnings
from foldtree2.src.mono_decoders import HeteroGAE_geo_Decoder
import torch_geometric
import glob
import h5py
from scipy import sparse
from copy import deepcopy

import pebble
import time
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, to_undirected
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphNorm, Linear, AGNNConv, TransformerConv, GATv2Conv, GCNConv, SAGEConv, MFConv, GENConv, JumpingKnowledge, HeteroConv
from einops import rearrange
from torch_geometric.nn.dense import dense_diff_pool as DiffPool
from torch.nn import ModuleDict, ModuleList, L1Loss
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.utils import negative_sampling
import os
import urllib.request
from urllib.error import HTTPError
import pytorch_lightning as L
import scipy.sparse
import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
import torch.nn as nn
import traceback
import faiss
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
EPS = 1e-15
datadir = '../../datasets/foldtree2/'

def weighted_jaccard_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	"""
	Compute the weighted Jaccard similarity between two matrices.
	Args:
		x (torch.Tensor): Tensor of shape (N, M).
		y (torch.Tensor): Tensor of shape (N, M).
		eps (float): Small value to avoid division by zero.
	Returns:
		torch.Tensor: Weighted Jaccard similarity for each sample (shape: (N,))
	"""
	# x, y: (N, M)
	# Compute min and max along last dimension
	min_xy = torch.minimum(x, y)
	max_xy = torch.maximum(x, y)
	num = min_xy.sum(dim=1)
	den = max_xy.sum(dim=1) + eps
	return num / den

def cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	"""
	Compute the cosine similarity between two matrices.
	Args:
		x (torch.Tensor): Tensor of shape (N, M).
		y (torch.Tensor): Tensor of shape (N, M).
		eps (float): Small value to avoid division by zero.
	Returns:
		torch.Tensor: Cosine similarity for each sample (shape: (N,))
	"""
	# x, y: (N, M)
	x_norm = x / (x.norm(dim=1, keepdim=True) + eps)
	y_norm = y / (y.norm(dim=1, keepdim=True) + eps)
	return (x_norm * y_norm).sum(dim=1)+eps

class EncodedFastaDataset(torch.utils.data.Dataset):
	"""
	PyTorch Dataset for encoded FASTA sequences.
	Each item is a tensor of mapped indices representing the encoded sequence.
	The input is a path to an encoded FASTA file (standard FASTA format).
	Stores ord mapping and allows access by identifier.
	"""
	def __init__(self, encoded_fasta_path):
		"""
		Args:
			encoded_fasta_path (str): Path to the encoded FASTA file.
		"""
		self.identifiers = []
		self.sequences = []
		self.id_to_idx = {}
		ords_set = set()
		with open(encoded_fasta_path, 'r') as f:
			seq = ''
			identifier = None
			for line in f:
				line = line.strip()
				if not line:
					continue
				if line.startswith('>'):
					if seq and identifier:
						self.identifiers.append(identifier)
						self.sequences.append(seq)
						self.id_to_idx[identifier] = len(self.identifiers) - 1
						ords_set.update(ord(c) for c in seq)
					identifier = line[1:].strip()
					seq = ''
				else:
					seq += line
			if seq and identifier:
				self.identifiers.append(identifier)
				self.sequences.append(seq)
				self.id_to_idx[identifier] = len(self.identifiers) - 1
				ords_set.update(ord(c) for c in seq)
		self.ord_list = sorted(ords_set)
		self.ord2idx = {o: i for i, o in enumerate(self.ord_list)}
		self.idx2ord = {i: o for i, o in enumerate(self.ord_list)}
		# Encode all sequences as mapped indices
		self.encoded_sequences = [
			torch.tensor([self.ord2idx[ord(c)] for c in seq], dtype=torch.long)
			for seq in self.sequences
		]

	def __len__(self):
		return len(self.encoded_sequences)

	def __getitem__(self, idx):
		# Returns a dict with identifier and tensor of mapped indices
		return {
			'identifier': self.identifiers[idx],
			'indices': self.encoded_sequences[idx]
		}

	def get_by_identifier(self, identifier):
		idx = self.id_to_idx[identifier]
		return self.__getitem__(idx)
	
class AttentionAggregation(nn.Module):
	"""
	Trainable attention-based aggregation for transformer outputs.
	Aggregates sequence embeddings to a fixed-size vector.
	"""
	def __init__(self, embed_dim, out_dim):
		super().__init__()
		self.query = nn.Parameter(torch.randn(1, embed_dim))
		self.attn = nn.Linear(embed_dim, 1)
		self.proj = nn.Sequential(
			nn.Linear(embed_dim, out_dim),
			nn.GELU(),
			nn.Linear(out_dim, out_dim)
		)

	def forward(self, x):
		# x: (seq_len, batch, embed_dim) or (seq_len, embed_dim)
		if x.dim() == 2:
			x = x.unsqueeze(1)  # (seq_len, 1, embed_dim)
		attn_scores = self.attn(x)  # (seq_len, batch, 1)
		attn_weights = torch.softmax(attn_scores, dim=0)  # (seq_len, batch, 1)
		agg = (attn_weights * x).sum(dim=0)  # (batch, embed_dim)
		out = self.proj(agg)  # (batch, out_dim)
		return out.squeeze(0) if out.shape[0] == 1 else out

class signature_transformer(torch.nn.Module):
	def __init__(self, encoder , decoder, hidden_channels, out_channels, dropout_p=0.05,
			  decoder_hidden=100,
			  nheads=8,
			  normalize=True,
			  nlayers=3,
			  attn_aggregate=False,
			  embedding_dim=128,
			  aggregation_strategy='mean',  # 'mean', 'max', 'attention', or 'concat_stats'
			 **kwargs
			  ):
		super(signature_transformer, self).__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.aggregation_strategy = aggregation_strategy

		#save all arguments to constructor
		self.args = locals()
		self.args.pop('self')
		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		self.normalize = normalize
		self.embedding_dim = embedding_dim
		#self.in_channels = decoder.decoders['contacts'].lastlin + 256 #256 for positional encoding
		self.in_channels = 200 + 256 #256 for positional encoding
		self.out_channels = out_channels
		self.decoder_hidden = decoder_hidden if isinstance(decoder_hidden, list) else [decoder_hidden, decoder_hidden, decoder_hidden]
		self.hidden_channels = hidden_channels
		#trainable embedding. clone the encoder embedding if provided
		#self.embedding1 = copy.deepcopy(encoder.vector_quantizer.embeddings)
		#self.embedding1.requires_grad = False
		#self.embedding2 = torch.nn.Embedding( encoder.num_embeddings , self.encoder.out_channels )
		self.input2transformer = torch.nn.Sequential(
			torch.nn.Linear(self.in_channels, hidden_channels[0]),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[0], hidden_channels[1]),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[1], hidden_channels[2]),
		)

		# vanilla pytorch transformer encoder layer
		self.transformer_encoder = torch.nn.TransformerEncoder(
			torch.nn.TransformerEncoderLayer(
				d_model=hidden_channels[2],
				nhead=nheads,
				dim_feedforward=hidden_channels[2] * 2,
				activation='gelu'
			),
			num_layers=nlayers
		)
		#batch norm and dropout
		self.encoder_norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_channels[2]) for _ in range(nlayers)])
		self.bn = torch.nn.BatchNorm1d(self.in_channels - 256 ) #batch norm on embedding dim only
		self.dropout = torch.nn.Dropout(p=dropout_p)
		
		
		if attn_aggregate == True:
			self.attn_agg = AttentionAggregation(self.hidden_channels[2], self.decoder_hidden[0])
			self.vec_out = torch.nn.Sequential(
			torch.nn.Linear(self.decoder_hidden[0], self.decoder_hidden[1]),
			torch.nn.GELU(),
			torch.nn.Linear(self.decoder_hidden[1], self.decoder_hidden[2]),
			torch.nn.GELU(),
			torch.nn.Linear(self.decoder_hidden[2], self.out_channels),
		)
		else:
			self.attn_agg = None
			# Adjust input dimension for concat_stats strategy
			if aggregation_strategy == 'concat_stats':
				input_dim = self.hidden_channels[2] * 4
			else:
				input_dim = self.hidden_channels[2]
			self.vec_out = torch.nn.Sequential(
			torch.nn.Linear(input_dim, self.decoder_hidden[1]),
			torch.nn.GELU(),
			torch.nn.Linear(self.decoder_hidden[1], self.decoder_hidden[2]),
			torch.nn.GELU(),
			torch.nn.Linear(self.decoder_hidden[2], self.out_channels),
		)
		
		# Global aggregation module for final vector output (separate from attn_agg)
		if aggregation_strategy == 'attention' and not attn_aggregate:
			self.global_attn_agg = AttentionAggregation(self.hidden_channels[2], self.out_channels)
		else:
			self.global_attn_agg = None
		self.n_signatures = out_channels
		#final sigmoid
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, data, **kwargs):
		#data is a an item from EncodedFastaDataset
		x_dict = data.x_dict
		edge_index_dict = data.edge_index_dict
		batch = data['res'].batch
		#embed the input sequences with the trainable embedding
		#x = self.embedding1(x_dict['zdiscrete']) #+ self.embedding2(x_dict['zdiscrete'])
		x = self.bn(x_dict['zdecoder'])
		#add positional encoding
		x = torch.cat( [ x , x_dict['positions']] , dim=-1)
		# proj to transformer input dim
		x = self.input2transformer(x)
		# Transformer expects (seq_len, batch, d_model), so add batch dim if needed
		if batch is not None:
			num_graphs = batch.max().item() + 1
			x_split = [x[batch == i] for i in range(num_graphs)]
			max_len = max([xi.shape[0] for xi in x_split])
			padded = []
			for xi in x_split:
				pad_len = max_len - xi.shape[0]
				if pad_len > 0:
					xi = torch.cat([xi, torch.zeros(pad_len, xi.shape[1], device=xi.device, dtype=xi.dtype)], dim=0)
				padded.append(xi)
			x = torch.stack(padded, dim=1)  # (seq_len, batch, d_model)
		else:
			x = x.unsqueeze(1)  # (seq_len, 1, d_model)
		
		out = x
		for i in range(self.transformer_encoder.num_layers):
			out = self.transformer_encoder.layers[i](out)
			#out = self.dropout(out)

		# Aggregate transformer outputs
		#copy z
		z = out.clone().detach()  # (batch, hidden_channels)
		# Apply aggregation strategy on each batch element
		if batch is not None:
			agg_out = []
			for i in range(num_graphs):
				seq_out = out[:, i, :]  # (seq_len, hidden_channels)
				if self.attn_agg is not None:
					agg_out.append(self.attn_agg(seq_out))
				elif self.aggregation_strategy == 'max':
					agg_out.append(seq_out.max(dim=0)[0])
				elif self.aggregation_strategy == 'attention' and self.global_attn_agg is not None:
					agg_out.append(self.global_attn_agg(seq_out))
				elif self.aggregation_strategy == 'concat_stats':
					stats = torch.cat([
						seq_out.mean(dim=0),
						seq_out.max(dim=0)[0],
						seq_out.min(dim=0)[0],
						seq_out.std(dim=0)
					], dim=-1)
					agg_out.append(stats)
				else:  # 'mean' (default)
					agg_out.append(seq_out.mean(dim=0))
			out = torch.stack(agg_out, dim=0)  # (num_graphs, decoder_hidden or hidden_channels)
			
		else:
			if self.attn_agg is not None:
				out = self.attn_agg(out)  # (1, decoder_hidden)
			elif self.aggregation_strategy == 'max':
				out = out.max(dim=0)[0]  # (hidden_channels)
			elif self.aggregation_strategy == 'attention' and self.global_attn_agg is not None:
				out = self.global_attn_agg(out)  # (out_channels)
			elif self.aggregation_strategy == 'concat_stats':
				out = torch.cat([
					out.mean(dim=0),
					out.max(dim=0)[0],
					out.min(dim=0)[0],
					out.std(dim=0)
				], dim=-1)
			else:  # 'mean' (default)
				out = out.mean(dim=0)  # (hidden_channels)
		# Project to output space
		vec = self.vec_out(out)  # (batch, out_channels)
		if self.normalize:
			vec = vec / (vec.norm(p=2, dim=1, keepdim=True) + EPS)
		return {'jaccard_vec': vec, 'z': z }

	def vec2numpy(self, vec):
		"""
		Convert a torch tensor vector to numpy array for FAISS.
		Args:
			vec (torch.Tensor): Input vector of shape (N,) or (1, N).
		Returns:
			np.ndarray: Vector as numpy array with shape (1, N).
		"""
		vec_np = vec.cpu().numpy()
		if vec_np.ndim == 1:
			vec_np = vec_np.reshape(1, -1)
		return vec_np.astype('float32')
	
	def pull_vectors(self, identifiers, vecfile='vectors.h5'):
		"""
		Pull precomputed vectors from an HDF5 file.
		Args:
			identifiers (list): List of sequence identifiers.
			vecfile (str): Path to the HDF5 file containing vectors.
		Returns:
			dict: Dictionary mapping identifiers to their vectors.
		"""
		vectors = {}
		with h5py.File(vecfile, 'r') as hf:
			for identifier in identifiers:
				if f"{identifier}_vec" in hf:
					vectors[identifier] = hf[f"{identifier}_vec"][:]
				else:
					raise KeyError(f"Identifier {identifier} not found in {vecfile}")
		return vectors
	
	def create_interactome(self, encoded_fasta = None , encoded_dataset = None, vecfile='vectors.h5', indexfile='faiss_index.bin', top_k=10):
		''' 
		Create a pairwise interaction dataframe using precomputed vectors and FAISS index.
		Run through each entry in dataset, query FAISS index for top_k nearest neighbors by cosine similarity,
		pull vectors, compute cosine similarity, and store pairwise scores in dataframe.
		Compile networkx graph and return graph and dataframe.
		Args:
			encoded_fasta (str): Path to encoded FASTA file (if encoded_dataset not provided).
			encoded_dataset (EncodedFastaDataset): Pre-loaded dataset.
			vecfile (str): Path to the HDF5 file containing vectors.
			indexfile (str): Path to the FAISS index file.
			top_k (int): Number of nearest neighbors to query.
		Returns:
			tuple: (interactome_df, G) - DataFrame of pairwise scores and NetworkX graph.
		'''	
		if encoded_dataset is None and encoded_fasta is not None:
			encoded_dataset = EncodedFastaDataset(encoded_fasta)
		elif encoded_dataset is None:
			raise ValueError("Must provide either encoded_dataset or encoded_fasta")

		#build vector database if files don't exist
		if not os.path.exists(vecfile) or not os.path.exists(indexfile):
			print("Vector database not found, building index...")
			self.build_vector_database(encoded_dataset=encoded_dataset, vecout=vecfile, indexout=indexfile)
		
		#load FAISS index and identifier mapping
		index = faiss.read_index(indexfile)
		with h5py.File(vecfile, 'r') as hf:
			identifier_list = [k.replace('_vec', '') for k in hf.keys() if k.endswith('_vec')]
		
		records = []
		#query each sequence
		for item in tqdm.tqdm(encoded_dataset, desc="Querying vectors"):
			identifier = item['identifier']
			#get query vector
			query_vec = self.vec2numpy(self.forward(item)['jaccard_vec'])
			#normalize for cosine similarity (FAISS inner product on normalized vectors = cosine similarity)
			faiss.normalize_L2(query_vec)
			#query FAISS index for top_k+1 nearest neighbors (includes self)
			distances, indices = index.search(query_vec, top_k + 1)
			#convert to lists and skip self-match
			distances = distances[0]
			indices = indices[0]
			#record pairwise scores (distances are cosine similarities for normalized vectors)
			for dist, idx in zip(distances, indices):
				neighbor = identifier_list[idx]
				if neighbor != identifier:  # skip self
					records.append({'protA': identifier, 'protB': neighbor, 'score': float(dist)})
					
		interactome_df = pd.DataFrame(records)
		#build networkx graph
		G = nx.from_pandas_edgelist(interactome_df, 'protA', 'protB', edge_attr='score')
		return interactome_df, G

	def build_vector_database(self, encoded_fasta = None, encoded_dataset = None, vecout='vectors.h5', indexout='faiss_index.bin'):
		"""
		Build a FAISS vector database for all sequences in the encoded dataset.
		Args:
			encoded_fasta (str): Path to encoded FASTA file (if encoded_dataset not provided).
			encoded_dataset (EncodedFastaDataset): Dataset of encoded sequences.
			vecout (str): Path to save vectors in HDF5 format.
			indexout (str): Path to save FAISS index.
		Returns:
			tuple: (vectors_df, index) - DataFrame with identifiers and vectors, FAISS index.
		"""
		if encoded_dataset is None and encoded_fasta is not None:
			encoded_dataset = EncodedFastaDataset(encoded_fasta)
		elif encoded_dataset is None:
			raise ValueError("Must provide either encoded_dataset or encoded_fasta")
			
		with torch.no_grad():
			# Compute vectors for all sequences
			records = []
			vectors_list = []
			identifiers_list = []
			
			for item in tqdm.tqdm(encoded_dataset, desc="Computing vectors"):
				identifier = item['identifier']
				vec = self.forward(item)['jaccard_vec'].cpu().numpy().astype('float32')
				if vec.ndim == 1:
					vec = vec.reshape(1, -1)
				vectors_list.append(vec[0])
				identifiers_list.append(identifier)
				records.append({'identifier': identifier, 'vec': vec[0]})
				
			vectors_df = pd.DataFrame(records).set_index('identifier')
			
			# Build FAISS index for cosine similarity (using inner product on normalized vectors)
			vectors_array = np.vstack(vectors_list).astype('float32')
			dimension = vectors_array.shape[1]
			
			# Normalize vectors for cosine similarity
			faiss.normalize_L2(vectors_array)
			
			# Create FAISS index (IndexFlatIP for inner product = cosine similarity on normalized vectors)
			index = faiss.IndexFlatIP(dimension)
			index.add(vectors_array)
			
		# Save vectors to HDF5
		with h5py.File(vecout, 'w') as hf:
			for identifier, vec in zip(identifiers_list, vectors_list):
				hf.create_dataset(f"{identifier}_vec", data=vec)

		# Save FAISS index
		faiss.write_index(index, indexout)
		
		return vectors_df, index


class HeteroGAE_geo_Decoder_pairwise(torch.nn.Module):
	def __init__(self, in_channels = {'res':10 , 'godnode4decoder':5 , 'foldx':23 },
				concat_positions = False, hidden_channels={'res_backbone_res': [20, 20, 20]},
				layers = 3,
				FFT2decoder_hidden = 10,
				contactdecoder_hidden = 10,
				nheads = 3 ,
				Xdecoder_hidden=30, 
				anglesdecoder_hidden=30,
				RTdecoder_hidden=30,
				metadata={}, 
				flavor = None,
				dropout= .001,
				output_fft = False,
				output_rt = False,
				output_angles = False,
				normalize = True,
				residual = True,
				output_edge_logits = False,
		ncat = 16,
		contact_mlp = True,
		jaccard_vec_dim = 128,  # new: output dimension for jaccard vector
		aggregation_strategy = 'mean'  # 'mean', 'max', 'attention', or 'concat_stats'
				):
		super(HeteroGAE_geo_Decoder_pairwise, self).__init__()
		# Setting the seed
		L.seed_everything(42)
		self.aggregation_strategy = aggregation_strategy
		
		self.bn = torch.nn.BatchNorm1d(in_channels['res'])

		if concat_positions == True:
			in_channels['res'] = in_channels['res'] + 256
		
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		self.convs = torch.nn.ModuleList()
		self.norms = torch.nn.ModuleList()
		self.normalize = normalize
		self.concat_positions = concat_positions
		in_channels_orig = copy.deepcopy(in_channels )
		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.in_channels = in_channels
		self.nlayers = layers
		self.output_fft = output_fft
		self.dropout = torch.nn.Dropout(p=dropout)
		self.jk = JumpingKnowledge(mode='cat')
		self.residual = residual
		finalout = list(hidden_channels.values())[-1][-1]
		for i in range(layers):
			layer = {}          
			
			for k,edge_type in enumerate( hidden_channels.keys() ):
				edgestr = '_'.join(edge_type)
				datain = edge_type[0]
				dataout = edge_type[2]
				if flavor == 'gat':
					layer[edge_type] =  GATv2Conv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False	)
				if flavor == 'mfconv':
					layer[edge_type] = MFConv( (-1, -1)  , hidden_channels[edge_type][i] , max_degree=5  , aggr = 'max' )
				if flavor == 'transformer' or edge_type == ('res','informs','godnode4decoder'):
					layer[edge_type] =  TransformerConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False  ) 
				if flavor == 'sage' :
					layer[edge_type] =  SAGEConv( (-1, -1) , hidden_channels[edge_type][i] ) 
				if k == 0 and i == 0:
					in_channels[dataout] = hidden_channels[edge_type][i]
				if k == 0 and i > 0:
					in_channels[dataout] = hidden_channels[edge_type][i-1]
				if k > 0 and i > 0:                    
					in_channels[dataout] = hidden_channels[edge_type][i]
				if k > 0 and i == 0:
					in_channels[dataout] = hidden_channels[edge_type][i]
			conv = HeteroConv( layer  , aggr='max')
			
			self.convs.append( conv )
			self.norms.append( GraphNorm(finalout) )
		self.sigmoid = nn.Sigmoid()
		if self.residual == True:
			lastlin = in_channels_orig['res']
		else:
			lastlin = Xdecoder_hidden[-1]

		self.lastlin = lastlin
		
		self.lin = torch.nn.Sequential(
				torch.nn.Dropout(dropout),
				DynamicTanh(finalout*layers , channels_last = True),
				torch.nn.Linear( finalout*layers , Xdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(  Xdecoder_hidden[0], Xdecoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(Xdecoder_hidden[1], lastlin),
				#torch.nn.Tanh(),
				#DynamicTanh(lastlin , channels_last = True),
				
				)
		

		self.in2model = torch.nn.Sequential(	
			torch.nn.Linear(in_channels_orig['res'], hidden_channels[('res', 'backbone', 'res')][0] ),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[('res', 'backbone', 'res')][0], hidden_channels[('res', 'backbone', 'res')][0] ),
			torch.nn.Tanh()
		)
		
		if output_fft == True:

			
			self.godnodedecoder = torch.nn.Sequential(
					torch.nn.Linear(in_channels['godnode4decoder'] , FFT2decoder_hidden[0]),
					torch.nn.GELU(),
					torch.nn.Linear(FFT2decoder_hidden[0], FFT2decoder_hidden[1] ) ,
					torch.nn.GELU(),
					#DynamicTanh(FFT2decoder_hidden[1] , channels_last = True),
					torch.nn.Linear(FFT2decoder_hidden[1], in_channels['fft2i'] +in_channels['fft2r'] ),
					#SIRENLayer( in_channels['fft2i'] +in_channels['fft2r'] , in_channels['fft2i'] +in_channels['fft2r'] , omega_0 = 30 )
				)
		else:
			self.godnodedecoder = None

		if contact_mlp == True:
			self.contact_mlp = torch.nn.Sequential(
				torch.nn.Dropout(dropout),
				torch.nn.Linear(2*lastlin, contactdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(contactdecoder_hidden[0], contactdecoder_hidden[1] ) ,
				torch.nn.GELU(),
				torch.nn.Linear(contactdecoder_hidden[1], 1) )
		else:
			self.contact_mlp = None

		if output_rt == True:
			if type(RTdecoder_hidden) is not list:
				RTdecoder_hidden = [RTdecoder_hidden, RTdecoder_hidden]
			#suggest r and t for each residue
			#feed into refinement network
			self.rt_mlp = torch.nn.Sequential(
				torch.nn.Dropout(dropout),
				torch.nn.Linear(lastlin, RTdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(RTdecoder_hidden[0], RTdecoder_hidden[1] ) ,
				torch.nn.GELU(),
				torch.nn.Linear(RTdecoder_hidden[1], 7) )
		else:
			self.rt_mlp = None

		if output_angles == True:
			if type(anglesdecoder_hidden) is not list:
				anglesdecoder_hidden = [anglesdecoder_hidden, anglesdecoder_hidden]
			self.angles_mlp = torch.nn.Sequential(
				torch.nn.Dropout(dropout),
				torch.nn.Linear(lastlin, anglesdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(anglesdecoder_hidden[0], anglesdecoder_hidden[1] ) ,
				torch.nn.GELU(),
				torch.nn.Linear(anglesdecoder_hidden[1], 3) ,
				torch.nn.Tanh()

			)
		else:
			self.angles_mlp = None

		if output_edge_logits == True:
			self.output_edge_logits = True
			self.edge_logits_mlp = torch.nn.Sequential(
				torch.nn.Linear(2*lastlin, 512),
				torch.nn.GELU(),
				torch.nn.Linear(512,512),
				torch.nn.GELU(),
				torch.nn.Linear(512,256),
				torch.nn.GELU(),
				torch.nn.Linear(256,256),
				torch.nn.GELU(),
				torch.nn.Linear(256,ncat),
				torch.nn.Sigmoid()
			)
		else:
			self.output_edge_logits = False
			self.edge_logits_mlp = None

		# New: projection for jaccard vector output with different aggregation strategies
		if aggregation_strategy == 'attention':
			self.global_attn_agg = AttentionAggregation(lastlin, jaccard_vec_dim)
			self.jaccard_vec_proj = None
		elif aggregation_strategy == 'concat_stats':
			# Concatenate mean, max, min, std (4x the input)
			self.jaccard_vec_proj = torch.nn.Sequential(
				torch.nn.Linear(lastlin * 4, jaccard_vec_dim),
				torch.nn.GELU(),
				torch.nn.Linear(jaccard_vec_dim, jaccard_vec_dim)
			)
			self.global_attn_agg = None
		else:  # 'mean' or 'max'
			self.jaccard_vec_proj = torch.nn.Sequential(
				torch.nn.Linear(lastlin, jaccard_vec_dim),
				torch.nn.GELU(),
				torch.nn.Linear(jaccard_vec_dim, jaccard_vec_dim)
			)
			self.global_attn_agg = None
		self.jaccard_vec_dim = jaccard_vec_dim

	def forward(self, data , contact_pred_index=None, **kwargs):
		#apply batch norm to res

		xdata, edge_index = data.x_dict, data.edge_index_dict
		xdata['res'] = self.dropout(xdata['res'])
		if self.concat_positions == True:
			xdata['res'] = torch.cat([xdata['res'], data['positions'].x], dim=1)

		#xdata['res'] = self.in2model(xdata['res'])

		#copy z for later concatenation
		inz = xdata['res'].clone()	
		x_dict_list = []
		for i,layer in enumerate(self.convs):
			if i > 0:
				prev = xdata['res'].clone()
			xdata = layer(xdata, edge_index)
			xdata['res'] = F.gelu(xdata['res'])
			xdata['res'] = self.norms[i](xdata['res'])
			if i > 0:
				xdata['res'] = xdata['res'] + prev
			x_dict_list.append(xdata['res'])

		xdata['res'] = self.jk(x_dict_list)
		z = xdata['res']
		z = self.lin(z)
		if self.residual == True:
			z = z + inz
		if self.normalize == True:
			z =  z / ( torch.norm(z, dim=1, keepdim=True) + 1e-10)

		# Project to jaccard vector (global embedding)
		# Aggregate over sequence dimension using specified strategy
		if self.aggregation_strategy == 'attention':
			# Attention-based aggregation
			if z.dim() == 2:
				z_agg = z.unsqueeze(1)  # (seq_len, 1, dim)
			else:
				z_agg = z
			jaccard_vec = self.global_attn_agg(z_agg)
		elif self.aggregation_strategy == 'max':
			# Max pooling
			z_agg = z.max(dim=0)[0] if z.dim() == 2 else z.max(dim=0)[0]
			jaccard_vec = self.jaccard_vec_proj(z_agg)
		elif self.aggregation_strategy == 'concat_stats':
			# Concatenate mean, max, min, std
			z_mean = z.mean(dim=0) if z.dim() == 2 else z.mean(dim=0)
			z_max = z.max(dim=0)[0] if z.dim() == 2 else z.max(dim=0)[0]
			z_min = z.min(dim=0)[0] if z.dim() == 2 else z.min(dim=0)[0]
			z_std = z.std(dim=0) if z.dim() == 2 else z.std(dim=0)
			z_concat = torch.cat([z_mean, z_max, z_min, z_std], dim=-1)
			jaccard_vec = self.jaccard_vec_proj(z_concat)
		else:  # 'mean' (default)
			# Mean pooling
			z_agg = z.mean(dim=0) if z.dim() == 2 else z.mean(dim=0)
			jaccard_vec = self.jaccard_vec_proj(z_agg)
		
		if self.normalize:
			jaccard_vec = jaccard_vec / (jaccard_vec.norm(p=2, dim=-1, keepdim=True) + 1e-10)
		# jaccard_vec: (jaccard_vec_dim,)

		#decode godnode
		fft2_pred = None
		if self.output_fft == True:
			zgodnode = xdata['godnode4decoder']
			fft2_pred = self.godnodedecoder( xdata['godnode4decoder'] )
		else:
			zgodnode = None
		
		rt_pred = None
		if self.rt_mlp is not None:
			rt_pred = self.rt_mlp( torch.cat( [ inz , z ] , axis = 1 ) )
		
		angles = None
		edge_logits = None

		if self.angles_mlp is not None:
			angles = self.angles_mlp( z )
			#tanh is -1 to 1, multiply by 2pi to get angles in radians
			angles = angles * 2 * np.pi

		edge_probs = None
		if contact_pred_index is not None:
			if self.contact_mlp is None:
				edge_probs = self.sigmoid( torch.sum( z[contact_pred_index[0]] * z[contact_pred_index[1]] , axis =1 ) )
			else:
				edge_scores = self.contact_mlp( torch.concat( [z[contact_pred_index[0]], z[contact_pred_index[1]] ] , axis = 1 ) ).squeeze()
				edge_probs = self.sigmoid(edge_scores)
			if self.edge_logits_mlp is not None:
				edge_logits = self.edge_logits_mlp( torch.concat( [z[contact_pred_index[0]], z[contact_pred_index[1]] ] , axis = 1 ) ).squeeze()
		
		if 'init' in kwargs and kwargs['init'] == True:
			# Initialize weights explicitly (Xavier initialization)
			for conv in self.convs:
				for c in conv.convs.values():
					for param in c.parameters():
						if param.dim() > 1:
							nn.init.xavier_uniform_(param)

		# Output matches transformer: jaccard_vec (global), z (per-position embedding)
		return  { 
			'jaccard_vec': jaccard_vec.squeeze(0) if jaccard_vec.dim() == 2 and jaccard_vec.shape[0] == 1 else jaccard_vec,
			'z': z,
			'edge_probs': edge_probs,
			'edge_logits': edge_logits,
			'zgodnode': zgodnode,
			'fft2pred': fft2_pred,
			'rt_pred': rt_pred,
			'angles': angles
		}
