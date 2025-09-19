#import libraries
#!/usr/bin/env python
# coding: utf-8

import copy
import importlib
import warnings
import torch_geometric
import glob
import h5py
from scipy import sparse
from copy import deepcopy
import pickle

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
from datasketch import WeightedMinHashGenerator, MinHashLSHForest, WeightedMinHash
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
    min_sum = torch.sum(torch.minimum(x, y), dim=1)
    max_sum = torch.sum(torch.maximum(x, y), dim=1)
    jaccard_sim = min_sum / (max_sum + eps)
    return jaccard_sim

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
	def __init__(self, encoder, hidden_channels, out_channels, dropout_p=0.05,
			  decoder_hidden=100,
			  n_signatures=256,
			  nheads=8,
			  nlayers=3,
			  ):
		super(signature_transformer, self).__init__()

		self.encoder = encoder


		print("Using embedding model:", self.embedding_model)

		self.wmg = WeightedMinHashGenerator(num_perm=n_signatures, seed=42)
		#save all arguments to constructor
		self.args = locals()
		self.args.pop('self')
		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False		
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.hidden_channels = hidden_channels
		#trainable embedding. clone the encoder embedding if provided
		self.embedding = encoder.vector_quantizer.embeddings.detach().clone()

		self.input2transformer = torch.nn.Sequential(
			torch.nn.Linear(in_channels, hidden_channels[0] * 2),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[0] * 2, hidden_channels[0]),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[0], hidden_channels[0]),
		)

		# vanilla pytorch transformer encoder layer
		self.transformer_encoder = torch.nn.TransformerEncoder(
			torch.nn.TransformerEncoderLayer(
				d_model=hidden_channels,
				nhead=nheads,
				dim_feedforward=hidden_channels[0] * 2,
				activation='gelu'
			),
			num_layers=nlayers
		)

		self.bn = torch.nn.BatchNorm1d(in_channels)
		self.dropout = torch.nn.Dropout(p=dropout_p)
		
		self.vec_out = torch.nn.Sequential(
			torch.nn.Linear(self.decoder_hidden, self.out_channels),
			torch.nn.GELU(),
			torch.nn.Linear(self.decoder_hidden, self.out_channels),
			torch.nn.GELU(),
			torch.nn.Linear(self.decoder_hidden, self.out_channels),
			torch.nn.Tanh()
		)
		self.attn_agg = AttentionAggregation(self.hidden_channels, self.decoder_hidden)

		
	def forward(self, data, **kwargs):
		#data is a an item from EncodedFastaDataset
		x_dict = data.x_dict
		edge_index_dict = data.edge_index_dict
		batch = data.batch if hasattr(data, 'batch') else None
		#embed the input sequences
		x = self.bn(x)
		x = self.dropout(x)
		# proj to transformer input dim
		x = self.input2transformer(x)
		# Transformer expects (seq_len, batch, d_model), so add batch dim if needed
		batch = x.batch if hasattr(x, 'batch') else None
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
			out = F.gelu(out)		
		# Aggregate transformer outputs
		#copy z
		z = out.clone().detach()  # (batch, hidden_channels)
		out = self.attn_agg(out)  # (batch, decoder_hidden)
		vec = self.vec_out(out)  # (batch, out_channels)
		return {'jaccard_vec': vec, 'z': z}

	def vec2weighted_minhash(self, vec):
		"""
		Convert a weighted vector to a Weighted MinHash signature.
		Args:
			vec (torch.Tensor): Input vector of shape (N,).
		Returns:
			np.ndarray: MinHash signature as a numpy array.
		"""

		mh = self.wmg.minhash(vec.cpu().numpy())
		return mh.hashvalues
	
	def pull_hashes(self, identifiers, sigfile='signatures.h5'):
		"""
		Pull precomputed MinHash signatures from an HDF5 file.
		Args:
			identifiers (list): List of sequence identifiers.
			sigfile (str): Path to the HDF5 file containing signatures.
		Returns:
			dict: Dictionary mapping identifiers to their MinHash signatures.
		"""
		signatures = {}
		with h5py.File(sigfile, 'r') as hf:
			for identifier in identifiers:
				if identifier in hf:
					signatures[identifier] = WeightedMinHash(  seed = 42 , hashvalues = hf[identifier][:])
				else:
					raise KeyError(f"Identifier {identifier} not found in {sigfile}")
		return signatures
	
	def create_interactome(self, encoded_fasta = None , encoded_dataset = None, sigfile='signatures.h5', lshfile='lsh_forest.pkl', top_k=10):
		''' 
		Create a pairwise interaction dataframe using precomputed MinHash signatures and LSH forest.
		run through each entry in df, query lsh forest for top_k nearest neighbors
		pull hashes, compute weighted jaccard similarity, and store pairwise scores in dataframe
		compile networkx graph and return graph and dataframe
		Args:
			df (pd.DataFrame): DataFrame with 'protA' and 'protB' columns.
			sigfile (str): Path to the HDF5 file containing signatures.
			lshfile (str): Path to the pickle file containing the LSH forest.
			top_k (int): Number of nearest neighbors to query.
		'''	
		if encoded_dataset is None and encoded_fasta is not None:
			raise ValueError("Must provide encoded_dataset if not providing encoded_fasta")
		if encoded_dataset is None:
			encoded_dataset = EncodedFastaDataset(encoded_fasta)

		#hash proteome if sigfile or lshfile do not exist
		if not os.path.exists(sigfile) or not os.path.exists(lshfile):
			print("Signatures or LSH file not found, hashing proteome...")
			self.hash_proteome( encoded_dataset=encoded_dataset, sigout=sigfile, lshout=lshfile )
		
		#load lsh forest
		with open(lshfile, 'rb') as f:
			lsh = pickle.load(f)
		records = []
		#pull all unique identifiers from df
		for item in encoded_dataset:
			identifier = item['identifier']
			#query lsh forest for top_k nearest neighbors
			neighbors = lsh.query(self.sig2weighted_minhash(self.forward(item)['jaccard_vec'])), top_k)
			#pull hashes for neighbors
			sigs = self.pull_hashes([identifier] + neighbors, sigfile=sigfile)
			q_hash = sigs[identifier]
			#compute weighted jaccard similarity for each neighbor
			for neighbor in neighbors:
				records.append({'protA': identifier, 'protB': neighbor, 'score': q_hash.jaccard(sigs[neighbor])})
		interactome_df = pd.DataFrame(records)
		#build networkx graph
		G = nx.from_pandas_edgelist(interactome_df, 'protA', 'protB', edge_attr='score')
		return interactome_df, G

	def hash_proteome( encoded_fasta = None, encoded_dataset = None, num_perm=256 , sigout='signatures.h5' , lshout='lsh_forest.pkl' ):
		"""
		Hash all sequences in the encoded dataset using Weighted MinHash.
		Args:
			encoded_dataset (EncodedFastaDataset): Dataset of encoded sequences.
			num_perm (int): Number of permutations for MinHash.
		Returns:
			pd.DataFrame: DataFrame with identifiers and their MinHash signatures.	
		"""
		#create hdf5 file to store the signatures
		if encoded_dataset is None and encoded_fasta is not None:
			raise ValueError("Must provide encoded_dataset if not providing encoded_fasta")
		if encoded_dataset is None:
			encoded_dataset = EncodedFastaDataset(encoded_fasta)
		with torch.no_grad():
			lsh = MinHashLSHForest(num_perm=self.n_signatures, seed=42, l=10)
			# Compute MinHash signatures for all sequences
			records = []
			for item in encoded_dataset:
				identifier = item['identifier']
				weights = self.forward(item)['jaccard_vec'].cpu().numpy()
				mh = self.wmg.minhash(weights)
				signature = mh.hashvalues
				lsh.insert(identifier, signature)
				records.append({'identifier': identifier, 'signature': signature , 'vec': weights })
			lsh.index()
			signature_df = pd.DataFrame(records).set_index('identifier')
		#dump signatures to hdf5
		with h5py.File(sigout, 'w') as hf:
			for idx, row in signature_df.iterrows():
				hf.create_dataset(idx, data=row['signature'])
				hf.create_dataset(f"{idx}_vec", data=row['vec'])

		#dump lsh to pickle
		with open(lshout, 'wb') as f:
			pickle.dump(lsh, f)
		
		return signature_df, lsh


