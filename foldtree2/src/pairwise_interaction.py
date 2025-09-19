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
from datasketch import WeightedMinHashGenerator, MinHashLSHForest
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
	def __init__(self, in_channels, hidden_channels, out_channels, dropout_p=0.05,
			  decoder_hidden=100,
			  n_signatures=256,
			  nheads=8,
			  nlayers=3
			  ):
		super(signature_transformer, self).__init__()

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

		#trainable embedding
		self.embedding = torch.nn.Embedding(n_signatures, in_channels)

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
		
		#copy z
		z = out.clone().detach()  # (batch, hidden_channels)
		out = self.attn_agg(out)  # (batch, decoder_hidden)
		vec = self.vec_out(out)  # (batch, out_channels)

		return {'jaccard_vec': vec, 'z': z}


class HeteroGAE_Pairwise_Decoder(torch.nn.Module):
	def __init__(self, in_channels = {'res':10 , 'godnode4decoder':5 , 'foldx':23}, xdim=100, hidden_channels={'res_backbone_res': [20, 20, 20]}, layers = 3
			,PINNdecoder_hidden = [10, 10 , 10], 
			contactdecoder_hidden = [10,10,10], 
			nheads = 8 , Xdecoder_hidden=30, metadata={}  ,
			 flavor = None, dropout= .1 , num_hashes = 100 , sample_size = 1000):
		super(HeteroGAE_Decoder, self).__init__()
		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		self.convs = torch.nn.ModuleList()
		in_channels_orig = copy.deepcopy(in_channels )
		self.wmg = WeightedMinHashGenerator( num_hashes = num_hashes , sample_size = sample_size , seed = 42)
		self.num_hashes = num_hashes
		self.sample_size = sample_size

		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.in_channels = in_channels
		self.nlayers = layers
		self.embeding_dim = xdim
		self.bn = torch.nn.BatchNorm1d(in_channels['res'])
		self.dropout = torch.nn.Dropout(p=dropout)		
		self.jk = JumpingKnowledge(mode='cat')
		
		for i in range(layers):
			layer = {}          
			for k,edge_type in enumerate( hidden_channels.keys() ):
				edgestr = '_'.join(edge_type)
				datain = edge_type[0]
				dataout = edge_type[2]
				if flavor == 'transformer' or edge_type == ('res','informs','godnode4decoder'):
					layer[edge_type] = torch.nn.Sequential( TransformerConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False) , torch.nn.LayerNorm(hidden_channels[edge_type][i]) )
				if flavor == 'sage':
					layer[edge_type] = torch.nn.Sequential( SAGEConv( (-1, -1) , hidden_channels[edge_type][i]) , torch.nn.LayerNorm(hidden_channels[edge_type][i]) )
				if ( 'res','backbone','res') == edge_type and i > 0:
					in_channels['res'] = hidden_channels[( 'res','backbone','res')][i-1] + in_channels['godnode4decoder']
				else:
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
		self.sigmoid = nn.Sigmoid()
		self.lin = torch.nn.Sequential(
				torch.nn.LayerNorm(self.hidden_channels[('res', 'backbone', 'res')][-1] *  layers),
				torch.nn.Linear( self.hidden_channels[('res', 'backbone', 'res')][-1] , Xdecoder_hidden),
		)
		
		self.godnodedecoder = torch.nn.Sequential(
				NormTanh(in_channels['godnode4decoder']),
				torch.nn.Linear(in_channels['godnode4decoder'] , PINNdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(PINNdecoder_hidden[0], PINNdecoder_hidden[1] ) ,
				torch.nn.GELU(),
				torch.nn.Linear(PINNdecoder_hidden[1], PINNdecoder_hidden[2] ) ,
				torch.nn.GELU(),
				NormTanh(),
				torch.nn.Linear(PINNdecoder_hidden[2], self.embeding_dim),
				)
	
		self.pair_foldx = torch.nn.Sequential(
				torch.nn.LayerNorm(self.embeding_dim*2),
				torch.nn.Linear(self.embeding_dim*2 , PINNdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(PINNdecoder_hidden[0], PINNdecoder_hidden[1] ) ,
				torch.nn.GELU(),
				torch.nn.Linear(PINNdecoder_hidden[1], PINNdecoder_hidden[2] ) ,
				torch.nn.GELU(),
				torch.nn.LayerNorm(PINNdecoder_hidden[2]),
				torch.nn.Linear(PINNdecoder_hidden[2], foldxdim),
				)

	def forward(self, x1data, **kwargs):
		#only useful for generating embeddings in eval mode
		#check if model is in eval mode
		if self.training:
			raise ValueError('forward1 only useful in eval mode')
		x1data, x1edge_index = x1data.x_dict, x1data.edge_index_dict
		#z = self.bn(z)
		#copy z for later concatenation
		inz1 = x1data['res'].copy()
		inzs = [ inz1 ]
		xdatas = [ x1data ]
		indices = [ x1edge_index ]
		decoder_inputs = []
		godnodes = []
		for inz,xdata,edge_index in zip(inzs,xdatas,indices):
			xsave = []
			for i,layer in enumerate(self.convs):
				xdata = layer(xdata, edge_index)
				for key in layer.convs.keys():
					key = key[2]
					xdata[key] = F.gelu(xdata[key])
				xsave.append(xdata['res'])
			xdata['res'] = self.jk(xsave)
			z = self.lin(xdata['res'])
			decoder_in =  torch.cat( [inz,  z] , axis = 1)
			decoder_inputs.append(decoder_in)
			xdata['godnode4decoder'] = self.godnodedecoder( xdata['godnode4decoder'] )
			godnodes.append(xdata['godnode4decoder'])
		z1 = decoder_inputs[0]
		g1 = godnodes[0]
		return z1 , g1
	
	def hash_foldome(self , dataloader , name = 'foldome'):
		Forest = MinHashLSHForest(num_perm=self.wmg.sample_size)
		#open hdf5 to store hash vals
		with h5py.File(name+'_embeddings.h5', 'w') as foldome:
			for data in tqdm.tqdm(dataloader):
				z1, g1 = self.forward(data)
				g1_hash = self.wmg.minhash(g1)
				Forest.add(data['identifier'] , g1_hash)
				foldome.create_dataset(data['identifier'] + '/hash' , data = g1_hash)
				foldome.create_dataset(data['identifier'] + '/z' , data = z1)
				foldome.create_dataset(data['identifier'] + '/godnode' , data = g1)
		Forest.index()
		#store the forest
		with open(name+'_forest.pkl', 'wb') as f:
			pickle.dump(Forest, f)
		return name+'_embeddings.h5', name+'_forest.pkl'
