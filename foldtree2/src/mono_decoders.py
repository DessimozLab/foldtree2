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
from foldtree2.src.chebconv import StableChebConv
from scipy.spatial.distance import cdist
EPS = 1e-15
datadir = '../../datasets/foldtree2/'


from foldtree2.src.losses import *
from foldtree2.src.layers import *
from foldtree2.src.dynamictan import *
from foldtree2.src.quantizers import *

from  torch_geometric.utils import to_undirected
#encoder super class


class HeteroGAE_geo_Decoder(torch.nn.Module):
	def __init__(self, in_channels = {'res':10 , 'godnode4decoder':5 , 'foldx':23 },
			   	concat_positions = False, hidden_channels={'res_backbone_res': [20, 20, 20]},
				layers = 3,
				FFT2decoder_hidden = 10,
				contactdecoder_hidden = 10,
				ssdecoder_hidden = 10,
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
				output_ss = False,
				normalize = True,
				residual = True,
				output_edge_logits = False,
				ncat = 16,
				contact_mlp = True ):
		super(HeteroGAE_geo_Decoder, self).__init__()
		# Setting the seed
		L.seed_everything(42)
		
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
				if flavor == 'transformer' or edge_type == ('res','informs','godnode4decoder') or edge_type == ('godnode4decoder','informs','res'):
					layer[edge_type] =  TransformerConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False  ) 
				if flavor == 'sage' :
					layer[edge_type] =  SAGEConv( (-1, -1) , hidden_channels[edge_type][i] ) 
				if flavor == 'cheb':
					layer[edge_type] = StableChebConv(-1 , hidden_channels[edge_type][i] , K=5)
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
				torch.nn.Linear( finalout*layers , Xdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(  Xdecoder_hidden[0], Xdecoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(Xdecoder_hidden[1], lastlin),				
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
					torch.nn.Linear(FFT2decoder_hidden[1], in_channels['fft2i'] +in_channels['fft2r'] ),
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

		if output_ss == True:
			self.output_ss = True
			self.ss_mlp = torch.nn.Sequential(
				torch.nn.LayerNorm(lastlin),
				torch.nn.Linear(lastlin, 128),
				torch.nn.GELU(),
				torch.nn.Linear(128,64),
				torch.nn.GELU(),
				torch.nn.Linear(64,3),
				torch.nn.LogSoftmax(dim=1)
			)
		else:
			self.output_ss = False
			self.ss_mlp = None



		if output_angles == True:
			if type(anglesdecoder_hidden) is not list:
				anglesdecoder_hidden = [anglesdecoder_hidden, anglesdecoder_hidden]
			self.angles_mlp = torch.nn.Sequential(
				#torch.nn.Dropout(dropout),
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
				#layernorm
				torch.nn.LayerNorm(2*lastlin),
				torch.nn.Linear(2*lastlin, anglesdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(anglesdecoder_hidden[0],anglesdecoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(anglesdecoder_hidden[1],anglesdecoder_hidden[2]),
				torch.nn.GELU(),
				torch.nn.Linear(anglesdecoder_hidden[2],ncat),
				torch.nn.GELU(),
				torch.nn.Sigmoid()
			)
		else:
			self.output_edge_logits = False
			self.edge_logits_mlp = None

	def forward(self, data , contact_pred_index, **kwargs):

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
		#decoder_in =  torch.cat( [inz,  z] , axis = 1)
		#amino acid prediction removed

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
		
		ss_pred = None
		if self.ss_mlp is not None:
			ss_pred = self.ss_mlp( z )

		angles = None
		edge_logits = None

		if self.angles_mlp is not None:
			angles = self.angles_mlp( z )
			#tanh is -1 to 1, multiply by pi to get angles in radians
			angles = angles * np.pi

		if contact_pred_index is None:
			return { 'edge_probs': None , 'zgodnode' :None , 'fft2pred':fft2_pred , 'rt_pred': None , 'angles': angles  , 'edge_logits': edge_logits  , 'ss_pred': ss_pred , 'z': z  }

		else:
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

		return  { 'edge_probs': edge_probs , 'edge_logits': edge_logits , 'zgodnode' :zgodnode , 'fft2pred':fft2_pred  , 'rt_pred': rt_pred , 'angles': angles , 'ss_pred': ss_pred , 'z': z , }


class CNN_geo_Decoder(torch.nn.Module):
	"""
	Muon-compatible CNN geometry decoder with modular architecture.
	Separates input, body, and head modules for compatibility with Muon optimizer.
	
	- input: Preprocessing and initial transformations (optimized with AdamW)
	- body: CNN convolution layers (weights optimized with Muon, gains/biases with AdamW)
	- head: Prediction heads for various tasks (optimized with AdamW)
	"""
	def __init__(self, in_channels={'res': 10, 'godnode4decoder': 5, 'foldx': 23},
				concat_positions=False, 
				conv_channels=[64, 128, 256],
				kernel_sizes=[3, 3, 3],
				FFT2decoder_hidden=[10, 10],
				contactdecoder_hidden=[10, 10],
				ssdecoder_hidden=[10, 10],
				Xdecoder_hidden=[30, 30], 
				anglesdecoder_hidden=[30, 30],
				RTdecoder_hidden=[30, 30],
				metadata={}, 
				dropout=.001,
				output_fft=False,
				output_rt=False,
				output_angles=False,
				output_ss=False,
				normalize=True,
				residual=True,
				output_edge_logits=False,
				ncat=16,
				contact_mlp=True,
				pool_type='global_mean',
				learn_positions=False):
		super(CNN_geo_Decoder, self).__init__()
		
		# Setting the seed
		L.seed_everything(42)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		
		self.normalize = normalize
		self.concat_positions = concat_positions
		self.learn_positions = learn_positions
		in_channels_orig = copy.deepcopy(in_channels)
		self.metadata = metadata
		self.output_fft = output_fft
		self.residual = residual
		self.pool_type = pool_type
		
		# Determine final output dimension
		if self.residual:
			lastlin = in_channels_orig['res']
		else:
			lastlin = Xdecoder_hidden[-1]
		self.lastlin = lastlin
		
		# ===================== INPUT MODULE =====================
		# Preprocessing: dropout, initial linear transformation
		self.input = nn.ModuleDict()
		
		input_dim = in_channels['res']
		if concat_positions:
			input_dim = input_dim + 256
		if learn_positions:
			self.concat_positions = False
			input_dim = input_dim + 32
		
		self.input['dropout'] = nn.Dropout(p=dropout)
		
		if learn_positions:
			self.input['position_mlp'] = Position_MLP(in_channels=256, hidden_channels=[128, 64], out_channels=32, dropout=dropout)
		
		self.input['input_lin'] = nn.Sequential(
			nn.Linear(input_dim, conv_channels[0]),
			nn.GELU(),
			nn.Linear(conv_channels[0], conv_channels[0]),
			nn.GELU()
		)
		
		# ===================== BODY MODULE =====================
		# CNN layers and batch normalization
		self.body = nn.ModuleDict()
		self.body['convs'] = nn.ModuleList()
		self.body['norms'] = nn.ModuleList()
		
		for i, (channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
			self.body['convs'].append(
				nn.Conv1d(
					channels if i == 0 else conv_channels[i-1], 
					channels, 
					kernel_size=kernel_size, 
					padding=kernel_size//2
				)
			)
			self.body['norms'].append(nn.BatchNorm1d(channels))
		
		finalout = conv_channels[-1]
		
		# Intermediate projection
		self.body['lin'] = nn.Sequential(
			nn.Linear(finalout, Xdecoder_hidden[0]),
			nn.GELU(),
			nn.Linear(Xdecoder_hidden[0], Xdecoder_hidden[1]),
			nn.GELU(),
			nn.Linear(Xdecoder_hidden[1], lastlin),
		)
		
		# ===================== HEAD MODULE =====================
		# All prediction heads (contact, angles, ss, rt, etc.)
		self.head = nn.ModuleDict()
		
		# FFT/Godnode decoder
		if output_fft:
			self.head['godnodedecoder'] = nn.Sequential(
				nn.Linear(in_channels['godnode4decoder'], FFT2decoder_hidden[0]),
				nn.GELU(),
				nn.Linear(FFT2decoder_hidden[0], FFT2decoder_hidden[1]),
				nn.GELU(),
				nn.Linear(FFT2decoder_hidden[1], in_channels['fft2i'] + in_channels['fft2r']),
			)
		
		# Contact prediction MLP
		if contact_mlp:
			self.head['contact_mlp'] = nn.Sequential(
				nn.Dropout(dropout),
				nn.Linear(2*lastlin, contactdecoder_hidden[0]),
				nn.GELU(),
				nn.Linear(contactdecoder_hidden[0], contactdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(contactdecoder_hidden[1], 1)
			)
		
		# Rotation-translation prediction
		if output_rt:
			if not isinstance(RTdecoder_hidden, list):
				RTdecoder_hidden = [RTdecoder_hidden, RTdecoder_hidden]
			self.head['rt_mlp'] = nn.Sequential(
				nn.Dropout(dropout),
				nn.Linear(lastlin, RTdecoder_hidden[0]),
				nn.GELU(),
				nn.Linear(RTdecoder_hidden[0], RTdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(RTdecoder_hidden[1], 7)
			)
		
		# Secondary structure prediction
		if output_ss:
			self.head['ss_mlp'] = nn.Sequential(
				nn.LayerNorm(lastlin),
				nn.Linear(lastlin, 128),
				nn.GELU(),
				nn.Linear(128, 64),
				nn.GELU(),
				nn.Linear(64, 3),
				nn.LogSoftmax(dim=1)
			)
		
		# Bond angles prediction
		if output_angles:
			if not isinstance(anglesdecoder_hidden, list):
				anglesdecoder_hidden = [anglesdecoder_hidden, anglesdecoder_hidden]
			self.head['angles_mlp'] = nn.Sequential(
				nn.Linear(lastlin, anglesdecoder_hidden[0]),
				nn.GELU(),
				nn.Linear(anglesdecoder_hidden[0], anglesdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(anglesdecoder_hidden[1], 3),
				nn.Tanh()
			)
		
		# Edge logits prediction
		if output_edge_logits:
			self.head['edge_logits_mlp'] = nn.Sequential(
				nn.LayerNorm(2*lastlin),
				nn.Linear(2*lastlin, anglesdecoder_hidden[0]),
				nn.GELU(),
				nn.Linear(anglesdecoder_hidden[0], anglesdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(anglesdecoder_hidden[1], anglesdecoder_hidden[2] if len(anglesdecoder_hidden) > 2 else anglesdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(anglesdecoder_hidden[2] if len(anglesdecoder_hidden) > 2 else anglesdecoder_hidden[1], ncat),
				nn.GELU(),
				nn.Sigmoid()
			)
		
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, data, contact_pred_index, **kwargs):
		xdata, _ = data.x_dict, data.edge_index_dict
		x = xdata['res']
		
		# ===================== INPUT PROCESSING =====================
		x = self.input['dropout'](x)
		
		if self.concat_positions:
			x = torch.cat([x, data['positions'].x], dim=1)
		if self.learn_positions:
			pos_enc = self.input['position_mlp'](data['positions'].x)
			x = torch.cat([x, pos_enc], dim=1)
		
		# Initial linear transformation
		x = self.input['input_lin'](x)
		
		# Copy for residual connection
		inz = x.clone()
		
		# ===================== BODY PROCESSING =====================
		# Handle batched data for CNN
		batch = data['res'].batch if hasattr(data['res'], 'batch') and data['res'].batch is not None else None
		
		if batch is not None:
			# Group by batch
			num_graphs = batch.max().item() + 1
			x_list = []
			for i in range(num_graphs):
				mask = batch == i
				x_batch = x[mask]  # Shape: (seq_len, features)
				
				# Transpose for Conv1d: (features, seq_len)
				x_batch = x_batch.transpose(0, 1).unsqueeze(0)  # (1, features, seq_len)
				
				# Apply CNN layers
				for conv, norm in zip(self.body['convs'], self.body['norms']):
					x_batch = conv(x_batch)
					x_batch = F.gelu(x_batch)
					x_batch = norm(x_batch)
				
				# Transpose back and remove batch dimension
				x_batch = x_batch.squeeze(0).transpose(0, 1)  # (seq_len, features)
				x_list.append(x_batch)
			
			# Concatenate back
			x = torch.cat(x_list, dim=0)
		else:
			# Single graph case
			x = x.transpose(0, 1).unsqueeze(0)  # (1, features, seq_len)
			
			for conv, norm in zip(self.body['convs'], self.body['norms']):
				x = conv(x)
				x = F.gelu(x)
				x = norm(x)
			
			x = x.squeeze(0).transpose(0, 1)  # (seq_len, features)
		
		# Final linear transformation
		z = self.body['lin'](x)
		
		if self.residual:
			z = z + inz
		if self.normalize:
			z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-10)
		
		# ===================== HEAD PROCESSING =====================
		# Godnode/FFT decoder
		fft2_pred = None
		if 'godnodedecoder' in self.head:
			zgodnode = xdata['godnode4decoder']
			fft2_pred = self.head['godnodedecoder'](xdata['godnode4decoder'])
		else:
			zgodnode = None
		
		# Rotation-translation prediction
		rt_pred = None
		if 'rt_mlp' in self.head:
			rt_pred = self.head['rt_mlp'](torch.cat([inz, z], axis=1))
		
		# Secondary structure prediction
		ss_pred = None
		if 'ss_mlp' in self.head:
			ss_pred = self.head['ss_mlp'](z)
		
		# Bond angles prediction
		angles = None
		if 'angles_mlp' in self.head:
			angles = self.head['angles_mlp'](z)
			angles = angles * 2 * np.pi
		
		# Contact prediction
		edge_logits = None
		if contact_pred_index is None:
			return {'edge_probs': None, 'zgodnode': None, 'fft2pred': fft2_pred, 
					'rt_pred': None, 'angles': angles, 'edge_logits': edge_logits, 
					'ss_pred': ss_pred, 'z': z}
		else:
			if 'contact_mlp' not in self.head:
				edge_probs = self.sigmoid(torch.sum(z[contact_pred_index[0]] * z[contact_pred_index[1]], axis=1))
			else:
				edge_scores = self.head['contact_mlp'](
					torch.concat([z[contact_pred_index[0]], z[contact_pred_index[1]]], axis=1)
				).squeeze()
				edge_probs = self.sigmoid(edge_scores)
			
			if 'edge_logits_mlp' in self.head:
				edge_logits = self.head['edge_logits_mlp'](
					torch.concat([z[contact_pred_index[0]], z[contact_pred_index[1]]], axis=1)
				).squeeze()
		
		# Weight initialization
		if 'init' in kwargs and kwargs['init']:
			for conv in self.body['convs']:
				for param in conv.parameters():
					if param.dim() > 1:
						nn.init.xavier_uniform_(param)
		
		return {'edge_probs': edge_probs, 'edge_logits': edge_logits, 'zgodnode': zgodnode, 
				'fft2pred': fft2_pred, 'rt_pred': rt_pred, 'angles': angles, 
				'ss_pred': ss_pred, 'z': z}

class Transformer_AA_Decoder(torch.nn.Module):
	"""
	Muon-compatible Transformer amino acid decoder with modular architecture.
	Separates input, body, and head modules for compatibility with Muon optimizer.
	
	- input: Preprocessing and initial projection (optimized with AdamW)
	- body: Transformer encoder layers (weights optimized with Muon, gains/biases with AdamW)
	- head: DNN/CNN decoders for amino acid prediction (optimized with AdamW)
	"""
	def __init__(
		self,
		in_channels={'res': 10},
		hidden_channels={'res_backbone_res': [20, 20, 20]},
		xdim=20,
		concat_positions=True,
		nheads=4,
		layers=2,
		AAdecoder_hidden=[128, 64, 32],
		amino_mapper=None,
		dropout=0.001,
		normalize=True,
		residual=True,
		learn_positions=False,
		**kwargs
	):
		super(Transformer_AA_Decoder, self).__init__()
		L.seed_everything(42)

		self.concat_positions = concat_positions
		self.learn_positions = learn_positions
		self.normalize = normalize
		self.residual = residual
		self.amino_acid_indices = amino_mapper
		
		input_dim = in_channels['res']
		if concat_positions:
			input_dim = input_dim + 256
		if learn_positions:
			self.concat_positions = False
			input_dim = input_dim + 32
		d_model = hidden_channels[('res', 'backbone', 'res')][0]

		print(d_model, nheads, layers, dropout)

		# ===================== INPUT MODULE =====================
		# Preprocessing and initial projection
		self.input = nn.ModuleDict()
		
		self.input['dropout'] = nn.Dropout(dropout)
		
		if learn_positions:
			self.input['position_mlp'] = Position_MLP(in_channels=256, hidden_channels=[128, 64], out_channels=32, dropout=dropout)
		
		self.input['proj'] = nn.Sequential(
			nn.Linear(input_dim, d_model),
			nn.GELU(),
			nn.Linear(d_model, d_model),
			nn.GELU(),
			nn.Tanh(),
		)

		# ===================== BODY MODULE =====================
		# Transformer encoder layers
		self.body = nn.ModuleDict()
		
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model, 
			nhead=nheads, 
			dropout=dropout
		)
		self.body['transformer_encoder'] = nn.TransformerEncoder(
			encoder_layer, 
			num_layers=layers
		)
		
		if self.residual:
			self.body['dmodel2input'] = nn.Sequential(
				nn.Linear(d_model, input_dim),
				nn.GELU(),
				nn.Linear(input_dim, input_dim),
			)

		# ===================== HEAD MODULE =====================
		# Amino acid prediction heads
		self.head = nn.ModuleDict()
		
		# Optional CNN decoder
		if use_cnn_decoder := kwargs.get('use_cnn_decoder', False):
			self.head['prenorm'] = nn.LayerNorm(d_model)
			self.head['cnn_decoder'] = nn.Sequential(
				# Conv1d expects (batch, channels, seq_len)
				nn.Conv1d(d_model, AAdecoder_hidden[0], kernel_size=3, padding=1),
				nn.GELU(),
				nn.Conv1d(AAdecoder_hidden[0], AAdecoder_hidden[1], kernel_size=3, padding=1),
				nn.GELU(),
				nn.Conv1d(AAdecoder_hidden[1], AAdecoder_hidden[2], kernel_size=3, padding=1),
				nn.GELU(),
				nn.Conv1d(AAdecoder_hidden[2], 20, kernel_size=1),
			)
		else:
			# DNN decoder for amino acid prediction
			self.head['dnn_decoder'] = nn.Sequential(
				nn.Linear(d_model, AAdecoder_hidden[0]),
				nn.GELU(),
				nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(AAdecoder_hidden[1], AAdecoder_hidden[2]),
				nn.GELU(),
				nn.Linear(AAdecoder_hidden[2], 20),
				nn.LogSoftmax(dim=1)
			)
		
		# Optional secondary structure prediction head
		if output_ss := kwargs.get('output_ss', False):
			self.head['ss_head'] = nn.Sequential(
				nn.Linear(d_model, AAdecoder_hidden[0]),
				nn.GELU(),
				nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(AAdecoder_hidden[1], AAdecoder_hidden[2]),
				nn.GELU(),
				nn.Linear(AAdecoder_hidden[2], 3),
				nn.LogSoftmax(dim=1)
			)

	def forward(self, data, **kwargs):
		x = data.x_dict['res']
		
		# ===================== INPUT PROCESSING =====================
		if self.concat_positions:
			x = torch.cat([x, data['positions'].x], dim=1)
		if self.learn_positions:
			pos_enc = self.input['position_mlp'](data['positions'].x)
			x = torch.cat([x, pos_enc], dim=1)
		
		inz = x.clone()
		x = self.input['dropout'](x)
		x = self.input['proj'](x)
		
		# ===================== BODY PROCESSING =====================
		# Transformer expects (seq_len, batch, d_model)
		batch = data['res'].batch
		
		if batch is not None:
			# Find the number of graphs in the batch
			num_graphs = batch.max().item() + 1
			# Split x into a list of tensors, one per graph
			x_split = [x[batch == i] for i in range(num_graphs)]
			# Pad sequences to the same length
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
		
		# Apply transformer
		x = self.body['transformer_encoder'](x)  # (seq_len, batch, d_model)
		
		# Apply residual connection
		if self.residual and 'dmodel2input' in self.body:
			if batch is not None:
				x = x + self.body['dmodel2input'](x)
			else:
				x = x + self.body['dmodel2input'](x)
		
		# Apply normalization
		if self.normalize:
			x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-10)
		
		# ===================== HEAD PROCESSING =====================
		if batch is not None:
			# Remove padding and concatenate results for all graphs in the batch
			aa_list = []
			ss_list = []
			for i, xi in enumerate(x.split(1, dim=1)):  # xi: (seq_len, 1, d_model)
				# Remove batch dimension and padding
				seq_len = (batch == i).sum().item()
				
				if 'cnn_decoder' in self.head:
					# Apply CNN decoder
					xi = self.head['prenorm'](xi.squeeze(1))  # (seq_len, d_model)
					if 'ss_head' in self.head:
						ss_list.append(self.head['ss_head'](xi[:seq_len, :]))
					xi_cnn = xi.permute(1, 0).unsqueeze(0)  # (1, d_model, seq_len)
					xi_cnn = self.head['cnn_decoder'](xi_cnn)  # (1, 20, seq_len)
					xi_cnn = xi_cnn.permute(2, 0, 1).squeeze(1)  # (seq_len, 20)
					aa_list.append(F.log_softmax(xi_cnn[:seq_len, :], dim=-1))
				else:
					aa_list.append(self.head['dnn_decoder'](xi[:seq_len, 0]))
					if 'ss_head' in self.head:
						ss_list.append(self.head['ss_head'](xi[:seq_len, 0]))
			
			aa = torch.cat(aa_list, dim=0)
			ss = torch.cat(ss_list, dim=0) if ss_list else None
			return {'aa': aa, 'ss_pred': ss}
		else:
			ss = None
			if 'cnn_decoder' in self.head:
				# Apply CNN decoder
				x = self.head['prenorm'](x)
				if 'ss_head' in self.head:
					ss = self.head['ss_head'](x)
				x_cnn = x.permute(1, 2, 0)  # (batch, d_model, seq_len)
				x_cnn = self.head['cnn_decoder'](x_cnn)  # (batch, 20, seq_len)
				x_cnn = x_cnn.permute(2, 0, 1)  # (seq_len, batch, 20)
				aa = F.log_softmax(x_cnn, dim=-1)
			else:
				aa = self.head['dnn_decoder'](x)
				if 'ss_head' in self.head:
					ss = self.head['ss_head'](x)
			return {'aa': aa, 'ss_pred': ss}
	
	def x_to_amino_acid_sequence(self, x_r):
		indices = torch.argmax(x_r, dim=1)
		amino_acid_sequence = ''.join(self.amino_acid_indices[idx.item()] for idx in indices)
		return amino_acid_sequence


class Transformer_Geometry_Decoder(torch.nn.Module):
	"""
	Muon-compatible Transformer geometry decoder with modular architecture.
	Separates input, body, and head modules for compatibility with Muon optimizer.
	
	- input: Preprocessing and initial projection (optimized with AdamW)
	- body: Transformer encoder layers (weights optimized with Muon, gains/biases with AdamW)
	- head: Prediction heads for rotation-translation, secondary structure, and bond angles (optimized with AdamW)
	"""
	def __init__(
		self,
		in_channels={'res': 10},
		hidden_channels={'res_backbone_res': [20, 20, 20]},
		concat_positions=True,
		nheads=4,
		layers=2,
		RTdecoder_hidden=[128, 64, 32],
		ssdecoder_hidden=[128, 64, 32],
		anglesdecoder_hidden=[128, 64, 32],
		dropout=0.001,
		normalize=True,
		residual=True,
		learn_positions=False,
		output_rt=True,
		output_ss=True,
		output_angles=True,
		**kwargs
	):
		super(Transformer_Geometry_Decoder, self).__init__()
		L.seed_everything(42)

		self.concat_positions = concat_positions
		self.learn_positions = learn_positions
		self.normalize = normalize
		self.residual = residual
		self.output_rt = output_rt
		self.output_ss = output_ss
		self.output_angles = output_angles
		
		input_dim = in_channels['res']
		if concat_positions:
			input_dim = input_dim + 256
		if learn_positions:
			self.concat_positions = False
			input_dim = input_dim + 32
		d_model = hidden_channels[('res', 'backbone', 'res')][0]

		print(f"Transformer_Geometry_Decoder: d_model={d_model}, nheads={nheads}, layers={layers}, dropout={dropout}")

		# ===================== INPUT MODULE =====================
		# Preprocessing and initial projection
		self.input = nn.ModuleDict()
		
		self.input['dropout'] = nn.Dropout(dropout)
		
		if learn_positions:
			self.input['position_mlp'] = Position_MLP(in_channels=256, hidden_channels=[128, 64], out_channels=32, dropout=dropout)
		
		self.input['proj'] = nn.Sequential(
			nn.Linear(input_dim, d_model),
			nn.GELU(),
			nn.Linear(d_model, d_model),
			nn.GELU(),
			nn.Tanh(),
		)

		# ===================== BODY MODULE =====================
		# Transformer encoder layers
		self.body = nn.ModuleDict()
		
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model, 
			nhead=nheads, 
			dropout=dropout
		)
		self.body['transformer_encoder'] = nn.TransformerEncoder(
			encoder_layer, 
			num_layers=layers
		)
		
		if self.residual:
			self.body['dmodel2input'] = nn.Sequential(
				nn.Linear(d_model, input_dim),
				nn.GELU(),
				nn.Linear(input_dim, input_dim),
			)

		# ===================== HEAD MODULE =====================
		# Geometry prediction heads
		self.head = nn.ModuleDict()
		
		# Rotation-translation prediction head (7D output: quaternion + translation)
		if output_rt:
			if not isinstance(RTdecoder_hidden, list):
				RTdecoder_hidden = [RTdecoder_hidden, RTdecoder_hidden]
			self.head['rt_head'] = nn.Sequential(
				nn.Linear(d_model, RTdecoder_hidden[0]),
				nn.GELU(),
				nn.Linear(RTdecoder_hidden[0], RTdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(RTdecoder_hidden[1], RTdecoder_hidden[2] if len(RTdecoder_hidden) > 2 else RTdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(RTdecoder_hidden[2] if len(RTdecoder_hidden) > 2 else RTdecoder_hidden[1], 7)  # 4 for quaternion + 3 for translation
			)
		
		# Secondary structure prediction head (3-class: helix, sheet, coil)
		if output_ss:
			if not isinstance(ssdecoder_hidden, list):
				ssdecoder_hidden = [ssdecoder_hidden, ssdecoder_hidden]
			self.head['ss_head'] = nn.Sequential(
				nn.LayerNorm(d_model),
				nn.Linear(d_model, ssdecoder_hidden[0]),
				nn.GELU(),
				nn.Linear(ssdecoder_hidden[0], ssdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(ssdecoder_hidden[1], ssdecoder_hidden[2] if len(ssdecoder_hidden) > 2 else ssdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(ssdecoder_hidden[2] if len(ssdecoder_hidden) > 2 else ssdecoder_hidden[1], 3),
				nn.LogSoftmax(dim=1)
			)
		
		# Bond angles prediction head (phi, psi, omega)
		if output_angles:
			if not isinstance(anglesdecoder_hidden, list):
				anglesdecoder_hidden = [anglesdecoder_hidden, anglesdecoder_hidden]
			self.head['angles_head'] = nn.Sequential(
				nn.Linear(d_model, anglesdecoder_hidden[0]),
				nn.GELU(),
				nn.Linear(anglesdecoder_hidden[0], anglesdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(anglesdecoder_hidden[1], anglesdecoder_hidden[2] if len(anglesdecoder_hidden) > 2 else anglesdecoder_hidden[1]),
				nn.GELU(),
				nn.Linear(anglesdecoder_hidden[2] if len(anglesdecoder_hidden) > 2 else anglesdecoder_hidden[1], 3),
				nn.Tanh()  # Output in [-1, 1], will be scaled to [-π, π]
			)

	def forward(self, data, contact_pred_index=None, **kwargs):
		x = data.x_dict['res']
		
		# ===================== INPUT PROCESSING =====================
		if self.concat_positions:
			x = torch.cat([x, data['positions'].x], dim=1)
		if self.learn_positions:
			pos_enc = self.input['position_mlp'](data['positions'].x)
			x = torch.cat([x, pos_enc], dim=1)
		
		x = self.input['dropout'](x)
		x = self.input['proj'](x)
		
		# ===================== BODY PROCESSING =====================
		# Transformer expects (seq_len, batch, d_model)
		batch = data['res'].batch if hasattr(data['res'], 'batch') else None
		
		if batch is not None:
			# Find the number of graphs in the batch
			num_graphs = batch.max().item() + 1
			# Split x into a list of tensors, one per graph
			x_split = [x[batch == i] for i in range(num_graphs)]
			# Pad sequences to the same length
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
		
		# Apply transformer
		x = self.body['transformer_encoder'](x)  # (seq_len, batch, d_model)
		
		# Apply residual connection
		if self.residual and 'dmodel2input' in self.body:
			x = x + self.body['dmodel2input'](x)
		
		# Apply normalization
		if self.normalize:
			x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-10)
		
		# ===================== HEAD PROCESSING =====================
		rt_pred = None
		ss_pred = None
		angles = None
		
		if batch is not None:
			# Remove padding and concatenate results for all graphs in the batch
			rt_list = []
			ss_list = []
			angles_list = []
			
			for i, xi in enumerate(x.split(1, dim=1)):  # xi: (seq_len, 1, d_model)
				# Remove batch dimension and padding
				seq_len = (batch == i).sum().item()
				xi = xi[:seq_len, 0]  # (seq_len, d_model)
				
				if self.output_rt and 'rt_head' in self.head:
					rt_list.append(self.head['rt_head'](xi))
				
				if self.output_ss and 'ss_head' in self.head:
					ss_list.append(self.head['ss_head'](xi))
				
				if self.output_angles and 'angles_head' in self.head:
					angles_list.append(self.head['angles_head'](xi))
			
			if rt_list:
				rt_pred = torch.cat(rt_list, dim=0)
			if ss_list:
				ss_pred = torch.cat(ss_list, dim=0)
			if angles_list:
				angles = torch.cat(angles_list, dim=0)
				angles = angles * np.pi  # Scale from [-1, 1] to [-π, π]
		else:
			# Single graph case
			x = x.squeeze(1)  # (seq_len, d_model)
			
			if self.output_rt and 'rt_head' in self.head:
				rt_pred = self.head['rt_head'](x)
			
			if self.output_ss and 'ss_head' in self.head:
				ss_pred = self.head['ss_head'](x)
			
			if self.output_angles and 'angles_head' in self.head:
				angles = self.head['angles_head'](x)
				angles = angles * np.pi  # Scale from [-1, 1] to [-π, π]
		
		return {
			'rt_pred': rt_pred,
			'ss_pred': ss_pred,
			'angles': angles,
			'z': x.squeeze(1) if batch is None else None  # Return latent for potential use
		}

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

class Transformer_Foldx_Decoder(torch.nn.Module):
	def __init__(
		self,
		input_dim=10,
		foldx_dim=23,
		concat_positions=True,
		d_model=128,
		nhead=8,
		num_layers=2,
		foldx_hidden=[128, 64, 32],
		attn_hidden=64,
		dropout=0.001,
		normalize=True,
		residual=True,
	):
		super(Transformer_Foldx_Decoder, self).__init__()
		L.seed_everything(42)

		self.concat_positions = concat_positions
		self.normalize = normalize
		self.residual = residual

		if concat_positions:
			input_dim = input_dim + 256

		self.input_proj = nn.Linear(input_dim, d_model)
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		self.attn_pool = AttentionPooling(d_model, attn_hidden)

		self.lin = torch.nn.Sequential(
			torch.nn.Dropout(dropout),
			DynamicTanh(d_model, channels_last=True),
			torch.nn.Linear(d_model, foldx_hidden[0]),
			torch.nn.GELU(),
			torch.nn.Linear(foldx_hidden[0], foldx_hidden[1]),
			torch.nn.GELU(),
			torch.nn.Linear(foldx_hidden[1], foldx_hidden[2]),
			torch.nn.GELU(),
			DynamicTanh(foldx_hidden[2], channels_last=True),
			torch.nn.Linear(foldx_hidden[2], foldx_dim)
		)

	def forward(self, data, **kwargs):
		x = data.x_dict['res']
		if self.concat_positions:
			x = torch.cat([x, data['positions'].x], dim=1)
		x = self.input_proj(x)
		x = self.transformer_encoder(x)
		# Attention pooling over all residues (tokens)
		pooled = self.attn_pool(x)
		if self.residual:
			# No residual connection here, as pooled is a single vector
			pass
		if self.normalize:
			pooled = pooled / (torch.norm(pooled, dim=-1, keepdim=True) + 1e-10)
		foldx_out = self.lin(pooled)
		return { 'foldx_out' : foldx_out }


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

class MultiMonoDecoder(torch.nn.Module):
	"""
	A decoder that combines several mono decoders (e.g., sequence, contacts) and calls them in the forward pass.
	Pass a list of tasks (e.g., ['sequence', 'contacts']) and their configs to __init__.
	"""
	def __init__(self,  configs):
		super().__init__()
		self.decoders = torch.nn.ModuleDict()
		self.configs = configs
		#initialize decoders based on configs
		for task in configs.keys():
			print(f"Initializing decoder for task: {task}")
			print( task == 'sequence' , task == 'sequence_transformer' , task == 'contacts' , task == 'geometry' , task == 'foldx')
			if task == 'sequence':
				self.decoders['sequence'] = HeteroGAE_AA_Decoder(**configs['sequence'])
			if task == 'sequence_transformer':
				self.decoders['sequence_transformer'] = Transformer_AA_Decoder(**configs['sequence_transformer'])
			elif task == 'contacts':
				self.decoders['contacts'] = HeteroGAE_geo_Decoder(**configs['contacts'])
			elif task == 'foldx':
				self.decoders['foldx'] = HeteroGAE_geo_Decoder(**configs['foldx'])
			elif task == 'pinn':
				#throw an error if pinn decoder is not implemented
				raise NotImplementedError("PINN decoder is not implemented yet.")
			elif task == 'geometry_transformer':
				#throw an error if geometry_transformer decoder is not implemented
				self.decoders['geometry_transformer'] = Transformer_Geometry_Decoder(**configs['geometry_transformer'])
			elif task == 'geometry_cnn':
				self.decoders['geometry_cnn'] = CNN_geo_Decoder(**configs['geometry_cnn'])
	def forward(self, data, contact_pred_index=None, **kwargs):
		results = {}
		for task, decoder in self.decoders.items():
			#if a decoder returns a value for a key that already exists in results that is none
			#and existing value is not none, keep the existing value
			#otherwise, update the results with the new value
			for key, value in decoder(data, contact_pred_index=contact_pred_index, **kwargs).items():
				if key in results and results[key] is not None and value is None:
					continue
				else:
					results[key] = value
		return results



