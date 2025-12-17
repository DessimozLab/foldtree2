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
from foldtree2.src.dynamictan import *
from foldtree2.src.quantizers import *
from foldtree2.src.se3_struct_decoder import *

from  torch_geometric.utils import to_undirected
#encoder super class



class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

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
			#tanh is -1 to 1, multiply by 2pi to get angles in radians
			angles = angles * 2 * np.pi

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
	def __init__(self, in_channels = {'res':10 , 'godnode4decoder':5 , 'foldx':23 },
				concat_positions = False, 
				conv_channels = [64, 128, 256],
				kernel_sizes = [3, 3, 3],
				FFT2decoder_hidden = [10, 10],
				contactdecoder_hidden = [10, 10],
				ssdecoder_hidden = [10, 10],
				Xdecoder_hidden=[30, 30], 
				anglesdecoder_hidden=[30, 30],
				RTdecoder_hidden=[30, 30],
				metadata={}, 
				dropout= .001,
				output_fft = False,
				output_rt = False,
				output_angles = False,
				output_ss = False,
				normalize = True,
				residual = True,
				output_edge_logits = False,
				ncat = 16,
				contact_mlp = True,
				pool_type = 'global_mean' ):
		super(CNN_geo_Decoder, self).__init__()
		# Setting the seed
		L.seed_everything(42)
		
		if concat_positions == True:
			in_channels['res'] = in_channels['res'] + 256
		
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		
		self.normalize = normalize
		self.concat_positions = concat_positions
		in_channels_orig = copy.deepcopy(in_channels)
		self.metadata = metadata
		self.output_fft = output_fft
		self.dropout = torch.nn.Dropout(p=dropout)
		self.residual = residual
		self.pool_type = pool_type
		
		# CNN layers
		self.convs = torch.nn.ModuleList()
		self.norms = torch.nn.ModuleList()
		
		input_dim = in_channels['res']
		for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
			# 1D convolution for sequence-like data
			self.convs.append(
				torch.nn.Conv1d(input_dim if i == 0 else conv_channels[i-1], 
								out_channels, 
								kernel_size=kernel_size, 
								padding=kernel_size//2)
			)
			self.norms.append(torch.nn.BatchNorm1d(out_channels))
			
		finalout = conv_channels[-1]
		
		if self.residual == True:
			lastlin = in_channels_orig['res']
		else:
			lastlin = Xdecoder_hidden[-1]

		self.lastlin = lastlin

		self.lin = torch.nn.Sequential(
				torch.nn.Linear(finalout, Xdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(Xdecoder_hidden[0], Xdecoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(Xdecoder_hidden[1], lastlin),				
				)
		
		if output_fft == True:	
			self.godnodedecoder = torch.nn.Sequential(
					torch.nn.Linear(in_channels['godnode4decoder'], FFT2decoder_hidden[0]),
					torch.nn.GELU(),
					torch.nn.Linear(FFT2decoder_hidden[0], FFT2decoder_hidden[1]),
					torch.nn.GELU(),
					torch.nn.Linear(FFT2decoder_hidden[1], in_channels['fft2i'] + in_channels['fft2r']),
				)
		else:
			self.godnodedecoder = None

		if contact_mlp == True:
			self.contact_mlp = torch.nn.Sequential(
				torch.nn.Dropout(dropout),
				torch.nn.Linear(2*lastlin, contactdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(contactdecoder_hidden[0], contactdecoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(contactdecoder_hidden[1], 1))
		else:
			self.contact_mlp = None

		if output_rt == True:
			if type(RTdecoder_hidden) is not list:
				RTdecoder_hidden = [RTdecoder_hidden, RTdecoder_hidden]
			self.rt_mlp = torch.nn.Sequential(
				torch.nn.Dropout(dropout),
				torch.nn.Linear(lastlin, RTdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(RTdecoder_hidden[0], RTdecoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(RTdecoder_hidden[1], 7))
		else:
			self.rt_mlp = None

		if output_ss == True:
			self.output_ss = True
			self.ss_mlp = torch.nn.Sequential(
				torch.nn.LayerNorm(lastlin),
				torch.nn.Linear(lastlin, 128),
				torch.nn.GELU(),
				torch.nn.Linear(128, 64),
				torch.nn.GELU(),
				torch.nn.Linear(64, 3),
				torch.nn.LogSoftmax(dim=1)
			)
		else:
			self.output_ss = False
			self.ss_mlp = None

		if output_angles == True:
			if type(anglesdecoder_hidden) is not list:
				anglesdecoder_hidden = [anglesdecoder_hidden, anglesdecoder_hidden]
			self.angles_mlp = torch.nn.Sequential(
				torch.nn.Linear(lastlin, anglesdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(anglesdecoder_hidden[0], anglesdecoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(anglesdecoder_hidden[1], 3),
				torch.nn.Tanh()
			)
		else:
			self.angles_mlp = None

		if output_edge_logits == True:
			self.output_edge_logits = True
			self.edge_logits_mlp = torch.nn.Sequential(
				torch.nn.LayerNorm(2*lastlin),
				torch.nn.Linear(2*lastlin, anglesdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(anglesdecoder_hidden[0], anglesdecoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(anglesdecoder_hidden[1], anglesdecoder_hidden[2] if len(anglesdecoder_hidden) > 2 else anglesdecoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(anglesdecoder_hidden[2] if len(anglesdecoder_hidden) > 2 else anglesdecoder_hidden[1], ncat),
				torch.nn.GELU(),
				torch.nn.Sigmoid()
			)
		else:
			self.output_edge_logits = False
			self.edge_logits_mlp = None

		self.sigmoid = nn.Sigmoid()

	def forward(self, data, contact_pred_index, **kwargs):
		xdata, _ = data.x_dict, data.edge_index_dict
		x = xdata['res']
		x = self.dropout(x)
		
		if self.concat_positions == True:
			x = torch.cat([x, data['positions'].x], dim=1)

		# Copy for residual connection
		inz = x.clone()
		
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
				for j, (conv, norm) in enumerate(zip(self.convs, self.norms)):
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
			
			for conv, norm in zip(self.convs, self.norms):
				x = conv(x)
				x = F.gelu(x)
				x = norm(x)
			
			x = x.squeeze(0).transpose(0, 1)  # (seq_len, features)

		# Final linear transformation
		z = self.lin(x)
		
		if self.residual == True:
			z = z + inz
		if self.normalize == True:
			z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-10)

		# Decode godnode
		fft2_pred = None
		if self.output_fft == True:
			zgodnode = xdata['godnode4decoder']
			fft2_pred = self.godnodedecoder(xdata['godnode4decoder'])
		else:
			zgodnode = None
		
		rt_pred = None
		if self.rt_mlp is not None:
			rt_pred = self.rt_mlp(torch.cat([inz, z], axis=1))
		
		ss_pred = None
		if self.ss_mlp is not None:
			ss_pred = self.ss_mlp(z)

		angles = None
		edge_logits = None

		if self.angles_mlp is not None:
			angles = self.angles_mlp(z)
			angles = angles * 2 * np.pi

		if contact_pred_index is None:
			return {'edge_probs': None, 'zgodnode': None, 'fft2pred': fft2_pred, 
					'rt_pred': None, 'angles': angles, 'edge_logits': edge_logits, 
					'ss_pred': ss_pred, 'z': z}
		else:
			if self.contact_mlp is None:
				edge_probs = self.sigmoid(torch.sum(z[contact_pred_index[0]] * z[contact_pred_index[1]], axis=1))
			else:
				edge_scores = self.contact_mlp(torch.concat([z[contact_pred_index[0]], z[contact_pred_index[1]]], axis=1)).squeeze()
				edge_probs = self.sigmoid(edge_scores)
			if self.edge_logits_mlp is not None:
				edge_logits = self.edge_logits_mlp(torch.concat([z[contact_pred_index[0]], z[contact_pred_index[1]]], axis=1)).squeeze()

		if 'init' in kwargs and kwargs['init'] == True:
			# Initialize weights explicitly (Xavier initialization)
			for conv in self.convs:
				for param in conv.parameters():
					if param.dim() > 1:
						nn.init.xavier_uniform_(param)

		return {'edge_probs': edge_probs, 'edge_logits': edge_logits, 'zgodnode': zgodnode, 
				'fft2pred': fft2_pred, 'rt_pred': rt_pred, 'angles': angles, 
				'ss_pred': ss_pred, 'z': z}


class Transformer_AA_Decoder(torch.nn.Module):
	def __init__(
		self,
		in_channels={'res':10},
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
		**kwargs
	):
		super(Transformer_AA_Decoder, self).__init__()
		L.seed_everything(42)

		self.concat_positions = concat_positions
		self.normalize = normalize
		self.residual = residual
		self.amino_acid_indices = amino_mapper
		input_dim = in_channels['res']
		if concat_positions:
			input_dim = input_dim + 256
		d_model = hidden_channels[('res' , 'backbone', 'res')][0]

		print( d_model , nheads , layers , dropout)

		self.input_proj = torch.nn.Sequential( 
			#DynamicTanh(input_dim, channels_last=True),
			torch.nn.Dropout(dropout),
			nn.Linear(input_dim, d_model), 
			torch.nn.GELU(),
			nn.Linear(d_model, d_model),
			torch.nn.Tanh(),
		)

		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nheads, dropout=dropout)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

		self.lin = torch.nn.Sequential(
			#layernorm
			torch.nn.LayerNorm(d_model),
			torch.nn.Linear(d_model, AAdecoder_hidden[0]),
			torch.nn.GELU(),
			torch.nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1]),
			torch.nn.GELU(),
			torch.nn.Linear(AAdecoder_hidden[1], AAdecoder_hidden[2]),
			torch.nn.GELU(),
			torch.nn.Linear(AAdecoder_hidden[2], xdim),
			torch.nn.LogSoftmax(dim=1)
		)

		self.cnn_decoder = None
		if use_cnn_decoder := kwargs.get('use_cnn_decoder', False):
			self.prenorm = torch.nn.LayerNorm(d_model)
			self.cnn_decoder = torch.nn.Sequential(
				# Reshape to (batch, channels, seq_len) for 1D conv
				#
				# Add channel dimension and transpose for conv1d
				# Conv1d expects (batch, channels, seq_len)
				torch.nn.Conv1d(d_model, AAdecoder_hidden[0], kernel_size=3, padding=1),
				torch.nn.GELU(),
				torch.nn.Conv1d(AAdecoder_hidden[0], AAdecoder_hidden[1], kernel_size=3, padding=1),
				torch.nn.GELU(),
				torch.nn.Conv1d(AAdecoder_hidden[1], AAdecoder_hidden[2], kernel_size=3, padding=1),
				torch.nn.GELU(),
				torch.nn.Conv1d(AAdecoder_hidden[2], xdim, kernel_size=1),
				# Transpose back to (seq_len, batch, features) and apply softmax
			)

	def forward(self, data, **kwargs):
		x = data.x_dict['res']
		if self.concat_positions:
			x = torch.cat([x, data['positions'].x], dim=1)
		inz = x.clone()
		x = self.input_proj(x)
		# Transformer expects (seq_len, batch, d_model), so add batch dim if needed
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
		
		x = self.transformer_encoder(x)  # (N, batch, d_model)
		
		if self.cnn_decoder is not None:
			# Apply CNN decoder
			x = self.prenorm(x)
			x_cnn = x.permute(1, 2, 0)  # (batch, d_model, seq_len)
			x_cnn = self.cnn_decoder(x_cnn)  # (batch, xdim, seq_len)
			x_cnn = x_cnn.permute(2, 0, 1)  # (seq_len, batch, xdim)
			aa = F.log_softmax(x_cnn, dim=-1)
		else:
			aa = self.lin(x)
		
		if batch is not None:
			# Remove padding and concatenate results for all graphs in the batch
			aa_list = []
			for i, xi in enumerate(x.split(1, dim=1)):  # xi: (seq_len, 1, d_model)
				# Remove batch dimension and padding (assume original lengths from batch)
				seq_len = (batch == i).sum().item()
				aa_list.append(self.lin(xi[:seq_len, 0]))
			aa = torch.cat(aa_list, dim=0)
		return  {'aa': aa }

	def x_to_amino_acid_sequence(self, x_r):
		indices = torch.argmax(x_r, dim=1)
		amino_acid_sequence = ''.join(self.amino_acid_indices[idx.item()] for idx in indices)
		return amino_acid_sequence


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
			results.update(decoder(data, contact_pred_index=contact_pred_index, **kwargs))
		return results



