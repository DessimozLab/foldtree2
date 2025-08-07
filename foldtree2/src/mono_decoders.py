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


from foldtree2.src.losses import *
from foldtree2.src.dynamictan import *
from foldtree2.src.quantizers import *

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
		
		if output_fft == True:
			#todo implement siren layer to output spatial frequencies
			#massage output to distmat
			#take fft2 of distmat
			#calc spatial and fft losses
			
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


	def forward(self, data , contact_pred_index, **kwargs):	
		
		#apply batch norm to res

		xdata, edge_index = data.x_dict, data.edge_index_dict
		xdata['res'] = self.dropout(xdata['res'])
		if self.concat_positions == True:
			xdata['res'] = torch.cat([xdata['res'], data['positions'].x], dim=1)
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
		
		angles = None
		if self.angles_mlp is not None:
			angles = self.angles_mlp( z )
			#tanh is -1 to 1, multiply by pi to get angles in radians
			angles = angles * np.pi

		if contact_pred_index is None:
			return { 'edge_probs': None , 'zgodnode' :None , 'fft2pred':fft2_pred , 'rt_pred': None , 'angles': angles }

		else:
			if self.contact_mlp is None:
				edge_probs = self.sigmoid( torch.sum( z[contact_pred_index[0]] * z[contact_pred_index[1]] , axis =1 ) )
			else:
				edge_scores = self.contact_mlp( torch.concat( [z[contact_pred_index[0]], z[contact_pred_index[1]] ] , axis = 1 ) ).squeeze()
				edge_probs = self.sigmoid(edge_scores)

		if 'init' in kwargs and kwargs['init'] == True:
			# Initialize weights explicitly (Xavier initialization)
			for conv in self.convs:
				for c in conv.convs.values():
					for param in c.parameters():
						if param.dim() > 1:
							nn.init.xavier_uniform_(param)

		return  { 'edge_probs': edge_probs , 'zgodnode' :zgodnode , 'fft2pred':fft2_pred  , 'rt_pred': rt_pred , 'angles': angles }


class Transformer_Geo_Decoder(torch.nn.Module):
	def __init__(
		self,
		in_channels={'res': 10, 'godnode4decoder': 5, 'foldx': 23},
		hidden_channels={'res_backbone_res': [128, 128, 128]},
		concat_positions=False,
		nheads=4,
		layers=2,
		Xdecoder_hidden=[128, 64, 32],
		FFT2decoder_hidden=[64, 32],
		contactdecoder_hidden=[64, 32],
		anglesdecoder_hidden=[64, 32],
		RTdecoder_hidden=[64, 32],
		metadata={},
		dropout=0.001,
		output_fft=False,
		output_rt=False,
		output_angles=False,
		normalize=True,
		residual=True,
		contact_mlp=True,
		**kwargs
	):
		super().__init__()
		L.seed_everything(42)
		self.concat_positions = concat_positions
		self.normalize = normalize
		self.residual = residual
		self.output_fft = output_fft
		self.output_rt = output_rt
		self.output_angles = output_angles
		self.contact_mlp_flag = contact_mlp
		self.metadata = metadata

		input_dim = in_channels['res']
		if concat_positions:
			input_dim += 256
		d_model = hidden_channels[('res', 'backbone', 'res')][0] if ('res', 'backbone', 'res') in hidden_channels else list(hidden_channels.values())[0][0]

		self.input_proj = torch.nn.Sequential(
			torch.nn.Dropout(dropout),
			nn.Linear(input_dim, d_model),
			torch.nn.GELU(),
			nn.Linear(d_model, d_model),
			torch.nn.Tanh(),
		)

		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nheads, dropout=dropout)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

		self.dropout = torch.nn.Dropout(p=dropout)

		# Output head for residue embeddings
		self.lin = torch.nn.Sequential(
			torch.nn.Dropout(dropout),
			DynamicTanh(d_model, channels_last=True),
			torch.nn.Linear(d_model, Xdecoder_hidden[0]),
			torch.nn.GELU(),
			torch.nn.Linear(Xdecoder_hidden[0], Xdecoder_hidden[1]),
			torch.nn.GELU(),
			torch.nn.Linear(Xdecoder_hidden[1], input_dim if residual else Xdecoder_hidden[-1]),
		)

		# RT decoder
		if output_rt:
			self.rt_mlp = torch.nn.Sequential(
				torch.nn.Dropout(dropout),
				torch.nn.Linear(input_dim if residual else Xdecoder_hidden[-1], RTdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(RTdecoder_hidden[0], RTdecoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(RTdecoder_hidden[1], 7),
			)
		else:
			self.rt_mlp = None

		# Angles decoder
		if output_angles:
			self.angles_mlp = torch.nn.Sequential(
				torch.nn.Dropout(dropout),
				torch.nn.Linear(input_dim if residual else Xdecoder_hidden[-1], anglesdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(anglesdecoder_hidden[0], anglesdecoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(anglesdecoder_hidden[1], 3),
				torch.nn.Tanh(),
			)
		else:
			self.angles_mlp = None
		
		#attention based aggregation for godnode
		# Attention-based aggregation for godnode/global state
		self.godnodedecoder = None
		if output_fft:
			# Use attention pooling to aggregate residue embeddings into a fixed-size godnode embedding
			self.godnode_attn_pool = AttentionPooling(d_model, d_model)
			self.godnodedecoder = torch.nn.Sequential(
				torch.nn.Linear(d_model, FFT2decoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(FFT2decoder_hidden[0], FFT2decoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(FFT2decoder_hidden[1], in_channels.get('fft2i', 0) + in_channels.get('fft2r', 0))
			)
		self.sigmoid = nn.Sigmoid()

	def forward(self, data, contact_pred_index=None, **kwargs):
		x = data.x_dict['res']
		if self.concat_positions:
			x = torch.cat([x, data['positions'].x], dim=1)
		inz = x.clone()
		x = self.input_proj(x)
		# Transformer expects (seq_len, batch, d_model), so add batch dim if needed
		batch = data['res'].batch if hasattr(data['res'], 'batch') else None
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

		x_layers = []
		out = x
		for i in range(self.transformer_encoder.num_layers):
			out = self.transformer_encoder.layers[i](out)
			out = F.gelu(out)
		z = self.lin(out)
		if self.residual:
			z = z + inz
		if self.normalize:
			z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-10)

		# decode godnode
		fft2_pred = None
		zgodnode = None
		
		if self.output_fft and self.godnodedecoder is not None:
			# Attention-based pooling over residue embeddings to get a godnode embedding
			# x: (seq_len, batch, d_model) or (seq_len, 1, d_model)
			if x.dim() == 3:
				# Pool over the first dimension (seq_len)
				# If batch > 1, pool each graph separately
				if x.shape[1] == 1:
					# Single graph
					zgodnode = self.godnode_attn_pool(x[:, 0, :])
					zgodnode = zgodnode.unsqueeze(0)  # (1, d_model)
				else:
					zgodnode = torch.stack([self.godnode_attn_pool(x[:, i, :]) for i in range(x.shape[1])], dim=0)
			else:
				# x is (seq_len, d_model)
				zgodnode = self.godnode_attn_pool(x)
				zgodnode = zgodnode.unsqueeze(0)
			# Decode the godnode embedding to get FFT2 prediction
			zgodnode = zgodnode.squeeze(0)  # (d_model,)
			if zgodnode.dim() == 1:
				zgodnode = zgodnode.unsqueeze(0)  # Ensure it's (1, d_model)
			if zgodnode.shape[0] != self.godnodedecoder[0].in_features:
				raise ValueError(f"zgodnode shape mismatch: expected {self.godnodedecoder[0].in_features}, got {zgodnode.shape[0]}")
			# Pass through the godnode decoder
			fft2_pred = self.godnodedecoder(zgodnode)
			
		rt_pred = None
		if self.rt_mlp is not None:
			rt_pred = self.rt_mlp(z)

		angles = None
		if self.angles_mlp is not None:
			angles = self.angles_mlp(z)
			angles = angles * np.pi

		return {  'fft2pred': fft2_pred, 'rt_pred': rt_pred, 'angles': angles}

class HeteroGAE_AA_Decoder(torch.nn.Module):
	def __init__(self, in_channels={'res': 10},
			  	 xdim=20, concat_positions=True, 
				 hidden_channels={'res_backbone_res': [20, 20, 20]}, 
				 layers=3,
				AAdecoder_hidden=[20, 20, 20], 
				amino_mapper=None, 
				flavor=None, 
				dropout=0.001, 
				normalize=True, 
				residual=True , **kwargs):
		
		super(HeteroGAE_AA_Decoder, self).__init__()
		L.seed_everything(42)

		if concat_positions:
			in_channels['res'] = in_channels['res'] + 256

		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		self.convs = torch.nn.ModuleList()
		self.norms = torch.nn.ModuleList()
		self.normalize = normalize
		self.concat_positions = concat_positions
		in_channels_orig = copy.deepcopy(in_channels)
		self.hidden_channels = hidden_channels
		self.in_channels = in_channels
		self.amino_acid_indices = amino_mapper
		self.nlayers = layers
		self.dropout = torch.nn.Dropout(p=dropout)
		self.jk = JumpingKnowledge(mode='cat')
		self.residual = residual
		finalout = list(hidden_channels.values())[-1][-1]
		for i in range(layers):
			layer = {}
			for k, edge_type in enumerate(hidden_channels.keys()):
				datain = edge_type[0]
				dataout = edge_type[2]
				if flavor == 'gat':
					layer[edge_type] = GATv2Conv((-1, -1), hidden_channels[edge_type][i], heads=3, concat=False)
				if flavor == 'mfconv':
					layer[edge_type] = MFConv((-1, -1), hidden_channels[edge_type][i], max_degree=5, aggr='mean')
				if flavor == 'transformer' or edge_type == ('res', 'informs', 'godnode4decoder'):
					layer[edge_type] = TransformerConv((-1, -1), hidden_channels[edge_type][i], heads=3, concat=False)
				if flavor == 'sage' :
					layer[edge_type] = SAGEConv((-1, -1), hidden_channels[edge_type][i])
				if k == 0 and i == 0:
					in_channels[dataout] = hidden_channels[edge_type][i]
				if k == 0 and i > 0:
					in_channels[dataout] = hidden_channels[edge_type][i-1]
				if k > 0 and i > 0:
					in_channels[dataout] = hidden_channels[edge_type][i]
				if k > 0 and i == 0:
					in_channels[dataout] = hidden_channels[edge_type][i]
			conv = HeteroConv(layer, aggr='mean')
			self.convs.append(conv)
			self.norms.append(GraphNorm(finalout))
		if self.residual:
			lastlin = in_channels_orig['res']
		else:
			lastlin = AAdecoder_hidden[-1]

		self.lin = torch.nn.Sequential(
			torch.nn.Dropout(dropout),
			#DynamicTanh(finalout * layers, channels_last=True),
			torch.nn.Linear(finalout * layers, AAdecoder_hidden[0]),
			torch.nn.GELU(),
			torch.nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1]),
			torch.nn.GELU(),
			torch.nn.Linear(AAdecoder_hidden[1], lastlin),
			torch.nn.Tanh(),
			#DynamicTanh(lastlin, channels_last=True),
		)

		self.aadecoder = torch.nn.Sequential(
			torch.nn.Dropout(dropout),
			#DynamicTanh(lastlin + in_channels_orig['res'], channels_last=True),
			torch.nn.Linear(lastlin + in_channels_orig['res'], AAdecoder_hidden[0]),
			torch.nn.GELU(),
			torch.nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1]),
			torch.nn.GELU(),
			#torch.nn.Linear(AAdecoder_hidden[1], AAdecoder_hidden[2]),
			#torch.nn.GELU(),
			#DynamicTanh(AAdecoder_hidden[1], channels_last=True),
			torch.nn.Linear(AAdecoder_hidden[1], xdim),
			torch.nn.LogSoftmax(dim=1)
		)

	def forward(self, data, **kwargs):
		xdata, edge_index = data.x_dict, data.edge_index_dict
		xdata['res'] = self.dropout(xdata['res'])
		if self.concat_positions:
			xdata['res'] = torch.cat([xdata['res'], data['positions'].x], dim=1)
		inz = xdata['res'].clone()
		x_dict_list = []
		for i, layer in enumerate(self.convs):
			if i > 0:
				prev = xdata['res'].clone()
			xdata = layer(xdata, edge_index)
			xdata['res'] = F.gelu(xdata['res'])
			xdata['res'] = self.norms[i](xdata['res'])
			if i > 0:
				xdata['res'] = xdata['res'] + prev
			x_dict_list.append(xdata['res'])
		xdata['res'] = self.jk(x_dict_list)
		z = self.lin(xdata['res'])
		if self.residual:
			z = z + inz
		if self.normalize:
			z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-10)
		decoder_in = torch.cat([inz, z], axis=1)
		aa = self.aadecoder(decoder_in)
		return { 'aa':aa }

	def x_to_amino_acid_sequence(self, x_r):
		indices = torch.argmax(x_r, dim=1)
		amino_acid_sequence = ''.join(self.amino_acid_indices[idx.item()] for idx in indices)
		return amino_acid_sequence

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
			torch.nn.Linear(d_model, AAdecoder_hidden[0]),
			torch.nn.GELU(),
			torch.nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1]),
			torch.nn.GELU(),
			torch.nn.Linear(AAdecoder_hidden[1], AAdecoder_hidden[2]),
			#torch.nn.GELU(),
			#DynamicTanh(AAdecoder_hidden[2], channels_last=True),
			#torch.nn.Linear(AAdecoder_hidden[2], xdim),
			#torch.nn.GELU(),
			torch.nn.LogSoftmax(dim=1)
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
		self.contactdecoder = torch.nn.Sequential(
				torch.nn.Linear( 2 * ( Xdecoder_hidden + in_channels) , contactdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(contactdecoder_hidden[0], contactdecoder_hidden[1] ) ,
				torch.nn.GELU(),
				torch.nn.Linear(contactdecoder_hidden[1], contactdecoder_hidden[2] ) ,
				torch.nn.GELU(),
				NormTanh(),
				torch.nn.Linear(contactdecoder_hidden[2], 1),
				torch.nn.Sigmoid()
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

	def forward2(self, x1data, x2data, contact_pred_index, **kwargs):
		x1data, x1edge_index = x1data.x_dict, x1data.edge_index_dict
		x2data, x2edge_index = x2data.x_dict, x2data.edge_index_dict
		#z = self.bn(z)
		#copy z for later concatenation
		inz1 = x1data['res'].copy()
		inz2 = x2data['res'].copy()
		inzs = [ inz1, inz2 ]
		xdatas = [ x1data, x2data ]
		indices = [ x1edge_index, x2edge_index ]
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
		z1, z2 = decoder_inputs
		g1 , g2 = godnodes
		foldx_pred = self.godnodedecoder( torch.cat(godnodes, dim=1) )
		interaction_prob = jaccard_distance_multiset(g1 , g2)
		if contact_pred_index is None:
			return interaction_prob, None, godnodes , foldx_pred
		edge_probs = self.contactdecoder(torch.cat( ( z1[contact_pred_index[0]] , z2[contact_pred_index[1]] ) ,dim= 1) )
		return interaction_prob, edge_probs, godnodes , foldx_pred
	
	def forward1(self, x1data, **kwargs):
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
				z1, g1 = self.forward1(data)
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
		for task in configs.keys():
			print(f"Initializing decoder for task: {task}")
			print( task == 'sequence' , task == 'sequence_transformer' , task == 'contacts' , task == 'geometry' , task == 'foldx')
			if task == 'sequence':
				self.decoders['sequence'] = HeteroGAE_AA_Decoder(**configs['sequence'])
			if task == 'sequence_transformer':
				self.decoders['sequence_transformer'] = Transformer_AA_Decoder(**configs['sequence_transformer'])
			elif task == 'contacts':
				self.decoders['contacts'] = HeteroGAE_geo_Decoder(**configs['contacts'])
			elif task == 'contacts_transformer':
				self.decoders['contacts_transformer'] = Transformer_Foldx_Decoder(**configs['contacts_transformer'])
			elif task == 'foldx':
				self.decoders['foldx'] = HeteroGAE_geo_Decoder(**configs['foldx'])
			elif task == 'pinn':
				#throw an error if pinn decoder is not implemented
				raise NotImplementedError("PINN decoder is not implemented yet.")

	def forward(self, data, contact_pred_index=None, **kwargs):
		results = {}
		for task, decoder in self.decoders.items():
			results.update(decoder(data, contact_pred_index=contact_pred_index, **kwargs))
		return results



