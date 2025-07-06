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

from losses import *
from dynamictan import *
from quantizers import *

from  torch_geometric.utils import to_undirected
#encoder super class
class Encoder(torch.nn.Module):
	def __init__() :
		super(Encoder, self).__init__()
		#save all arguments to constructor
		self.args = locals()
		self.args.pop('self')
		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		self.vector_quantizer = None
		
	def forward(self, x_dict, edge_index_dict):
		raise NotImplementedError('forward method not implemented')

	def structlist_loader(self, structlist, batch_size = 1):
		#load a list of structures into a dataloader
		dataloader = DataLoader(structlist, batch_size=batch_size, shuffle=False)
		return dataloader	

	def encode_structures_fasta(self, dataloader, filename = 'structalign.strct.fasta' , verbose = False , alphabet = None , replace = False):
		#write an encoded fasta for use with mafft and iqtree. only doable with alphabet size of less that 248
		#0x01 – 0xFF excluding > (0x3E), = (0x3D), < (0x3C), - (0x2D), Space (0x20), Carriage Return (0x0d) and Line Feed (0x0a)
		replace_dict = { '>' : chr(249), '=' : chr(250), '<' : chr(251), '-' : chr(252), ' ' : chr(253) , '\r' : chr(254), '\n' : chr(255) }
		#check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		
		if alphabet is not None:
			print('using alphabet')
			print(alphabet)
		
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z,qloss = self.forward(data.x_dict , data.edge_index_dict)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = data.identifier
				f.write(f'>{identifier}\n')
				outstr = ''
				for char in strdata[0]:
					#start at 0x01
					if alphabet is not None:
						char = alphabet[char]
					else:
						char = chr(char+1)
					
					if replace and char in replace_dict:
						char = replace_dict[char]
					outstr += char
					f.write(char)

				f.write('\n')

				if verbose == True:
					print(identifier, outstr)
		return filename


class mk1_Encoder(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, 
	num_embeddings, commitment_cost, metadata={} , edge_dim = 1,
	 encoder_hidden = 100 , dropout_p = 0.05 , EMA = False , 
	 reset_codes = True  , nheads = 3 , flavor= 'sage' , fftin = False):
		super(mk1_Encoder, self).__init__()

		#save all arguments to constructor
		self.args = locals()
		self.args.pop('self')

		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		self.num_embeddings = num_embeddings
		self.convs = torch.nn.ModuleList()
		self.norms = torch.nn.ModuleList()
		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.in_channels = in_channels
		self.encoder_hidden = encoder_hidden
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		#batch norm
		self.bn = torch.nn.BatchNorm1d(in_channels)
		
		self.dropout = torch.nn.Dropout(p=dropout_p)
		self.jk = JumpingKnowledge(mode='cat')
		self.fftin = fftin

		if self.fftin == True:
			in_channels = in_channels + 2 * 80

		self.ffin = torch.nn.Sequential(
			torch.nn.Linear(in_channels, hidden_channels[0] * 2 ),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[0] * 2 , hidden_channels[0]),
			torch.nn.GELU(),
			DynamicTanh(hidden_channels[0] , channels_last = True),
			)

		for i in range(1,len(hidden_channels)):
			if flavor == 'gat':
				self.convs.append(
					torch.nn.ModuleDict({
						'_'.join(edge_type): GATv2Conv(hidden_channels[i-1], 
						hidden_channels[i] , heads = nheads , dropout = dropout_p,
						concat = False )
						for edge_type in metadata['edge_types']
					})
				)

			if flavor == 'transformer':
				self.convs.append(
					torch.nn.ModuleDict({
						'_'.join(edge_type): TransformerConv( hidden_channels[i-1], 
						hidden_channels[i] , heads = nheads , dropout = dropout_p,
						concat = False )
						for edge_type in metadata['edge_types']
					})
				)

			if flavor == 'sage':
				self.convs.append(
					torch.nn.ModuleDict({
						'_'.join(edge_type): SAGEConv( hidden_channels[i-1], 
						hidden_channels[i] )
						for edge_type in metadata['edge_types']
					})
				)
			self.norms.append( 
				GraphNorm(hidden_channels[i])
				)
		
		self.lin = torch.nn.Sequential(
			DynamicTanh(hidden_channels[-1]* (len(hidden_channels)-1)  , channels_last = True),
			torch.nn.Linear(hidden_channels[-1] * (len(hidden_channels)-1),self.encoder_hidden ) , 
			torch.nn.GELU(),
			torch.nn.Linear(self.encoder_hidden, self.encoder_hidden) ,	
			torch.nn.GELU(),
			DynamicTanh(self.encoder_hidden , channels_last = True),
			)
		
		self.out_dense= torch.nn.Sequential(
			torch.nn.Linear(self.encoder_hidden + 20 , self.encoder_hidden) ,
			torch.nn.GELU(),
			torch.nn.Linear( self.encoder_hidden, self.encoder_hidden //2 ) ,
			torch.nn.GELU(),
			torch.nn.Linear(self.encoder_hidden//2, self.out_channels) ,	
			torch.nn.GELU(),
			DynamicTanh(self.out_channels , channels_last = True),
			#torch.nn.Tanh()
			)
		
		if EMA == False:
			self.vector_quantizer = VectorQuantizer(num_embeddings, out_channels, commitment_cost)
		else:
			self.vector_quantizer = VectorQuantizerEMA(num_embeddings, out_channels, commitment_cost , reset = reset_codes)
		
	def forward(self, data , edge_attr_dict = None , **kwargs):
		
		x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
		x_dict['res'] = self.bn(x_dict['res'])
		if 'debug' in kwargs and kwargs['debug'] == True:
			print( x_dict['res'].shape , 'x_dict[res] shape')
		
		if self.fftin == True:
			x_dict['res'] = torch.cat([x_dict['res'], data['fourier1dr'].x , data['fourier1di'].x ], dim=1)
		x = self.dropout(x_dict['res'])
		x_save= []
		# Apply the first layer
		x = self.ffin(x)
		for i, convs in enumerate(self.convs):
			# Apply the graph convolutions and average over all edge types
			if edge_attr_dict is not None:
				x = [conv(x, edge_index_dict[tuple(edge_type.split('_'))], edge_attr_dict[tuple(edge_type.split('_'))]) for edge_type, conv in convs.items()]
			else:
				x = [conv(x, edge_index_dict[tuple(edge_type.split('_'))]) for edge_type, conv in convs.items()]
			x = torch.stack(x, dim=0).mean(dim=0)
			x = F.gelu(x)
			x = self.norms[i](x)
			#if i < len(self.hidden_channels) - 1 else x
			x_save.append(x)
		x = self.jk(x_save)
		x = self.lin(x)
		#use aa sequence as input
		x = self.out_dense( torch.cat([ x , x_dict['AA']], dim=1) )
		#normalize the output to have norm 1
		z_quantized, vq_loss = self.vector_quantizer(x)
		if 'debug' in kwargs and kwargs['debug'] == True:
			print('z_quantized shape:', z_quantized.shape)
			print('vq_loss:', vq_loss)
			print('x shape:', x.shape)

			print('x_dict keys:', x_dict.keys())
			print('edge_index_dict keys:', edge_index_dict.keys())
		return z_quantized, vq_loss

	def encode_structures_fasta(self, dataloader, filename = 'structalign.strct.fasta' , verbose = False , alphabet = None , replace = False , **kwargs):
		"""
		Write an encoded fasta for use with mafft and raxmlng. Only doable with alphabet size of less than 248.
		0x01 – 0xFF excluding > (0x3E), = (0x3D), < (0x3C), - (0x2D), Space (0x20), Carriage Return (0x0d) and Line Feed (0x0a)
		"""
		#replace characters with special characters to avoid issues with fasta format
		replace_dict = { '>' : chr(249), '=' : chr(250), '<' : chr(251), '-' : chr(252), ' ' : chr(253) , '\r' : chr(254), '\n' : chr(255) }
		#check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		
		if alphabet is not None:
			print('using alphabet')
			print(alphabet)
		
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				if 'debug' in kwargs and kwargs['debug'] == True:
					print('res shape' ,  data['res'].x.shape[0] )
				data = data.to(self.device)
				z,qloss = self.forward(data)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = data.identifier
				f.write(f'>{identifier}\n')
				outstr = ''
				for char in strdata[0]:
					#start at 0x01
					if alphabet is not None:
						char = alphabet[char]
					else:
						char = chr(char+1)
					if replace and char in replace_dict:
						char = replace_dict[char]
					outstr += char
				if 'debug' in kwargs and kwargs['debug'] == True:
					print('len outstring' , len(outstr) )
					assert len(outstr) == data['res'].x.shape[0], f"Output string length {len(outstr)} does not match AA length {data['res'].x.shape[0]} for identifier {identifier}"
					assert len(outstr) == data['AA'].x.shape[0], f"Output string length {len(outstr)} does not match AA length {data['AA'].x.shape[0]} for identifier {identifier}"
					assert len(outstr) == z.shape[0], f"Output string length {len(outstr)} does not match z shape {z.shape[0]} for identifier {identifier}"
				f.write(outstr)
				f.write('\n')
		return filename 

class HeteroGAE_Decoder(torch.nn.Module):
	def __init__(self, in_channels = {'res':10 , 'godnode4decoder':5 , 'foldx':23}, xdim=20, concat_positions = False, hidden_channels={'res_backbone_res': [20, 20, 20]}, layers = 3,  AAdecoder_hidden = 20 
			  ,PINNdecoder_hidden = 10, contactdecoder_hidden = 10, nheads = 3 , Xdecoder_hidden=30, metadata={}, amino_mapper= None,
			    flavor = None, dropout= .001 , output_foldx = False, normalize = True , residual = True , contact_mlp = True ):
		super(HeteroGAE_Decoder, self).__init__()
		# Setting the seed
		L.seed_everything(42)
		

		if type( Xdecoder_hidden) == int:
			Xdecoder_hidden = [Xdecoder_hidden, Xdecoder_hidden, Xdecoder_hidden]
		
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
		self.output_foldx = output_foldx
		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.in_channels = in_channels
		self.amino_acid_indices = amino_mapper
		self.nlayers = layers
		self.bn = torch.nn.BatchNorm1d(in_channels['res'])
		self.norm_in = torch.nn.LayerNorm(in_channels['res'])
		self.bn_foldx = torch.nn.BatchNorm1d(in_channels['foldx'])
		self.norm_foldx = torch.nn.LayerNorm(in_channels['foldx'])
		self.revmap_aa = { v:k for k,v in amino_mapper.items() }
		self.dropout = torch.nn.Dropout(p=dropout)
		self.jk = JumpingKnowledge(mode='cat')# , channels =100 , num_layers = layers) 
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
					layer[edge_type] = MFConv( (-1, -1)  , hidden_channels[edge_type][i] , max_degree=10  , aggr = 'max' )
				if flavor == 'transformer' or edge_type == ('res','informs','godnode4decoder'):
					layer[edge_type] =  TransformerConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False  ) 
				if flavor == 'sage':
					layer[edge_type] =  SAGEConv( (-1, -1) , hidden_channels[edge_type][i] ) # , aggr = SoftmaxAggregation() ) 
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
			#self.norms.append( DynamicTanh(finalout , channels_last=True) )
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
				torch.nn.GELU(),
				DynamicTanh(lastlin , channels_last = True),
				)
		
		self.aadecoder = torch.nn.Sequential(
				torch.nn.Dropout(dropout),
				DynamicTanh(lastlin + in_channels_orig['res']  , channels_last = True),
				torch.nn.Linear(lastlin + in_channels_orig['res'] , AAdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1] ) ,
				torch.nn.GELU(),
				torch.nn.Linear(AAdecoder_hidden[1],AAdecoder_hidden[2]) ,
				torch.nn.GELU(),
				DynamicTanh(AAdecoder_hidden[2] , channels_last = True),
				torch.nn.Linear(AAdecoder_hidden[2] , xdim) ,
				torch.nn.LogSoftmax(dim=1) )
	
		if output_foldx == True:
			self.godnodedecoder = torch.nn.Sequential(
					torch.nn.Linear(in_channels['godnode4decoder'] , PINNdecoder_hidden[0]),
					torch.nn.GELU(),
					torch.nn.Linear(PINNdecoder_hidden[0], PINNdecoder_hidden[1] ) ,
					torch.nn.GELU(),
					DynamicTanh(PINNdecoder_hidden[1] , channels_last = True),
					torch.nn.Linear(PINNdecoder_hidden[1], in_channels['foldx']) )
		
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

	
	def forward(self, data , contact_pred_index, **kwargs):		
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
		decoder_in =  torch.cat( [inz,  z] , axis = 1)
		#decode aa
		aa = self.aadecoder(decoder_in)
		#decode godnode
		foldx_out = None
		if self.output_foldx == True:
			foldx_out = self.godnodedecoder( xdata['godnode4decoder'] )
		
		edge_probs = None
		if contact_pred_index is not None:
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
		return {'aa': aa, 'foldx_out': foldx_out, 'edge_probs': edge_probs}
		
	
	def x_to_amino_acid_sequence(self, x_r):
		"""
		Converts the reconstructed 20-dimensional matrix to a sequence of amino acids.

		Args:
			x_r (Tensor): Reconstructed 20-dimensional tensor.

		Returns:
			str: A string representing the sequence of amino acids.
		"""
		# Find the index of the maximum value in each row to get the predicted amino acid
		indices = torch.argmax(x_r, dim=1)
		
		# Convert indices to amino acids
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
	with open(filename, 'r') as f:
		#read all chars of file into a string
		for line in tqdm.tqdm(f):
			if line[0] == '>':
				seqdict[ID] = seqstr[:-1]
				ID = line[1:].strip()
				seqstr = ''
			else:
				seqstr += line
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

	