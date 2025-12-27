#!/usr/bin/env python
# coding: utf-8

import importlib
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

from foldtree2.src.losses import *
from foldtree2.src.quantizers import *

# Note: datadir is defined but may not be used throughout the module
datadir = '../../datasets/foldtree2/'
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
	 reset_codes = True  , nheads = 3 , flavor= 'sage' , fftin = False,
	 use_commitment_scheduling = False, commitment_warmup_steps = 5000,
	 commitment_schedule = 'cosine', commitment_start = 0.1, concat_positions = False , **kwargs):
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
		#layer norm
		self.ln = torch.nn.LayerNorm(in_channels)
		
		self.concat_positions = concat_positions

		self.dropout = torch.nn.Dropout(p=dropout_p)
		self.jk = JumpingKnowledge(mode='cat')
		self.fftin = fftin
		self.in_channels = in_channels
		if self.concat_positions == True:
			self.in_channels += 256

		self.inmlp = torch.nn.Sequential(
			torch.nn.Dropout(dropout_p),
			torch.nn.LayerNorm(self.in_channels),
			torch.nn.Linear(self.in_channels, hidden_channels[0] * 2 ),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[0] * 2 , hidden_channels[0]),
			torch.nn.GELU(),
			)

		if self.fftin == True:
			self.ffin = torch.nn.Sequential(
				torch.nn.Dropout(dropout_p),
				torch.nn.LayerNorm(2 * 80),
				torch.nn.Linear(2 * 80 , hidden_channels[0] * 2 ),
				torch.nn.GELU(),
				torch.nn.Linear(hidden_channels[0] * 2 , hidden_channels[0]),
				torch.nn.GELU())
		else:
			self.ffin = None

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
			torch.nn.Linear(hidden_channels[-1] * (len(hidden_channels)-1),self.encoder_hidden ) , 
			torch.nn.GELU(),
			torch.nn.Linear(self.encoder_hidden, self.encoder_hidden) ,	
			torch.nn.GELU(),
			)
		
		self.out_dense= torch.nn.Sequential(
			torch.nn.Linear(self.encoder_hidden + 20 , self.encoder_hidden) ,
			torch.nn.GELU(),
			torch.nn.Linear( self.encoder_hidden, self.encoder_hidden ) ,
			torch.nn.GELU(),
			torch.nn.Linear(self.encoder_hidden, self.out_channels) ,	
			torch.nn.Tanh()
			)
		
		if EMA == False:
			self.vector_quantizer = VectorQuantizer(num_embeddings, out_channels, commitment_cost)
		else:
			self.vector_quantizer = VectorQuantizerEMA(
				num_embeddings, out_channels, commitment_cost, 
				reset=reset_codes,
				use_commitment_scheduling=use_commitment_scheduling,
				commitment_warmup_steps=commitment_warmup_steps,
				commitment_schedule=commitment_schedule,
				commitment_start=commitment_start,
				commitment_end=commitment_cost
			)
		
	def forward(self, data , edge_attr_dict = None , **kwargs):
		
		x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
		x_dict['res'] = self.ln(x_dict['res'])
				
		if self.concat_positions == True:
			x_dict['res'] = torch.cat([x_dict['res'], data['positions'].x], dim=1)

		
		if 'debug' in kwargs and kwargs['debug'] == True:
			print( x_dict['res'].shape , 'x_dict[res] shape')
		xin = self.dropout(x_dict['res'])
		x_save= []
		# Apply the first layer
		x = self.inmlp(xin)		
		#add fourier features if present
		if self.fftin == True:
			#data['fourier1dr'].x = self.dropout(data['fourier1dr'].x)
			#data['fourier1di'].x = self.dropout(data['fourier1di'].x)
			x+= self.ffin(torch.cat([data['fourier1dr'].x, data['fourier1di'].x], dim=1))

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

	def encode_structures_fasta(self, dataloader, filename = 'structalign.strct.fasta' , verbose = False , alphabet = None , replace = True , **kwargs):
		"""
		Write an encoded fasta for use with mafft and raxmlng. Only doable with alphabet size of less than 248.
		0x01 – 0xFF excluding > (0x3E), = (0x3D), < (0x3C), - (0x2D), Space (0x20), Carriage Return (0x0d) and Line Feed (0x0a)
		"""
		#replace characters with special characters to avoid issues with fasta format
		replace_dict = {chr(0):chr(246) , '"':chr(248) , '#':chr(247), '>' : chr(249), '=' : chr(250), '<' : chr(251), '-' : chr(252), ' ' : chr(253) , '\r' : chr(254), '\n' : chr(255) }
		
		#check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		
		if alphabet is not None:
			print('using alphabet')
			print(alphabet)
		
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader) , desc='Encoding structures to FASTA' ):
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


class mk1_MuonEncoder(torch.nn.Module):
	"""
	Muon-compatible encoder with modular architecture.
	Separates input, body, and head modules for compatibility with Muon optimizer.
	
	- input: Preprocessing and initial MLPs (optimized with AdamW)
	- body: Graph convolution layers (weights optimized with Muon, gains/biases with AdamW)
	- head: Output projection and quantization (optimized with AdamW)
	"""
	def __init__(self, in_channels, hidden_channels, out_channels, 
				num_embeddings, commitment_cost, metadata={}, edge_dim=1,
				encoder_hidden=100, dropout_p=0.05, EMA=False, 
				reset_codes=True, nheads=3, flavor='sage', fftin=False,
				use_commitment_scheduling=False, commitment_warmup_steps=5000,
				commitment_schedule='cosine', commitment_start=0.1, concat_positions=False , **kwargs):
		super(mk1_MuonEncoder, self).__init__()

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
		self.fftin = fftin
		
		# Determine input channels
		self.in_channels = in_channels
		if self.concat_positions:
			self.in_channels += 256

		# ===================== INPUT MODULE =====================
		# Preprocessing layers: LayerNorm, input MLPs, dropout
		self.input = nn.ModuleDict()
		
		self.input['ln'] = nn.LayerNorm(in_channels)
		self.input['dropout'] = nn.Dropout(p=dropout_p)
		
		self.input['inmlp'] = nn.Sequential(
			nn.Linear(self.in_channels, hidden_channels[0] * 2),
			nn.GELU(),
			nn.Linear(hidden_channels[0] * 2, hidden_channels[0]),
			nn.GELU(),
		)
		
		if self.fftin:
			self.input['ffin'] = nn.Sequential(
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
		x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
		
		# ===================== INPUT PROCESSING =====================
		x_dict['res'] = self.input['ln'](x_dict['res'])
		
		if self.concat_positions:
			x_dict['res'] = torch.cat([x_dict['res'], data['positions'].x], dim=1)
		
		if 'debug' in kwargs and kwargs['debug']:
			print(x_dict['res'].shape, 'x_dict[res] shape')
		
		xin = self.input['dropout'](x_dict['res'])
		x = self.input['inmlp'](xin)
		
		# Add fourier features if present
		if self.fftin and 'ffin' in self.input:
			x += self.input['ffin'](torch.cat([data['fourier1dr'].x, data['fourier1di'].x], dim=1))
		
		# ===================== BODY PROCESSING =====================
		x_save = []
		for i, convs in enumerate(self.body['convs']):
			# Apply graph convolutions and average over all edge types
			if edge_attr_dict is not None:
				x_list = [conv(x, edge_index_dict[tuple(edge_type.split('_'))], 
							  edge_attr_dict[tuple(edge_type.split('_'))]) 
						 for edge_type, conv in convs.items()]
			else:
				x_list = [conv(x, edge_index_dict[tuple(edge_type.split('_'))]) 
						 for edge_type, conv in convs.items()]
			
			x = torch.stack(x_list, dim=0).mean(dim=0)
			x = F.gelu(x)
			x = self.body['norms'][i](x)
			x_save.append(x)
		
		x = self.body['jk'](x_save)
		
		# ===================== HEAD PROCESSING =====================
		x = self.head['lin'](x)
		x = self.head['out_dense'](torch.cat([x, x_dict['AA']], dim=1))
		
		# Quantization
		z_quantized, vq_loss = self.vector_quantizer(x)
		
		if 'debug' in kwargs and kwargs['debug']:
			print('z_quantized shape:', z_quantized.shape)
			print('vq_loss:', vq_loss)
			print('x shape:', x.shape)
			print('x_dict keys:', x_dict.keys())
			print('edge_index_dict keys:', edge_index_dict.keys())
		
		return z_quantized, vq_loss

	def encode_structures_fasta(self, dataloader, filename='structalign.strct.fasta', 
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

	