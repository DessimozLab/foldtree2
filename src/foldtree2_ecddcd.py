#!/usr/bin/env python
# coding: utf-8

from utils import *
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

#decoder super class
class Decoder(torch.nn.Module):
	def __init__() :
		super(Decoder, self).__init__()
		#save all arguments to constructor
		self.args = locals()
		self.args.pop('self')
		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	def	forward(self, z):
		raise NotImplementedError('forward method not implemented')
	

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
			)
		
		self.out_dense= torch.nn.Sequential(
			torch.nn.Linear(self.encoder_hidden + 20 , self.encoder_hidden) ,
			torch.nn.GELU(),
			torch.nn.Linear( self.encoder_hidden, self.out_channels) ,
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
		
		return z_quantized, vq_loss

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
					f.write(char)

				f.write('\n')

				if verbose == True:
					print(identifier, outstr)
		return filename 

class DenoisingTransformer(nn.Module):
	def __init__(self, input_dim=3, d_model=128, nhead=8, num_layers=2 , dropout=0.001):
		super(DenoisingTransformer, self).__init__()
		self.input_proj = nn.Linear(input_dim, d_model)
		# Transformer encoder: PyTorch transformer expects input of shape (seq_len, batch, d_model)
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		# Project back to the 12-dimensional output space
		#self.output_proj = nn.Linear(d_model, 14 )
		self.output_proj_rt = nn.Sequential(
			DynamicTanh( d_model , channels_last = True),
			nn.Linear(d_model, 50),
			nn.GELU(),
			nn.Linear(50, 25),
			nn.GELU(),
			nn.Linear(25, 7) ,
			DynamicTanh(7 , channels_last = True)
			)
		
		self.output_proj_angles = nn.Sequential(
			DynamicTanh( d_model , channels_last = True),
			nn.Linear(d_model, 10),
			nn.GELU(),
			nn.Linear(10, 5),
			nn.GELU(),
			DynamicTanh(5 , channels_last = True),
			nn.Linear(5, 3) )
	
	def forward(self, z , positions):

		"""
		Args:
		  r: Tensor of shape ( N, 3, 3) containing input rotation matrices.
		  t: Tensor of shape ( N, 3) containing input translation vectors.
		  angles: Tensor of shape ( N, 2) containing angles.
		  positions: Tensor of shape ( N, M) containing positional encoding.

		Returns:
		  r_refined: Tensor of shape ( N, 3, 3) with denoised (and re-orthogonalized) rotations.
		  t_refined: Tensor of shape ( N, 3) with denoised translations.
		"""
		N, _ = z.shape
		if positions is not None:
			x = torch.cat([z , positions], dim=-1)
		else:
			x = z
		# Project to the transformer dimension: (N, d_model)
		x = self.input_proj(x)
		# PyTorch's transformer expects shape (seq_len, batch, d_model)
		x = self.transformer_encoder(x)  # Process all positions (sequence length) per batch
		# Project back to the 12-dim space
		refined_features_rt = self.output_proj_rt(x)  # ( N, 7)
		refined_features_angles = self.output_proj_angles(x)  # ( N, 3)

		# Split the features back into rotation and translation parts
		r_refined_flat = refined_features_rt[..., :4]  # ( N, 4)
		r_refined = quaternion_to_rotation_matrix(r_refined_flat)  # ( N, 3, 3)
		t_refined = refined_features_rt[..., 4:]  # ( N, 3)
		
		return r_refined, t_refined , refined_features_angles

	@staticmethod
	def orthogonalize(r):
		"""
		Re-orthogonalizes each 3x3 matrix in the batch so that it is a valid rotation matrix.
		
		Args:
		  r: Tensor of shape ( N, 9)
		
		Returns:
		  r_ortho: Tensor of shape ( N, 3, 3) where each matrix is orthogonal.
		"""
		N, _ = r.shape
		# Flatten sequence dimensions: (N, 3, 3)
		r_flat = r.reshape(-1, 3, 3)
		U, S, Vh = torch.linalg.svd(r, full_matrices=False)
		r_ortho = torch.matmul(U, Vh)
		# Reshape back to ( N, 3, 3)
		r_ortho = r_ortho.reshape( N, 3, 3)
		return r_ortho

class HeteroGAE_Decoder(torch.nn.Module):
	def __init__(self, in_channels = {'res':10 , 'godnode4decoder':5 , 'foldx':23}, xdim=20, concat_positions = False, hidden_channels={'res_backbone_res': [20, 20, 20]}, layers = 3,  AAdecoder_hidden = 20 
			  ,PINNdecoder_hidden = 10, contactdecoder_hidden = 10, nheads = 3 , Xdecoder_hidden=30, metadata={}, amino_mapper= None,
			    flavor = None, dropout= .001 , output_foldx = False, normalize = True , residual = True , contact_mlp = True ):
		super(HeteroGAE_Decoder, self).__init__()
		# Setting the seed
		L.seed_everything(42)

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
					layer[edge_type] = MFConv( (-1, -1)  , hidden_channels[edge_type][i] , max_degree=5  , aggr = 'mean' )
				if flavor == 'transformer' or edge_type == ('res','informs','godnode4decoder'):
					layer[edge_type] =  TransformerConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False  ) 
				if flavor == 'sage' or edge_type == ('res','backbone','res'):
					layer[edge_type] =  SAGEConv( (-1, -1) , hidden_channels[edge_type][i] ) # , aggr = SoftmaxAggregation() ) 
				if k == 0 and i == 0:
					in_channels[dataout] = hidden_channels[edge_type][i]
				if k == 0 and i > 0:
					in_channels[dataout] = hidden_channels[edge_type][i-1]
				if k > 0 and i > 0:                    
					in_channels[dataout] = hidden_channels[edge_type][i]
				if k > 0 and i == 0:
					in_channels[dataout] = hidden_channels[edge_type][i]
			conv = HeteroConv( layer  , aggr='mean')
			
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
		if self.output_foldx == True:
			zgodnode = xdata['godnode4decoder']
			foldx_pred = self.godnodedecoder( xdata['godnode4decoder'] )
		else:
			foldx_pred = None
			zgodnode = None
		
		if contact_pred_index is None:
			return aa, None, zgodnode , foldx_pred
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

		return aa,  edge_probs , zgodnode , foldx_pred
		
	
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

