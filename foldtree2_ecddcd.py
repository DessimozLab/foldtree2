#!/usr/bin/env python
# coding: utf-8

from utils import *
from  torch_geometric.utils import to_undirected


class VectorQuantizerEMA(nn.Module):
	def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99 , epsilon=1e-5, reset_threshold=100000, reset = True , klweight = 1 , diversityweight= 1 , entropyweight = 1 ):
		super(VectorQuantizerEMA, self).__init__()
		self.embedding_dim = embedding_dim
		self.num_embeddings = num_embeddings
		self.commitment_cost = commitment_cost
		self.decay = decay
		self.epsilon = epsilon
		self.reset_threshold = reset_threshold
		self.reset = reset
		# Initialize the codebook with uniform distribution
		self.diversityweight = diversityweight
		self.klweight= klweight
		self.entropyweight = entropyweight

		self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
		self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
		self.entropyweight= entropyweight
		self.diversityweight = diversityweight
		self.klweight = klweight

		# EMA variables
		self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
		self.ema_w = nn.Parameter(self.embeddings.weight.clone())

		# Track usage of embeddings
		self.register_buffer('embedding_usage_count', torch.zeros(num_embeddings, dtype=torch.long))

	def forward(self, x):
		# Flatten input
		flat_x = x.view(-1, self.embedding_dim)

		# Compute distances between input and codebook embeddings
		distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
					 + torch.sum(self.embeddings.weight**2, dim=1)
					 - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))

		# Get the encoding that has the minimum distance
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
		encodings.scatter_(1, encoding_indices, 1)

		# Quantize the latents by mapping to the nearest embeddings
		quantized = torch.matmul(encodings, self.embeddings.weight).view_as(x)

		# Compute the commitment loss
		e_latent_loss = F.mse_loss(quantized.detach(), x)
		q_latent_loss = F.mse_loss(quantized, x.detach())
		loss = q_latent_loss + self.commitment_cost * e_latent_loss

		# Regularization
		entropy_reg = entropy_regularization(encodings)
		diversity_reg = diversity_regularization(encodings)
		kl_div_reg = kl_divergence_regularization(encodings)

		# Combine all losses
		total_loss = loss - self.entropyweight*entropy_reg + self.diversityweight*diversity_reg + self.klweight*kl_div_reg

		# EMA updates
		if self.training:
			encodings_sum = encodings.sum(0)
			dw = torch.matmul(encodings.t(), flat_x)

			self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * encodings_sum
			self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)

			n = self.ema_cluster_size.sum()
			self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n)

			self.embeddings.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)

			# Update usage count
			self.embedding_usage_count += encodings_sum.long()
			
			if self.reset== True:
				# Reset unused embeddings
				self.reset_unused_embeddings()

		# Straight-through estimator for the backward pass
		quantized = x + (quantized - x).detach()

		return quantized, total_loss

	def reset_unused_embeddings(self):
		"""
		Resets the embeddings that have not been used for a certain number of iterations.
		"""
		unused_embeddings = self.embedding_usage_count < self.reset_threshold
		num_resets = unused_embeddings.sum().item()
		if num_resets > 0:
			with torch.no_grad():
				self.embeddings.weight[unused_embeddings] = torch.randn((num_resets, self.embedding_dim), device=self.embeddings.weight.device)
			# Reset usage counts for the reset embeddings
			self.embedding_usage_count[unused_embeddings] = 0

	def discretize_z(self, x):
		# Flatten input
		flat_x = x.view(-1, self.embedding_dim)
		# Compute distances between input and codebook embeddings
		distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
					 + torch.sum(self.embeddings.weight**2, dim=1)
					 - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))
		# Get the encoding that has the minimum distance
		closest_indices = torch.argmin(distances, dim=1)
		
		# Convert indices to characters
		char_list = [chr(idx.item()) for idx in closest_indices]
		return closest_indices, char_list

	def string_to_hex(self, s):
		# if string is ascii, convert to hex
		if all(ord(c) < 248 for c in s):
			return s.encode().hex()
		else:
			#throw an error
			raise ValueError('String contains non-ASCII characters')
		
	def string_to_embedding(self, s):
		
		# Convert characters back to indices
		indices = torch.tensor([ord(c) for c in s], dtype=torch.long, device=self.embeddings.weight.device)
		
		# Retrieve embeddings from the codebook
		embeddings = self.embeddings(indices)
		return embeddings
	
	def ord_to_embedding(self, s):
		# Convert characters back to indices
		indices = torch.tensor([c for c in s], dtype=torch.long, device=self.embeddings.weight.device)
		# Retrieve embeddings from the codebook
		embeddings = self.embeddings(indices)
		return embeddings


# Define the regularization functions outside the class

def entropy_regularization(encodings):
	probabilities = encodings.mean(dim=0)
	entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))
	return entropy

def diversity_regularization(encodings):
	probabilities = encodings.mean(dim=0)
	diversity_loss = torch.sum((probabilities - 1 / probabilities.size(0)) ** 2)
	return diversity_loss

def kl_divergence_regularization(encodings):
	probabilities = encodings.mean(dim=0)
	kl_divergence = torch.sum(probabilities * torch.log(probabilities * probabilities.size(0) + 1e-10))
	return kl_divergence


# residual block feed forward
class ResidualBlockFF(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels , outchannels , nlayers = 2):
		super(ResidualBlockFF, self).__init__()
		self.ff_stack1 = torch.nn.Sequential( 
			Linear(in_channels, hidden_channels),
			torch.nn.ReLU(),
			#add nlayers
			*[torch.nn.Sequential(Linear(hidden_channels, hidden_channels), torch.nn.ReLU()) for i in range(nlayers)]
			, Linear(hidden_channels, in_channels) , torch.nn.ReLU()
			)
		self.final = torch.nn.Linear(in_channels, outchannels)

	def forward(self, x):
		xcopy = x
		x = self.ff_stack1(x)
		x = x+xcopy
		x = self.final(x)
		return F.relu(x)


class VectorQuantizer(nn.Module):
	def __init__(self, num_embeddings, embedding_dim, commitment_cost):
		super(VectorQuantizer, self).__init__()


		self.embedding_dim = embedding_dim
		self.num_embeddings = num_embeddings
		self.commitment_cost = commitment_cost

		self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
		self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

	def forward(self, x):
		# Flatten input
		flat_x = x.view(-1, self.embedding_dim)

		# Calculate distances
		distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
					 + torch.sum(self.embeddings.weight**2, dim=1)
					 - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))

		# Get the encoding that has the min distance
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
		encodings.scatter_(1, encoding_indices, 1)

		# Quantize the latents
		quantized = torch.matmul(encodings, self.embeddings.weight).view_as(x)

		# Loss
		e_latent_loss = F.mse_loss(quantized.detach(), x)
		q_latent_loss = F.mse_loss(quantized, x.detach())
		loss = q_latent_loss + self.commitment_cost * e_latent_loss

		# Straight-through estimator
		quantized = x + (quantized - x).detach()
		return quantized, loss

	def discretize_z(self, x):
		# Flatten input
		flat_x = x.view(-1, self.embedding_dim)
		# Compute distances between input and codebook embeddings
		distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
					 + torch.sum(self.embeddings.weight**2, dim=1)
					 - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))
		# Get the encoding that has the minimum distance
		closest_indices = torch.argmin(distances, dim=1)
		
		# Convert indices to characters
		char_list = [chr(idx.item()) for idx in closest_indices]
		return closest_indices, char_list

	def string_to_hex(self, s):
		# if string is ascii, convert to hex
		if all(ord(c) < 248 for c in s):
			return s.encode().hex()
		else:
			#throw an error
			raise ValueError('String contains non-ASCII characters')
		
	def string_to_embedding(self, s):
		
		# Convert characters back to indices
		indices = torch.tensor([ord(c) for c in s], dtype=torch.long, device=self.embeddings.weight.device)
		
		# Retrieve embeddings from the codebook
		embeddings = self.embeddings(indices)
		
		return embeddings


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
	 reset_codes = True  , nheads = 3 , flavor= 'sage'):
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
		if flavor == 'gat':
			for i in range(len(hidden_channels)):
				self.convs.append(
					torch.nn.ModuleDict({
						'_'.join(edge_type): GATv2Conv(in_channels if i == 0 else hidden_channels[i-1], 
						hidden_channels[i] , heads = nheads , dropout = dropout_p,
						concat = False )
						for edge_type in metadata['edge_types']
					})
				)			
		if flavor == 'sage':
			for i in range(len(hidden_channels)):
				self.convs.append(
					torch.nn.ModuleDict({
						'_'.join(edge_type): SAGEConv(in_channels if i == 0 else hidden_channels[i-1], 
						hidden_channels[i] )
						for edge_type in metadata['edge_types']
					})
				)

		self.lin = torch.nn.Sequential(
			torch.nn.LayerNorm(hidden_channels[-1] * len(hidden_channels)), 
			torch.nn.Linear(hidden_channels[-1] * len(hidden_channels), hidden_channels[-1] ))
		self.out_dense= torch.nn.Sequential(
			torch.nn.Linear(hidden_channels[-1] + 20 , self.encoder_hidden) ,
			torch.nn.GELU(),
			torch.nn.Linear(self.encoder_hidden, self.encoder_hidden) ,
			torch.nn.GELU(),
			torch.nn.Linear(self.encoder_hidden, self.out_channels) ,
			torch.nn.LayerNorm(self.out_channels),
			)
		if EMA == False:
			self.vector_quantizer = VectorQuantizer(num_embeddings, out_channels, commitment_cost)
		else:
			self.vector_quantizer = VectorQuantizerEMA(num_embeddings, out_channels, commitment_cost , reset = reset_codes)
		
	def forward(self, data , edge_attr_dict = None):
		data['res', 'backbone', 'res'].edge_index = to_undirected(data['res' , 'backbone' , 'res'].edge_index) 
		x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
		x_dict['res'] = self.bn(x_dict['res'])
		x = self.dropout(x_dict['res'])
		x_save= []
		for i, convs in enumerate(self.convs):
			# Apply the graph convolutions and average over all edge types
			if edge_attr_dict is not None:
				x = [conv(x, edge_index_dict[tuple(edge_type.split('_'))], edge_attr_dict[tuple(edge_type.split('_'))]) for edge_type, conv in convs.items()]
			else:
				x = [conv(x, edge_index_dict[tuple(edge_type.split('_'))]) for edge_type, conv in convs.items()]
			x = torch.stack(x, dim=0).mean(dim=0)
			x = F.gelu(x) if i < len(self.hidden_channels) - 1 else x
			x_save.append(x)
		x = self.jk(x_save)
		x = self.lin(x)
		x = self.out_dense( torch.cat([ x , x_dict['AA']], dim=1) )
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
	def __init__(self, input_dim=12, d_model=128, nhead=8, num_layers=2):
		super(DenoisingTransformer, self).__init__()
		# Linear projection from 12 (9 for rotation + 3 for translation) to d_model
		self.input_proj = nn.Linear(input_dim, d_model)
		
		# Transformer encoder: PyTorch transformer expects input of shape (seq_len, batch, d_model)
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.01)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		# Project back to the 12-dimensional output space
		#self.output_proj = nn.Linear(d_model, 14 )
		self.output_proj_rt = nn.Sequential(
			nn.Linear(d_model, 50),
			nn.GELU(),
			nn.Linear(50, 25),
			nn.GELU(),
			nn.LayerNorm(25),
			nn.Linear(25, 12) )
		
		self.output_proj_angles = nn.Sequential(
			nn.Linear(d_model, 10),
			nn.GELU(),
			nn.Linear(10, 5),
			nn.GELU(),
			nn.LayerNorm(5),
			nn.Linear(5, 3) )
	
	def forward(self, embeddings , positions):

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
		N, _ = embeddings.shape
		# Flatten each 3x3 rotation matrix into 9 numbers: ( N, 9)
		x = torch.cat([embeddings , positions], dim=-1)
		# Project to the transformer dimension: (N, d_model)
		x = self.input_proj(x)
		# PyTorch's transformer expects shape (seq_len, batch, d_model)
		x = self.transformer_encoder(x)  # Process all positions (sequence length) per batch
		# Project back to the 12-dim space
		refined_features_rt = self.output_proj_rt(x)  # ( N, 12)
		refined_features_angles = self.output_proj_angles(x)  # ( N, 2)		
		
		# Split the features back into rotation and translation parts
		r_refined_flat = refined_features_rt[..., :9]  # ( N, 9)
		t_refined = refined_features_rt[..., 9:]         # ( N, 3)
		# Reshape the rotation back to ( N, 3, 3)
		#r_refined = r_refined_flat.reshape( N, 3, 3)
		# Re-orthogonalize each rotation matrix using SVD
		r_refined = self.orthogonalize(r_refined_flat)
		#r_refined = r_refined_flat.reshape( N, 3, 3)
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
	def __init__(self, in_channels = {'res':10 , 'godnode4decoder':5 , 'foldx':23}, xdim=20, hidden_channels={'res_backbone_res': [20, 20, 20]}, layers = 3,  AAdecoder_hidden = 20 
			  ,PINNdecoder_hidden = 10, contactdecoder_hidden = 10, 
			  nheads = 3 , Xdecoder_hidden=30, metadata={}, 
			  amino_mapper= None  , flavor = None, dropout= .1 ,
				output_foldx = False , contact_mlp = False , denoise = False):
		super(HeteroGAE_Decoder, self).__init__()
		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		self.convs = torch.nn.ModuleList()
		in_channels_orig = copy.deepcopy(in_channels )
		#self.bn = torch.nn.BatchNorm1d(encoder_out_channels)
		self.output_foldx = output_foldx
		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.in_channels = in_channels
		self.amino_acid_indices = amino_mapper
		self.nlayers = layers
		self.bn = torch.nn.BatchNorm1d(in_channels['res'])
		self.bn_foldx = torch.nn.BatchNorm1d(in_channels['foldx'])
		self.revmap_aa = { v:k for k,v in amino_mapper.items() }
		self.dropout = torch.nn.Dropout(p=dropout)
		self.jk = JumpingKnowledge(mode='cat')# , channels =100 , num_layers = layers) 
		for i in range(layers):
			layer = {}          
			for k,edge_type in enumerate( hidden_channels.keys() ):
				edgestr = '_'.join(edge_type)
				datain = edge_type[0]
				dataout = edge_type[2]
				#if ( 'res','informs','godnode4decoder') == edge_type:
				#	layer[edgestr] = TransformerConv( in_channels[datain] , hidden_channels[edge_type][i], heads = nheads , concat= False)
				#else:
				if flavor == 'transformer' or edge_type == ('res','informs','godnode4decoder'):
					layer[edge_type] = TransformerConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False  )
				if flavor == 'sage':
					layer[edge_type] = SAGEConv( (-1, -1) , hidden_channels[edge_type][i] )
				if flavor == 'mfconv':
					layer[edge_type] = MFConv( (-1, -1)  , hidden_channels[edge_type][i] , max_degree=5 )  
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
				torch.nn.Linear( self.hidden_channels[('res', 'backbone', 'res')][-1] *  layers , Xdecoder_hidden),
				torch.nn.GELU(),
				torch.nn.Linear(Xdecoder_hidden, Xdecoder_hidden),
				torch.nn.GELU(),
				torch.nn.LayerNorm(Xdecoder_hidden),
				)
	
		self.aadecoder = torch.nn.Sequential(
				torch.nn.Linear(Xdecoder_hidden + in_channels_orig['res'] , AAdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1] ) ,
				torch.nn.GELU(),
				torch.nn.Linear(AAdecoder_hidden[1],AAdecoder_hidden[2]) ,
				torch.nn.GELU(),
				torch.nn.LayerNorm(AAdecoder_hidden[2]),
				torch.nn.Linear(AAdecoder_hidden[2] , xdim) ,
				torch.nn.LogSoftmax(dim=1) )
	
		if output_foldx == True:
			self.godnodedecoder = torch.nn.Sequential(
					torch.nn.Linear(in_channels['godnode4decoder'] , PINNdecoder_hidden[0]),
					torch.nn.GELU(),
					torch.nn.Linear(PINNdecoder_hidden[0], PINNdecoder_hidden[1] ) ,
					torch.nn.GELU(),
					torch.nn.Linear(PINNdecoder_hidden[1], in_channels['foldx']) )
		
		if denoise == True:
			dim = 256+Xdecoder_hidden + in_channels_orig['res']
			self.denoiser = DenoisingTransformer(input_dim= dim, d_model=128, nhead=4, num_layers=2)
		else:
			self.denoiser = None
		
		if contact_mlp == True:
			self.contact_decoder = torch.nn.Sequential(
				torch.nn.Linear( 2*(Xdecoder_hidden + in_channels_orig['res']) , contactdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(contactdecoder_hidden[0], contactdecoder_hidden[1] ) ,
				torch.nn.GELU(),								
				torch.nn.LayerNorm(contactdecoder_hidden[1]),
				torch.nn.Linear(contactdecoder_hidden[1], 1 ) ,
				torch.nn.Sigmoid()
				)
		else:
			self.contact_decoder = None

	def print_config(self):
		print('decoder convs' ,  self.convs)
		print( 'batchnorm' , self.bn)
		print( 'dropout' , self.dropout)
		print('aadecoder', self.aadecoder)
		print('lin' ,  self.lin)
		print( 'sigmoid' ,  self.sigmoid)
		print('godnodedecoder' , self.godnodedecoder)
		print('t_decoder' , self.t_decoder)
		print('r_decoder' , self.r_decoder)
		print( 'angledecoder' , self.angledecoder)
		print('denoiser' , self.denoiser)
	
	def forward(self, data , contact_pred_index, **kwargs):
		xdata, edge_index = data.x_dict, data.edge_index_dict
		#xdata['res'] = self.bn(xdata['res'])
		#copy z for later concatenation
		inz = xdata['res'].clone()	
		x_dict_list = []
		for i,layer in enumerate(self.convs):
			xdata = layer(xdata, edge_index)
			for key in xdata.keys():
				xdata[key] = F.gelu(xdata[key])
			x_dict_list.append(xdata['res'])
		xdata['res'] = self.jk(x_dict_list)
		z = xdata['res']
		z = self.lin(z)
		decoder_in =  torch.cat( [inz,  z] , axis = 1)
		#decode aa
		aa = self.aadecoder(decoder_in)

		if self.output_foldx == True:
			zgodnode = xdata['godnode4decoder']
			foldx_pred = self.godnodedecoder( xdata['godnode4decoder'] )
		else:
			foldx_pred = None
			zgodnode = None
		#decode contacts
		if self.denoiser:
			unique_batches = torch.unique(data['res'].batch)	
			rs = []
			ts = []
			angles_list = []
			for b in unique_batches:
				idx = (data['res'].batch == b).nonzero(as_tuple=True)[0]
				if idx.numel() > 2:
					ri,ti, anglesi = self.denoiser(decoder_in[idx] ,  data['positions'].x[idx])
					ri = ri.view(-1, 3 , 3)
					ti = ti.view(-1, 3)
					rs.append(ri)
					ts.append(ti)
					angles_list.append(anglesi)
				else:
					rs.append(r[idx])
					ts.append(t[idx])			
					angles_list.append(angles[idx])
			angles = torch.cat(angles_list, dim=0)	
			t = torch.cat(ts, dim=0)
			r = torch.cat(rs, dim=0)
			r = r.view(-1, 3, 3)
			t = t.view(-1, 3)
		else:
			r = None
			t = None
			angles = None
				
		if contact_pred_index is None:
			return aa, None, zgodnode , foldx_pred , r , t , angles
		if contact_pred_index is not None and self.contact_decoder is None:
			#compute similarity matrix
			sim_matrix = (z[contact_pred_index[0]] * z[contact_pred_index[1]]).sum(dim=1)
			#find contacts
			edge_probs = self.sigmoid(sim_matrix)
			return aa,  edge_probs , zgodnode , foldx_pred , r , t , angles
		else:
			edge_probs = self.contact_decoder(torch.cat( ( decoder_in[contact_pred_index[0]] , decoder_in[contact_pred_index[1]] ) ,dim= 1) )
			
		return aa,  edge_probs , zgodnode , foldx_pred , r , t , angles
		
	
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

	def load(self, modelfile):
		self.load_state_dict(torch.load(modelfile))
		self.eval()
		return self

	def save(self, modelfile):
		torch.save(self.state_dict(), modelfile)
		return modelfile
	
	def ret_config(self):
		return { 'encoder_out_channels': self.in_channels, 'xdim': 20, 'hidden_channels': self.hidden_channels, 'out_channels_hidden': self.out_channels_hidden, 'metadata': self.metadata, 'amino_mapper': self.amino_acid_indices }

	def save_config(self, configfile):
		with open(configfile , 'w') as f:
			json.dump(self.ret_config(), f)
		return configfile

	def load_from_config(config):
		return HeteroGAE_Encoder(**config)

def jaccard_distance_multiset(A: torch.Tensor,
                              B: torch.Tensor,
                              dim: int = -1,
                              eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the generalized (multiset) Jaccard distance between two tensors A and B.
    Both A and B should be nonnegative and have the same shape.
    
    :param A: Tensor of shape (..., n_features)
    :param B: Tensor of shape (..., n_features)
    :param dim: Dimension along which to compute Jaccard. Default is the last dimension.
    :param eps: Small constant to avoid division by zero.
    :return: Tensor of Jaccard distances of shape (...).
    """
    # Ensure A and B have the same shape
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape.")

    # Compute sum of minima and maxima along the chosen dimension
    min_sum = torch.minimum(A, B).sum(dim=dim)
    max_sum = torch.maximum(A, B).sum(dim=dim)

    # Compute Jaccard similarity
    jaccard_similarity = min_sum / (max_sum + eps)

    return jaccard_similarity

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
					layer[edge_type] = TransformerConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False)
				if flavor == 'sage':
					layer[edge_type] = SAGEConv( (-1, -1) , hidden_channels[edge_type][i])
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
				torch.nn.LayerNorm(contactdecoder_hidden[2]),
				torch.nn.Linear(contactdecoder_hidden[2], 1),
				torch.nn.Sigmoid()
				)
		self.godnodedecoder = torch.nn.Sequential(
				torch.nn.LayerNorm(in_channels['godnode4decoder']),
				torch.nn.Linear(in_channels['godnode4decoder'] , PINNdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(PINNdecoder_hidden[0], PINNdecoder_hidden[1] ) ,
				torch.nn.GELU(),
				torch.nn.Linear(PINNdecoder_hidden[1], PINNdecoder_hidden[2] ) ,
				torch.nn.GELU(),
				torch.nn.LayerNorm(self.embeding_dim),
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


'''
def recon_loss(data, pos_edge_index, decoder , poslossmod=1, neglossmod=1 , distweight = False) -> Tensor:
	r"""Given latent variables :obj:`z`, computes the binary cross
	entropy loss for positive edges :obj:`pos_edge_index` and negative
	sampled edges.

	Args:
		data (HeteroData): The input data containing node features and edge indices.
		pos_edge_index (torch.Tensor): The positive edges to train against.
		decoder (torch.nn.Module, optional): The decoder model. (default: :obj:`None`)
		poslossmod (float, optional): The positive loss modifier. (default: :obj:`1`)
		neglossmod (float, optional): The negative loss modifier. (default: :obj:`1`)
	"""
	pos = decoder( data, pos_edge_index )[1]
	if torch.any(pos) < 0:
		#set to 1 if any value is less than 0
		pos[pos < 0] = 0
	pos_loss = -torch.log(pos + EPS)	
	#weigh the loss with plddt values using elementwise multiplication
	#pos_loss = pos_loss * data['plddt'].x[pos_edge_index[0]].view(-1,1) * data['plddt'].x[pos_edge_index[1]].view(-1,1)
	if distweight == True:
		coord1 = data['coords'].x[pos_edge_index[0]]
		coord2 = data['coords'].x[pos_edge_index[1]]
		dist = torch.norm(coord1 - coord2, dim=1)
		pos_loss = pos_loss * dist.view(-1,1)
	pos_loss = pos_loss.mean()
	neg_edge_index = negative_sampling( pos_edge_index, data['res'].x.size(0))
	neg = decoder( data , neg_edge_index)[1]
	if torch.any(1- neg) < 0:
		neg[neg>1] = 1
	neg_loss = -torch.log((1 - neg) + EPS)
	#neg_loss = neg_loss * data['plddt'].x[neg_edge_index[0]].view(-1,1) * data['plddt'].x[neg_edge_index[1]].view(-1,1)
	if distweight == True:
		coord1 = data['coords'].x[neg_edge_index[0]]
		coord2 = data['coords'].x[neg_edge_index[1]]
		dist = torch.norm(coord1 - coord2, dim=1)
		neg_loss = neg_loss * dist.view(-1,1)
	neg_loss = neg_loss.mean()
	return poslossmod * pos_loss + neglossmod * neg_loss
'''



def recon_loss(data , pos_edge_index: Tensor , decoder = None , poslossmod = 1 , neglossmod= 1, distweight = False) -> Tensor:
    r"""Given latent variables :obj:`z`, computes the binary cross
    entropy loss for positive edges :obj:`pos_edge_index` and negative
    sampled edges.

    Args:
        z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
        pos_edge_index (torch.Tensor): The positive edges to train against.
        neg_edge_index (torch.Tensor, optional): The negative edges to
            train against. If not given, uses negative sampling to
            calculate negative edges. (default: :obj:`None`)
    """
    pos =decoder(data, pos_edge_index )[1]
    #turn pos edge index into a binary matrix
    pos_loss = -torch.log( pos + EPS).mean()
    neg_edge_index = negative_sampling(pos_edge_index, data['res'].x.size(0))
    neg = decoder(data ,  neg_edge_index )[1]
    neg_loss = -torch.log( ( 1 - neg) + EPS ).mean()
    return poslossmod*pos_loss + neglossmod*neg_loss




#amino acid onehot loss for x reconstruction
def aa_reconstruction_loss(x, recon_x):
	"""
	compute the loss over the node feature reconstruction.
	using categorical cross entropy
	"""
	x = torch.argmax(x, dim=1)
	#recon_x = torch.argmax(recon_x, dim=1)
	return F.cross_entropy(recon_x, x)

def gaussian_loss(mu , logvar , beta= 1.5):
	'''
	
	add beta to disentangle the features
	
	'''
	kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
	return beta*kl_loss

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



def fape_loss(true_R, true_t, pred_R, pred_t, batch, plddt= None, d_clamp=10.0, eps=1e-8 , temperature = .25 , reduction = 'mean' , soft = False):
	"""
	Computes the Frame Aligned Point Error (FAPE) loss.
	
	For each structure in the batch, for every pair of residues (i, j),
	the local coordinates of the difference (t[j] - t[i]) are computed
	in the corresponding residue i frame using both true and predicted rotations.
	The loss is then the average clamped L2 distance between
	these local coordinates.
	
	Args:
		true_R (Tensor): True rotation matrices, shape (N, 3, 3)
		true_t (Tensor): True translation vectors, shape (N, 3)
		pred_R (Tensor): Predicted rotation matrices, shape (N, 3, 3)
		pred_t (Tensor): Predicted translation vectors, shape (N, 3)
		batch (Tensor): Batch indices for each residue, shape (N,)
		d_clamp (float, optional): Clamping threshold for error. (default: 10.0)
		eps (float, optional): Small constant for numerical stability. (default: 1e-8)
		
	Returns:
		Tensor: The scalar FAPE loss.
	"""
	losses = []
	unique_batches = torch.unique(batch)
	for b in unique_batches:
		idx = (batch == b).nonzero(as_tuple=True)[0]
		if idx.numel() < 2:
			continue

		# Compute pairwise differences for the predicted translations.
		diff_pred = pred_t[idx].unsqueeze(1) - pred_t[idx].unsqueeze(0)  # shape: (m, m, 3)
		# Transform differences into the local predicted frames.
		local_pred = torch.einsum("mij,mnj->mni", pred_R[idx].transpose(1,2 ), diff_pred)
		
		# Compute pairwise differences for the true translations.
		diff_true = true_t[idx].unsqueeze(1) - true_t[idx].unsqueeze(0)  # shape: (m, m, 3)
		# Transform differences into the local true frames.
		local_true = torch.einsum("mij,mnj->mni", true_R[idx].transpose(1,2), diff_true)
		
		# Compute the L2 error per residue pair and clamp it.
		if soft == False:
			error = torch.norm(local_pred - local_true + eps, dim=-1)
			if plddt is not None:
				error = error*plddt[idx]
			error = torch.clamp(error, max=d_clamp)
			losses.append(error.mean())
		else:				
			# Compute pairwise squared Euclidean distances
			dist_sq = torch.cdist(local_pred, local_true, p=2).pow(2)
			dist_sq = torch.clamp(dist_sq, max=d_clamp**2)
			# Compute soft alignment probabilities using a Gaussian kernel
			soft_alignment = F.softmax(-dist_sq / temperature, dim=-1)
			# Compute soft FAPE loss
			weighted_distances = (soft_alignment * dist_sq).sum(dim=-1)  # (B, N)
			fape_loss = weighted_distances.mean() if reduction == 'mean' else weighted_distances.sum()
			if plddt is not None:
				fape_loss = fape_loss * plddt[idx]
			losses.append(fape_loss)
	if losses:
		return torch.stack(losses).mean()
	else:
		return torch.tensor(0.0, device=true_R.device)

