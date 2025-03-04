#!/usr/bin/env python
# coding: utf-8

from utils import *
from losses import *

from  torch_geometric.utils import to_undirected

class VectorQuantizerEMA(nn.Module):
	def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99 , epsilon=1e-5, reset_threshold=100000, reset = True , klweight = 0 , diversityweight=0 , entropyweight = 1 , jsweight = 0):
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
		self.jsweight = jsweight

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
		if self.entropyweight > 0:
			entropy_reg = entropy_regularization(encodings)
		else:
			entropy_reg = 0
		if self.diversityweight > 0:
			diversity_reg = diversity_regularization(encodings)
		else:
			diversity_reg = 0
		if self.klweight > 0:
			kl_div_reg = kl_divergence_regularization(encodings)
		else:
			kl_div_reg = 0
		if self.jsweight > 0:
			jensen_shannon = jensen_shannon_regularization(encodings)
		else:
			jensen_shannon = 0
		# Combine all losses
		total_loss = loss - self.entropyweight*entropy_reg + self.diversityweight*diversity_reg + self.klweight*kl_div_reg - self.jsweight*jensen_shannon

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

def jensen_shannon_regularization(encodings):
    # 1) Compute the average distribution p
    p = encodings.mean(dim=0)
    
    # 2) Define uniform distribution u
    K = p.size(0)
    u = torch.ones_like(p) / K
    
    # 3) Compute the midpoint m = (p + u) / 2
    m = 0.5 * (p + u)
    
    # 4) Use the definition of JSD(p || u):
    # JSD(p || u) = 0.5 * KL(p || m) + 0.5 * KL(u || m)
    # KL(x || y) = sum( x_i * log(x_i / y_i) )
    eps = 1e-10
    
    kl_p_m = torch.sum(p * torch.log((p + eps) / (m + eps)))
    kl_u_m = torch.sum(u * torch.log((u + eps) / (m + eps)))
    
    jsd = 0.5 * kl_p_m + 0.5 * kl_u_m
    return jsd



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
		for i in range(len(hidden_channels)):
			if flavor == 'gat':
				self.convs.append(
					torch.nn.ModuleDict({
						'_'.join(edge_type): GATv2Conv(in_channels if i == 0 else hidden_channels[i-1], 
						hidden_channels[i] , heads = nheads , dropout = dropout_p,
						concat = False )
						for edge_type in metadata['edge_types']
					})
				)
			if flavor == 'sage':
				self.convs.append(
					torch.nn.ModuleDict({
						'_'.join(edge_type): SAGEConv(in_channels if i == 0 else hidden_channels[i-1], 
						hidden_channels[i] )
						for edge_type in metadata['edge_types']
					})
				)
				
			self.norms.append( 
				torch.nn.LayerNorm(hidden_channels[i])
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
			torch.nn.Tanh()
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
			x = self.norms[i](x)
			x = F.gelu(x) if i < len(self.hidden_channels) - 1 else x
			x_save.append(x)
		x = self.jk(x_save)
		x = self.lin(x)
		x = self.out_dense( torch.cat([ x , x_dict['AA']], dim=1) )
		#normalize the output to have norm 1
		x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-10)
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
	
def quaternion_to_rotation_matrix(quat):
	"""
	Convert a batch of quaternions (x, y, z, w) into 3x3 rotation matrices.
	
	Parameters:
	- quat: (batch, N, 4) Tensor of quaternions (x, y, z, w)

	Returns:
	- rot_matrices: (batch, N, 3, 3) Tensor of rotation matrices
	"""
	assert quat.shape[-1] == 4, "Quaternions should have shape (*, 4)"
	
	norm = torch.norm(quat, dim=-1, keepdim=True)
	quat = quat / norm
	x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]


	# Compute rotation matrix elements
	xx, yy, zz = x * x, y * y, z * z
	xy, xz, yz = x * y, x * z, y * z
	wx, wy, wz = w * x, w * y, w * z

	rot_matrices = torch.stack([
		torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
		torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
		torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
	], dim=-2)

	return rot_matrices  # Shape: (batch, N, 3, 3)


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
			nn.LayerNorm(d_model),
			nn.Linear(d_model, 50),
			nn.GELU(),
			nn.Linear(50, 25),
			nn.GELU(),
			nn.Linear(25, 7) ,
			nn.LayerNorm(7) )
		
		self.output_proj_angles = nn.Sequential(
			nn.Linear(d_model, 10),
			nn.GELU(),
			nn.Linear(10, 5),
			nn.GELU(),
			nn.LayerNorm(5),
			nn.Linear(5, 3) )
	
	def forward(self, angles , positions):

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
		N, _ = angles.shape
		# Flatten each 3x3 rotation matrix into 9 numbers: ( N, 9)
		if positions is not None:
			x = torch.cat([angles , positions], dim=-1)
		else:
			x = angles
		# Project to the transformer dimension: (N, d_model)
		x = self.input_proj(x)
		# PyTorch's transformer expects shape (seq_len, batch, d_model)
		x = self.transformer_encoder(x)  # Process all positions (sequence length) per batch
		# Project back to the 12-dim space
		refined_features_rt = self.output_proj_rt(x)  # ( N, 4)
		refined_features_angles = self.output_proj_angles(x)  # ( N, 3)
		#learn residual for angles
		refined_features_angles += angles
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
			  ,PINNdecoder_hidden = 10, contactdecoder_hidden = 10, 
			  nheads = 3 , Xdecoder_hidden=30, metadata={}, 
			  amino_mapper= None  , flavor = None, dropout= .001 , geodecoder_hidden = 10 ,
				output_foldx = False , geometry = False , denoise = False , normalize = True):
		super(HeteroGAE_Decoder, self).__init__()
		# Setting the seed
		L.seed_everything(42)

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
				if flavor == 'gat':
					layer[edge_type] =  GATv2Conv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False	)
				if flavor == 'transformer' or edge_type == ('res','informs','godnode4decoder'):
					layer[edge_type] =  TransformerConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False  ) 
				if flavor == 'sage':
					layer[edge_type] =  SAGEConv( (-1, -1) , hidden_channels[edge_type][i] ) # , aggr = SoftmaxAggregation() ) 
				if flavor == 'mfconv':
					layer[edge_type] = MFConv( (-1, -1)  , hidden_channels[edge_type][i] , max_degree=5  , aggr = 'max' )
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
			self.norms.append( torch.nn.LayerNorm(hidden_channels[('res','backbone','res')][i]) )

		self.sigmoid = nn.Sigmoid()
		self.lin = torch.nn.Sequential(
				torch.nn.LayerNorm(sum( self.hidden_channels[('res', 'backbone', 'res')] )),
				torch.nn.Linear( sum(self.hidden_channels[('res', 'backbone', 'res')]) , Xdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(Xdecoder_hidden[0], Xdecoder_hidden[1]),
				torch.nn.GELU(),
				torch.nn.Linear(Xdecoder_hidden[1], Xdecoder_hidden[2]),
				torch.nn.GELU(),
				torch.nn.Linear(Xdecoder_hidden[2], Xdecoder_hidden[2]),
				torch.nn.GELU(),
				torch.nn.LayerNorm(Xdecoder_hidden[2]),
				)
		
		self.aatransformer = TransformerConv( (-1, -1) , AAdecoder_hidden[0] , heads = nheads , concat= False  )
		self.aadecoder = torch.nn.Sequential(
				torch.nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1] ) ,
				torch.nn.GELU(),
				#torch.nn.Linear(AAdecoder_hidden[1],AAdecoder_hidden[2]) ,
				#torch.nn.GELU(),
				torch.nn.LayerNorm(AAdecoder_hidden[1]),
				torch.nn.Linear(AAdecoder_hidden[1] , xdim) ,
				torch.nn.LogSoftmax(dim=1) )
	
		if output_foldx == True:
			self.godnodedecoder = torch.nn.Sequential(
					torch.nn.Linear(in_channels['godnode4decoder'] , PINNdecoder_hidden[0]),
					torch.nn.GELU(),
					torch.nn.Linear(PINNdecoder_hidden[0], PINNdecoder_hidden[1] ) ,
					torch.nn.GELU(),
					torch.nn.LayerNorm(PINNdecoder_hidden[1]),
					torch.nn.Linear(PINNdecoder_hidden[1], in_channels['foldx']) )
		
		if denoise == True:
			dim = 256+10
			self.denoiser = DenoisingTransformer(input_dim= dim, d_model=100, nhead=5, num_layers=2 , dropout=0.001)
		else:
			self.denoiser = None
		
		if geometry == True:
			self.geometry_decoder = TransformerConv( (-1, -1) , geodecoder_hidden[0] , heads = nheads , concat= False  )
			self.geo_out = torch.nn.Sequential(
				torch.nn.Linear(geodecoder_hidden[0] , geodecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(geodecoder_hidden[0], geodecoder_hidden[1] ) ,
				torch.nn.GELU(),
				torch.nn.Linear(geodecoder_hidden[1],geodecoder_hidden[2]) ,
				torch.nn.GELU(),
				torch.nn.LayerNorm(geodecoder_hidden[2]),
				torch.nn.Linear(geodecoder_hidden[2] , 10 )  )
		else:
			self.geometry_decoder = None
	
	def forward(self, data , contact_pred_index, **kwargs):
		data['res', 'backbone', 'res'].edge_index = to_undirected(data['res' , 'backbone' , 'res'].edge_index) 
		xdata, edge_index = data.x_dict, data.edge_index_dict
		#xdata['res'] = self.bn(xdata['res'])
		if self.concat_positions == True:
			xdata['res'] = torch.cat([xdata['res'], data['positions'].x], dim=1)
		xdata['res'] = self.dropout(xdata['res'])
		#copy z for later concatenation
		inz = xdata['res'].clone()	
		x_dict_list = []
		for i,layer in enumerate(self.convs):
			xdata = layer(xdata, edge_index)
			for key in xdata.keys():
				xdata[key] = F.gelu(xdata[key])
			xdata['res'] = self.norms[i](xdata['res'])
			x_dict_list.append(xdata['res'])
		xdata['res'] = self.jk(x_dict_list)
		z = xdata['res']
		z = self.lin(z)
		z = torch.tanh(z)
		if self.normalize == True:
			z =  z / ( torch.norm(z, dim=1, keepdim=True) + 1e-10)

		decoder_in =  torch.cat( [inz,  z] , axis = 1)
		#decode aa
		
		aa_in = torch.cat([decoder_in, data['positions'].x], dim=1)
		aa_in = self.aatransformer( aa_in , edge_index['res' , 'window' , 'res'] )
		aa = self.aadecoder(aa_in)

		#decode geometry
		if self.geometry_decoder is not None:
			#add positional encoding to decoder input
			geo_in = torch.cat([decoder_in, data['positions'].x], dim=1)
			rta = self.geometry_decoder( geo_in , edge_index['res' , 'window' , 'res'] )
			rta = self.geo_out(rta)
			r = rta[..., :4]  # ( N, 4)
			r = quaternion_to_rotation_matrix(r)  # ( N, 3, 3)
			t = rta[..., 4:7]  # ( N, 3)
			angles = rta[..., 7:]
		else:
			angles = None
			r = None
			t = None
		
		#decode godnode
		if self.output_foldx == True:
			zgodnode = xdata['godnode4decoder']
			foldx_pred = self.godnodedecoder( xdata['godnode4decoder'] )
		else:
			foldx_pred = None
			zgodnode = None
		#decode geometry with small transformer to refine coordinates etc
		if self.denoiser is not None:
			unique_batches = torch.unique(data['res'].batch)	
			rs = []
			ts = []
			angles_list = []
			for b in unique_batches:
				idx = (data['res'].batch == b).nonzero(as_tuple=True)[0]
				if idx.numel() > 2:
					ri,ti, anglesi = self.denoiser(angles2[idx] ,  data['positions'].x[idx])
					ri = ri.view(-1, 3 , 3)
					ti = ti.view(-1, 3)
					rs.append(ri)
					ts.append(ti)
					angles_list.append(anglesi)
				else:
					rs.append(r[idx])
					ts.append(t[idx])			
					angles_list.append(angles[idx])
			angles2 = torch.cat(angles_list, dim=0)	
			t2 = torch.cat(ts, dim=0)
			r2 = torch.cat(rs, dim=0)
			r2 = r.view(-1, 3, 3)
			t2 = t.view(-1, 3)
		else:
			r2 = None
			t2 = None
			angles2 = None
		
		if contact_pred_index is None:
			return aa, None, zgodnode , foldx_pred , r , t , angles , r2,t2, angles2
		else:
			edge_probs = self.sigmoid( torch.sum( z[contact_pred_index[0]] * z[contact_pred_index[1]] , axis =1 ) )
		
		return aa,  edge_probs , zgodnode , foldx_pred , r , t , angles , r2, t2, angles2
		
	
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

class Transformer_Decoder(torch.nn.Module):
	def __init__(self, in_channels = {'res':10 , 'godnode4decoder':5 , 'foldx':23}, xdim=20,  layers = 3,  AAdecoder_hidden = [20] * 3 
			  ,PINNdecoder_hidden = [10] * 3, contactdecoder_hidden = [10 ] * 2, 
			  nheads = 3 , Xdecoder_hidden=30, metadata={}, concat_positions = True,
			  amino_mapper= None  ,  dropout= .001 ,  
				output_foldx = False , denoise = False , geometry = False):
		super(Transformer_Decoder, self).__init__()

		#aimed at testing what removing the local nature of the grpah based decoder does.
		#this is a transformer based decoder that takes the entire graph as input
		if concat_positions == True:
			input_positions = 256
		else:
			input_positions = 0
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
		self.in_channels = in_channels
		self.amino_acid_indices = amino_mapper
		self.nlayers = layers
		self.sigmoid = nn.Sigmoid()
		self.bn = torch.nn.BatchNorm1d(in_channels['res'])
		self.bn_foldx = torch.nn.BatchNorm1d(in_channels['foldx'])
		self.revmap_aa = { v:k for k,v in amino_mapper.items() }
		self.dropout = torch.nn.Dropout(p=dropout)
		self.att_pooling = AttentionPooling(embedding_dim=Xdecoder_hidden + in_channels_orig['res'] + input_positions, hidden_dim=256)
		
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=Xdecoder_hidden, 
												nhead=nheads,
												dropout=dropout , 
												dim_feedforward=512,
												activation='gelu',    # Use GELU instead of ReLU
												norm_first=True   )
		
		
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layers)
		self.concat_positions = concat_positions
		
		self.lin = torch.nn.Sequential(
				torch.nn.LayerNorm( in_channels['res'] + input_positions ),
				torch.nn.Linear(in_channels['res'] + input_positions, Xdecoder_hidden),
				torch.nn.GELU(),
				torch.nn.Linear(Xdecoder_hidden, Xdecoder_hidden),
				torch.nn.GELU(),
				torch.nn.LayerNorm(Xdecoder_hidden),
				)
		
		self.aadecoder = torch.nn.Sequential(
				torch.nn.Linear(Xdecoder_hidden + in_channels_orig['res'] + input_positions , AAdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1] ) ,
				torch.nn.GELU(),
				torch.nn.LayerNorm(AAdecoder_hidden[1]),
				torch.nn.Linear(AAdecoder_hidden[1] , xdim) ,
				torch.nn.LayerNorm(xdim),
				torch.nn.LogSoftmax(dim=1) )
		
		if output_foldx == True:
			self.godnodedecoder = torch.nn.Sequential(
					torch.nn.Linear(Xdecoder_hidden + in_channels_orig['res'] + input_positions , PINNdecoder_hidden[0]),
					torch.nn.GELU(),
					torch.nn.Linear(PINNdecoder_hidden[0], PINNdecoder_hidden[1] ) ,
					torch.nn.GELU(),
					torch.nn.LayerNorm(PINNdecoder_hidden[1]),
					torch.nn.Linear(PINNdecoder_hidden[1], in_channels['foldx']) )
		
			
		if geometry == True:
			self.geometry_decoder = TransformerConv( (-1, -1) , 100 , heads = nheads , concat= False  )
			self.geo_out = torch.nn.Sequential(
				torch.nn.Linear(AAdecoder_hidden[0] , AAdecoder_hidden[0]),
				torch.nn.GELU(),
				torch.nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1] ) ,
				torch.nn.GELU(),
				torch.nn.Linear(AAdecoder_hidden[1],AAdecoder_hidden[2]) ,
				torch.nn.GELU(),
				torch.nn.LayerNorm(AAdecoder_hidden[2]),
				torch.nn.Linear(AAdecoder_hidden[2] , 10 )  )
		else:
			self.geometry_decoder = None
	

		if denoise == True:
			#always concat positions
			#use transformer to denoise local geometrz from transformer conv
			dim = 10 + 256
			self.denoiser = DenoisingTransformer(input_dim= dim, d_model=50, nhead=5, num_layers=2 , dropout=0.001)
		else:
			self.denoiser = None
	
	def forward(self, data , contact_pred_index, **kwargs):
		xdata, edge_index = data.x_dict, data.edge_index_dict

		xdata['res'] = self.bn(xdata['res'])

		if self.concat_positions == True:
			xdata['res'] = torch.cat([xdata['res'], xdata['positions']], dim=-1)

		zin = xdata['res'].clone()
		xdata['res'] = self.dropout(xdata['res'])
		xdata['res'] = self.lin(xdata['res'] )
		unique_batches = torch.unique(data['res'].batch)
		for b in unique_batches:
			idx = (data['res'].batch == b).nonzero(as_tuple=True)[0]
			xdata['res'][idx] = self.transformer_encoder(xdata['res'][idx])
		z = xdata['res'].clone()
		decoder_in =  torch.cat( [ xdata['res'] , zin ] , axis = -1)
		#decode aa
		aa = self.aadecoder(decoder_in)
		#decode geometry
		if self.geometry_decoder is not None:
			rta = self.geometry_decoder( decoder_in , edge_index['res' , 'window' , 'res'] )
			rta = self.geo_out(rta)
			r = rta[..., :4]  # ( N, 4)
			r = quaternion_to_rotation_matrix(r)  # ( N, 3, 3)
			t = rta[..., 4:7]  # ( N, 3)
			angles = rta[..., 7:]
		else:
			angles = None
			r = None
			t = None

		if self.output_foldx == True:
			godnodes = []
			for b in unique_batches:				
				idx = (data['res'].batch == b).nonzero(as_tuple=True)[0]
				pooled = self.att_pooling(decoder_in[idx]).unsqueeze(0)
				godnodes.append(pooled)
			zgodnode = torch.cat(godnodes, dim=0)
			foldx_pred = self.godnodedecoder( zgodnode )
		else:
			foldx_pred = None
			zgodnode = None
		
		#decode with small transformer to refine coordinates etc
		if self.denoiser is not None:
			unique_batches = torch.unique(data['res'].batch)
			rs = []
			ts = []
			angles_list = []
			for b in unique_batches:
				idx = (data['res'].batch == b).nonzero(as_tuple=True)[0]
				if idx.numel() > 2:
					ri,ti, anglesi = self.denoiser(angles2[idx] ,  None)
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
			t2 = torch.cat(ts, dim=0)
			r2 = torch.cat(rs, dim=0)
			r2 = r2.view(-1, 3, 3)
			t2 = t2.view(-1, 3)
		else:
			r2 = None
			t2 = None
			angles2 = None
		
		#decode contacts
		if contact_pred_index is None:
			edge_probs = None		
		else:
			edge_probs = self.sigmoid( torch.sum( z[contact_pred_index[0]] * z[contact_pred_index[1]] , axis =1 ) )	
		return aa,  edge_probs , zgodnode , foldx_pred , r , t , angles , r2, t2, angles2
		
	
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

