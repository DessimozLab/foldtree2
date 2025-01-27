import constants
import utils

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
	
from torch_geometric.nn import HeteroConv
class HeteroGAE_Encoder(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, layers ,  out_channels, num_embeddings, commitment_cost, metadata={} , encoder_hidden = 100 , dropout_p = 0.01 , EMA = False , average = False, reset_codes = True , nheads = 3 , separated = False , flavor = 'transformer'):
		super(HeteroGAE_Encoder, self).__init__()

		#save all arguments to constructor
		self.args = locals()
		self.args.pop('self')

		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False        
		self.convs = torch.nn.ModuleList()

		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.in_channels = in_channels
		self.encoder_hidden = encoder_hidden
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		#batch norm
		self.nlayers = layers
		self.bn = torch.nn.BatchNorm1d(in_channels['res'])
		self.dropout = torch.nn.Dropout(p=dropout_p)
		self.separated = separated
		self.average = average
		
		for i in range(layers):
			layer = {}          
			for k,edge_type in enumerate( hidden_channels.keys() ):
				edgestr = '_'.join(edge_type)

				datain = edge_type[0]
				dataout = edge_type[2]

				#if ( 'res','informs','godnode4decoder') == edge_type:
				#	layer[edgestr] = TransformerConv( in_channels[datain] , hidden_channels[edge_type][i], heads = nheads , concat= False)
				#else:
				if flavor == 'transformer' or edge_type == ('res','informs','godnode'):
					layer[edge_type] = TransformerConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False)
				if flavor == 'gat':
					layer[edge_type] = GATConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads )
				if flavor == 'gcn':
					layer[edge_type] = GCNConv( (-1, -1) , hidden_channels[edge_type][i])
				if flavor == 'sage':
					layer[edge_type] = SAGEConv( (-1, -1) , hidden_channels[edge_type][i])
				if flavor == 'mfconv':
					layer[edge_type] = MFConv( (-1, -1)  , hidden_channels[edge_type][i])  
				if flavor == 'FiLM':
					layer[edge_type] = FiLMConv( in_channels[datain] , hidden_channels[edge_type][i],                                               
					nn = 
					torch.nn.Sequential(
					torch.nn.Linear( in_channels[datain] , hidden_channels[edge_type][i]),
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] , hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] , 2 * hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU() )
					)

				if flavor == 'GIN':
					layer[edge_type] = GINConv(                              
					nn = 
					torch.nn.Sequential(
					torch.nn.Linear( in_channels[datain] , hidden_channels[edge_type][i]),
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] , hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] ,  hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU() )
					)
				if ( 'res','backbone','res') == edge_type and i > 0:
					in_channels['res'] = hidden_channels[( 'res','backbone','res')][i-1] + in_channels['godnode']
				else:
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
		
		print('encoder convolutions')
		print(self.convs)
	
		#output dense layer
		self.out_dense = self.out_dense= torch.nn.Sequential(
			Linear(hidden_channels[('res', 'contactPoints', 'res')][-1] + 20 , encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, out_channels),
			torch.nn.SiLU()
			)
		
		#initialize the vector quantizer
		self.vector_quantizer = VectorQuantizerEMA(num_embeddings, out_channels, commitment_cost , reset = reset_codes)
	
	def forward(self, xdata, edge_index):

		#xdata = {key: xdata[key] for key in ['res']}
		#edge_index_dict = {key: edge_index_dict[key] for key in [ ('res','backbone','res') ,  ('res','contactPoints', 'res') ] }
		xdata['res'] = self.bn(xdata['res'])
		xdata['res'] = self.dropout(xdata['res'])
		xaa = xdata['AA']
		for i,layer in enumerate(self.convs):
			xdata = layer(xdata, edge_index)
			#add relu to all convolutions in this layer
			for key in layer.convs.keys():
				key = key[2]
				xdata[key] = F.relu(xdata[key])
		xres = xdata['res']
		x = self.out_dense( torch.cat([xres,xaa], dim=1) )
		#x = self.out_dense( x )
		z_quantized,  qloss = self.vector_quantizer(x)
		return z_quantized, qloss

	def encode_structures( dataloader, encoder, filename = 'structalign.strct' ):
		#write with contacts 
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z,qloss = self.forward(data.x_dict , data.edge_index_dict)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = structlist[i]
				f.write(f'\n//////startprot//////{identifier}//////\n')
				for char in strdata[1]:
					f.write(char)
				f.write(f'\n//////contacts//////{identifier}//////\n')
				#write the contacts stored in the data object
				contacts = data.edge_index_dict[( 'res','contactPoints','res')]
				#write a json object with the contacts
				contacts = contacts.detach().cpu().numpy()
				#convert edge index to a json object
				contacts = contacts.tolist()
				f.write(json.dumps(contacts))
				f.write(f'\n//////endprot//////\n')
				f.write('\n')
		return filename

	def encode_structures_fasta(self, dataloader, filename = 'structalign.strct.fasta' , verbose = False):
		#write an encoded fasta for use with mafft and raxml. only doable with alphabet size of less that 248
		#0x01 – 0xFF excluding > (0x3E), = (0x3D), < (0x3C), - (0x2D), Space (0x20), Carriage Return (0x0d) and Line Feed (0x0a)
		replace_dict = {chr(0):chr(246) , '"':chr(248) , '#':chr(247), '>' : chr(249), '=' : chr(250),
		 '<' : chr(251), '-' : chr(252), ' ' : chr(253) , '\r' : chr(254), '\n' : chr(255) }
		
		#check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z,qloss = self.forward(data.x_dict, data.edge_index_dict)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = data.identifier
				f.write(f'>{identifier}\n')
				outstr = ''
				for char in strdata[0]:
					char = chr(char)
					if char in replace_dict:
						char = replace_dict[char]
					outstr += char
					f.write(char)
				f.write('\n')
				if verbose == True:
					print(identifier, outstr)
		return filename
	
	def encode_structures_numbers(self, dataloader, filename = 'structalign.strct.fasta' ):
		#write an encoded fasta with just numbers of the discrete characters
		#check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z,qloss = self.forward(data.x_dict, data.edge_index_dict)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = data.identifier
				f.write(f'\n>{identifier}\n')
				for num in strdata[0]:
					f.write(str(num)+ ',')
				f.write('\n')
		return filename


	def load(self, modelfile):
		self.load_state_dict(torch.load(modelfile))
		self.eval()
		return self

	def save(self, modelfile):
		torch.save(self.state_dict(), modelfile)
		return modelfile
	
	def ret_config(self):
		return {'in_channels': self.in_channels, 'hidden_channels': self.hidden_channels, 'out_channels': self.out_channels, 'num_embeddings': self.vector_quantizer.num_embeddings, 'commitment_cost': self.vector_quantizer.commitment_cost, 'metadata': self.metadata}

	def save_config(self, configfile):
		with open(configfile , 'w') as f:
			json.dump(self.ret_config(), f)
		return configfile

	def load_from_config(config):
		return HeteroGAE_Encoder(**config)
	
	def encode_structures( dataloader, encoder, filename = 'structalign.strct' ):
		#write with contacts 
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z,qloss = self.forward(data.x_dict , data.edge_index_dict)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = structlist[i]
				f.write(f'\n//////startprot//////{identifier}//////\n')
				for char in strdata[1]:
					f.write(char)
				f.write(f'\n//////contacts//////{identifier}//////\n')
				#write the contacts stored in the data object
				contacts = data.edge_index_dict[( 'res','contactPoints','res')]
				#write a json object with the contacts
				contacts = contacts.detach().cpu().numpy()
				#convert edge index to a json object
				contacts = contacts.tolist()
				f.write(json.dumps(contacts))
				f.write(f'\n//////endprot//////\n')
				f.write('\n')
		return filename

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
	
	def encode_structures_numbers(self, dataloader, filename = 'structalign.strct.fasta' ):
		#write an encoded fasta with just numbers of the discrete characters
		#check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z,qloss = self.forward(data.x_dict, data.edge_index_dict)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = data.identifier
				f.write(f'\n>{identifier}\n')
				for num in strdata[0]:
					f.write(str(num)+ ',')
				f.write('\n')
		return filename


	def load(self, modelfile):
		self.load_state_dict(torch.load(modelfile))
		self.eval()
		return self

	def save(self, modelfile):
		torch.save(self.state_dict(), modelfile)
		return modelfile
	
	def ret_config(self):
		return {'in_channels': self.in_channels, 'hidden_channels': self.hidden_channels, 'out_channels': self.out_channels, 'num_embeddings': self.vector_quantizer.num_embeddings, 'commitment_cost': self.vector_quantizer.commitment_cost, 'metadata': self.metadata}

	def save_config(self, configfile):
		with open(configfile , 'w') as f:
			json.dump(self.ret_config(), f)
		return configfile

	def load_from_config(config):
		return HeteroGAE_Encoder(**config)


class TransformerDecoder(nn.Module):
	def __init__(self, embedding_dim, num_layers=6, nhead=8, dropout=0.1, seq_len=128):
		super(TransformerDecoder, self).__init__()
		self.embedding_dim = embedding_dim
		self.seq_len = seq_len

		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		self.convs = torch.nn.ModuleList()

		#self.bn = torch.nn.BatchNorm1d(encoder_out_channels)
		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.in_channels = encoder_out_channels
		self.amino_acid_indices = amino_mapper
		self.bn = torch.nn.BatchNorm1d(encoder_out_channels)
		self.revmap_aa = { v:k for k,v in amino_mapper.items() }
		self.dropout = torch.nn.Dropout(p=dropout)
		
		# Transformer decoder
		self.transformer = nn.Transformer(
			d_model=embedding_dim,
			nhead=nhead,
			num_encoder_layers=num_layers,
			num_decoder_layers=num_layers,
			dropout=dropout
		)
		
		# Linear projection from transformer output to sequence logits (e.g., for amino acids)
		self.to_sequence = nn.Linear(embedding_dim, 20)  # 20 for amino acid sequence classification (AAs)
		
		# Optional contact prediction head
		self.contact_head = nn.Linear(embedding_dim, 1)  # Sigmoid for binary contact prediction



	def forward(self, quantized_latents):
		"""
		Args:
			quantized_latents (Tensor): Quantized embeddings from the encoder (VQ output), shape (batch_size, seq_len, embedding_dim)
		Returns:
			sequence_logits (Tensor): Logits for sequence reconstruction (batch_size, seq_len, 20)
			contact_map (Tensor): Predicted contact map (batch_size, seq_len, seq_len)
		"""
		# Apply transformer decoder
		# (For simplicity, we'll use quantized_latents as both encoder and decoder input in the transformer)
		transformer_output = self.transformer(quantized_latents, quantized_latents)
		
		# Project to sequence space (e.g., amino acids)
		sequence_logits = self.to_sequence(transformer_output)
		
		# Predict contacts
		batch_size, seq_len, _ = transformer_output.shape
		contact_map = torch.sigmoid(self.contact_head(transformer_output)).view(batch_size, seq_len, 1)  # Pairwise contacts

		return sequence_logits, contact_map


class HeteroGAE_Decoder(torch.nn.Module):
	def __init__(self, in_channels = {'res':10 , 'godnode4decoder':5 , 'foldx':23}, xdim=20, hidden_channels={'res_backbone_res': [20, 20, 20]}, layers = 3,  AAdecoder_hidden = 20 
			  ,PINNdecoder_hidden = 10, contactdecoder_hidden = 10, nheads = 3 , Xdecoder_hidden=30, metadata={}, amino_mapper= None  , flavor = None, dropout= .1):
		super(HeteroGAE_Decoder, self).__init__()
		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		self.convs = torch.nn.ModuleList()
		
		in_channels_orig = copy.deepcopy(in_channels )

		#self.bn = torch.nn.BatchNorm1d(encoder_out_channels)
		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.in_channels = in_channels
		self.amino_acid_indices = amino_mapper
		self.nlayers = layers
		self.bn = torch.nn.BatchNorm1d(in_channels['res'])

		self.revmap_aa = { v:k for k,v in amino_mapper.items() }
		self.dropout = torch.nn.Dropout(p=dropout)
		
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
					layer[edge_type] = TransformerConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False)
				if flavor == 'gat':
					layer[edge_type] = GATConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads )
				if flavor == 'gcn':
					layer[edge_type] = GCNConv( (-1, -1) , hidden_channels[edge_type][i])
				if flavor == 'sage':
					layer[edge_type] = SAGEConv( (-1, -1) , hidden_channels[edge_type][i])
				if flavor == 'mfconv':
					layer[edge_type] = MFConv( (-1, -1)  , hidden_channels[edge_type][i])  
				if flavor == 'FiLM':
					layer[edge_type] = FiLMConv( in_channels[datain] , hidden_channels[edge_type][i],                                               
					nn = 
					torch.nn.Sequential(
					torch.nn.Linear( in_channels[datain] , hidden_channels[edge_type][i]),
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] , hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] , 2 * hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU() )
					)

				if flavor == 'GIN':
					layer[edge_type] = GINConv(                              
					nn = 
					torch.nn.Sequential(
					torch.nn.Linear( in_channels[datain] , hidden_channels[edge_type][i]),
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] , hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] ,  hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU() )
					)

				
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
			

			conv = HeteroConv( layer  , aggr='mean')
			self.convs.append( conv )
		
		print('decoder convs')
		print( self.convs)

		print( 'batchnorm' , self.bn)
		print( 'dropout' , self.dropout)
		


		self.sigmoid = nn.Sigmoid()

		self.lin = torch.nn.Sequential(
				torch.nn.Linear( self.hidden_channels[('res', 'backbone', 'res')][-1] , Xdecoder_hidden),
		)
		
		print( in_channels_orig['res'] )  
		print( Xdecoder_hidden)

		self.aadecoder = torch.nn.Sequential(
				torch.nn.Linear(in_channels_orig['res'] + Xdecoder_hidden , AAdecoder_hidden[0]),
				torch.nn.ReLU(),
				torch.nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1] ) ,
				torch.nn.ReLU(),
				torch.nn.Linear(AAdecoder_hidden[1], AAdecoder_hidden[2] ) ,
				torch.nn.ReLU(),

				torch.nn.Linear(AAdecoder_hidden[2], xdim),
				torch.nn.LogSoftmax(dim=1) )
		
		self.contactdecoder = torch.nn.Sequential(
				torch.nn.Linear( Xdecoder_hidden , contactdecoder_hidden[0]),
				torch.nn.ReLU(),
				torch.nn.Linear(contactdecoder_hidden[0], contactdecoder_hidden[1] ) ,
				torch.nn.ReLU(),
				torch.nn.Linear(contactdecoder_hidden[1], contactdecoder_hidden[2] ) ,
				torch.nn.ReLU(),

				torch.nn.Linear(contactdecoder_hidden[2], Xdecoder_hidden),
				)
		

		self.godnodedecoder = torch.nn.Sequential(
				torch.nn.Linear(in_channels['godnode4decoder'] , PINNdecoder_hidden[0]),
				torch.nn.ReLU(),
				torch.nn.Linear(PINNdecoder_hidden[0], PINNdecoder_hidden[1] ) ,
				torch.nn.ReLU(),
				torch.nn.Linear(PINNdecoder_hidden[1], PINNdecoder_hidden[2] ) ,
				torch.nn.ReLU(),
				torch.nn.Linear(PINNdecoder_hidden[2], in_channels['foldx']) )
		
		print('aadecoder', self.aadecoder)
		print('lin' ,  self.lin)
		print( 'sigmoid' ,  self.sigmoid)

	def forward(self, z , xdata, edge_index, contact_pred_index, **kwargs):
		z = self.bn(z)
		#copy z for later concatenation
		inz = z
		xdata['res'] = z
		for i,layer in enumerate(self.convs):
			xdata = layer(xdata, edge_index)
			for key in layer.convs.keys():
				key = key[2]
				xdata[key] = F.relu(xdata[key])
			#context = xdata['godnode4decoder'].repeat(xdata['res'].shape[],1)
			#xdata['res'] = torch.cat( [xdata['res'], xdata['godnode4decoder'] ] , axis = 1)
		z = xdata['res']
		zgodnode = xdata['godnode4decoder']
		#pass through resnet decoder first
		#decoder_in =  torch.cat( [inz,  z] , axis = 1)
		#z_decoder = self.decoder( decoder_in )
		z = self.lin( z )
		#decode aa
		aa = self.aadecoder( torch.cat( [inz,  z ], axis = 1 ) )
		foldx_pred = self.godnodedecoder( xdata['godnode4decoder'] )
		if contact_pred_index is None:
			return aa, None, zgodnode , foldx_pred
		sim_matrix = (z[contact_pred_index[0]] * z[contact_pred_index[1]]).sum(dim=1)
		#find contacts
		edge_probs = self.sigmoid(sim_matrix)
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


def recon_loss(z , xdata,edge_index, pos_edge_index, decoder , poslossmod=1, neglossmod=1) -> Tensor:
	r"""Given latent variables :obj:`z`, computes the binary cross
	entropy loss for positive edges :obj:`pos_edge_index` and negative
	sampled edges.

	Args:
		xdata (HeteroData): The input data containing node features and edge indices.
		pos_edge_index (torch.Tensor): The positive edges to train against.
		decoder (torch.nn.Module, optional): The decoder model. (default: :obj:`None`)
		poslossmod (float, optional): The positive loss modifier. (default: :obj:`1`)
		neglossmod (float, optional): The negative loss modifier. (default: :obj:`1`)
	"""
	
	pos = decoder(z , xdata, edge_index, pos_edge_index )[1]
	pos_loss = -torch.log(pos + EPS).mean()
	neg_edge_index = negative_sampling( pos_edge_index, xdata['res'].size(0))
	neg = decoder(z , xdata, edge_index , neg_edge_index)[1]
	neg_loss = -torch.log((1 - neg) + EPS).mean()
	
	return poslossmod * pos_loss + neglossmod * neg_loss

#define loss for x reconstruction   
def x_reconstruction_loss(x, recon_x):
	"""
	compute the loss over the node feature reconstruction.
	"""
	return F.l1_loss(recon_x, x)


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
