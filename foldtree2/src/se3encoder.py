import torch
from se3_transformer_pytorch import SE3Transformer
from foldtree2.src.dynamictan import *
from foldtree2.src.quantizers import *

class se3_Encoder(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, 
	num_embeddings, commitment_cost, metadata={}, edge_dim=1,
	encoder_hidden=100, dropout_p=0.05, EMA=False, 
	reset_codes=True, nheads=3, flavor='sage', fftin=False):
		super(se3_Encoder, self).__init__()
		
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
		self.encoder_hidden = encoder_hidden
		self.num_embeddings = num_embeddings
		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.fftin = fftin
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		# SE3Transformer for 3D structure processing
		self.se3 = SE3Transformer(
			dim=hidden_channels[0],
			heads=nheads,
			depth=len(hidden_channels) if isinstance(hidden_channels, list) else 1,
			dim_head=hidden_channels[0] if isinstance(hidden_channels, list) else 64,
			num_degrees=2,
			valid_radius=10,
			attend_sparse_neighbors=True,
			num_neighbors=0,
			max_sparse_neighbors=8
		)

		self.bn = torch.nn.BatchNorm1d(in_channels)
		self.dropout = torch.nn.Dropout(p=dropout_p)
		
		if fftin == True:
			in_channels+= 2 * 80

		
		self.ffin = torch.nn.Sequential(
			torch.nn.Linear(in_channels, hidden_channels[0] * 2),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[0] * 2, hidden_channels[0]),
			torch.nn.GELU(),
			DynamicTanh(hidden_channels[0], channels_last=True),
		)
		
		self.lin = torch.nn.Sequential(
			DynamicTanh(hidden_channels[-1] * (len(hidden_channels)-1), channels_last=True),
			torch.nn.Linear(hidden_channels[-1] * (len(hidden_channels)-1), self.encoder_hidden),
			torch.nn.GELU(),
			torch.nn.Linear(self.encoder_hidden, self.encoder_hidden),
			torch.nn.GELU(),
		)

		self.out_dense = torch.nn.Sequential(
			torch.nn.Linear(self.encoder_hidden + 20, self.out_channels),
			torch.nn.GELU(),
			DynamicTanh(self.out_channels, channels_last=True),
			torch.nn.Tanh()
		)
		
		if not EMA:
			self.vector_quantizer = VectorQuantizer(num_embeddings, out_channels, commitment_cost)
		else:
			self.vector_quantizer = VectorQuantizerEMA(num_embeddings, out_channels, commitment_cost, reset=reset_codes)

	def forward(self, data, edge_attr_dict=None, **kwargs):
		if isinstance(data, dict):
			x_dict, edge_index_dict = data, kwargs.get('edge_index_dict', {})
		else:
			x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

		x_dict['res'] = self.bn(x_dict['res'])

		if self.fftin == True:
			x_dict['res'] = torch.cat([x_dict['res'], data['fourier1dr'].x, data['fourier1di'].x], dim=1)
		x = self.dropout(x_dict['res'])

		# apply ffin
		x = self.ffin(x)

		# Get coordinates for SE3 transformer
		coords = data['coords'].x 

		# Build adjacency matrix from edge_index
		N = x.shape[0]
		adj_mat = torch.zeros((N, N), dtype=torch.bool, device=x.device)
		edge_index = edge_index_dict.get(('res', 'backbone', 'res'), None)
		if edge_index is not None:
			adj_mat[edge_index[0], edge_index[1]] = True		
		
		# Apply SE3 transformer by batch
		batches = data['res'].batch if 'batch' in data else None
		
		if batches is not None and batches.dim() == 1:
			transformed = []
			for batch_idx in range(batches.max().item() + 1):
				mask = (batches == batch_idx)
				x_batch = x[mask]
				coords_batch = coords[mask]
				adj_mat_batch = adj_mat[mask][:, mask]
				if x_batch.shape[0] == 0:
					continue
				# Transform the batch
				transformed_batch = self.se3(x_batch, coords_batch, mask, adj_mat=adj_mat_batch)					
				transformed.append(transformed_batch)
			transformed = torch.cat(transformed, dim=0)	
		else:
			# If no batch information, process the entire graph at once
			transformed = self.se3(x, coords, None, adj_mat=adj_mat)
		
		# Remove batch dimension if it was added
		if transformed.dim() == 3 and x.shape[0] == 1:
			transformed = transformed.squeeze(0)
		
		x = self.lin(transformed)
		
		# Combine with amino acid sequence information
		x = self.out_dense(torch.cat([x, x_dict['AA']], dim=1))
		
		z_quantized, vq_loss = self.vector_quantizer(x)
		
		return z_quantized, vq_loss

	def structlist_loader(self, structlist, batch_size=1):
		#load a list of structures into a dataloader
		dataloader = DataLoader(structlist, batch_size=batch_size, shuffle=False)
		return dataloader

	def encode_structures_fasta(self, dataloader, filename='structalign.strct.fasta', verbose=False, alphabet=None, replace=False):
		#write an encoded fasta for use with mafft and iqtree. only doable with alphabet size of less that 248
		#0x01 – 0xFF excluding > (0x3E), = (0x3D), < (0x3C), - (0x2D), Space (0x20), Carriage Return (0x0d) and Line Feed (0x0a)
		replace_dict = { '>' : chr(249), '=' : chr(250), '<' : chr(251), '-' : chr(252), ' ' : chr(253) , '\r' : chr(254), '\n' : chr(255) }
		#check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		
		if alphabet is not None:
			print('using alphabet')
			print(alphabet)
		
		with open(filename, 'w') as f:
			for i, data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z, qloss = self.forward(data)
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
