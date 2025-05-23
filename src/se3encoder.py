import torch
from se3_transformer_pytorch import SE3Transformer
from src.dynamictan import *
from src.quantizers import *

class se3_Encoder(torch.nn.Module):
	def __init__(self, in_channels, out_channels, num_embeddings, commitment_cost, encoder_hidden=100, dropout_p=0.05, EMA=True, reset_codes=False):
		super(mk1_Encoder, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.encoder_hidden = encoder_hidden
		self.num_embeddings = num_embeddings

		self.se3 = SE3Transformer(
			dim=in_channels,
			heads=8,
			depth=1,
			dim_head=64,
			num_degrees=2,
			valid_radius=10,
			attend_sparse_neighbors=True,
			num_neighbors=0,
			max_sparse_neighbors=8
		)

		self.dropout = torch.nn.Dropout(p=dropout_p)
		self.lin = torch.nn.Sequential(
			DynamicTanh(in_channels , channels_last = True),
			torch.nn.Linear(in_channels, encoder_hidden),
			torch.nn.GELU(),
		)

		self.out_dense = torch.nn.Sequential(
			DynamicTanh(encoder_hidden + 20 , channels_last = True),
			torch.nn.Linear(encoder_hidden + 20 , out_channels),
			torch.nn.GELU(),
			torch.nn.Tanh()
		)
		
		if not EMA:
			self.vector_quantizer = VectorQuantizer(num_embeddings, out_channels, commitment_cost)
		else:
			self.vector_quantizer = VectorQuantizerEMA(num_embeddings, out_channels, commitment_cost, reset=reset_codes)

	def forward(self, data, **kwargs):
		# Assume 'res' node type, adjust as needed
		x = data['res'].x  # (N, in_channels)
		coors = data['coords'].x  # (N, 3)
		mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)  # or use data['res'].mask if available
		batches = data['res'].batch  # (N,)
		# Mask is a boolean tensor indicating valid nodes
		
		# Adjacency matrix: should be (N, N) bool tensor
		# Build from edge_index if needed
		N = x.shape[0]
		adj_mat = torch.zeros((N, N), dtype=torch.bool, device=x.device)
		edge_index = data['res'].edge_index
		adj_mat[edge_index[0], edge_index[1]] = True

		# Add batch dimension if needed
		if x.dim() == 2:
			x = x.unsqueeze(0)
			coors = coors.unsqueeze(0)
			mask = mask.unsqueeze(0)
			adj_mat = adj_mat.unsqueeze(0)

		x = self.dropout(x)
		out = self.se3(x, coors, mask, adj_mat=adj_mat)  # (B, N, out_dim)
		out = out.squeeze(0)  # remove batch dim if present

		x = self.lin(out)
		x = self.out_dense(x)

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
