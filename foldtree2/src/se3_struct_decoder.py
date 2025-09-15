import torch
from se3_transformer_pytorch import SE3Transformer
from foldtree2.src.dynamictan import *
from foldtree2.src.quantizers import *


def derive_rt_from_coords(coords):
	#comput the R and T from one residue to the next using pytorch
	# coords: (N, 3)
	assert coords.dim() == 2 and coords.size(1) == 3, "coords must be of shape (N, 3)"
	
	
class se3_denoiser(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, 
	num_embeddings, commitment_cost, metadata={}, edge_dim=1,
	encoder_hidden=100, dropout_p=0.05):
		super(se3_denoiser, self).__init__()

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
		
		#refinement pass
		if self.rtin == True:
			in_channels += 7 + 3 + 3 # rt, angles, pred_coords
			#refined_coors = coors + model(atom_feats, coors, mask, return_type = 1) # (2, 32, 3)

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


		# SE3Transformer for 3D structure processing
		self.se3 = SE3Transformer(
		dim = 64,
		depth = 2,
		input_degrees = 1,
		num_degrees = 2,
		output_degrees = 2,
		reduce_dim_out = True,
		differentiable_coors = True,
		global_feats_dim = 32 # this must be set to the dimension of the global features, in this example, 32

		).to(self.device)

		self.bn = torch.nn.BatchNorm1d(in_channels)
		self.dropout = torch.nn.Dropout(p=dropout_p)
				
		self.input2transformer = torch.nn.Sequential(
			torch.nn.Linear(in_channels, hidden_channels[0] * 2),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[0] * 2, hidden_channels[0]),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[0], hidden_channels[0]),
		)


		self.out_angles = torch.nn.Sequential(
			torch.nn.Linear(self.encoder_hidden, self.out_channels),
			torch.nn.GELU(),
			DynamicTanh(self.out_channels, channels_last=True),
			torch.nn.Tanh()
		)


	def forward(self, data, edge_attr_dict=None, **kwargs):
		if isinstance(data, dict):
			x_dict, edge_index_dict = data, kwargs.get('edge_index_dict', {})
		else:
			x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
		x_dict['res'] = self.bn(x_dict['res'])
		x_dict['res'] = self.dropout(x_dict['res'])
		mask  = torch.ones( x_dict['res'].shape).bool().cuda()
		#proj to transformer input dim
		x = self.input2transformer(x_dict['res'])
		coords = x_dict['coords'].view(-1, 3)
		# Transformer expects (seq_len, batch, d_model), so add batch dim if needed
		batch = x_dict['res'].batch if hasattr(x_dict['res'], 'batch') else None
		if batch is not None:
			num_graphs = batch.max().item() + 1
			x_split = [x[batch == i] for i in range(num_graphs)]
			coords_split = [coords[batch == i] for i in range(num_graphs)]
			max_len = max([xi.shape[0] for xi in x_split])
			x_padded = []
			coords_padded = []
			for xi, ci in zip(x_split, coords_split):
				pad_len = max_len - xi.shape[0]
				if pad_len > 0:
					xi = torch.cat([xi, torch.zeros(pad_len, xi.shape[1], device=xi.device, dtype=xi.dtype)], dim=0)
					ci = torch.cat([ci, torch.zeros(pad_len, ci.shape[1], device=ci.device, dtype=ci.dtype)], dim=0)
				x_padded.append(xi)
				coords_padded.append(ci)
			x = torch.stack(x_padded, dim=1)         # (seq_len, batch, d_model)
			coords = torch.stack(coords_padded, dim=1) # (seq_len, batch, 3)
		else:
			x = x.unsqueeze(1)         # (seq_len, 1, d_model)
			coords = coords.unsqueeze(1) # (seq_len, 1, 3)

		# SE3 transformer
		out = self.se3(x, coords, mask)
		z = out.type0 # invariant type 0    - (1, 128)
		newcoords = out.type1 # equivariant type 1  - (1, 128, 3)
		#derive rt from newcoords


		angles = self.out_angles(z)


		return {'angles': angles, 'R': self.out_R(x), 'T': self.out_T(x), 'z': self.out_dense(x)}


class struct_transformer_decoder(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, dropout_p=0.05,
			  decoder_hidden=100,
			  out_angles=True,
			  out_coords=True,
			  out_RT=False,
			  nheads=8,
			  nlayers=3

			  ):
		super(struct_transformer_decoder, self).__init__()
		
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
		self.hidden_channels = hidden_channels
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# vanilla pytorch transformer encoder layer
		self.transformer_encoder = torch.nn.TransformerEncoder(
			torch.nn.TransformerEncoderLayer(
				d_model=hidden_channels,
				nhead=nheads,
				dim_feedforward=hidden_channels[0] * 2,
				activation='gelu'
			),
			num_layers=nlayers
		)

		self.bn = torch.nn.BatchNorm1d(in_channels)
		self.dropout = torch.nn.Dropout(p=dropout_p)
		
		self.input2transformer = torch.nn.Sequential(
			torch.nn.Linear(in_channels, hidden_channels[0] * 2),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[0] * 2, hidden_channels[0]),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[0], hidden_channels[0]),
		)

		self.angle_out = torch.nn.Sequential(
			torch.nn.Linear(self.decoder_hidden, self.out_channels),
			torch.nn.GELU(),
			DynamicTanh(self.out_channels, channels_last=True),
			torch.nn.Tanh()
		)

		self.out_R = torch.nn.Sequential(
			torch.nn.Linear(self.encoder_hidden, 4),
			torch.nn.GELU(),
			DynamicTanh(4, channels_last=True),
			torch.nn.Tanh()
		)
		self.out_T = torch.nn.Sequential(
			torch.nn.Linear(self.encoder_hidden, 3),
			torch.nn.GELU(),
			DynamicTanh(3, channels_last=True),
			torch.nn.Tanh()
		)

		self.out_dense = torch.nn.Sequential(
			torch.nn.Linear(self.encoder_hidden + 20, self.out_channels),
			torch.nn.GELU(),
			DynamicTanh(self.out_channels, channels_last=True),
			torch.nn.Tanh()
		)
		
	def forward(self, data, **kwargs):
		if isinstance(data, dict):
			x_dict, edge_index_dict = data, kwargs.get('edge_index_dict', {})
		else:
			x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
		x_dict['res'] = self.bn(x_dict['res'])
		x = self.dropout(x_dict['res'])
		# proj to transformer input dim
		x = self.input2transformer(x)

		# Transformer expects (seq_len, batch, d_model), so add batch dim if needed
		batch = x_dict['res'].batch if hasattr(x_dict['res'], 'batch') else None
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

		out = x
		for i in range(self.transformer_encoder.num_layers):
			out = self.transformer_encoder.layers[i](out)
			out = F.gelu(out)		

		if self.angle_out is not None:
			angles = self.angle_out(out)
		else:
			angles = None
					
		if self.out_R is not None:
			R = self.out_R(x)
		else:
			R = None
			T = None

		z = self.out_dense(x)
		return {'angles': angles, 'R': R, 'T': T, 'z': z}

	