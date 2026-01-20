import torch
torch.set_default_dtype(torch.float64)  # recommended for equivariant network training

from gotennet_pytorch import GotenNet
from foldtree2.src.dynamictan import *
from foldtree2.src.quantizers import *
import pytorch_lightning as L


def derive_rt_from_coords(coords):
	#comput the R and T from one residue to the next using pytorch
	# coords: (N, 3)
	assert coords.dim() == 2 and coords.size(1) == 3, "coords must be of shape (N, 3)"
	
	
class se3_denoiser(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, 
	num_embeddings, commitment_cost, metadata={}, edge_dim=1,
	encoder_hidden=100, dropout_p=0.05, max_degree=2, depth=1, heads=2, dim_head=32,
	dim_edge_refinement=256, return_coors=True, num_atom_types=20):
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
		self.return_coors = return_coors
		self.num_atom_types = num_atom_types
		
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# GotenNet for 3D structure processing
		self.gotennet = GotenNet(
			dim = hidden_channels[0] if isinstance(hidden_channels, list) else hidden_channels,
			max_degree = max_degree,
			depth = depth,
			heads = heads,
			dim_head = dim_head,
			dim_edge_refinement = dim_edge_refinement,
			return_coors = return_coors
		).to(self.device)

		self.bn = torch.nn.BatchNorm1d(in_channels)
		self.dropout = torch.nn.Dropout(p=dropout_p)
		
		# Project input features to atom IDs (discrete tokens for GotenNet)
		self.input2atomids = torch.nn.Sequential(
			torch.nn.Linear(in_channels, hidden_channels[0] * 2),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[0] * 2, hidden_channels[0]),
			torch.nn.GELU(),
			torch.nn.Linear(hidden_channels[0], num_atom_types),
		)

		# Output layers for angles
		gotennet_out_dim = hidden_channels[0] if isinstance(hidden_channels, list) else hidden_channels
		self.out_angles = torch.nn.Sequential(
			torch.nn.Linear(gotennet_out_dim, self.encoder_hidden),
			torch.nn.GELU(),
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
		
		# Normalize and dropout input features
		x_dict['res'] = self.bn(x_dict['res'])
		x_dict['res'] = self.dropout(x_dict['res'])
		
		# Project input features to atom IDs
		atom_logits = self.input2atomids(x_dict['res'])
		atom_ids = torch.argmax(atom_logits, dim=-1)  # (num_nodes,)
		
		# Get coordinates
		coords = data['coords'].x if hasattr(data, 'coords') else data.x_dict['coords']
		coords = coords.view(-1, 3)  # (num_nodes, 3)
		
		# Create adjacency matrix from edge_index
		# For GotenNet, we need a dense adjacency matrix
		batch = data['res'].batch if hasattr(data['res'], 'batch') else None
		
		if batch is not None:
			# Handle batched data
			num_graphs = batch.max().item() + 1
			atom_ids_list = []
			coords_list = []
			adj_mat_list = []
			
			for i in range(num_graphs):
				mask = batch == i
				num_nodes = mask.sum().item()
				
				# Extract atom_ids and coords for this graph
				atom_ids_list.append(atom_ids[mask])
				coords_list.append(coords[mask])
				
				# Build adjacency matrix for this graph
				adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=self.device)
				# Use backbone edges or contact edges if available
				if ('res', 'backbone', 'res') in edge_index_dict:
					edge_index = edge_index_dict[('res', 'backbone', 'res')]
				elif ('res', 'contactPoints', 'res') in edge_index_dict:
					edge_index = edge_index_dict[('res', 'contactPoints', 'res')]
				else:
					# Use first available edge type
					edge_index = list(edge_index_dict.values())[0] if edge_index_dict else None
				
				if edge_index is not None:
					# Filter edges for this graph
					node_indices = torch.where(mask)[0]
					node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
					
					for edge_idx in range(edge_index.shape[1]):
						src, dst = edge_index[0, edge_idx].item(), edge_index[1, edge_idx].item()
						if src in node_mapping and dst in node_mapping:
							adj[node_mapping[src], node_mapping[dst]] = True
				
				adj_mat_list.append(adj)
			
			# Pad to same length
			max_len = max(aid.shape[0] for aid in atom_ids_list)
			atom_ids_padded = []
			coords_padded = []
			adj_mat_padded = []
			
			for aid, c, adj in zip(atom_ids_list, coords_list, adj_mat_list):
				pad_len = max_len - aid.shape[0]
				if pad_len > 0:
					# Pad with -1 for atom_ids (GotenNet treats negative as padding)
					atom_ids_padded.append(torch.cat([aid, torch.full((pad_len,), -1, device=self.device, dtype=aid.dtype)]))
					coords_padded.append(torch.cat([c, torch.zeros(pad_len, 3, device=self.device, dtype=c.dtype)]))
					# Pad adjacency matrix
					adj_pad = torch.zeros((max_len, max_len), dtype=torch.bool, device=self.device)
					adj_pad[:adj.shape[0], :adj.shape[1]] = adj
					adj_mat_padded.append(adj_pad)
				else:
					atom_ids_padded.append(aid)
					coords_padded.append(c)
					adj_mat_padded.append(adj)
			
			atom_ids_batch = torch.stack(atom_ids_padded)  # (batch, max_len)
			coords_batch = torch.stack(coords_padded)      # (batch, max_len, 3)
			adj_mat_batch = torch.stack(adj_mat_padded)    # (batch, max_len, max_len)
		else:
			# Single graph case
			num_nodes = atom_ids.shape[0]
			adj_mat = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=self.device)
			
			# Build adjacency matrix from edge_index
			if ('res', 'backbone', 'res') in edge_index_dict:
				edge_index = edge_index_dict[('res', 'backbone', 'res')]
			elif ('res', 'contactPoints', 'res') in edge_index_dict:
				edge_index = edge_index_dict[('res', 'contactPoints', 'res')]
			else:
				edge_index = list(edge_index_dict.values())[0] if edge_index_dict else None
			
			if edge_index is not None:
				adj_mat[edge_index[0], edge_index[1]] = True
			
			atom_ids_batch = atom_ids.unsqueeze(0)  # (1, num_nodes)
			coords_batch = coords.unsqueeze(0)      # (1, num_nodes, 3)
			adj_mat_batch = adj_mat.unsqueeze(0)    # (1, num_nodes, num_nodes)
		
		# Forward pass through GotenNet
		invariant, coors_out = self.gotennet(atom_ids_batch, adj_mat=adj_mat_batch, coors=coords_batch)
		
		# invariant shape: (batch, num_nodes, dim)
		# coors_out shape: (batch, num_nodes, 3) if return_coors=True
		
		# Flatten batch dimension for compatibility with downstream processing
		if batch is not None:
			# Unpad and concatenate
			z_list = []
			for i in range(num_graphs):
				mask = batch == i
				num_nodes = mask.sum().item()
				z_list.append(invariant[i, :num_nodes])
			z = torch.cat(z_list, dim=0)  # (total_nodes, dim)
		else:
			z = invariant.squeeze(0)  # (num_nodes, dim)
		
		# Predict angles from invariant features
		angles = self.out_angles(z)
		
		return {
			'angles': angles,
			'z': z,
			'coors_out': coors_out if self.return_coors else None
		}


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

	