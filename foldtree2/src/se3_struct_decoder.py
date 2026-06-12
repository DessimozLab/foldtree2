import torch
torch.set_default_dtype(torch.float64)  # recommended for equivariant network training

from gotennet_pytorch import GotenNet
from foldtree2.src.dynamictan import *
from foldtree2.src.quantizers import *
from foldtree2.src.folding_refiner import QuaternionFoldingRefiner
from foldtree2.src.losses.fape import rotation_matrix_to_quaternion
import pytorch_lightning as L


def _normalize_vec(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _safe_orthogonal(v: torch.Tensor) -> torch.Tensor:
	"""Build a deterministic unit vector orthogonal to v."""
	x_axis = torch.zeros_like(v)
	x_axis[..., 0] = 1.0
	y_axis = torch.zeros_like(v)
	y_axis[..., 1] = 1.0
	use_x = v[..., 0].abs() < 0.9
	base = torch.where(use_x.unsqueeze(-1), x_axis, y_axis)
	ortho = torch.cross(v, base, dim=-1)
	return _normalize_vec(ortho)


def _compute_local_frame_from_ca(ca_coords: torch.Tensor):
	"""Build local frames from CA coordinates only.

	Returns:
		rotmat: (N, 3, 3)
		trans: (N, 3) global translations (CA positions)
		quat: (N, 4) unit quaternions (w, x, y, z)
	"""
	if ca_coords is None:
		return None, None, None

	if ca_coords.ndim != 2 or ca_coords.shape[-1] != 3:
		raise RuntimeError(f'Expected CA coords with shape [N,3], got {tuple(ca_coords.shape)}')

	coords = torch.nan_to_num(ca_coords, nan=0.0, posinf=0.0, neginf=0.0)
	n = coords.shape[0]
	if n == 0:
		return None, None, None

	if n == 1:
		rot = torch.eye(3, dtype=coords.dtype, device=coords.device).unsqueeze(0)
		trans = coords
		quat = rotation_matrix_to_quaternion(rot)
		return rot, trans, quat

	prev_ca = torch.roll(coords, shifts=1, dims=0)
	next_ca = torch.roll(coords, shifts=-1, dims=0)
	prev_ca[0] = coords[0] + (coords[0] - coords[1])
	next_ca[-1] = coords[-1] + (coords[-1] - coords[-2])

	v_prev = coords - prev_ca
	v_next = next_ca - coords

	e1 = _normalize_vec(v_prev + v_next)
	degenerate_e1 = (v_prev + v_next).norm(dim=-1) < 1e-8
	if degenerate_e1.any():
		e1[degenerate_e1] = _normalize_vec(v_next[degenerate_e1])

	normal = torch.cross(v_prev, v_next, dim=-1)
	e2 = _normalize_vec(normal)
	degenerate_e2 = normal.norm(dim=-1) < 1e-8
	if degenerate_e2.any():
		e2[degenerate_e2] = _safe_orthogonal(e1[degenerate_e2])

	e3 = _normalize_vec(torch.cross(e1, e2, dim=-1))
	e2 = _normalize_vec(torch.cross(e3, e1, dim=-1))

	rot = torch.stack([e1, e2, e3], dim=-1)
	rot = torch.nan_to_num(rot, nan=0.0, posinf=0.0, neginf=0.0)
	trans = coords
	quat = rotation_matrix_to_quaternion(rot)
	quat = torch.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0)
	quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)

	return rot, trans, quat


def _frame_outputs_from_coords(coords_flat):
	if coords_flat is None:
		return {
			'quat_pred': None,
			'quat_unit_pred': None,
			'trans_pred': None,
			'trans_local_pred': None,
			'rotmat_pred': None,
			'local_frames_pred': None,
			'rt_pred': None,
		}

	rot, trans, quat = _compute_local_frame_from_ca(coords_flat)
	if rot is None or trans is None or quat is None:
		return {
			'quat_pred': None,
			'quat_unit_pred': None,
			'trans_pred': None,
			'trans_local_pred': None,
			'rotmat_pred': None,
			'local_frames_pred': None,
			'rt_pred': None,
		}

	trans_local = torch.einsum('...ij,...j->...i', rot.transpose(-1, -2), trans)
	local_frames = torch.cat([rot, trans_local.unsqueeze(-1)], dim=-1)
	rt_pred = torch.cat([quat, trans], dim=-1)

	return {
		'quat_pred': quat,
		'quat_unit_pred': quat,
		'trans_pred': trans,
		'trans_local_pred': trans_local,
		'rotmat_pred': rot,
		'local_frames_pred': local_frames,
		'rt_pred': rt_pred,
	}
	
	
class se3_denoiser(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, 
	num_embeddings, commitment_cost, metadata={}, edge_dim=1,
	encoder_hidden=100, dropout_p=0.05, max_degree=2, depth=1, heads=2, dim_head=32,
	dim_edge_refinement=256, return_coors=True, num_atom_types=20):
		super().__init__()

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
			num_atoms = num_atom_types,
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


	def _extract_atom_ids(self, data, x_dict, ft2_token_ids=None, aa_identity_ids=None):
		# Primary label source for SE3 nodes: FoldTree2 discrete token ids from the
		# encoder + quantizer combination.
		if ft2_token_ids is not None:
			return ft2_token_ids.to(dtype=torch.long, device=self.device).view(-1)

		# Secondary label source: amino-acid identities reconstructed by an AA decoder.
		if aa_identity_ids is not None:
			aa_ids = aa_identity_ids.to(dtype=torch.long, device=self.device).view(-1)
			if self.num_atom_types < 20:
				aa_ids = aa_ids % self.num_atom_types
			else:
				aa_ids = aa_ids.clamp_max(self.num_atom_types - 1)
			return aa_ids

		# Optional in-graph fallback if upstream code stores tokens as a node feature.
		if isinstance(data, dict):
			token_store = data.get('ft2_tokens', None)
			token_x = getattr(token_store, 'x', None) if token_store is not None else None
		else:
			token_store = data['ft2_tokens'] if hasattr(data, 'node_types') and ('ft2_tokens' in data.node_types) else None
			token_x = token_store.x if (token_store is not None and hasattr(token_store, 'x')) else None
		if token_x is not None:
			if token_x.ndim == 2 and token_x.shape[-1] == 1:
				token_x = token_x.squeeze(-1)
			return token_x.to(dtype=torch.long, device=self.device).view(-1)

		# Last-resort fallback for compatibility when token ids are unavailable.
		atom_logits = self.input2atomids(x_dict['res'])
		return torch.argmax(atom_logits, dim=-1).to(dtype=torch.long, device=self.device)


	def forward(self, data, edge_attr_dict=None, **kwargs):
		if isinstance(data, dict):
			x_dict, edge_index_dict = data, kwargs.get('edge_index_dict', {})
		else:
			x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
		
		# Normalize and dropout input features
		x_dict['res'] = self.bn(x_dict['res'])
		x_dict['res'] = self.dropout(x_dict['res'])
		
		atom_ids = self._extract_atom_ids(
			data,
			x_dict,
			ft2_token_ids=kwargs.get('ft2_token_ids'),
			aa_identity_ids=kwargs.get('aa_identity_ids'),
		)

		# Fail fast with a readable message instead of a CUDA device-side assert.
		atom_embed = getattr(getattr(self.gotennet, 'node_init', None), 'atom_embed', None)
		if atom_embed is not None and hasattr(atom_embed, 'num_embeddings'):
			max_atom_id = int(atom_ids.max().detach().cpu()) if atom_ids.numel() > 0 else -1
			if max_atom_id >= atom_embed.num_embeddings:
				raise RuntimeError(
					f'atom_ids out of range for GotenNet atom embedding: max_id={max_atom_id} '
					f'num_embeddings={atom_embed.num_embeddings}. '
					'Ensure GotenNet is initialized with num_atoms >= num_atom_types.'
				)
		
		# Get predicted coordinates only. We intentionally refuse to consume
		# ground-truth coords here so SE3 always refines first-stage predictions.
		coords = kwargs.get('coords_pred', None)
		if coords is None:
			coords = kwargs.get('coord_pred', None)
		if coords is None and isinstance(data, dict):
			for k in ('coords_pred', 'coord_pred'):
				store = data.get(k, None)
				val = getattr(store, 'x', None) if store is not None else None
				if val is not None:
					coords = val
					break
		if coords is None and not isinstance(data, dict):
			if hasattr(data, 'node_types'):
				for k in ('coords_pred', 'coord_pred'):
					if k in data.node_types:
						store = data[k]
						if hasattr(store, 'x') and store.x is not None:
							coords = store.x
							break
		if coords is None:
			raise RuntimeError(
				"SE3 decoder requires predicted coordinates via 'coords_pred' or 'coord_pred'; "
				"it will not use known ground-truth 'coords'."
			)
		coords = coords.view(-1, 3)  # (num_nodes, 3)
		
		# Create adjacency matrix from edge_index
		# For GotenNet, we need a dense adjacency matrix
		batch = data['res'].batch if hasattr(data['res'], 'batch') else None
		
		num_graphs = 1
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
				
				#add the contacts by looking at the distance 
				# Compute pairwise distances
				dists = torch.cdist(coords[mask], coords[mask])  # (num nodes, num_nodes)
				contact_threshold = 8.0  # Angstroms, adjust as needed
				contact_edges = (dists < contact_threshold) & (dists > 0)  # Exclude self-edges
				adj |= contact_edges
				
				#reformat this into edge_index format and add to the adj matrix
				#edge_index = contact_edges.nonzero(as_tuple=False).t()
				#if edge_index is not None:
				#	# Filter edges for this graph
				#	node_indices = torch.where(mask)[0]
				#	node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
				#	for edge_idx in range(edge_index.shape[1]):
				#		src, dst = edge_index[0, edge_idx].item(), edge_index[1, edge_idx].item()
				#		if src in node_mapping and dst in node_mapping:
				#			adj[node_mapping[src], node_mapping[dst]] = True
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
			
			# Build adjacency matrix from point cloud distances
			dists = torch.cdist(coords, coords)  # (num_nodes, num_nodes)
			contact_threshold = 8.0  # Angstroms, adjust as needed
			contact_edges = (dists < contact_threshold) & (dists > 0)  # Exclude self-edges
			adj_mat |= contact_edges
			
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
			coords_list = []
			for i in range(num_graphs):
				mask = batch == i
				num_nodes = mask.sum().item()
				z_list.append(invariant[i, :num_nodes])
				if coors_out is not None:
					coords_list.append(coors_out[i, :num_nodes])
			z = torch.cat(z_list, dim=0)  # (total_nodes, dim)
			coors_out_flat = torch.cat(coords_list, dim=0) if len(coords_list) > 0 else None
		else:
			z = invariant.squeeze(0)  # (num_nodes, dim)
			coors_out_flat = coors_out.squeeze(0) if coors_out is not None else None
		
		# Predict angles from invariant features
		angles = self.out_angles(z)
		frame_outputs = _frame_outputs_from_coords(coors_out_flat)
		
		return {
			'angles': angles,
			'z': z,
			'coors_out': coors_out if self.return_coors else None,
			'coors_out_flat': coors_out_flat if self.return_coors else None,
			'quat_pred': frame_outputs['quat_pred'],
			'quat_unit_pred': frame_outputs['quat_unit_pred'],
			'trans_pred': frame_outputs['trans_pred'],
			'trans_local_pred': frame_outputs['trans_local_pred'],
			'rotmat_pred': frame_outputs['rotmat_pred'],
			'local_frames_pred': frame_outputs['local_frames_pred'],
			'rt_pred': frame_outputs['rt_pred'],
		}


class struct_transformer_decoder(QuaternionFoldingRefiner):
	"""
	Backward-compatible alias for the standalone folding refiner module.

	Prefer importing `QuaternionFoldingRefiner` from
	`foldtree2.src.folding_refiner` for new training scripts.
	"""
	pass

	