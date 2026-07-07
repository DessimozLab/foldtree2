import torch
torch.set_default_dtype(torch.float64)  # recommended for equivariant network training
import importlib
import torch.nn as nn
import torch.nn.functional as F

from gotennet_pytorch import GotenNet
from foldtree2.src.dynamictan import *
from foldtree2.src.quantizers import *
from foldtree2.src.folding_refiner import QuaternionFoldingRefiner
from foldtree2.src.manifold_hyper_connections import ManifoldHyperConnections
from foldtree2.src.losses.fape import rotation_matrix_to_quaternion

from torch_geometric.nn import TransformerConv, GATConv, GCNConv, global_mean_pool

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
			'trans_steps_pred': None,
			'trans_coords_pred': None,
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
			'trans_steps_pred': None,
			'trans_coords_pred': None,
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
		'trans_steps_pred': None,
		'trans_coords_pred': trans,
		'trans_local_pred': trans_local,
		'rotmat_pred': rot,
		'local_frames_pred': local_frames,
		'rt_pred': rt_pred,
	}


def _sincos_logits_to_angles(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	"""Convert [sin, cos] paired logits into bounded angles in radians."""
	pairs = logits.view(*logits.shape[:-1], 3, 2)
	pairs = pairs / pairs.norm(dim=-1, keepdim=True).clamp_min(eps)
	return torch.atan2(pairs[..., 0], pairs[..., 1])


def _frame_outputs_from_rt_pred(rt_pred):
	"""Compatibility helper to derive frame outputs from [quat(4), trans(3)]."""
	if rt_pred is None:
		return {
			'quat_pred': None,
			'quat_unit_pred': None,
			'trans_pred': None,
			'trans_steps_pred': None,
			'trans_coords_pred': None,
			'trans_local_pred': None,
			'rotmat_pred': None,
			'local_frames_pred': None,
		}

	quat = torch.nan_to_num(rt_pred[..., :4], nan=0.0, posinf=0.0, neginf=0.0)
	quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
	trans = torch.nan_to_num(rt_pred[..., 4:], nan=0.0, posinf=0.0, neginf=0.0)

	w, x, y, z = quat.unbind(dim=-1)
	one = torch.ones_like(w)
	two = one * 2.0

	r00 = one - two * (y * y + z * z)
	r01 = two * (x * y - z * w)
	r02 = two * (x * z + y * w)
	r10 = two * (x * y + z * w)
	r11 = one - two * (x * x + z * z)
	r12 = two * (y * z - x * w)
	r20 = two * (x * z - y * w)
	r21 = two * (y * z + x * w)
	r22 = one - two * (x * x + y * y)

	rotmat = torch.stack([
		torch.stack([r00, r01, r02], dim=-1),
		torch.stack([r10, r11, r12], dim=-1),
		torch.stack([r20, r21, r22], dim=-1),
	], dim=-2)

	trans_local = torch.einsum('...ij,...j->...i', rotmat.transpose(-1, -2), trans)
	local_frames = torch.cat([rotmat, trans_local.unsqueeze(-1)], dim=-1)
	return {
		'quat_pred': quat,
		'quat_unit_pred': quat,
		'trans_pred': trans,
		'trans_steps_pred': trans,
		'trans_coords_pred': None,
		'trans_local_pred': trans_local,
		'rotmat_pred': rotmat,
		'local_frames_pred': local_frames,
	}


class Position_MLP(nn.Module):
	"""Fallback compact position encoder used by legacy decoder paths."""
	def __init__(self, in_channels=256, hidden_channels=None, out_channels=32, dropout=0.01):
		super().__init__()
		if hidden_channels is None:
			hidden_channels = [128, 128, 128]
		self.net = nn.Sequential(
			nn.Linear(in_channels, hidden_channels[0]),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_channels[0], hidden_channels[1]),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_channels[1], out_channels),
		)

	def forward(self, x):
		return self.net(x)


	
	
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


class Transformer_Geometry_Decoder(torch.nn.Module):
	"""
	SE(3)-invariant graph geometry decoder.

	Consumes residue embeddings z and contact probabilities, performs invariant message
	passing on a residue graph, and outputs:
	- updated residue embeddings (z)
	- backbone angles (phi/psi/omega)
	- rigid frames (quat + translation)
	- CA-like coordinates reconstructed by cumulative sum of translations
	"""
	def __init__(
		self,
		in_channels={'res': 10},
		hidden_channels={'res_backbone_res': [20, 20, 20]},
		concat_positions=True,
		nheads=4,
		layers=2,
		RTdecoder_hidden=[128, 64, 32],
		ssdecoder_hidden=[128, 64, 32],
		anglesdecoder_hidden=[128, 64, 32],
		dropout=0.01,
		normalize=True,
		residual=True,
		learn_positions=False,
		output_rt=True,
		output_ss=True,
		output_angles=True,
		**kwargs
	):
		super().__init__()
		L.seed_everything(42)

		self.concat_positions = concat_positions
		self.learn_positions = learn_positions
		self.normalize = normalize
		self.residual = residual
		self.output_rt = output_rt
		self.output_ss = output_ss
		self.output_angles = output_angles
		self.position_feature_dim = 256

		if self.learn_positions:
			# Match mono_decoders behavior: learned position embedding replaces raw concat path.
			self.concat_positions = False
			self.position_mlp = Position_MLP(
				in_channels=self.position_feature_dim,
				hidden_channels=[128, 128, 128],
				out_channels=32,
				dropout=dropout,
			)
		else:
			self.position_mlp = None

		self.contact_threshold = float(kwargs.get('contact_threshold', 0.05))
		self.max_neighbors = int(kwargs.get('max_neighbors', 32))
		self.backbone_layers = max(1, int(kwargs.get('backbone_layers', 1)))
		self.num_layers = max(1, int(layers))
		self.edge_dim = int(kwargs.get('edge_dim', 1))
		self.use_mhc = bool(kwargs.get('use_mhc', False))
		self.mhc_streams = max(1, int(kwargs.get('mhc_streams', 4)))
		self.mhc_sinkhorn_iters = max(1, int(kwargs.get('mhc_sinkhorn_iters', 5)))
		self.mhc_temperature = float(kwargs.get('mhc_temperature', 1.0))
		self.mhc_eps = float(kwargs.get('mhc_eps', 1e-6))
		self.translation_limit = float(kwargs.get('translation_limit', 20.0))
		self.head_kernel_size = max(1, int(kwargs.get('head_kernel_size', 5)))
		if self.head_kernel_size % 2 == 0:
			self.head_kernel_size += 1
		self.head_padding = self.head_kernel_size // 2
		self._transformer_conv_cls = kwargs.get('transformer_conv_cls', None)
		if self._transformer_conv_cls is None:
			try:
				tg_nn = importlib.import_module('torch_geometric.nn')
				self._transformer_conv_cls = getattr(tg_nn, 'TransformerConv')
			except Exception as exc:
				raise RuntimeError(
					'Transformer_Geometry_Decoder requires torch_geometric.nn.TransformerConv. '
					'Install torch-geometric or pass transformer_conv_cls explicitly.'
				) from exc

		if isinstance(in_channels, dict):
			base_input_dim = int(in_channels.get('res', next(iter(in_channels.values()))))
		else:
			base_input_dim = int(in_channels)

		input_dim = base_input_dim

		if self.learn_positions:
			input_dim = input_dim + 32
		elif self.concat_positions:
			input_dim = input_dim + self.position_feature_dim

		default_internal = self._resolve_d_model(hidden_channels, fallback=base_input_dim)
		self.d_model = int(kwargs.get('internal_dim', default_internal))
		print(
			f"Transformer_Geometry_Decoder(SE3Graph): d_model={self.d_model}, "
			f"backbone_layers={self.backbone_layers}, layers={self.num_layers}, contact_threshold={self.contact_threshold}, "
			f"max_neighbors={self.max_neighbors}, translation_limit={self.translation_limit}, "
			f"head_kernel_size={self.head_kernel_size}, use_mhc={self.use_mhc}, "
			f"mhc_streams={self.mhc_streams if self.use_mhc else 1}"
		)

		self.input_dropout = nn.Dropout(dropout)
		self.input_proj = nn.Sequential(
			nn.LayerNorm(input_dim, eps=1e-6),
			nn.Linear(input_dim, self.d_model),
			nn.GELU(),
			nn.Linear(self.d_model, self.d_model),
		)
		self.node_norm = nn.LayerNorm(self.d_model, eps=1e-6)

		self.graph_convs = nn.ModuleList()

		for _ in range(self.num_layers):
			self.graph_convs.append(
				self._transformer_conv_cls(
					in_channels=self.d_model,
					out_channels=self.d_model,
					heads=nheads,
					concat=False,
					dropout=dropout,
					edge_dim=self.edge_dim,
				)
			)

		if self.use_mhc:
			self.mhc = ManifoldHyperConnections(
				d_model=self.d_model,
				num_streams=self.mhc_streams,
				sinkhorn_iters=self.mhc_sinkhorn_iters,
				temperature=self.mhc_temperature,
				eps=self.mhc_eps,
				dropout=dropout,
			)
		else:
			self.mhc = None
			
		angle_hidden = anglesdecoder_hidden[0] if isinstance(anglesdecoder_hidden, list) else int(anglesdecoder_hidden)
		self.angle_head = nn.Sequential(
			nn.Conv1d(self.d_model, angle_hidden, kernel_size=self.head_kernel_size, padding=self.head_padding),
			nn.GELU(),
			nn.Conv1d(angle_hidden, angle_hidden, kernel_size=self.head_kernel_size, padding=self.head_padding),
			nn.GELU(),
			nn.Conv1d(angle_hidden, 3, kernel_size=1),
			nn.Tanh()
		)

		rt_hidden = RTdecoder_hidden[0] if isinstance(RTdecoder_hidden, list) else int(RTdecoder_hidden)
		self.r_head = nn.Sequential(
			nn.Conv1d(self.d_model, self.d_model, kernel_size=self.head_kernel_size, padding=self.head_padding),
			nn.GELU(),
			nn.Conv1d(self.d_model, rt_hidden, kernel_size=self.head_kernel_size, padding=self.head_padding),
			nn.GELU(),
			nn.Conv1d(rt_hidden, rt_hidden, kernel_size=self.head_kernel_size, padding=self.head_padding),
			nn.GELU(),
			nn.Conv1d(rt_hidden, 4, kernel_size=1),
			nn.Tanh()
		)

		self.t_head = nn.Sequential(
			nn.Conv1d(self.d_model, self.d_model, kernel_size=self.head_kernel_size, padding=self.head_padding),
			nn.GELU(),
			nn.Conv1d(self.d_model, rt_hidden, kernel_size=self.head_kernel_size, padding=self.head_padding),
			nn.GELU(),
			nn.Conv1d(rt_hidden, rt_hidden, kernel_size=self.head_kernel_size, padding=self.head_padding),
			nn.GELU(),
			nn.Conv1d(rt_hidden, 3, kernel_size=1),
			nn.Tanh()
		)

	def _resolve_d_model(self, hidden_channels, fallback: int) -> int:
		if isinstance(hidden_channels, dict):
			for key in [('res', 'backbone', 'res'), 'res_backbone_res', 'res']:
				if key in hidden_channels:
					val = hidden_channels[key]
					return int(val[0] if isinstance(val, (list, tuple)) else val)
			first_val = next(iter(hidden_channels.values()))
			return int(first_val[0] if isinstance(first_val, (list, tuple)) else first_val)
		if isinstance(hidden_channels, (list, tuple)) and len(hidden_channels) > 0:
			first = hidden_channels[0]
			return int(first if not isinstance(first, (list, tuple)) else first[0])
		if hidden_channels is None:
			return int(fallback)
		if isinstance(hidden_channels, (int, float)):
			return int(hidden_channels)
		return int(fallback)

	def _get_batch(self, data, kwargs):
		if kwargs.get('batch', None) is not None:
			return kwargs['batch']
		if hasattr(data, '__getitem__'):
			try:
				store = data['res']
				if hasattr(store, 'batch') and store.batch is not None:
					return store.batch
			except Exception:
				pass
		return None

	def _coords_from_translations(self, trans_pred, batch):
		if trans_pred is None:
			return None
		if batch is None:
			coords = torch.cumsum(trans_pred.float(), dim=0)
			return coords.to(dtype=trans_pred.dtype)

		coords = torch.zeros_like(trans_pred)
		for b in torch.unique(batch, sorted=True):
			idx = torch.where(batch == b)[0]
			if idx.numel() == 0:
				continue
			coords[idx] = torch.cumsum(trans_pred[idx].float(), dim=0).to(dtype=coords.dtype)
		return coords

	def _pack_coords_for_batch(self, coords_flat, batch):
		if batch is None:
			return coords_flat.unsqueeze(0)

		num_graphs = int(batch.max().item()) + 1
		coords_by_graph = [coords_flat[batch == i] for i in range(num_graphs)]
		max_len = max(c.shape[0] for c in coords_by_graph) if len(coords_by_graph) > 0 else 0
		padded = []
		for c in coords_by_graph:
			if c.shape[0] < max_len:
				pad = torch.zeros(max_len - c.shape[0], c.shape[1], dtype=c.dtype, device=c.device)
				c = torch.cat([c, pad], dim=0)
			padded.append(c)
		return torch.stack(padded, dim=0)

	def _pack_node_features_for_batch(self, z, batch):
		n_nodes = z.shape[0]
		if batch is None:
			idx = torch.arange(n_nodes, device=z.device, dtype=torch.long)
			z_seq = z.unsqueeze(0)
			mask = torch.ones((1, n_nodes), dtype=torch.bool, device=z.device)
			return z_seq, mask, [idx]

		num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
		idx_by_graph = [torch.where(batch == i)[0] for i in range(num_graphs)]
		max_len = max((idx.numel() for idx in idx_by_graph), default=0)
		z_seq = torch.zeros((num_graphs, max_len, z.shape[1]), dtype=z.dtype, device=z.device)
		mask = torch.zeros((num_graphs, max_len), dtype=torch.bool, device=z.device)

		for i, idx in enumerate(idx_by_graph):
			if idx.numel() == 0:
				continue
			z_seq[i, :idx.numel()] = z[idx]
			mask[i, :idx.numel()] = True

		return z_seq, mask, idx_by_graph

	def _unpack_node_features_from_batch(self, z_seq, idx_by_graph, n_nodes):
		if len(idx_by_graph) == 1 and idx_by_graph[0].numel() == n_nodes:
			return z_seq[0, :n_nodes]

		z_flat = torch.zeros((n_nodes, z_seq.shape[-1]), dtype=z_seq.dtype, device=z_seq.device)
		for i, idx in enumerate(idx_by_graph):
			if idx.numel() == 0:
				continue
			z_flat[idx] = z_seq[i, :idx.numel()]
		return z_flat

	def _build_edge_attr(self, probs: torch.Tensor, dtype: torch.dtype):
		if self.edge_dim <= 1:
			return probs.unsqueeze(-1).to(dtype=dtype)
		zeros = torch.zeros((probs.shape[0], self.edge_dim - 1), device=probs.device, dtype=dtype)
		return torch.cat([probs.unsqueeze(-1).to(dtype=dtype), zeros], dim=-1)

	def forward(self, data, **kwargs):
		z_in = kwargs.get('z', None)
		if z_in is None:
			z_in = data.x_dict['res']
		edge_attr = data['res', 'contact_proba', 'res'].edge_attr
		edge_index = data['res', 'contact_proba', 'res'].edge_index
		edge_attr = edge_attr.clamp(1e-6, 1.0-1e-6)
		
		if self.learn_positions:
			if self.position_mlp is None:
				raise RuntimeError('learn_positions=True requires position_mlp to be initialized')
			pos_enc = self.position_mlp(data['positions']['x'])
			z_in = torch.cat([z_in, pos_enc], dim=-1)

		batch = self._get_batch(data, kwargs)
		z = self.input_dropout(z_in)
		z = self.input_proj(z)
		#z = self.node_norm(z)
		
		# Stage 1: contact-graph message passing.
		# Build undirected graph for transformer convolutions.
		
		if edge_index.numel() > 0:
			if self.use_mhc and self.mhc is not None:
				stream_state = self.mhc.init_streams(z)
				for conv in self.graph_convs:
					z_layer = self.mhc.readout(stream_state, for_block=True)
					h = conv(z_layer, edge_index, edge_attr=edge_attr)
					stream_state = self.mhc.step(stream_state, h, residual=self.residual)
				z = self.mhc.readout(stream_state)
			else:
				for conv in self.graph_convs:
					h = conv(z, edge_index, edge_attr=edge_attr)
					z = z + h if self.residual else z

		if self.normalize:
			z = F.normalize(z, p=2, dim=-1)

		z_seq, _, idx_by_graph = self._pack_node_features_for_batch(z, batch)
		z_seq_ch = z_seq.transpose(1, 2)

		quat_seq = self.r_head(z_seq_ch).transpose(1, 2)
		quat_pred_raw = self._unpack_node_features_from_batch(quat_seq, idx_by_graph, z.shape[0])

		trans_seq_logits = self.t_head(z_seq_ch).transpose(1, 2)
		trans_logits = self._unpack_node_features_from_batch(trans_seq_logits, idx_by_graph, z.shape[0])

		# Centered sigmoid keeps translations bounded to [-translation_limit, translation_limit].
		trans_pred = trans_logits * self.translation_limit
		rt_pred = torch.cat([quat_pred_raw, trans_pred], dim=-1)
		frame_outputs_all = _frame_outputs_from_rt_pred(rt_pred)
		coords = self._coords_from_translations(frame_outputs_all['trans_pred'], batch)
		
		if self.output_angles:
			angles_seq = self.angle_head(z_seq_ch).transpose(1, 2)
			angles_logits = self._unpack_node_features_from_batch(angles_seq, idx_by_graph, z.shape[0])
			angles = angles_logits * torch.pi
		else:
			angles = None

		coords_out = self._pack_coords_for_batch(coords, batch)

		if self.output_rt:
			frame_outputs = {
				'quat_pred': frame_outputs_all['quat_pred'],
				'quat_unit_pred': frame_outputs_all['quat_unit_pred'],
				'trans_pred': frame_outputs_all['trans_pred'],
				'trans_steps_pred': frame_outputs_all['trans_pred'],
				'trans_coords_pred': coords_out,
				'trans_local_pred': frame_outputs_all['trans_local_pred'],
				'rotmat_pred': frame_outputs_all['rotmat_pred'],
				'local_frames_pred': frame_outputs_all['local_frames_pred'],
				'rt_pred': rt_pred,
			}
		else:
			frame_outputs = {
				'quat_pred': None,
				'quat_unit_pred': None,
				'trans_pred': None,
				'trans_steps_pred': None,
				'trans_coords_pred': None,
				'trans_local_pred': None,
				'rotmat_pred': None,
				'local_frames_pred': None,
				'rt_pred': None,
			}

		return {
			'quat_pred': frame_outputs['quat_pred'],
			'quat_unit_pred': frame_outputs['quat_unit_pred'],
			'trans_pred': frame_outputs['trans_pred'],
			'trans_steps_pred': frame_outputs['trans_steps_pred'],
			'trans_coords_pred': frame_outputs['trans_coords_pred'],
			'trans_local_pred': frame_outputs['trans_local_pred'],
			'rotmat_pred': frame_outputs['rotmat_pred'],
			'local_frames_pred': frame_outputs['local_frames_pred'],
			'rt_pred': frame_outputs['rt_pred'],
			'angles': angles,
			'z': z,
			'coords_pred': coords_out,
			'contact_probs': edge_attr,
			'ords': data.x_dict.get('ords', None),
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
		coords = kwargs.get('coords_pred_atoms', None)
		atom_level = coords is not None
		if coords is None:
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
		if coords.ndim == 3:
			atom_level = True
			num_residues, atoms_per_residue, _ = coords.shape
			coords = coords.reshape(-1, 3)
			atom_type_ids = kwargs.get('atom_type_ids', None)
			if atom_type_ids is not None:
				if atom_type_ids.shape != (num_residues, atoms_per_residue):
					raise RuntimeError(
						f"Expected atom_type_ids with shape {(num_residues, atoms_per_residue)}, "
						f"got {tuple(atom_type_ids.shape)}"
					)
				atom_ids = atom_type_ids.to(dtype=torch.long, device=self.device).reshape(-1)
			else:
				atom_ids = atom_ids.view(num_residues, 1).expand(num_residues, atoms_per_residue).reshape(-1)
		else:
			coords = coords.view(-1, 3)  # (num_nodes, 3)
			num_residues = coords.shape[0]
			atoms_per_residue = 1
		
		#use dot product results to add adges
		x_dict['dot_prod'] = edge_attr_dict.get('dot_prod', None) if edge_attr_dict is not None else None

		# Create adjacency matrix from edge_index
		# For GotenNet, we need a dense adjacency matrix
		residue_batch = data['res'].batch if hasattr(data['res'], 'batch') else None
		batch = residue_batch
		if atom_level and residue_batch is not None:
			batch = residue_batch.repeat_interleave(atoms_per_residue)
		
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
		coors_out_atoms = None
		if atom_level and coors_out_flat is not None:
			coors_out_atoms = coors_out_flat.reshape(num_residues, atoms_per_residue, 3)
		
		return {
			'angles': angles,
			'z': z,
			'coors_out': coors_out if self.return_coors else None,
			'coors_out_flat': coors_out_flat if self.return_coors else None,
			'coors_out_atoms': coors_out_atoms if self.return_coors else None,
			'quat_pred': frame_outputs['quat_pred'],
			'quat_unit_pred': frame_outputs['quat_unit_pred'],
			'trans_pred': frame_outputs['trans_pred'],
			'trans_steps_pred': frame_outputs.get('trans_steps_pred', None),
			'trans_coords_pred': frame_outputs.get('trans_coords_pred', frame_outputs['trans_pred']),
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

	
