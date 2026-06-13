import torch
import torch.nn as nn
import torch.nn.functional as F


class ManifoldHyperConnections(nn.Module):
	"""mHC-style multi-stream residual mixer with Sinkhorn-constrained mixing.

	The module is shape-agnostic for tensors whose last dimension is feature size.
	Examples:
	- [N, D] node features
	- [S, B, D] sequence features
	"""
	def __init__(
		self,
		d_model: int,
		num_streams: int = 4,
		sinkhorn_iters: int = 5,
		temperature: float = 1.0,
		eps: float = 1e-6,
		dropout: float = 0.0,
	):
		super().__init__()
		self.d_model = int(d_model)
		self.num_streams = max(1, int(num_streams))
		self.sinkhorn_iters = max(1, int(sinkhorn_iters))
		self.temperature = float(temperature)
		self.eps = float(eps)

		self.pre_mix_logits = nn.Parameter(torch.zeros(self.num_streams, self.num_streams))
		self.post_mix_logits = nn.Parameter(torch.zeros(self.num_streams, self.num_streams))
		self.to_block_logits = nn.Parameter(torch.zeros(self.num_streams))
		self.to_stream_logits = nn.Parameter(torch.zeros(self.num_streams))
		self.readout_logits = nn.Parameter(torch.zeros(self.num_streams))
		self.stream_offsets = nn.Parameter(torch.zeros(self.num_streams, self.d_model))
		self.delta_dropout = nn.Dropout(dropout)

	def _sinkhorn(self, logits: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
		scaled = logits / max(self.temperature, self.eps)
		mat = torch.exp(scaled).clamp_min(self.eps).to(dtype=dtype, device=device)
		for _ in range(self.sinkhorn_iters):
			mat = mat / mat.sum(dim=-1, keepdim=True).clamp_min(self.eps)
			mat = mat / mat.sum(dim=-2, keepdim=True).clamp_min(self.eps)
		return mat

	def init_streams(self, z: torch.Tensor) -> torch.Tensor:
		if self.num_streams == 1:
			return z.unsqueeze(0)

		base = z.unsqueeze(0).expand(self.num_streams, *z.shape)
		offset_shape = [self.num_streams] + [1] * (z.ndim - 1) + [self.d_model]
		offsets = torch.tanh(self.stream_offsets).view(*offset_shape).to(dtype=z.dtype, device=z.device)
		streams = base + offsets
		return F.normalize(streams, p=2, dim=-1, eps=self.eps)

	def readout(self, streams: torch.Tensor, for_block: bool = False) -> torch.Tensor:
		logits = self.to_block_logits if for_block else self.readout_logits
		weights = F.softmax(logits, dim=0).to(dtype=streams.dtype, device=streams.device)
		reduce_expr = 's,s...d->...d'
		return torch.einsum(reduce_expr, weights, streams)

	def step(self, streams: torch.Tensor, layer_delta: torch.Tensor, residual: bool = True) -> torch.Tensor:
		pre_mix = self._sinkhorn(self.pre_mix_logits, streams.dtype, streams.device)
		post_mix = self._sinkhorn(self.post_mix_logits, streams.dtype, streams.device)

		mixed_streams = torch.einsum('ij,j...d->i...d', pre_mix, streams)

		delta = F.normalize(layer_delta, p=2, dim=-1, eps=self.eps)
		delta = self.delta_dropout(delta)
		delta_weights = F.softmax(self.to_stream_logits, dim=0).to(dtype=streams.dtype, device=streams.device)
		# Match delta.unsqueeze(0) rank: [1, ...delta.shape...]
		broadcast_shape = [self.num_streams] + [1] * delta.ndim
		delta_streams = delta.unsqueeze(0) * delta_weights.view(*broadcast_shape)

		updated = mixed_streams + delta_streams if residual else delta_streams
		updated = torch.einsum('ij,j...d->i...d', post_mix, updated)
		return F.normalize(updated, p=2, dim=-1, eps=self.eps)
