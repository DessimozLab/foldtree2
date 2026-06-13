import copy
import math
from typing import Optional, Callable, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from foldtree2.src.layers import SwiGLU
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax
from typing import overload, Tuple


def _get_activation_fn(
	activation: Union[str, Callable[[Tensor], Tensor]],
	dim_feedforward: Optional[int] = None,
) -> Callable[[Tensor], Tensor]:
	if callable(activation):
		return activation
	if isinstance(activation, str):
		activation = activation.lower()
	if activation == "relu":
		return F.relu
	if activation == "gelu":
		return F.gelu
	if activation == "swiglu":
		if dim_feedforward is None:
			raise ValueError("dim_feedforward is required when activation='swiglu'")
		return SwiGLU(dim_feedforward)
	raise ValueError(f"Unsupported activation: {activation}")


class ExclusiveSelfAttention(nn.Module):
	"""
	Multi-head Exclusive Self Attention.

	This follows the XSA idea:

		Y = Attention(Q, K, V)
		Vn = normalize(V)
		Z = Y - <Y, Vn> Vn

	where V is the token's own value vector per head.

	Input shape:
		batch_first=True:
			x: [B, T, D]

		batch_first=False:
			x: [T, B, D]

	Output shape:
		same as x
	"""

	def __init__(
		self,
		embed_dim: int,
		num_heads: int,
		dropout: float = 0.0,
		bias: bool = True,
		batch_first: bool = False,
		eps: float = 1e-8,
	):
		super().__init__()

		if embed_dim % num_heads != 0:
			raise ValueError(
				f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
			)

		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads
		self.dropout = dropout
		self.batch_first = batch_first
		self.eps = eps

		self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

	def forward(
		self,
		x: Tensor,
		attn_mask: Optional[Tensor] = None,
		key_padding_mask: Optional[Tensor] = None,
		is_causal: bool = False,
	) -> Tensor:
		"""
		Args:
			x:
				[B, T, D] if batch_first=True, otherwise [T, B, D]

			attn_mask:
				Optional attention mask. Should be compatible with
				torch.nn.functional.scaled_dot_product_attention.

			key_padding_mask:
				Optional bool mask of shape [B, T], where True means masked.

			is_causal:
				Whether to use causal attention.

		Returns:
			Tensor with same shape as x.
		"""

		need_transpose = not self.batch_first
		if need_transpose:
			# [T, B, D] -> [B, T, D]
			x = x.transpose(0, 1)

		B, T, D = x.shape
		H = self.num_heads
		Dh = self.head_dim

		q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
		k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
		v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]

		attn_mask = self._merge_masks(
			attn_mask=attn_mask,
			key_padding_mask=key_padding_mask,
			batch_size=B,
			num_heads=H,
			tgt_len=T,
			src_len=T,
			device=x.device,
		)

		y = F.scaled_dot_product_attention(
			q,
			k,
			v,
			attn_mask=attn_mask,
			dropout_p=self.dropout if self.training else 0.0,
			is_causal=is_causal,
		)

		# XSA projection removal.
		#
		# Remove the component of the attention output y_i that lies along
		# the self value vector v_i.
		#
		# y, v: [B, H, T, Dh]
		v_norm = v / v.norm(dim=-1, keepdim=True).clamp_min(self.eps)
		z = y - (y * v_norm).sum(dim=-1, keepdim=True) * v_norm

		z = z.transpose(1, 2).contiguous().view(B, T, D)
		out = self.out_proj(z)

		if need_transpose:
			# [B, T, D] -> [T, B, D]
			out = out.transpose(0, 1)

		return out

	@staticmethod
	def _merge_masks(
		attn_mask: Optional[Tensor],
		key_padding_mask: Optional[Tensor],
		batch_size: int,
		num_heads: int,
		tgt_len: int,
		src_len: int,
		device: torch.device,
	) -> Optional[Tensor]:
		"""
		Converts key_padding_mask into an additive attention mask compatible with SDPA.

		SDPA accepts:
			bool mask: True means keep
			float mask: additive, usually 0 or -inf

		To avoid ambiguity, we construct a float additive mask.
		"""

		final_mask = None

		if attn_mask is not None:
			if attn_mask.dtype == torch.bool:
				# PyTorch SDPA bool mask uses True = allowed.
				# nn.Transformer-style bool masks often use True = blocked.
				# To avoid silently doing the wrong thing, convert assuming
				# Transformer convention: True means masked.
				attn_mask = attn_mask.masked_fill(attn_mask, float("-inf"))
				attn_mask = attn_mask.masked_fill(~attn_mask.bool(), 0.0)

			final_mask = attn_mask.to(device)

		if key_padding_mask is not None:
			# key_padding_mask: [B, S], True means ignore key.
			padding_mask = torch.zeros(
				batch_size,
				1,
				1,
				src_len,
				device=device,
				dtype=torch.float32,
			)
			padding_mask = padding_mask.masked_fill(
				key_padding_mask[:, None, None, :].to(device),
				float("-inf"),
			)

			final_mask = padding_mask if final_mask is None else final_mask + padding_mask

		return final_mask

class XSATransformerEncoderLayer(nn.Module):
	"""
	Drop-in-ish XSA variant of torch.nn.TransformerEncoderLayer.

	Replaces standard MultiheadAttention with ExclusiveSelfAttention.

	Supports:
		- norm_first=True / False
		- batch_first=True / False
		- src_mask
		- src_key_padding_mask
		- causal attention
		- activation="swiglu"
	"""

	def __init__(
		self,
		d_model: int,
		nhead: int,
		dim_feedforward: int = 2048,
		dropout: float = 0.1,
		activation: Union[str, Callable[[Tensor], Tensor]] = "swiglu",
		layer_norm_eps: float = 1e-5,
		batch_first: bool = False,
		norm_first: bool = False,
		bias: bool = True,
		eps: float = 1e-8,
	):
		super().__init__()

		self.self_attn = ExclusiveSelfAttention(
			embed_dim=d_model,
			num_heads=nhead,
			dropout=dropout,
			bias=bias,
			batch_first=batch_first,
			eps=eps,
		)

		self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

		self.norm_first = norm_first
		self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
		self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)

		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = _get_activation_fn(activation, dim_feedforward)
	
	def forward(
		self,
		src: Tensor,
		src_mask: Optional[Tensor] = None,
		src_key_padding_mask: Optional[Tensor] = None,
		is_causal: bool = False,
		**kwargs,
	) -> Tensor:
		x = src

		if self.norm_first:
			x = x + self._xsa_block(
				self.norm1(x),
				src_mask,
				src_key_padding_mask,
				is_causal,
			)
			x = x + self._ff_block(self.norm2(x))
		else:
			x = self.norm1(
				x + self._xsa_block(
					x,
					src_mask,
					src_key_padding_mask,
					is_causal,
				)
			)
			x = self.norm2(x + self._ff_block(x))

		return x

	def _xsa_block(
		self,
		x: Tensor,
		attn_mask: Optional[Tensor],
		key_padding_mask: Optional[Tensor],
		is_causal: bool,
	) -> Tensor:
		x = self.self_attn(
			x,
			attn_mask=attn_mask,
			key_padding_mask=key_padding_mask,
			is_causal=is_causal,
		)
		return self.dropout1(x)

	def _ff_block(self, x: Tensor) -> Tensor:
		x = self.linear2(self.dropout(self.activation(self.linear1(x))))
		return self.dropout2(x)
	

class TransformerConv(MessagePassing):
	r"""Local graph transformer convolution used as the base for XSA variants."""

	_alpha: OptTensor

	def __init__(
		self,
		in_channels: Union[int, Tuple[int, int]],
		out_channels: int,
		heads: int = 1,
		concat: bool = True,
		beta: bool = False,
		dropout: float = 0.,
		edge_dim: Optional[int] = None,
		bias: bool = True,
		root_weight: bool = True,
		**kwargs,
	):
		kwargs.setdefault('aggr', 'add')
		super().__init__(node_dim=0, **kwargs)

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.heads = heads
		self.beta = beta and root_weight
		self.root_weight = root_weight
		self.concat = concat
		self.dropout = dropout
		self.edge_dim = edge_dim
		self._alpha = None

		if isinstance(in_channels, int):
			in_channels = (in_channels, in_channels)

		self.lin_key = Linear(in_channels[0], heads * out_channels)
		self.lin_query = Linear(in_channels[1], heads * out_channels)
		self.lin_value = Linear(in_channels[0], heads * out_channels)
		if edge_dim is not None:
			self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
		else:
			self.lin_edge = self.register_parameter('lin_edge', None)

		if concat:
			self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
			if self.beta:
				self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
			else:
				self.lin_beta = self.register_parameter('lin_beta', None)
		else:
			self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
			if self.beta:
				self.lin_beta = Linear(3 * out_channels, 1, bias=False)
			else:
				self.lin_beta = self.register_parameter('lin_beta', None)

		self.reset_parameters()

	def reset_parameters(self):
		super().reset_parameters()
		self.lin_key.reset_parameters()
		self.lin_query.reset_parameters()
		self.lin_value.reset_parameters()
		if self.lin_edge is not None:
			self.lin_edge.reset_parameters()
		self.lin_skip.reset_parameters()
		if self.lin_beta is not None:
			self.lin_beta.reset_parameters()

	def forward(
		self,
		x: Union[Tensor, PairTensor],
		edge_index: Adj,
		edge_attr: OptTensor = None,
		return_attention_weights: Optional[bool] = None,
	) -> Union[
		Tensor,
		Tuple[Tensor, Tuple[Tensor, Tensor]],
		Tuple[Tensor, SparseTensor],
	]:
		H, C = self.heads, self.out_channels

		if isinstance(x, Tensor):
			x = (x, x)

		query = self.lin_query(x[1]).view(-1, H, C)
		key = self.lin_key(x[0]).view(-1, H, C)
		value = self.lin_value(x[0]).view(-1, H, C)
		out = self.propagate(edge_index, query=query, key=key, value=value,
							 edge_attr=edge_attr)

		alpha = self._alpha
		self._alpha = None

		if self.concat:
			out = out.view(-1, self.heads * self.out_channels)
		else:
			out = out.mean(dim=1)

		if self.root_weight:
			x_r = self.lin_skip(x[1])
			if self.lin_beta is not None:
				beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
				beta = beta.sigmoid()
				out = beta * x_r + (1 - beta) * out
			else:
				out = out + x_r

		if isinstance(return_attention_weights, bool):
			assert alpha is not None
			if isinstance(edge_index, Tensor):
				return out, (edge_index, alpha)
			if isinstance(edge_index, SparseTensor):
				return out, edge_index.set_value(alpha, layout='coo')

		return out

	def message(
		self,
		query_i: Tensor,
		key_j: Tensor,
		value_j: Tensor,
		edge_attr: OptTensor,
		index: Tensor,
		ptr: OptTensor,
		size_i: Optional[int],
	) -> Tensor:
		if self.lin_edge is not None:
			assert edge_attr is not None
			edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
			key_j = key_j + edge_attr

		alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
		alpha = softmax(alpha, index, ptr, size_i)
		self._alpha = alpha
		alpha = F.dropout(alpha, p=self.dropout, training=self.training)

		out = value_j
		if edge_attr is not None:
			out = out + edge_attr

		return out * alpha.view(-1, self.heads, 1)

	def __repr__(self) -> str:
		return (f'{self.__class__.__name__}({self.in_channels}, '
				f'{self.out_channels}, heads={self.heads})')


class XSATransformerConv(MessagePassing):
	r"""XSA variant of :class:`TransformerConv`.

	This layer computes graph transformer attention as usual, then removes the
	component of the aggregated message that lies along each target node's own
	projected value vector before skip/root fusion.
	"""

	_alpha: OptTensor

	def __init__(
		self,
		in_channels: Union[int, Tuple[int, int]],
		out_channels: int,
		heads: int = 1,
		concat: bool = True,
		beta: bool = False,
		dropout: float = 0.,
		edge_dim: Optional[int] = None,
		bias: bool = True,
		root_weight: bool = True,
		eps: float = 1e-8,
		**kwargs,
	):
		kwargs.setdefault('aggr', 'add')
		super().__init__(node_dim=0, **kwargs)

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.heads = heads
		self.beta = beta and root_weight
		self.root_weight = root_weight
		self.concat = concat
		self.dropout = dropout
		self.edge_dim = edge_dim
		self.eps = eps
		self._alpha = None

		if isinstance(in_channels, int):
			in_channels = (in_channels, in_channels)

		self.lin_key = Linear(in_channels[0], heads * out_channels)
		self.lin_query = Linear(in_channels[1], heads * out_channels)
		self.lin_value = Linear(in_channels[0], heads * out_channels)
		self.lin_self_value = Linear(in_channels[1], heads * out_channels)
		if edge_dim is not None:
			self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
		else:
			self.lin_edge = self.register_parameter('lin_edge', None)

		if concat:
			self.lin_skip = Linear(in_channels[1], heads * out_channels,
								   bias=bias)
			if self.beta:
				self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
			else:
				self.lin_beta = self.register_parameter('lin_beta', None)
		else:
			self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
			if self.beta:
				self.lin_beta = Linear(3 * out_channels, 1, bias=False)
			else:
				self.lin_beta = self.register_parameter('lin_beta', None)

		self.reset_parameters()

	def reset_parameters(self):
		super().reset_parameters()
		self.lin_key.reset_parameters()
		self.lin_query.reset_parameters()
		self.lin_value.reset_parameters()
		self.lin_self_value.reset_parameters()
		if self.lin_edge is not None:
			self.lin_edge.reset_parameters()
		self.lin_skip.reset_parameters()
		if self.lin_beta is not None:
			self.lin_beta.reset_parameters()

	def forward(
		self,
		x: Union[Tensor, PairTensor],
		edge_index: Adj,
		edge_attr: OptTensor = None,
		return_attention_weights: Optional[bool] = None,
	) -> Union[
			Tensor,
			Tuple[Tensor, Tuple[Tensor, Tensor]],
			Tuple[Tensor, SparseTensor],
	]:
		H, C = self.heads, self.out_channels

		if isinstance(x, Tensor):
			x = (x, x)

		query = self.lin_query(x[1]).view(-1, H, C)
		key = self.lin_key(x[0]).view(-1, H, C)
		value = self.lin_value(x[0]).view(-1, H, C)
		self_value = self.lin_self_value(x[1]).view(-1, H, C)

		out = self.propagate(edge_index, query=query, key=key, value=value,
							 edge_attr=edge_attr)

		alpha = self._alpha
		self._alpha = None

		self_value_norm = self_value / self_value.norm(
			dim=-1,
			keepdim=True,
		).clamp_min(self.eps)
		out = out - (out * self_value_norm).sum(
			dim=-1,
			keepdim=True,
		) * self_value_norm

		if self.concat:
			out = out.view(-1, self.heads * self.out_channels)
		else:
			out = out.mean(dim=1)

		if self.root_weight:
			x_r = self.lin_skip(x[1])
			if self.lin_beta is not None:
				beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
				beta = beta.sigmoid()
				out = beta * x_r + (1 - beta) * out
			else:
				out = out + x_r

		if isinstance(return_attention_weights, bool):
			assert alpha is not None
			if isinstance(edge_index, Tensor):
				return out, (edge_index, alpha)
			if isinstance(edge_index, SparseTensor):
				return out, edge_index.set_value(alpha, layout='coo')

		return out

	def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
				edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
				size_i: Optional[int]) -> Tensor:

		if self.lin_edge is not None:
			assert edge_attr is not None
			edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
													  self.out_channels)
			key_j = key_j + edge_attr

		alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
		alpha = softmax(alpha, index, ptr, size_i)
		self._alpha = alpha
		alpha = F.dropout(alpha, p=self.dropout, training=self.training)

		out = value_j
		if edge_attr is not None:
			out = out + edge_attr

		out = out * alpha.view(-1, self.heads, 1)
		return out

	def __repr__(self) -> str:
		return (f'{self.__class__.__name__}({self.in_channels}, '
				f'{self.out_channels}, heads={self.heads})')
