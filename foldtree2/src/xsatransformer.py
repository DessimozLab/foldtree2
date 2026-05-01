import copy
from typing import Optional, Callable, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

def _get_activation_fn(
	activation: Union[str, Callable[[Tensor], Tensor]]
) -> Callable[[Tensor], Tensor]:
	if callable(activation):
		return activation
	if activation == "relu":
		return F.relu
	if activation == "gelu":
		return F.gelu
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
	"""

	def __init__(
		self,
		d_model: int,
		nhead: int,
		dim_feedforward: int = 2048,
		dropout: float = 0.1,
		activation: Union[str, Callable[[Tensor], Tensor]] = "relu",
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

		self.activation = _get_activation_fn(activation)
	
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