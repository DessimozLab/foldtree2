"""Axial attention building blocks for ft2 sequence reconstruction.

This module is a port/adaptation of the MSA Transformer's axial attention
components (https://github.com/rmrao/msa-transformer) by Roshan Rao, used
under the original license.

The MSA Transformer processes a multiple sequence alignment (MSA) laid out as
a 2D grid of tokens with shape ``(num_rows, num_cols, batch_size, embed_dim)``
where ``num_rows`` indexes the aligned sequences and ``num_cols`` indexes the
residue positions within each sequence. Two complementary flavours of
self-attention operate *axially* over this grid:

* ``RowSelfAttention``    -- different sequences (rows) attend to each other at
  the same residue position (column). Attention weights have shape
  ``(num_heads, batch_size, num_cols_i, num_cols_j)`` because for each column
  position we compute a similarity between every pair of rows and then sum the
  per-position scores across columns before the softmax. This is why the
  original code calls the layout ``attn_shape = "hnij"``.

* ``ColumnSelfAttention`` -- different residue positions (columns) attend to
  each other within the same sequence (row). Attention weights have shape
  ``(num_heads, num_cols, batch_size, num_rows_i, num_rows_j)`` and use the
  ``"icnhd,jcnhd->hcnij"`` einsum.

``AxialTransformerLayer`` wires a row-attention block, a column-attention
block, and a GELU feed-forward network together as Pre-LN
(``NormalizedResidualBlock``) residual stages, matching the original MSA
Transformer layer used for protein MSAs.

Key adaptations from the original for FoldTree2
------------------------------------------------
* No dependency on the ``evo`` package -- only ``torch`` / ``torch.nn`` are
  used.
* ``self_attn_padding_mask`` is documented as shape ``[B, R]`` (batch,
  rows), consistent with the rest of the reconstruction module which masks
  padded sequences in the alignment.
* The row-attention scaling ``self.scaling / sqrt(num_rows)`` is preserved
  verbatim because it is what makes the summed-over-columns attention
  numerically well-behaved; column attention uses the plain
  ``self.scaling``.
* ``max_tokens_per_msa`` controls a chunked/batched forward used at
  inference time (when ``torch.is_grad_enabled()`` is ``False``) to bound
  peak memory.
"""

from typing import Optional, Tuple

import math

import torch
import torch.nn as nn


__all__ = [
    "RowSelfAttention",
    "ColumnSelfAttention",
    "AxialTransformerLayer",
    "NormalizedResidualBlock",
    "FeedForwardNetwork",
]


class RowSelfAttention(nn.Module):
    """Self-attention over the *rows* of a 2D MSA layout.

    The input tensor ``x`` has shape ``(num_rows, num_cols, batch_size,
    embed_dim)`` where ``num_rows`` indexes the aligned sequences and
    ``num_cols`` indexes the residue positions within a sequence. Row
    self-attention lets each sequence attend to every other sequence at the
    same residue position. Because the per-position query/key dot products
    are summed across columns before the softmax, the attention weight tensor
    has shape ``(num_heads, batch_size, num_cols_i, num_cols_j)`` and is
    emitted with the einsum label layout ``"hnij"`` (h = heads, n = batch, i/j
    = column indices). The query is scaled by
    ``self.scaling / sqrt(num_rows)`` so the summed scores keep a comparable
    magnitude regardless of alignment depth.

    Parameters
    ----------
    embed_dim : int
        Total width of the input embedding. Must be divisible by
        ``num_heads``.
    num_heads : int
        Number of attention heads. ``head_dim = embed_dim // num_heads``.
    dropout : float, optional
        Dropout probability applied to the attention probabilities.
    max_tokens_per_msa : int, optional
        Soft cap on ``num_rows * num_cols`` above which the forward pass is
        chunked at inference time (i.e. when gradients are not being
        computed). Used to bound peak memory on large alignments.

    Attributes
    ----------
    attn_shape : str
        Always ``"hnij"`` -- the einsum output layout for row attention
        weights, preserved from the original MSA Transformer.
    scaling : float
        Base ``head_dim ** -0.5`` scale factor. The effective query scale
        used in :meth:`align_scaling` is ``self.scaling / sqrt(num_rows)``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_tokens_per_msa: int = 2 ** 16,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa
        # Preserved verbatim from the original MSA Transformer: the einsum
        # output layout for summed-over-rows row attention weights.
        self.attn_shape = "hnij"

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, q: torch.Tensor) -> float:
        """Return the effective query scale ``self.scaling / sqrt(num_rows)``.

        The number of rows is taken from the first dimension of ``q``. The
        ``1 / sqrt(num_rows)`` factor compensates for the fact that row
        attention sums query/key dot products across all residue positions
        before the softmax, so deeper alignments would otherwise produce
        scores with growing magnitude.
        """
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def compute_attention_weights(
        self,
        x: torch.Tensor,
        scaling: float,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute raw (pre-softmax) row-attention weights.

        Parameters
        ----------
        x : torch.Tensor
            Slice of the input MSA tensor with shape
            ``(num_rows, num_cols, batch_size, embed_dim)``.
        scaling : float
            Effective query scale (output of :meth:`align_scaling`).
        self_attn_mask : torch.Tensor, optional
            Optional additive attention mask. Currently unsupported -- the
            original raises ``NotImplementedError`` and so do we.
        self_attn_padding_mask : torch.Tensor, optional
            Boolean tensor of shape ``(batch_size, num_rows)`` where ``True``
            marks padded (masked) rows that should not contribute attention
            mass.

        Returns
        -------
        torch.Tensor
            Attention weights of shape ``(num_heads, batch_size,
            num_cols_i, num_cols_j)`` (the ``"hnij"`` layout).
        """
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        q = q * scaling
        if self_attn_padding_mask is not None:
            # ``self_attn_padding_mask`` has shape [B, R] with ``True`` marking
            # padded rows (sequences). Row attention sums query/key dot
            # products across the alignment axis, so padded sequences must be
            # forced to contribute zero query AND key energy -- otherwise they
            # would skew the residue-residue scores. We zero both q and k for
            # padded rows; the resulting summed scores then naturally exclude
            # those rows and no separate ``masked_fill`` on the [H, B, C, C]
            # weight tensor is required.
            row_keep = (~self_attn_padding_mask).to(q)  # [B, R], 1 = keep
            row_keep = row_keep.permute(1, 0).view(
                num_rows, 1, batch_size, 1, 1
            )  # [R, 1, B, 1, 1]
            q = q * row_keep
            k = k * row_keep

        attn_weights = torch.einsum(
            f"rinhd,rjnhd->{self.attn_shape}", q, k
        )

        if self_attn_mask is not None:
            raise NotImplementedError(
                "RowSelfAttention does not yet support an additive "
                "self_attn_mask."
            )

        return attn_weights

    def compute_attention_update(
        self,
        x: torch.Tensor,
        attn_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention probabilities to the values and project out.

        Parameters
        ----------
        x : torch.Tensor
            Input slice with shape ``(num_rows, num_cols, batch_size,
            embed_dim)``.
        attn_probs : torch.Tensor
            Softmaxed attention probabilities with the ``"hnij"`` layout.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as ``x``.
        """
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        context = torch.einsum(
            f"{self.attn_shape},rjnhd->rinhd", attn_probs, v
        )
        context = context.contiguous().view(
            num_rows, num_cols, batch_size, embed_dim
        )
        output = self.out_proj(context)
        return output

    def _batched_forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunked inference-time forward.

        Because row attention sums scores across the column axis, we
        accumulate the raw attention weights in float (preserving additivity)
        for each chunk of rows, run a single global softmax, and then apply
        the probabilities to the values in chunks.
        """
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        attns = 0
        scaling = self.align_scaling(x)
        for start in range(0, num_rows, max_rows):
            attn_weights = self.compute_attention_weights(
                x[start : start + max_rows],
                scaling,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=(
                    self_attn_padding_mask[:, start : start + max_rows]
                    if self_attn_padding_mask is not None
                    else None
                ),
            )
            attns = attns + attn_weights
        attn_probs = attns.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)

        outputs = []
        for start in range(0, num_rows, max_rows):
            output = self.compute_attention_update(
                x[start : start + max_rows], attn_probs
            )
            outputs.append(output)

        output = torch.cat(outputs, 0)
        return output, attn_probs

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run row self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input MSA tensor of shape ``(num_rows, num_cols, batch_size,
            embed_dim)``.
        self_attn_mask : torch.Tensor, optional
            Additive attention mask (currently unsupported).
        self_attn_padding_mask : torch.Tensor, optional
            Boolean mask of shape ``(batch_size, num_rows)``; ``True`` marks
            padded rows.

        Returns
        -------
        output : torch.Tensor
            Tensor with the same shape as ``x``.
        attn_probs : torch.Tensor
            Attention probabilities with the ``"hnij"`` layout, useful for
            contact prediction / interpretability.
        """
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if (
            num_rows * num_cols > self.max_tokens_per_msa
        ) and not torch.is_grad_enabled():
            return self._batched_forward(x, self_attn_mask, self_attn_padding_mask)
        scaling = self.align_scaling(x)
        attn_weights = self.compute_attention_weights(
            x, scaling, self_attn_mask, self_attn_padding_mask
        )
        attn_probs = attn_weights.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)
        output = self.compute_attention_update(x, attn_probs)
        return output, attn_probs


class ColumnSelfAttention(nn.Module):
    """Self-attention over the *columns* of a 2D MSA layout.

    The input tensor ``x`` has shape ``(num_rows, num_cols, batch_size,
    embed_dim)``. Column self-attention lets each residue position attend to
    every other residue position within the same sequence (row). The
    attention weight tensor has shape ``(num_heads, num_cols, batch_size,
    num_rows_i, num_rows_j)`` and uses the ``"icnhd,jcnhd->hcnij"`` einsum.
    Unlike :class:`RowSelfAttention`, the query is scaled by the plain
    ``self.scaling`` -- there is no summation across an alignment axis to
    compensate for.

    Parameters
    ----------
    embed_dim : int
        Total width of the input embedding. Must be divisible by
        ``num_heads``.
    num_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability applied to the attention probabilities.
    max_tokens_per_msa : int, optional
        Soft cap on ``num_rows * num_cols`` above which the forward pass is
        chunked along the column axis at inference time.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_tokens_per_msa: int = 2 ** 16,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def _batched_forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunked inference-time forward along the column axis."""
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_cols = max(1, self.max_tokens_per_msa // num_rows)
        outputs = []
        attns = []
        for start in range(0, num_cols, max_cols):
            # The padding mask is per-row [B, R] and is independent of which
            # column slice we process, so it is forwarded unchanged to every
            # chunk.
            output, attn = self(
                x[:, start : start + max_cols],
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
            outputs.append(output)
            attns.append(attn)
        output = torch.cat(outputs, 1)
        attns = torch.cat(attns, 1)
        return output, attns

    def compute_attention_update(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the column-attention output and probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(num_rows, num_cols, batch_size,
            embed_dim)``.
        self_attn_mask : torch.Tensor, optional
            Additive attention mask (currently unsupported).
        self_attn_padding_mask : torch.Tensor, optional
            Boolean mask of shape ``(batch_size, num_rows)``; ``True`` marks
            padded rows. It is applied so that padded sequences never receive
            attention mass.

        Returns
        -------
        output : torch.Tensor
            Tensor with the same shape as ``x``.
        attn_probs : torch.Tensor
            Attention probabilities of shape ``(num_heads, num_cols,
            batch_size, num_rows_i, num_rows_j)``.
        """
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if num_rows == 1:
            # With a single row the attention matrix is trivially all ones;
            # this also avoids a degenerate softmax over a fully-masked row
            # when the only row is padded.
            attn_probs = torch.ones(
                self.num_heads,
                num_cols,
                batch_size,
                num_rows,
                num_rows,
                device=x.device,
                dtype=x.dtype,
            )
            output = self.out_proj(self.v_proj(x))
        else:
            q = self.q_proj(x).view(
                num_rows, num_cols, batch_size, self.num_heads, self.head_dim
            )
            k = self.k_proj(x).view(
                num_rows, num_cols, batch_size, self.num_heads, self.head_dim
            )
            v = self.v_proj(x).view(
                num_rows, num_cols, batch_size, self.num_heads, self.head_dim
            )
            q = q * self.scaling

            attn_weights = torch.einsum("icnhd,jcnhd->hcnij", q, k)

            if self_attn_mask is not None:
                raise NotImplementedError(
                    "ColumnSelfAttention does not yet support an additive "
                    "self_attn_mask."
                )
            if self_attn_padding_mask is not None:
                # ``self_attn_padding_mask`` is [B, R] with ``True`` marking
                # padded rows (sequences). Column attention is
                # sequence-sequence, so padded sequences must never receive
                # attention mass as keys. We broadcast [B, R] to the
                # [H, C, B, R_i, R_j] weight tensor along the key axis (R_j).
                key_mask = (
                    self_attn_padding_mask.unsqueeze(0).unsqueeze(0).unsqueeze(3)
                )  # [1, 1, B, 1, R]
                attn_weights = attn_weights.masked_fill(key_mask, -10000)

            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            context = torch.einsum("hcnij,jcnhd->icnhd", attn_probs, v)
            context = context.contiguous().view(
                num_rows, num_cols, batch_size, embed_dim
            )
            output = self.out_proj(context)
        return output, attn_probs

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run column self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input MSA tensor of shape ``(num_rows, num_cols, batch_size,
            embed_dim)``.
        self_attn_mask : torch.Tensor, optional
            Additive attention mask (currently unsupported).
        self_attn_padding_mask : torch.Tensor, optional
            Boolean mask of shape ``(batch_size, num_rows)``; ``True`` marks
            padded rows.

        Returns
        -------
        output : torch.Tensor
            Tensor with the same shape as ``x``.
        attn_probs : torch.Tensor
            Attention probabilities of shape ``(num_heads, num_cols,
            batch_size, num_rows_i, num_rows_j)``.
        """
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if (
            num_rows * num_cols
        ) > self.max_tokens_per_msa and not torch.is_grad_enabled():
            return self._batched_forward(
                x,
                self_attn_mask,
                self_attn_padding_mask,
            )
        return self.compute_attention_update(
            x, self_attn_mask, self_attn_padding_mask
        )


class NormalizedResidualBlock(nn.Module):
    """Pre-LayerNorm residual wrapper around a sublayer.

    The block applies ``LayerNorm`` to the input, runs the wrapped sublayer,
    drops out the sublayer output, and adds the result to the original
    (un-normalised) residual. This matches the Pre-LN convention used by the
    MSA Transformer.

    Parameters
    ----------
    layer : nn.Module
        The wrapped sublayer (e.g. a :class:`RowSelfAttention`,
        :class:`ColumnSelfAttention`, or :class:`FeedForwardNetwork`).
    embedding_dim : int
        Width of the input features, used to size the LayerNorm.
    dropout : float, optional
        Dropout applied to the sublayer output before the residual add.
    """

    def __init__(
        self,
        layer: nn.Module,
        embedding_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layer = layer
        self.dropout_module = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """Apply the Pre-LN residual block.

        The wrapped sublayer may return either a single tensor or a
        ``(output, *extras)`` tuple. Extras (such as attention probabilities)
        are forwarded transparently so callers still receive them.
        """
        residual = x
        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        x = self.dropout_module(x)
        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        return x


class FeedForwardNetwork(nn.Module):
    """Simple GELU-activated position-wise feed-forward network.

    Parameters
    ----------
    embedding_dim : int
        Input/output width.
    ffn_embedding_dim : int
        Inner (hidden) width of the MLP.
    activation_dropout : float, optional
        Dropout applied between the two linear layers.
    max_tokens_per_msa : int, optional
        Retained for API symmetry with the attention modules; the FFN itself
        is not chunked but the value is exposed for callers that mirror the
        attention modules' configuration.
    """

    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2 ** 14,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.max_tokens_per_msa = max_tokens_per_msa
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the FFN: ``fc2(GELU(fc1(x)))`` with activation dropout."""
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x


class AxialTransformerLayer(nn.Module):
    """One axial MSA Transformer block.

    The block is the composition, in order, of

    1. row self-attention (sequences attend to sequences at the same
       residue position),
    2. column self-attention (residue positions attend to residue
       positions within a sequence), and
    3. a GELU feed-forward network,

    each wrapped in a Pre-LN :class:`NormalizedResidualBlock` residual stage.
    This mirrors the original MSA Transformer's ``AxialTransformerLayer``.

    Parameters
    ----------
    embedding_dim : int, optional
        Width of the token embeddings.
    ffn_embedding_dim : int, optional
        Inner width of the feed-forward network.
    num_attention_heads : int, optional
        Number of attention heads for both row and column attention.
    dropout : float, optional
        Residual-stage dropout.
    attention_dropout : float, optional
        Forwarded as the ``dropout`` argument of the row and column
        attention modules (dropout on the attention probabilities).
    activation_dropout : float, optional
        Dropout inside the feed-forward network.
    max_tokens_per_msa : int, optional
        Soft cap controlling chunked inference-time forward for the
        attention modules.

    Forward signature
    -----------------
    ``forward(x, self_attn_mask=None, self_attn_padding_mask=None,
    need_head_weights=False)``

    * ``x`` has shape ``(num_rows, num_cols, batch_size, embed_dim)``.
    * ``self_attn_padding_mask`` has shape ``(batch_size, num_rows)`` with
      ``True`` marking padded rows.
    * When ``need_head_weights`` is ``False`` the layer returns just ``x``;
      when it is ``True`` it returns ``(x, column_attn, row_attn)`` so the
      column and row attention probabilities can be used downstream (e.g.
      for contact prediction).
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2 ** 14,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout

        row_self_attention = RowSelfAttention(
            embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )
        column_self_attention = ColumnSelfAttention(
            embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )
        feed_forward_layer = FeedForwardNetwork(
            embedding_dim,
            ffn_embedding_dim,
            activation_dropout=activation_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        self.row_self_attention = self.build_residual(row_self_attention)
        self.column_self_attention = self.build_residual(column_self_attention)
        self.feed_forward_layer = self.build_residual(feed_forward_layer)

    def build_residual(self, layer: nn.Module) -> NormalizedResidualBlock:
        """Wrap a sublayer in a Pre-LN :class:`NormalizedResidualBlock`."""
        return NormalizedResidualBlock(
            layer,
            self.embedding_dim,
            self.dropout_prob,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_head_weights: bool = False,
    ):
        """Run the axial transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input MSA features of shape ``(num_rows, num_cols, batch_size,
            embed_dim)``.
        self_attn_mask : torch.Tensor, optional
            Additive attention mask (currently unsupported by the attention
            modules).
        self_attn_padding_mask : torch.Tensor, optional
            Boolean mask of shape ``(batch_size, num_rows)``; ``True`` marks
            padded rows.
        need_head_weights : bool, optional
            If ``True``, also return the column and row attention
            probabilities.

        Returns
        -------
        torch.Tensor or tuple
            ``x`` (the updated features, same shape as the input) when
            ``need_head_weights`` is ``False``, otherwise
            ``(x, column_attn, row_attn)``.
        """
        x, row_attn = self.row_self_attention(
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        x, column_attn = self.column_self_attention(
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        x = self.feed_forward_layer(x)
        if need_head_weights:
            return x, column_attn, row_attn
        return x
