"""MSA Transformer adapter for ft2 structural sequence reconstruction.

Ports the MSA Transformer architecture (github.com/rmrao/msa-transformer,
originally based on ESM/MSA Transformer from Facebook AI Research) to
FoldTree2's ``ft2`` structural-token vocabulary. The model consumes a batch
of tokenised MSAs of shape ``(B, N_seq, L)`` — discrete codes produced by the
VQ-VAE — and emits per-sequence per-residue feature tensors of shape
``(B, N_seq, L, embed_dim)`` via axial attention.

Structural information flows jointly across two axes:

* **Row attention** (rows = sequences) lets each residue attend to the same
  residue in every other sequence of the MSA — the cross-sequence signal that
  lets known extant geometry inform ancestral positions.
* **Column attention** (cols = residues) lets each residue attend to every
  other residue within its own sequence — the within-chain structural context.

Downstream heads (see :mod:`foldtree2.src.reconstruction.reconstruction_heads`)
consume the resulting features to predict distances, contacts, or frames.

The module is intentionally a faithful port: the only departure from the
upstream MSA Transformer is the vocabulary (``ft2`` codes instead of amino
acids) and the addition of an explicit ``msa_mask`` argument so callers can
pass a precomputed validity mask.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from foldtree2.src.reconstruction.axial_attention import AxialTransformerLayer


__all__ = ["MsaReconstructionHead", "LearnedPositionalEmbedding"]


class LearnedPositionalEmbedding(nn.Embedding):
    """Learned positional embedding that skips padding tokens.

    Ports the MSA Transformer positional embedding: positions are computed
    as a cumulative count of non-padding tokens along the residue axis, so
    two sequences with different effective lengths still align
    positionally after padding. The ``padding_idx`` slot is reserved so the
    cumulative index never collides with a learned position id.

    Parameters
    ----------
    num_embeddings : int
        Number of learnable residue positions (``max_seqlen``).
    embedding_dim : int
        Dimensionality of the embedding vectors.
    padding_idx : int
        Token id reserved for padding; positions are offset so this id is
        never assigned to a real residue.

    Notes
    -----
    The cumsum runs along the **last** dimension (``dim=-1``) so the layer
    accepts both single-sequence tensors ``(B, L)`` and full MSA tensors
    ``(B, N_seq, L)``. In the latter case positions are computed independently
    for each sequence in the MSA, which is the desired behaviour: every row
    is positionally aligned to the residue axis.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
    ) -> None:
        if padding_idx is not None:
            # Reserve room for the padding id and a 1-based offset so that
            # positions for the first valid residue land on a real row of
            # the embedding table (mirrors the upstream MSA Transformer).
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx=padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() < 2:
            raise ValueError(
                "LearnedPositionalEmbedding expects input of dim >= 2 "
                "(batch, [num_seqs,] seq_len); got shape "
                f"{tuple(input.shape)}"
            )
        # 1 where the token is a real residue, 0 where it is padding.
        mask = input.ne(self.padding_idx).int()
        # Cumulative count of valid residues, then masked back to zero on
        # padded positions and offset by padding_idx so the first valid
        # residue maps to row ``padding_idx + 1``.
        positions = (
            torch.cumsum(mask, dim=-1).type_as(mask) * mask
        ).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class MsaReconstructionHead(nn.Module):
    """MSA Transformer reconstruction head for ``ft2`` structural tokens.

    Consumes a batch of tokenised MSAs of shape ``(B, N_seq, L)`` — ``ft2``
    VQ-VAE codes — and produces per-sequence per-residue feature tensors of
    shape ``(B, N_seq, L, embed_dim)`` via axial attention. Row attention
    (over sequences) and column attention (over residues) are interleaved so
    structural information flows jointly across the two axes.

    Architecture::

        tokens (B, N_seq, L)
          -> embed_tokens + embed_positions + msa_position_embedding
          -> dropout + zero padded positions
          -> [AxialTransformerLayer] * num_layers   (layout R, C, B, D)
          -> LayerNorm
          -> features (B, N_seq, L, embed_dim)

    Parameters
    ----------
    alphabet_size : int, optional
        Size of the ``ft2`` token vocabulary. Must include PAD/BOS/EOS
        slots; use 31 for a 30-character structural alphabet. Default 31.
    embed_dim : int, optional
        Token and hidden dimension. Default 256.
    num_heads : int, optional
        Number of attention heads per axial attention block. Default 8.
    num_layers : int, optional
        Number of axial transformer blocks. Default 6.
    dropout : float, optional
        Dropout applied to embeddings and residuals. Default 0.1.
    attention_dropout : float, optional
        Dropout applied within attention weights. Default 0.1.
    activation_dropout : float, optional
        Dropout applied to the feed-forward activation. Default 0.1.
    max_seqlen : int, optional
        Maximum supported residue positions. Default 1024.
    max_msa_depth : int, optional
        Maximum supported number of MSA rows. Default 256.
    max_tokens_per_msa : int, optional
        Cap on (non-padding) tokens retained per MSA during batched
        inference; rows beyond the cap are masked out. Default 16384.
    padding_idx : int, optional
        Token id reserved for padding. Default 0.

    Raises
    ------
    ValueError
        If ``alphabet_size`` is not larger than ``padding_idx``, or if the
        inputs to :meth:`forward` violate the documented shapes / bounds.
    """

    def __init__(
        self,
        alphabet_size: int = 31,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_seqlen: int = 2048,
        max_msa_depth: int = 256,
        max_tokens_per_msa: int = 2 ** 14,
        padding_idx: int = 0,
        geom_dim: int = 0,
    ) -> None:
        super().__init__()

        if alphabet_size <= padding_idx:
            raise ValueError(
                f"alphabet_size ({alphabet_size}) must be strictly greater "
                f"than padding_idx ({padding_idx}); the vocabulary must "
                f"contain at least one non-padding token."
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )

        self.alphabet_size = alphabet_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.max_seqlen = max_seqlen
        self.max_msa_depth = max_msa_depth
        self.max_tokens_per_msa = max_tokens_per_msa
        self.dropout = dropout
        self.geom_dim = int(geom_dim)

        # Token lookup. ``padding_idx`` zeroes the gradient and the row so
        # padded tokens contribute neither forward nor backward.
        self.embed_tokens = nn.Embedding(
            alphabet_size, embed_dim, padding_idx=padding_idx
        )
        self.embed_positions = LearnedPositionalEmbedding(
            max_seqlen, embed_dim, padding_idx=padding_idx
        )

        # Optional projection for continuous geometry features (CA coords,
        # pairwise distances, contact embeddings, etc.).  Known sequences
        # supply these; masked sequences pass zeros so only the token
        # embedding drives those rows.  The row attention then propagates
        # geometry from known to masked sequences during axial processing.
        if self.geom_dim > 0:
            self.geom_proj = nn.Linear(self.geom_dim, embed_dim)
            # Learnable marker added to MSA rows whose geometry has been
            # withheld.  This gives axial attention an explicit signal of
            # which rows carry structural information so it can propagate
            # geometry from rows that have it to rows that do not.
            self.no_geom_embedding = nn.Parameter(
                0.02 * torch.randn(1, 1, 1, embed_dim)
            )
        else:
            self.geom_proj = None
            self.no_geom_embedding = None

        # Per-MSA-row depth embedding. Broadcast across the batch and
        # residue axes; only the leading ``num_seqs`` rows are ever read.
        self.msa_position_embedding = nn.Parameter(
            0.01 * torch.randn(1, max_msa_depth, 1, 1)
        )

        # Axial transformer stack. The feed-forward width follows the
        # standard 4x expansion used by the MSA Transformer.
        ffn_embedding_dim = embed_dim * 4
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    embedding_dim=embed_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    max_tokens_per_msa=max_tokens_per_msa,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _maybe_trim_msa(
        self,
        tokens: torch.Tensor,
        msa_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Trim an MSA to ``max_tokens_per_msa`` valid residues per batch row.

        Rows are kept greedily by validity count (densest first) and the
        remaining rows are masked out. This mirrors the upstream MSA
        Transformer inference-time trimming and keeps memory bounded for
        very deep alignments.

        Returns the trimmed ``msa_mask`` (a new tensor; the input is not
        mutated). ``tokens`` is left untouched — masking the padded rows is
        enough because the forward pass zeroes padded positions and the
        axial layers refuse to attend to ``padding_mask`` positions.
        """
        batch_size, num_seqs, seq_len = tokens.shape
        max_tokens = self.max_tokens_per_msa
        if max_tokens is None or num_seqs * seq_len <= max_tokens:
            return msa_mask

        tokens_per_seq = msa_mask.sum(dim=-1)  # (B, num_seqs)
        # Indices that would sort rows by validity, densest first.
        order = tokens_per_seq.argsort(dim=1, descending=True)
        sorted_counts = tokens_per_seq.gather(1, order)
        running = sorted_counts.cumsum(dim=1)
        keep_in_order = running <= max_tokens
        # Scatter the keep flag back to the original row ordering.
        row_keep = torch.zeros_like(keep_in_order)
        row_keep.scatter_(1, order, keep_in_order)
        return msa_mask & row_keep.unsqueeze(-1)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        tokens: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
        need_head_weights: bool = False,
        geom_features: Optional[torch.Tensor] = None,
        geom_known_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run axial attention over a batch of ``ft2`` MSAs.

        Parameters
        ----------
        tokens : torch.Tensor
            Long tensor of shape ``(batch_size, num_seqs, seq_len)`` holding
            ``ft2`` token indices. Padded positions must equal
            ``padding_idx``.
        msa_mask : torch.Tensor, optional
            Bool tensor of shape ``(batch_size, num_seqs, seq_len)`` where
            ``True`` marks a valid residue and ``False`` marks padding. If
            omitted, the mask is derived from ``tokens != padding_idx``.
        need_head_weights : bool, optional
            If ``True``, also return per-layer row and column attention
            weights for downstream analysis / visualisation.

        Returns
        -------
        dict
            ``"features"`` : ``torch.Tensor`` of shape
            ``(B, N_seq, L, embed_dim)`` — per-sequence per-residue
            features, after the final ``LayerNorm``.
            ``"representations"`` : ``dict[int | str, torch.Tensor]``
            mapping layer index to intermediate features. Key ``0`` is the
            post-embedding (pre-block) representation, keys ``1..num_layers``
            are the outputs of each axial block, and ``"final"`` is the
            post-``LayerNorm`` features (identical to ``"features"``).
            ``"row_attentions"`` : ``torch.Tensor`` of shape
            ``(B, num_layers, H, N_seq, N_seq)`` — sequence-sequence
            attention. Only present when ``need_head_weights=True``.
            ``"col_attentions"`` : ``torch.Tensor`` of shape
            ``(B, num_layers, H, seq_len, seq_len)`` — residue-residue
            attention. Only present when ``need_head_weights=True``.

        Raises
        ------
        ValueError
            If ``tokens`` is not rank-3, exceeds the configured bounds, or
            if ``msa_mask`` does not match the ``tokens`` shape.
        """
        # ---- Validate inputs ------------------------------------------- #
        if tokens.dim() != 3:
            raise ValueError(
                "tokens must be rank-3 (batch_size, num_seqs, seq_len); "
                f"got {tokens.dim()}D shape {tuple(tokens.shape)}"
            )
        if tokens.dtype not in (torch.long, torch.int32, torch.int64):
            raise ValueError(
                f"tokens must be an integer tensor of ft2 code indices; "
                f"got dtype {tokens.dtype}"
            )
        batch_size, num_seqs, seq_len = tokens.shape

        if seq_len > self.max_seqlen:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seqlen ({self.max_seqlen}). "
                "Increase max_seqlen or truncate the input."
            )
        if num_seqs > self.max_msa_depth:
            raise ValueError(
                f"num_seqs ({num_seqs}) exceeds max_msa_depth "
                f"({self.max_msa_depth}). Increase max_msa_depth or trim the "
                "MSA."
            )

        # Build / validate the validity mask (True = valid residue).
        if msa_mask is None:
            msa_mask = tokens.ne(self.padding_idx)
        else:
            if msa_mask.shape != tokens.shape:
                raise ValueError(
                    f"msa_mask shape {tuple(msa_mask.shape)} does not match "
                    f"tokens shape {tuple(tokens.shape)}"
                )
            if msa_mask.dtype != torch.bool:
                msa_mask = msa_mask.to(torch.bool)

        # Optionally trim very deep MSAs to keep memory bounded.
        msa_mask = self._maybe_trim_msa(tokens, msa_mask)

        # ---- Embeddings ------------------------------------------------ #
        x = self.embed_tokens(tokens)                       # (B, N, L, D)
        x = x + self.embed_positions(tokens)                 # (B, N, L, D)
        x = x + self.msa_position_embedding[:, :num_seqs, :, :]
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Zero out padded positions so they carry no signal into the
        # residual stream. ``msa_mask`` is True at valid residues, so we
        # multiply by the mask (keeping valid positions and zeroing pads).
        x = x * msa_mask.unsqueeze(-1).type_as(x)

        # ---- Integrate geometry features for rows that carry geometry ----
        # Each sequence with available structure supplies per-residue
        # geometry (pseudo-torsions, bond-angle cosines, contact densities,
        # nearest-neighbour distances).  Rows whose geometry is withheld get
        # zero geometry features plus a learnable "no geometry" marker, so
        # axial attention can explicitly distinguish them and propagate
        # structural information from the rows that do carry geometry.
        if self.geom_proj is not None and geom_features is not None:
            if geom_features.shape != tokens.shape + (self.geom_dim,):
                raise ValueError(
                    f"geom_features expected shape "
                    f"{tuple(tokens.shape) + (self.geom_dim,)}, "
                    f"got {tuple(geom_features.shape)}"
                )
            if geom_known_mask is None:
                has_geom = torch.ones(
                    batch_size, num_seqs, dtype=torch.bool, device=tokens.device
                )
            else:
                if geom_known_mask.shape != (batch_size, num_seqs):
                    raise ValueError(
                        f"geom_known_mask expected shape "
                        f"({batch_size}, {num_seqs}), got "
                        f"{tuple(geom_known_mask.shape)}"
                    )
                has_geom = geom_known_mask.to(torch.bool)
            row_geom = has_geom.unsqueeze(-1) & msa_mask          # (B, N, L)
            row_no_geom = (~has_geom).unsqueeze(-1) & msa_mask    # (B, N, L)
            geom_embed = self.geom_proj(geom_features)           # (B, N, L, D)
            x = x + geom_embed * row_geom.unsqueeze(-1).type_as(x)
            if self.no_geom_embedding is not None:
                x = x + self.no_geom_embedding * row_no_geom.unsqueeze(-1).type_as(x)
        elif self.geom_proj is not None and geom_features is None:
            # Geometry is enabled but not provided — continue with tokens only.
            pass

        # ---- Axial transformer (layout: R, C, B, D) ------------------- #
        # The axial layers expect the attention axes first: R = sequences
        # (rows), C = residues (columns). Batch is the slowest axis.
        x = x.permute(1, 2, 0, 3)                            # (N, L, B, D)

        # Axial attention padding mask in the convention True = pad, which
        # is what the RowSelfAttention / ColumnSelfAttention modules expect.
        # RowSelfAttention expects a 2D mask (batch, num_rows) where True
        # marks an entirely-padded row (sequence).  A row is padded when
        # ALL its residue positions are masked.
        row_padding_mask = (~msa_mask).all(dim=-1)           # (B, N) — rows only
        # Keep the full 3D mask for per-column use (ColumnSelfAttention
        # receives ``self_attn_padding_mask`` as a keyword but the ported
        # implementation only uses row-level masks in the column-attention
        # path as well — the actual per-position masking is done before the
        # axial layers by zeroing padded positions in the embedding).
        padding_mask = row_padding_mask                      # (B, N)

        representations = {0: x.permute(2, 0, 1, 3)}          # (B, N, L, D)
        row_attentions = []
        col_attentions = []

        for layer_idx, layer in enumerate(self.layers, start=1):
            if need_head_weights:
                x, row_attn, col_attn = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_head_weights=True,
                )
                row_attentions.append(row_attn)
                col_attentions.append(col_attn)
            else:
                x = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_head_weights=False,
                )
            representations[layer_idx] = x.permute(2, 0, 1, 3)  # (B, N, L, D)

        x = self.layer_norm(x)
        features = x.permute(2, 0, 1, 3)                    # (B, N, L, D)
        representations["final"] = features

        output: dict = {
            "features": features,
            "representations": representations,
        }
        if need_head_weights:
            # Stack per-layer attentions along a new layer axis. Each layer
            # is expected to return weights of shape
            # (B, H, ...) so stacking on dim=1 yields
            # (B, num_layers, H, ...).
            output["row_attentions"] = torch.stack(row_attentions, dim=1)
            output["col_attentions"] = torch.stack(col_attentions, dim=1)
        return output

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters in the module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}("
            f"alphabet_size={self.alphabet_size}, "
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}, "
            f"max_seqlen={self.max_seqlen}, "
            f"max_msa_depth={self.max_msa_depth}, "
            f"padding_idx={self.padding_idx})"
        )
