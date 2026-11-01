"""Two-step MSA → geometry pipeline: MSA Transformer → SE(3) decoder.

The idea: the MSA Transformer (axial attention over the ft2 alignment) enriches
each sequence's features with cross-sequence structural context.  The SE(3)
decoder then converts those enriched per-residue features into real 3D
coordinates — using the known structures as a crutch for the unknown ones.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from foldtree2.src.reconstruction.msa_transformer_adapter import MsaReconstructionHead
from foldtree2.src.models.se3_struct_decoder import se3_denoiser


class MsaSe3Pipeline(nn.Module):
    """Two-step prediction: ft2 MSA → axial attention → SE(3) decoder.

    Step 1 — MSA Transformer processes the full ft2 MSA with axial attention,
    producing per-sequence per-residue features ``(B, N_seq, L, D)``.

    Step 2 — For each sequence, the SE(3) decoder (GotenNet) consumes those
    features and predicts CA coordinates and backbone angles.  The SE(3)
    parameters are shared across all sequences (same decoder applied to each
    row of the MSA independently).

    The SE(3) decoder also receives the discrete ft2 token ids for its atom-type
    embedding, and an initial coordinate seed (from the distance head or from the
    known structures).

    Args:
        alphabet_size: ft2 vocabulary size.
        embed_dim: MSA Transformer hidden dimension.
        num_heads: MSA Transformer attention heads.
        num_layers: MSA Transformer layers.
        se3_in_channels: Input dimension for the SE(3) decoder (usually same as embed_dim).
        se3_hidden: SE(3) decoder hidden dimension.
        se3_depth: SE(3) GotenNet depth.
        se3_heads: SE(3) attention heads.
        se3_num_atom_types: Number of atom type embeddings in GotenNet.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        alphabet_size: int = 31,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        se3_in_channels: int = 256,
        se3_hidden: int = 128,
        se3_depth: int = 3,
        se3_heads: int = 4,
        se3_num_atom_types: int = 40,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Step 1: MSA Transformer
        self.msa_encoder = MsaReconstructionHead(
            alphabet_size=alphabet_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seqlen=2048,
            geom_dim=9,
        )

        # Optional projection if MSA embed_dim ≠ SE(3) input dim
        if embed_dim != se3_in_channels:
            self.feature_proj = nn.Linear(embed_dim, se3_in_channels)
        else:
            self.feature_proj = nn.Identity()

        # Step 2: SE(3) decoder (shared across all sequences)
        self.se3_decoder = se3_denoiser(
            in_channels=se3_in_channels,
            hidden_channels=[se3_hidden],
            out_channels=3,
            num_embeddings=se3_num_atom_types,
            commitment_cost=0.25,
            dropout_p=0.0,
            depth=se3_depth,
            heads=se3_heads,
            dim_head=16,
            num_atom_types=se3_num_atom_types,
            coordinate_update="residual",
            zero_init_coordinate_update=True,
            coordinate_normalization="rms_radius",
            coordinate_residual_gain=32.0,
            angle_output="sincos",
        )

    def forward(
        self,
        tokens: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
        seed_coords: Optional[torch.Tensor] = None,
        known_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict:
        """Run the two-step pipeline.

        Args:
            tokens: (B, N_seq, L) ft2 token indices.
            msa_mask: (B, N_seq, L) bool mask.
            seed_coords: (B, N_seq, L, 3) initial CA coords (NaN for unknown).
            known_mask: (B, N_seq) bool; True for sequences with known coords.

        Returns:
            dict with:
                features: (B, N_seq, L, D) MSA Transformer features.
                coords: (B, N_seq, L, 3) SE(3) refined coordinates.
                angles: (B, N_seq, L, 6) SE(3) predicted backbone angles.
                attentions: optional MSA Transformer attention weights.
        """
        B, N_seq, L = tokens.shape

        # Step 1: MSA Transformer → per-seq features with cross-seq context
        enc_out = self.msa_encoder(tokens, msa_mask=msa_mask, need_head_weights=False)
        features = enc_out["features"]  # (B, N_seq, L, D)

        # Project to SE(3) input dimension
        se3_feats = self.feature_proj(features)  # (B, N_seq, L, se3_in)

        # Step 2: Run SE(3) decoder per-sequence (shared weights)
        all_coords, all_angles = [], []

        for b in range(B):
            for n in range(N_seq):
                # Build HeteroData for this sequence
                g = HeteroData()
                g["res"].x = se3_feats[b, n]  # (L, se3_in)

                # Coordinate seed: from argument or zero
                if seed_coords is not None and torch.isfinite(seed_coords[b, n]).all():
                    coords_seed = seed_coords[b, n]  # (L, 3)
                else:
                    coords_seed = torch.zeros(L, 3, device=se3_feats.device)

                # Residue mask per sequence
                if msa_mask is not None:
                    residue_mask = msa_mask[b, n]  # (L,) bool
                else:
                    residue_mask = torch.ones(L, dtype=torch.bool, device=se3_feats.device)

                # Run SE(3) decoder
                out = self.se3_decoder(
                    g,
                    coords_pred=coords_seed,
                    ft2_token_ids=tokens[b, n],
                    edge_attr_dict={
                        "node_mask": residue_mask,
                        "use_distance_contacts": True,
                    },
                )
                all_coords.append(out["coors_out_flat"])   # (L, 3)
                all_angles.append(out["angles"])            # (L, 6)

        coords = torch.stack(all_coords, dim=0).view(B, N_seq, L, 3)
        angles = torch.stack(all_angles, dim=0).view(B, N_seq, L, 6)

        return {
            "features": features,
            "coords": coords,
            "angles": angles,
            "attentions": enc_out.get("row_attentions", None),
        }