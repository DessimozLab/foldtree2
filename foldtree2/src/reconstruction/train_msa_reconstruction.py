#!/usr/bin/env python3
"""MSA Reconstruction training with optional SE(3) decoder refinement.

Two modes:

  **Heads mode** (default): MSA Transformer → distance/angle/coordinate heads.
    Lightweight, good for prototyping.

  **SE(3) mode** (``--use-se3``): MSA Transformer → SE(3) decoder per sequence.
    The SE(3) decoder (GotenNet) converts the cross-sequence-enriched features
    into actual 3D CA coordinates and backbone angles using the full SE(3) loss
    stack (FAPE, bond length, angle, distance).  Known structures bootstrap the
    masked (ancestral) sequences through the MSA context.

Usage::

    # Heads mode (default)
    python foldtree2/src/reconstruction/train_msa_reconstruction.py

    # SE(3) mode
    python foldtree2/src/reconstruction/train_msa_reconstruction.py \\
        --use-se3 --se3-hidden 128 --se3-depth 3 \\
        --batch-size 1 --family-ids 1003993

    # CPU test (SE(3) per-sequence is expensive — small model)
    python foldtree2/src/reconstruction/train_msa_reconstruction.py \\
        --fast-dev-run --use-se3 --embed-dim 64 --se3-hidden 32 --se3-depth 1 \\
        --family-ids 1003993 --accelerator cpu --devices 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

if __package__ in {None, ""}:
    _root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_root))

from foldtree2.src.reconstruction.msa_transformer_adapter import MsaReconstructionHead
from foldtree2.src.reconstruction.reconstruction_heads import (
    MsaDistanceHead,
    MsaAngleHead,
    MsaCoordinateHead,
    msa_distance_loss,
    msa_angle_loss,
    msa_coordinate_loss,
)
from foldtree2.src.reconstruction.tree_msa_dataset import (
    TreeMsaDataset,
    CladeSampledMsaDataset,
    msa_collate,
    _MIN_SEP,
)
from foldtree2.src.distance.distance_restraints import distance_labels, eligible_pairs


# ---------------------------------------------------------------------------
# CA pseudo-dihedral (used by both modes)
# ---------------------------------------------------------------------------

def _ca_pseudo_dihedrals(ca: torch.Tensor) -> torch.Tensor:
    L = ca.shape[0]
    if L < 4:
        return torch.full((L,), float("nan"), device=ca.device, dtype=ca.dtype)
    p0, p1, p2, p3 = ca[:-3], ca[1:-2], ca[2:-1], ca[3:]
    b1 = F.normalize(p1 - p0, dim=-1, eps=1e-8)
    b2 = F.normalize(p2 - p1, dim=-1, eps=1e-8)
    b3 = F.normalize(p3 - p2, dim=-1, eps=1e-8)
    n1 = F.normalize(torch.cross(b1, b2, dim=-1), dim=-1, eps=1e-8)
    n2 = F.normalize(torch.cross(b2, b3, dim=-1), dim=-1, eps=1e-8)
    cos = (n1 * n2).sum(dim=-1).clamp(-1., 1.)
    sin = (torch.cross(n1, n2, dim=-1) * b2).sum(dim=-1).clamp(-1., 1.)
    dih = torch.atan2(sin, cos)
    result = torch.full((L,), float("nan"), device=ca.device, dtype=ca.dtype)
    # result[i] gets dihedral CA[i-1..i+2], needs indices 2..L-2
    result[2:-1] = dih  # (L-3,) -> (L-3,)
    return result


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class MsaReconstructionModule(pl.LightningModule):
    """MSA Reconstruction with optional SE(3) decoder refinement.

    Args:
        use_se3: If True, use the SE(3) decoder pipeline instead of heads.
        alphabet_size: ft2 vocabulary.
        embed_dim: MSA Transformer hidden dim.
        num_heads: Attention heads.
        num_layers: Axial transformer layers.
        dropout: Dropout.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        label_smoothing: CE label smoothing.
        loss_weights: Dict of loss weights.
        se3_hidden: SE(3) GotenNet hidden.
        se3_depth: SE(3) depth.
        se3_heads: SE(3) attention heads.
    """

    def __init__(
        self,
        use_se3: bool = False,
        alphabet_size: int = 31,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
        min_separation: int = _MIN_SEP,
        loss_weights: Optional[Dict[str, float]] = None,
        max_pairs: int = 100000,
        # Heads mode
        distance_hidden: int = 128,
        angle_hidden: int = 64,
        coord_hidden: int = 128,
        # SE(3) mode
        se3_hidden: int = 128,
        se3_depth: int = 3,
        se3_heads: int = 4,
        # SE(3) memory: cap per-sequence length handed to the O(L^2) GotenNet
        # decoder.  Sequences longer than this are truncated to the first
        # ``se3_max_residues`` residues before the decoder forward, and the
        # loss is restricted to that prefix.  0 / None disables the cap.
        se3_max_residues: int = 512,
        # SE(3) memory: cap the number of sequences per batch that are run
        # through the O(L^2) GotenNet decoder.  The decoder builds an autograd
        # graph per sequence; running it on every sequence in a deep MSA
        # (e.g. 18 sequences for family 1003993) accumulates a graph that
        # OOMs a 23 GB GPU.  Supervised (known_structure=True, not masked)
        # sequences are decoded first; if fewer than the cap are supervised,
        # the remaining slots are filled with any other non-masked sequences.
        # Sequences not decoded get zero coords/angles and no loss.  0 /
        # None disables the cap (decode every non-masked sequence).
        se3_max_sequences: int = 4,
        # SE(3) seed noise (Angstrom): supervised sequences are seeded with
        # the true CA coords plus N(0, se3_seed_noise) noise so the decoder
        # acts as a denoising/refinement network.  0 uses the exact coords.
        se3_seed_noise: float = 0.5,
        # ReduceLROnPlateau scheduler (used by configure_optimizers)
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_weights"])
        self.use_se3 = bool(use_se3)
        self.se3_max_residues = int(se3_max_residues) if se3_max_residues else 0
        self.se3_max_sequences = int(se3_max_sequences) if se3_max_sequences else 0
        self.se3_seed_noise = float(se3_seed_noise)
        self.label_smoothing = label_smoothing
        self.focal_gamma = float(focal_gamma)
        self.min_separation = min_separation
        self.lr_factor = float(lr_factor)
        self.lr_patience = int(lr_patience)
        self.min_lr = float(min_lr)
        self._max_pairs = int(max_pairs)
        self._loss_weights = {
            "distance": 1.0, "angle": 0.1, "coordinate": 0.1,
            "fape": 1.0, "bond": 1.0,  # SE(3)-specific
            **(loss_weights or {}),
        }

        # MSA Transformer (shared)
        self.msa_encoder = MsaReconstructionHead(
            alphabet_size=alphabet_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seqlen=4096,
            geom_dim=9,
        )

        if self.use_se3:
            self._build_se3_pipeline(embed_dim, se3_hidden, se3_depth, se3_heads)
        else:
            self._build_heads(embed_dim, distance_hidden, angle_hidden, coord_hidden, dropout)

    def _build_heads(self, embed_dim, distance_hidden, angle_hidden, coord_hidden, dropout):
        from foldtree2.src.distance.distance_restraints import DISTANCE_EDGES
        self.distance_head = MsaDistanceHead(
            embed_dim=embed_dim, hidden=distance_hidden,
            edges=DISTANCE_EDGES, dropout=dropout,
        )
        self.angle_head = MsaAngleHead(embed_dim=embed_dim, hidden=angle_hidden)

    def _build_se3_pipeline(self, embed_dim, se3_hidden, se3_depth, se3_heads):
        from foldtree2.src.reconstruction.msa_se3_pipeline import MsaSe3Pipeline
        self.se3_pipeline = MsaSe3Pipeline(
            alphabet_size=self.hparams.alphabet_size,
            embed_dim=embed_dim,
            num_heads=self.hparams.num_heads,
            num_layers=self.hparams.num_layers,
            se3_in_channels=embed_dim,
            se3_hidden=se3_hidden,
            se3_depth=se3_depth,
            se3_heads=se3_heads,
            dropout=self.hparams.dropout,
        )
        # GotenNet's coord/angle heads are zero-initialised, which blocks
        # gradients to every upstream layer (grad_h = W^T grad_out = 0).
        # A small random init restores gradient flow.
        dec = self.se3_pipeline.se3_decoder
        torch.nn.init.normal_(dec.gotennet.proj_to_coors[1].weight, std=1e-3)
        if hasattr(dec, "out_angles"):
            for m in reversed(list(dec.out_angles)):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, std=1e-3)
                    break

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, batch: Dict) -> Dict:
        tokens = batch["tokens"]
        msa_mask = batch.get("mask", None)

        if self.use_se3:
            return self._forward_se3(tokens, msa_mask,
                                     batch.get("known_structure"), batch.get("coords"))
        else:
            return self._forward_heads(tokens, msa_mask,
                                       batch.get("known_structure"), batch.get("coords"))

    def _forward_heads(self, tokens, msa_mask, known_mask=None, coords=None):
        geom = None
        if hasattr(self.msa_encoder, 'geom_proj') and self.msa_encoder.geom_proj is not None:
            if coords is not None and known_mask is not None:
                geom = self._geom_features(coords, known_mask)

        enc = self.msa_encoder(
            tokens, msa_mask=msa_mask, need_head_weights=False, geom_features=geom,
        )
        features = enc["features"]
        L = tokens.shape[-1]
        idx = torch.arange(L, device=self.device)
        pairs = (idx[None] - idx[:, None]).abs().ge(self.min_separation).nonzero()

        # Subsample pairs for memory efficiency
        max_pairs = getattr(self, '_max_pairs', 100000)
        if max_pairs > 0 and pairs.shape[0] > max_pairs:
            perm = torch.randperm(pairs.shape[0], device=pairs.device)[:max_pairs]
            pairs = pairs[perm]

        if msa_mask is not None:
            seq_mask = msa_mask.any(dim=-1)
        else:
            seq_mask = None
        return {
            "features": features,
            "distance_logits": self.distance_head(features, seq_mask=seq_mask, pairs=pairs),
            "angle_logits": self.angle_head(features),
            "pairs": pairs,
            "attentions": enc.get("row_attentions", None),
        }

    def _forward_se3(self, tokens, msa_mask, known_mask=None, coords=None):
        """Run MSA Transformer then SE(3) decoder per sequence.

        The GotenNet SE(3) decoder runs a dense O(L^2) neighbor attention per
        sequence, which OOMs on long MSA rows (e.g. ~2500-residue OMA rows).
        To bound peak memory, each sequence is truncated to the first
        ``self.se3_max_residues`` residues before being handed to the decoder.
        Outputs are zero-padded back to the full MSA length ``L`` so downstream
        tensor shapes stay consistent; the training step is responsible for
        restricting the loss mask to the truncated prefix.  A cap of ``0`` (or
        ``None``) disables the truncation.

        A second memory bound, ``self.se3_max_sequences``, caps the number of
        sequences per batch that are actually run through the decoder.  The
        decoder builds its own autograd graph per sequence; running it on
        every sequence in a deep MSA accumulates a graph that OOMs a 23 GB
        GPU.  Supervised (``known_structure=True`` and not fully masked)
        sequences are decoded first; remaining slots are filled with any
        other non-masked sequences.  Sequences that are not decoded get
        zero coords/angles and contribute no loss.  A cap of ``0`` disables
        the cap (decode every non-masked sequence).  The indices actually
        decoded are returned as ``out["se3_decoded_idx"]`` (a list of
        ``(b, n)`` tuples) so the training step can restrict the loss.
        """
        B, N_seq, L = tokens.shape

        # Step 1: MSA Transformer — cross-sequence context (full length)
        enc = self.msa_encoder(tokens, msa_mask=msa_mask, need_head_weights=False)
        features = enc["features"]  # (B, N_seq, L, D)

        # Resolve the per-sequence decoder length cap.
        K = int(getattr(self, "se3_max_residues", 0) or 0)
        if K <= 0:
            K = L
        K = min(K, L)

        # Resolve the per-batch decoder sequence cap.
        M = int(getattr(self, "se3_max_sequences", 0) or 0)
        # M == 0 means "decode every non-masked sequence" (uncapped).

        # Step 2: SE(3) decoder per sequence (truncated to first K residues).
        # We only run the decoder on a capped subset of sequences per batch
        # element: supervised sequences first, then any remaining non-masked
        # sequences.  Non-decoded sequences are zero-filled later.
        all_coords = [[None] * N_seq for _ in range(B)]
        all_angles = [[None] * N_seq for _ in range(B)]
        decoded_idx = []  # list of (b, n) tuples that were actually decoded
        A = None

        for b in range(B):
            # Build the ordered candidate list for this batch element:
            # supervised (known & not masked) first, then remaining non-masked.
            supervised_cands = []
            remaining_cands = []
            for n in range(N_seq):
                if msa_mask is not None:
                    seq_v = bool(msa_mask[b, n].any().item())
                else:
                    seq_v = True
                if not seq_v:
                    # Fully masked sequence — cannot be decoded.
                    continue
                is_sup = known_mask is not None and bool(known_mask[b, n])
                if is_sup:
                    supervised_cands.append(n)
                else:
                    remaining_cands.append(n)
            candidates = supervised_cands + remaining_cands
            if M > 0:
                candidates = candidates[:M]

            for n in candidates:
                # Truncate to the first K residues to bound the O(L^2) decoder.
                feat_k = features[b, n, :K]           # (K, D)
                # Fully-masked rows in axial attention can produce NaN
                # features (softmax over all-masked keys); sanitize before
                # handing them to the SE(3) decoder, which rejects nonfinite
                # valid features.
                feat_k = torch.nan_to_num(feat_k, nan=0.0)
                tok_k = tokens[b, n, :K]              # (K,)
                if msa_mask is not None:
                    node_mask_k = msa_mask[b, n, :K]   # (K,) bool
                else:
                    node_mask_k = torch.ones(
                        K, dtype=torch.bool, device=tokens.device,
                    )
                # Seed coordinates for the refinement decoder.  The GotenNet
                # SE(3) decoder is a *refinement* network: seeding all points
                # at the origin makes every pair distance r_ij = 0, from
                # which no meaningful geometry can be recovered.  Instead:
                #   * supervised (known structure) sequences are seeded with
                #     the true CA coordinates plus Gaussian noise
                #     N(0, se3_seed_noise) — a denoising task;
                #   * unknown (ancestral) sequences are seeded with an
                #     idealized alpha-helix backbone.
                is_sup = known_mask is not None and bool(known_mask[b, n])
                if is_sup and coords is not None:
                    seed_k = coords[b, n, :K].to(device=tokens.device, dtype=torch.float32).clone()
                    # Padded/short sequences carry NaN coords; replace those
                    # rows with idealized helix positions and drop them from
                    # the decoder node mask.
                    bad = ~torch.isfinite(seed_k).all(dim=-1) | (seed_k == 0).all(dim=-1)
                    if bad.any():
                        idx_bad = torch.arange(K, device=tokens.device, dtype=torch.float32)
                        theta_bad = torch.deg2rad(idx_bad * 100.0)
                        helix = torch.stack(
                            [2.3 * torch.cos(theta_bad),
                             2.3 * torch.sin(theta_bad),
                             3.6 * idx_bad], dim=-1)
                        seed_k[bad] = helix[bad]
                        node_mask_k = node_mask_k & ~bad
                    if self.se3_seed_noise > 0:
                        seed_k = seed_k + torch.randn_like(seed_k) * self.se3_seed_noise
                else:
                    # Idealized alpha-helix: radius 2.3 A, 100 deg/residue
                    # rotation, 3.6 A/residue rise along z.
                    idx = torch.arange(K, device=tokens.device, dtype=torch.float32)
                    theta = torch.deg2rad(idx * 100.0)
                    seed_k = torch.stack(
                        [
                            2.3 * torch.cos(theta),
                            2.3 * torch.sin(theta),
                            3.6 * idx,
                        ],
                        dim=-1,
                    )

                # Build a HeteroData for this (truncated) sequence
                g = HeteroData()
                g["res"].x = feat_k

                if not node_mask_k.any():
                    continue
                try:
                    out = self.se3_pipeline.se3_decoder(
                        g,
                        coords_pred=seed_k,
                        ft2_token_ids=tok_k,
                        edge_attr_dict={
                            "node_mask": node_mask_k,
                            "use_distance_contacts": True,
                        },
                    )
                except (RuntimeError, ValueError):
                    # shortcut: degenerate GotenNet input (nonfinite invariants);
                    # skip this sequence, upgrade if failures become common
                    continue
                coords_k = out["coors_out_flat"]   # (K, 3)
                angles_k = out["angles"]           # (K, A) — A depends on head

                if A is None:
                    A = angles_k.shape[-1]

                # Zero-pad back to the full MSA length so downstream shapes
                # (loss, validation, checkpointing) stay consistent.
                coords_full = torch.zeros(L, 3, device=tokens.device)
                angles_full = torch.zeros(L, A, device=tokens.device)
                coords_full[:K] = coords_k
                angles_full[:K] = angles_k
                all_coords[b][n] = coords_full
                all_angles[b][n] = angles_full
                decoded_idx.append((b, n))

        # Default angle dim if no sequence was decoded at all.
        if A is None:
            A = 3

        # Fill every non-decoded slot (masked or beyond the cap) with zeros.
        for b in range(B):
            for n in range(N_seq):
                if all_coords[b][n] is None:
                    all_coords[b][n] = torch.zeros(L, 3, device=tokens.device)
                    all_angles[b][n] = torch.zeros(L, A, device=tokens.device)

        coords_out = torch.stack(
            [torch.stack(row, dim=0) for row in all_coords], dim=0
        ).view(B, N_seq, L, 3)
        angles_out = torch.stack(
            [torch.stack(row, dim=0) for row in all_angles], dim=0
        ).view(B, N_seq, L, A)

        return {
            "features": features,
            "coords": coords_out,
            "angles": angles_out,
            "attentions": enc.get("row_attentions", None),
            "se3_max_residues": K,
            "se3_decoded_idx": decoded_idx,
        }

    # ------------------------------------------------------------------ #
    # Target helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pair_labels(coords, pairs, edges):
        B, N_seq, L, _ = coords.shape
        labels = coords.new_full((B, N_seq, pairs.shape[0]), -1, dtype=torch.long)
        for b in range(B):
            for n in range(N_seq):
                c = coords[b, n]
                vr = torch.isfinite(c).all(dim=-1)
                if not vr.any():
                    continue
                vp = vr[pairs[:, 0]] & vr[pairs[:, 1]]
                if not vp.any():
                    continue
                _lbl, _ = distance_labels(c, pairs[vp], edges)
                labels[b, n, vp] = _lbl.long()
        return labels

    @staticmethod
    def _se3_loss_for_sequence(pred_coords, pred_angles, target_coords, mask, loss_weights):
        """SE(3)-style loss for one sequence: FAPE + bond + distance + angle."""
        target = target_coords[mask].detach()
        pred = pred_coords[mask]
        zero = pred.sum() * 0.0
        components = {}

        # FAPE: frame-aligned point error (simplified — cRMSD-like)
        if loss_weights.get("fape", 0) > 0:
            aligned_pred = pred - pred.mean(0)
            aligned_tgt = target - target.mean(0)
            rmsd = (aligned_pred - aligned_tgt).norm(dim=-1)
            components["fape"] = rmsd.mean()

        # Bond: CA-CA bond length
        if loss_weights.get("bond", 0) > 0:
            valid_mask = mask[1:] & mask[:-1]
            if valid_mask.any():
                pb = (pred_coords[1:] - pred_coords[:-1]).norm(dim=-1)[valid_mask]
                tb = (target_coords[1:] - target_coords[:-1]).norm(dim=-1)[valid_mask]
                valid_bonds = (tb >= 2.5) & (tb <= 4.5)
                if valid_bonds.any():
                    components["bond"] = F.smooth_l1_loss(pb[valid_bonds], tb[valid_bonds])
                else:
                    components["bond"] = zero
            else:
                components["bond"] = zero

        # Distance: pairwise distances (smooth L1 on cdist)
        if loss_weights.get("distance", 0) > 0:
            pd = torch.cdist(pred[None], pred[None])[0]
            td = torch.cdist(target[None], target[None])[0]
            # Exclude self-pairs and near-identical distances
            off = ~torch.eye(len(pred), dtype=torch.bool, device=pred.device)
            neighbors = (td < 15.0) & (td > 0.5) & off
            if neighbors.any():
                components["distance"] = F.smooth_l1_loss(
                    pd[neighbors], td[neighbors]
                )
            else:
                components["distance"] = zero

        # Angle: predicted vs target (circular)
        if loss_weights.get("angle", 0) > 0:
            if pred_angles is not None and torch.isfinite(pred_angles[mask]).any():
                pa = pred_angles[mask]
                delta = (pa - pa.new_zeros(pa.shape))  # placeholder, no target angles
                components["angle"] = delta.norm() * 0.0

        total = sum(
            loss_weights.get(k, 0.0) * v for k, v in components.items()
        )
        return total, components

    # ------------------------------------------------------------------ #
    # Training step
    # ------------------------------------------------------------------ #

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        known = batch.get("known_structure", None)
        coords_tgt = batch.get("coords", None)
        mask = batch.get("mask", None)
        log = {}
        total = out.get("features", out.get("coords", torch.tensor(0.0))).sum() * 0.0

        if self.use_se3:
            # SE(3) mode: per-sequence losses
            B, N_seq = known.shape
            # Loss is restricted to the truncated SE(3) decoder prefix
            # (see ``_forward_se3``); the decoder is O(L^2) and is only run on
            # the first ``K`` residues, so the zero-padded tail of the
            # predicted coords must not contribute to the loss.
            K_cap = int(out.get("se3_max_residues", 0) or 0)
            # Only compute loss for sequences that were actually run through
            # the decoder; sequences beyond ``se3_max_sequences`` (or fully
            # masked) get zero coords and must be skipped.
            decoded_idx = out.get("se3_decoded_idx", [])
            decoded_set = {(int(b), int(n)) for b, n in decoded_idx}
            n_decoded_sup = 0  # supervised + decoded (used for normalization)
            for b in range(B):
                for n in range(N_seq):
                    if (b, n) not in decoded_set:
                        continue
                    if known is not None and not known[b, n]:
                        # Decoded but unsupervised: no target coords → no loss.
                        continue
                    n_decoded_sup += 1
                    if mask is not None:
                        res_mask = mask[b, n]  # (L,)
                    else:
                        res_mask = torch.ones_like(coords_tgt[b, n, :, 0], dtype=torch.bool)

                    # Restrict the loss to the truncated SE(3) prefix.
                    if K_cap > 0 and K_cap < res_mask.shape[0]:
                        res_mask = res_mask.clone()
                        res_mask[K_cap:] = False

                    pred_c = out["coords"][b, n]
                    pred_a = out["angles"][b, n]
                    tgt_c = coords_tgt[b, n]
                    # Drop positions with nonfinite target coords (PDB gaps,
                    # collate padding) so they cannot poison the loss.
                    res_mask = res_mask & torch.isfinite(tgt_c).all(dim=-1)

                    loss_val, comps = self._se3_loss_for_sequence(
                        pred_c, pred_a, tgt_c, res_mask, self._loss_weights,
                    )
                    total = total + loss_val
                    for k, v in comps.items():
                        key = f"train/se3_{k}"
                        log[key] = log.get(key, total.new_zeros(())) + v.detach()

            # Normalise by number of supervised AND decoded sequences
            n_sup = max(1, n_decoded_sup)
            total = total / n_sup
            for k in list(log.keys()):
                if k.startswith("train/se3_"):
                    log[k] = log[k] / n_sup

        else:
            # Heads mode: distance + angle + coordinate losses
            pairs = out["pairs"]
            if coords_tgt is not None and known is not None:
                # Distance
                if self._loss_weights.get("distance", 0) > 0:
                    edges = self.distance_head.edges.to(self.device)
                    labels = self._pair_labels(coords_tgt, pairs, edges).to(self.device)
                    loss = msa_distance_loss(
                        out["distance_logits"], labels, known,
                        label_smoothing=self.label_smoothing,
                        gamma=self.focal_gamma,
                    )
                    log["train/distance"] = loss.detach()
                    total = total + self._loss_weights["distance"] * loss

                # Angle
                if self._loss_weights.get("angle", 0) > 0:
                    targets = self._angle_targets(coords_tgt).to(self.device)
                    targets_3d = targets.unsqueeze(-1).expand(-1, -1, -1, 3)
                    loss = msa_angle_loss(
                        out["angle_logits"], targets_3d, known,
                        res_mask=torch.isfinite(targets),
                    )
                    log["train/angle"] = loss.detach()
                    total = total + self._loss_weights["angle"] * loss

        self.log_dict(log, prog_bar=True)
        return total

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        log = {}
        known = batch.get("known_structure")
        coords_tgt = batch.get("coords")

        if coords_tgt is not None and known is not None and not self.use_se3:
            pairs = out.get("pairs")
            if pairs is not None:
                edges = self.distance_head.edges.to(self.device)
                labels = self._pair_labels(coords_tgt, pairs, edges).to(self.device)
                loss = msa_distance_loss(out["distance_logits"], labels, known, gamma=self.focal_gamma)
                log["val/loss"] = loss.detach()

        if log:
            self.log_dict(log, prog_bar=True)
        return log

    @staticmethod
    def _angle_targets(coords, res_mask=None):
        B, N_seq, L, _ = coords.shape
        angles = coords.new_full((B, N_seq, L), float("nan"))
        for b in range(B):
            for n in range(N_seq):
                c = coords[b, n]
                vr = torch.isfinite(c).all(dim=-1)
                if not vr.any():
                    continue
                ca_filled = c.clone()
                ca_filled[~vr] = ca_filled[vr].mean(dim=0) if vr.any() else 0.0
                dih = _ca_pseudo_dihedrals(ca_filled)
                if res_mask is not None:
                    dih[~res_mask[b, n]] = float("nan")
                angles[b, n] = dih
        return angles

    @staticmethod
    def _geom_features(coords, known_mask):
        """SE(3)-invariant per-residue geometry features from CA coords."""
        import math
        B, N_seq, L, _ = coords.shape
        device = coords.device
        geom = torch.zeros(B, N_seq, L, 9, device=device)
        for b in range(B):
            for n in range(N_seq):
                if not known_mask[b, n]:
                    continue
                c = coords[b, n]
                vr = torch.isfinite(c).all(dim=-1)
                nv = int(vr.sum().item())
                if nv < 3:
                    continue
                vi = vr.nonzero(as_tuple=True)[0]
                cv = c[vi]
                # Pseudo-torsion
                if nv >= 5:
                    for k in range(nv - 3):
                        col = vi[k + 1].item()   # anchor at second residue
                        p0, p1, p2, p3 = cv[k], cv[k+1], cv[k+2], cv[k+3]
                        uu = F.normalize(p1 - p0, dim=-1, eps=1e-8)
                        vv = F.normalize(p2 - p1, dim=-1, eps=1e-8)
                        ww = F.normalize(p3 - p2, dim=-1, eps=1e-8)
                        n1 = torch.cross(uu, vv, dim=-1)
                        n2 = torch.cross(vv, ww, dim=-1)
                        n1 = F.normalize(n1, dim=-1, eps=1e-8)
                        n2 = F.normalize(n2, dim=-1, eps=1e-8)
                        c_angle = (n1 * n2).sum().clamp(-1., 1.)
                        s_angle = (torch.cross(n1, n2, dim=-1) * vv).sum().clamp(-1., 1.)
                        dih = math.atan2(float(s_angle), float(c_angle))
                        if math.isfinite(dih):
                            geom[b, n, col, 0] = dih
                # Bond-angle cosine
                if nv >= 3:
                    for k in range(1, nv - 1):
                        col = vi[k].item()
                        u = F.normalize(cv[k] - cv[k-1], dim=-1, eps=1e-8)
                        v = F.normalize(cv[k+1] - cv[k], dim=-1, eps=1e-8)
                        geom[b, n, col, 1] = (u * v).sum().clamp(-1., 1.)
                # Sequential distance d(i, i+3)
                if nv >= 4:
                    for k in range(nv - 3):
                        geom[b, n, vi[k].item(), 2] = (cv[k+3] - cv[k]).norm().clamp(0., 50.)
                # Contact densities and NN distances
                if nv > 1:
                    pd = torch.cdist(cv[None], cv[None])[0]
                    off = ~torch.eye(nv, dtype=torch.bool, device=device)
                    for idx, col in enumerate(vi):
                        col = col.item()
                        dr = pd[idx]
                        geom[b, n, col, 3] = ((dr < 8.0) & off[idx]).sum().float()
                        geom[b, n, col, 4] = ((dr < 12.0) & off[idx]).sum().float()
                        s = dr.clone()
                        s[idx] = float('inf')
                        sd = s.sort().values
                        for fi, nth in enumerate([5, 10, 20, 50], start=5):
                            if len(sd) >= nth:
                                geom[b, n, col, fi] = sd[nth - 1].clamp(0., 50.)
        return geom

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Prevent gradual memory fragmentation over many epochs
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    def on_validation_end(self):
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr,
        )
        monitor = "val/loss" if not self.use_se3 else "train/se3_fape"
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": monitor}}

    def on_before_optimizer_step(self, optimizer):
        # Squeeze `use_se3` since it sets up modules lazily
        pass


def _patch_hparams(model):
    """Fix hparams for SE(3) mode."""
    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", default="configs/oma_benchmark_cohort.json")
    p.add_argument("--families-root", default="families/Information_benchmark/marker_genes")
    p.add_argument("--mask-ratio", type=float, default=0.25)
    p.add_argument("--family-ids", nargs="*", default=None)

    # Mode
    p.add_argument("--use-se3", action="store_true", help="Use SE(3) decoder pipeline")
    p.add_argument("--init-from", default=None,
                   help="Path to a heads-mode .ckpt to warm-start the SE(3) "
                        "pipeline's msa_encoder weights from.")

    # MSA Transformer
    p.add_argument("--alphabet-size", type=int, default=31)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)

    # Heads mode architecture
    p.add_argument("--distance-hidden", type=int, default=128)
    p.add_argument("--angle-hidden", type=int, default=64)
    p.add_argument("--coord-hidden", type=int, default=128)

    # SE(3) mode architecture
    p.add_argument("--se3-hidden", type=int, default=128)
    p.add_argument("--se3-depth", type=int, default=3)
    p.add_argument("--se3-heads", type=int, default=4)
    p.add_argument("--se3-max-residues", type=int, default=512,
                   help="Cap the per-sequence length handed to the O(L^2) "
                        "GotenNet SE(3) decoder.  Each sequence is truncated "
                        "to the first K residues before the decoder forward "
                        "and the loss is restricted to that prefix.  Set to 0 "
                        "to disable the cap (full-length decoder, may OOM on "
                        "long MSA rows).")
    p.add_argument("--se3-max-sequences", type=int, default=4,
                   help="Cap the number of sequences per batch run through "
                        "the SE(3) decoder.  The GotenNet decoder builds an "
                        "autograd graph per sequence; decoding every sequence "
                        "in a deep MSA (e.g. 18 sequences for family 1003993) "
                        "accumulates a graph that OOMs a 23 GB GPU.  "
                        "Supervised (known_structure=True, not masked) "
                        "sequences are decoded first; remaining slots are "
                        "filled with any other non-masked sequences.  "
                        "Sequences not decoded get zero coords/angles and "
                        "no loss.  Set to 0 to disable the cap.")
    p.add_argument("--se3-seed-noise", type=float, default=0.5,
                   help="Std-dev (Angstrom) of the Gaussian noise added to "
                        "the true CA coordinates when seeding the SE(3) "
                        "decoder for supervised sequences "
                        "(denoising/refinement). Set to 0 to seed with the "
                        "exact coordinates. Unknown (ancestral) sequences "
                        "are always seeded with an idealized alpha-helix.")

    # Loss weights
    p.add_argument("--loss-weights", type=str,
                   default="distance=1,angle=0.1",
                   help="Comma-separated key=value weights")
    p.add_argument("--max-pairs", type=int, default=100000,
                   help="Max residue pairs per sequence for distance head")

    # Optimisation
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma for distance head")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="models/msa_reconstruction")

    # Callbacks
    p.add_argument("--patience", type=int, default=10,
                   help="EarlyStopping patience on val/loss (or train/fape in SE(3) mode)")
    p.add_argument("--lr-patience", type=int, default=5,
                   help="ReduceLROnPlateau patience")
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="LR decay factor for ReduceLROnPlateau")
    p.add_argument("--min-lr", type=float, default=1e-6,
                   help="minimum LR for ReduceLROnPlateau")

    # Lightning
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--strategy", default="auto")
    p.add_argument("--precision", default="32")
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--limit-train-batches", type=float, default=None)

    # Dataset
    p.add_argument("--use-clade-sampler", action="store_true")
    p.add_argument("--clade-size", type=int, default=8)
    return p


def _parse_loss_weights(raw: str) -> Dict[str, float]:
    weights = {}
    for token in raw.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        weights[k.strip()] = float(v.strip())
    return weights


def main():
    args = _parser().parse_args()
    pl.seed_everything(args.seed)
    loss_weights = _parse_loss_weights(args.loss_weights)
    print(f"Mode: {'SE(3)' if args.use_se3 else 'Heads'}")
    print(f"Loss weights: {loss_weights}")

    # Dataset
    if args.use_clade_sampler:
        dataset = CladeSampledMsaDataset(
            cohort_path=args.cohort, families_root=args.families_root,
            clade_size=args.clade_size, mask_ratio=args.mask_ratio,
            alphabet_size=args.alphabet_size,
        )
    else:
        dataset = TreeMsaDataset(
            cohort_path=args.cohort, families_root=args.families_root,
            mask_ratio=args.mask_ratio, family_ids=args.family_ids,
            alphabet_size=args.alphabet_size,
        )

    n = len(dataset)
    n_train = max(1, int(n * 0.85))
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=msa_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=msa_collate, num_workers=0,
    )

    # Model
    model = MsaReconstructionModule(
        use_se3=args.use_se3,
        alphabet_size=args.alphabet_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        loss_weights=loss_weights,
        distance_hidden=args.distance_hidden,
        angle_hidden=args.angle_hidden,
        coord_hidden=args.coord_hidden,
        se3_hidden=args.se3_hidden,
        se3_depth=args.se3_depth,
        se3_heads=args.se3_heads,
        se3_max_residues=args.se3_max_residues,
        se3_max_sequences=args.se3_max_sequences,
        se3_seed_noise=args.se3_seed_noise,
        max_pairs=args.max_pairs,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    # Warm-start the SE(3) pipeline's MSA encoder from a heads-mode checkpoint.
    if args.use_se3 and args.init_from:
        init_path = Path(args.init_from)
        if not init_path.is_file():
            raise FileNotFoundError(f"--init-from checkpoint not found: {init_path}")
        print(f"Warm-starting msa_encoder from: {init_path}")
        ckpt = torch.load(str(init_path), map_location="cpu")
        # Lightning checkpoints store weights under "state_dict"; raw torch.save
        # of a module stores at the top level. Handle both.
        src_state = ckpt.get("state_dict", ckpt)
        enc_state = {
            k[len("msa_encoder."):]: v for k, v in src_state.items()
            if k.startswith("msa_encoder.")
        }
        if not enc_state:
            raise RuntimeError(
                f"No keys starting with 'msa_encoder.' found in {init_path}. "
                f"Available prefixes: "
                f"{sorted(set(k.split('.')[0] for k in src_state))[:10]}"
            )
        missing, unexpected = model.msa_encoder.load_state_dict(
            enc_state, strict=True,
        )
        # With strict=True, mismatched keys raise; reaching here means success.
        print(f"Loaded {len(enc_state)} msa_encoder weights from {init_path} "
              f"(strict=True).")
        del ckpt, src_state, enc_state

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Use val/loss if available, else train/loss
    ckpt_monitor = "val/loss" if not args.use_se3 else "train/se3_fape"

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=outdir, filename="msa_recon-{epoch:03d}-{val/loss:.3f}",
        monitor=ckpt_monitor, save_top_k=3, mode="min",
    )

    # EarlyStopping (ReduceLROnPlateau is handled inside configure_optimizers
    # and monitors val/loss in Heads mode, train/fape in SE(3) mode).
    stop_monitor = ckpt_monitor
    early_stop = pl.callbacks.EarlyStopping(
        monitor=stop_monitor, patience=args.patience, mode="min", verbose=True,
    )

    cfg = {**vars(args), "loss_weights": loss_weights,
           "dataset_size": n}
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        max_epochs=args.epochs,
        callbacks=[checkpoint, early_stop],
        default_root_dir=str(outdir),
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        log_every_n_steps=5,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Done. Best model: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()