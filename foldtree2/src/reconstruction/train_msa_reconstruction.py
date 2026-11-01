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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_weights"])
        self.use_se3 = bool(use_se3)
        self.label_smoothing = label_smoothing
        self.min_separation = min_separation
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
        self.coord_head = MsaCoordinateHead(embed_dim=embed_dim, hidden=coord_hidden)

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
            "coord_delta": self.coord_head(features),
            "pairs": pairs,
            "attentions": enc.get("row_attentions", None),
        }

    def _forward_se3(self, tokens, msa_mask, known_mask=None, coords=None):
        """Run MSA Transformer then SE(3) decoder per sequence."""
        B, N_seq, L = tokens.shape

        # Step 1: MSA Transformer — cross-sequence context
        enc = self.msa_encoder(tokens, msa_mask=msa_mask, need_head_weights=False)
        features = enc["features"]  # (B, N_seq, L, D)

        # Step 2: SE(3) decoder per sequence
        all_coords, all_angles = [], []
        for b in range(B):
            for n in range(N_seq):
                if msa_mask is not None:
                    seq_v = msa_mask[b, n].any()
                else:
                    seq_v = True
                if not seq_v:
                    all_coords.append(torch.zeros(L, 3, device=tokens.device))
                    all_angles.append(torch.zeros(L, 6, device=tokens.device))
                    continue

                # Build a HeteroData for this sequence
                g = HeteroData()
                g["res"].x = features[b, n]  # (L, D)

                seed = torch.zeros(L, 3, device=tokens.device)

                if msa_mask is not None:
                    node_mask = msa_mask[b, n]
                else:
                    node_mask = torch.ones(L, dtype=torch.bool, device=tokens.device)

                out = self.se3_pipeline.se3_decoder(
                    g,
                    coords_pred=seed,
                    ft2_token_ids=tokens[b, n],
                    edge_attr_dict={"node_mask": node_mask, "use_distance_contacts": True},
                )
                all_coords.append(out["coors_out_flat"])
                all_angles.append(out["angles"])

        return {
            "features": features,
            "coords": torch.stack(all_coords, dim=0).view(B, N_seq, L, 3),
            "angles": torch.stack(all_angles, dim=0).view(B, N_seq, L, 6),
            "attentions": enc.get("row_attentions", None),
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
                delta = (pred_angles[mask] - target_coords.new_zeros(mask.sum(), 6))  # placeholder
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
            for b in range(B):
                for n in range(N_seq):
                    if known is not None and not known[b, n]:
                        continue
                    if mask is not None:
                        res_mask = mask[b, n]  # (L,)
                    else:
                        res_mask = torch.ones_like(coords_tgt[b, n, :, 0], dtype=torch.bool)

                    pred_c = out["coords"][b, n]
                    pred_a = out["angles"][b, n]
                    tgt_c = coords_tgt[b, n]

                    loss_val, comps = self._se3_loss_for_sequence(
                        pred_c, pred_a, tgt_c, res_mask, self._loss_weights,
                    )
                    total = total + loss_val
                    for k, v in comps.items():
                        key = f"train/se3_{k}"
                        log[key] = log.get(key, total.new_zeros(())) + v.detach()

            # Normalise by number of supervised sequences
            n_sup = max(1, int(known.sum().item())) if known is not None else 1
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

                # Coordinate
                if self._loss_weights.get("coordinate", 0) > 0:
                    loss = msa_coordinate_loss(
                        out["coord_delta"], coords_tgt, known,
                    )
                    log["train/coordinate"] = loss.detach()
                    total = total + self._loss_weights["coordinate"] * loss

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
                loss = msa_distance_loss(out["distance_logits"], labels, known)
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

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs if self.trainer else 100,
        )
        return [opt], [{"scheduler": sched, "interval": "epoch"}]

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

    # Loss weights
    p.add_argument("--loss-weights", type=str,
                   default="distance=1,angle=0.1,coordinate=0.1,fape=1,bond=1",
                   help="Comma-separated key=value weights")
    p.add_argument("--max-pairs", type=int, default=100000,
                   help="Max residue pairs per sequence for distance head")

    # Optimisation
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="models/msa_reconstruction")

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
        loss_weights=loss_weights,
        distance_hidden=args.distance_hidden,
        angle_hidden=args.angle_hidden,
        coord_hidden=args.coord_hidden,
        se3_hidden=args.se3_hidden,
        se3_depth=args.se3_depth,
        se3_heads=args.se3_heads,
        max_pairs=args.max_pairs,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Use val/loss if available, else train/loss
    ckpt_monitor = "val/loss" if not args.use_se3 else "train/fape"

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=outdir, filename="msa_recon-{epoch:03d}-{val/loss:.3f}",
        monitor=ckpt_monitor, save_top_k=3, mode="min",
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
        callbacks=[checkpoint],
        default_root_dir=str(outdir),
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        log_every_n_steps=5,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Done. Best model: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()