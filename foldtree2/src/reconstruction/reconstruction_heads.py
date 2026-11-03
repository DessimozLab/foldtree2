"""Output heads for MSA-based reconstruction: distance distributions and contacts.

These heads sit on top of the MSA Transformer features (per-sequence,
per-residue representations) and produce distance/contact predictions for
every residue pair in every sequence of the MSA.  They reuse the binning
conventions from ``foldtree2.src.distance.distance_restraints``.
"""

from typing import Optional, Tuple

import math
import torch
from torch import nn
import torch.nn.functional as F

from foldtree2.src.distance.distance_restraints import DISTANCE_EDGES, validate_edges


class MsaDistanceHead(nn.Module):
    """Predict binned pairwise distance distributions for all sequences in an MSA.

    For each sequence independently, the head applies a pair-MLP to every
    valid residue pair (i, j) and outputs logits over distance bins.

    Args:
        embed_dim: Feature dimension from the MSA Transformer.
        hidden: Hidden dimension of the pair MLP.
        edges: Distance bin boundaries (Angstrom).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden: int = 128,
        edges=DISTANCE_EDGES,
        dropout: float = 0.1,
    ):
        super().__init__()
        edges = torch.tensor(edges, dtype=torch.float32)
        validate_edges(edges)
        if (edges == 8.0).sum().item() != 1:
            raise ValueError("Distance bins must have an exact 8-A contact boundary")
        self.register_buffer("edges", edges)
        self.contact_bins = int((edges <= 8.0).sum().item())
        self.n_bins = len(edges) + 1

        self.pair_mlp = nn.Sequential(
            # Two residue features, two normalised sequence-separation
            # channels (linear and log), and one normalised dot product of the
            # two residue features — without a relative term the head cannot
            # tell "same environment" from "opposite ends of the chain".
            nn.Linear(embed_dim * 2 + 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_bins),
        )

    def forward(
        self,
        features: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        pairs: Optional[torch.Tensor] = None,
        chunk_size: int = 2048,
    ) -> torch.Tensor:
        """Predict distance bin logits for each sequence in the MSA.

        Args:
            features: (B, N_seq, L, D) per-residue features.
            seq_mask: (B, N_seq) bool; True where sequence is valid.
            pairs: (P, 2) pair indices. If None, all i<j with sep>=12.
            chunk_size: Chunk size for memory-efficient evaluation.

        Returns:
            logits: (B, N_seq, P, n_bins) — gradient-preserving.
        """
        B, N_seq, L, D = features.shape
        if pairs is None:
            idx = torch.arange(L, device=features.device)
            sep = idx[None] - idx[:, None]
            pairs = (sep.abs() >= 12).nonzero()
        else:
            pairs = pairs.to(features.device)
        P = pairs.shape[0]
        log_L = math.log(max(L, 2))

        # Collect chunk outputs then stack to preserve gradient through the MLP.
        # Using a list avoids the ``new_zeros + in-place slice`` pattern which
        # would break autograd (in-place assignment to a non-leaf does not
        # propagate grad).
        # Subsample pairs for memory efficiency when max_pairs is set.
        all_out: list = []

        for b in range(B):
            for n in range(N_seq):
                if seq_mask is not None and not seq_mask[b, n]:
                    all_out.append(features.new_zeros(P, self.n_bins))
                    continue
                feat = features[b, n]  # (L, D)
                chunk_list = []
                for c_start in range(0, P, chunk_size):
                    c_end = min(c_start + chunk_size, P)
                    cp = pairs[c_start:c_end]
                    i, j = cp[:, 0], cp[:, 1]
                    sep = (j - i).abs().float()
                    sep_feat = torch.stack(
                        [sep / max(L, 2), torch.log(sep.clamp(min=1.0)) / log_L],
                        dim=-1,
                    )
                    fi, fj = feat[i], feat[j]
                    rel = ((fi * fj).sum(-1) / D).unsqueeze(-1)
                    pair_feat = torch.cat([fi, fj, sep_feat, rel], dim=-1)
                    chunk_list.append(self.pair_mlp(pair_feat))
                all_out.append(torch.cat(chunk_list, dim=0))

        return torch.stack(all_out, dim=0).view(B, N_seq, P, self.n_bins)

    def distance_labels(
        self, coords: torch.Tensor, pairs: torch.Tensor
    ) -> torch.Tensor:
        """Convert native coordinates into distance bin labels."""
        from foldtree2.src.distance.distance_restraints import distance_labels as _dl
        return _dl(coords, pairs, self.edges)


class MsaContactHead(nn.Module):
    """Predict contact probabilities (binary) per sequence in the MSA."""

    def __init__(self, embed_dim: int = 256, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.pair_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        pairs: Optional[torch.Tensor] = None,
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        """Predict contact logits (binary). Returns (B, N_seq, P)."""
        B, N_seq, L, D = features.shape
        if pairs is None:
            idx = torch.arange(L, device=features.device)
            sep = idx[None] - idx[:, None]
            pairs = (sep.abs() >= 12).nonzero()
        else:
            pairs = pairs.to(features.device)
        P = pairs.shape[0]

        all_out: list = []
        for b in range(B):
            for n in range(N_seq):
                if seq_mask is not None and not seq_mask[b, n]:
                    all_out.append(features.new_zeros(P))
                    continue
                feat = features[b, n]
                chunk_list = []
                for c_start in range(0, P, chunk_size):
                    c_end = min(c_start + chunk_size, P)
                    cp = pairs[c_start:c_end]
                    i, j = cp[:, 0], cp[:, 1]
                    chunk_list.append(self.pair_mlp(
                        torch.cat([feat[i], feat[j]], dim=-1)
                    ).squeeze(-1))
                all_out.append(torch.cat(chunk_list, dim=0))

        return torch.stack(all_out, dim=0).view(B, N_seq, P)


class MsaAngleHead(nn.Module):
    """Predicts backbone torsion angles (phi, psi, omega) from MSA features.

    Outputs paired (sin, cos) logits per residue so the angle prediction is
    well-behaved near +/-pi.  The loss uses the SE(3) circular-distance
    convention.
    """

    def __init__(self, embed_dim: int = 256, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),  # sin/cos for phi, psi, omega
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict backbone angle logits.

        Args:
            features: (B, N_seq, L, D) from MSA Transformer.

        Returns:
            logits: (B, N_seq, L, 6) paired (sin, cos) logits.
        """
        return self.mlp(features)


class MsaCoordinateHead(nn.Module):
    """Predicts CA coordinates per residue from MSA features.

    A lightweight regression head — not an IPA module.  Produces rough
    coordinate estimates that can be refined by downstream SE(3) modules.
    The coordinates are predicted in a local frame relative to the residue
    index (like a simple polymer model), so the loss is positional.
    """

    def __init__(self, embed_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),  # (dx, dy, dz) residual step
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict per-residue coordinate deltas.

        Args:
            features: (B, N_seq, L, D)

        Returns:
            delta: (B, N_seq, L, 3) coordinate deltas.
        """
        return self.mlp(features)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def msa_angle_loss(
    angle_logits: torch.Tensor,
    target_angles: torch.Tensor,
    known_mask: torch.Tensor,
    res_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Circular smooth-L1 angle loss (phi, psi, omega) with partial supervision.

    Args:
        angle_logits: (B, N_seq, L, 6) paired (sin, cos) per angle.
        target_angles: (B, N_seq, L, 3) true angles in radians.
        known_mask: (B, N_seq) bool; True = supervised sequences.
        res_mask: (B, N_seq, L) bool; True = valid residue positions.

    Returns:
        scalar loss over valid sequences and residues.
    """
    B, N_seq, L, _ = angle_logits.shape
    total_loss = angle_logits.sum() * 0.0
    n_valid = 0

    for b in range(B):
        for n in range(N_seq):
            if not known_mask[b, n]:
                continue
            # Normalise sin/cos pairs to unit circle
            pairs = angle_logits[b, n].view(L, 3, 2)
            pairs = F.normalize(pairs, dim=-1, eps=1e-8)
            pred_angles = torch.atan2(pairs[..., 0], pairs[..., 1])  # (L, 3)

            targets = target_angles[b, n]  # (L, 3)
            if res_mask is not None:
                valid_res = res_mask[b, n]
                if not valid_res.any():
                    continue
                pred_angles = pred_angles[valid_res]
                targets = targets[valid_res]

            # Circular difference in [-pi, pi]
            delta = (pred_angles - targets + math.pi) % (2 * math.pi) - math.pi
            loss = F.smooth_l1_loss(delta, torch.zeros_like(delta))
            total_loss = total_loss + loss
            n_valid += 1

    return total_loss / max(n_valid, 1)


def msa_coordinate_loss(
    coord_pred: torch.Tensor,
    coord_target: torch.Tensor,
    known_mask: torch.Tensor,
    res_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Coordinate MSE loss on CA positions with partial supervision.

    Uses smooth L1 loss (Huber) which is more robust to outliers than MSE.

    Args:
        coord_pred: (B, N_seq, L, 3) predicted CA positions (deltas or positions).
        coord_target: (B, N_seq, L, 3) true CA positions; NaN = no target.
        known_mask: (B, N_seq) bool; True = supervised sequences.
        res_mask: (B, N_seq, L) bool; True = valid residue positions.

    Returns:
        scalar loss over valid positions.
    """
    B, N_seq, L, _ = coord_pred.shape
    total_loss = coord_pred.sum() * 0.0
    n_valid = 0

    for b in range(B):
        for n in range(N_seq):
            if not known_mask[b, n]:
                continue
            c_pred = coord_pred[b, n]  # (L, 3)
            c_tgt = coord_target[b, n]  # (L, 3)
            valid = torch.isfinite(c_tgt).all(dim=-1)  # (L,) bool
            if res_mask is not None:
                valid = valid & res_mask[b, n]
            if not valid.any():
                continue
            loss = F.smooth_l1_loss(c_pred[valid], c_tgt[valid])
            total_loss = total_loss + loss
            n_valid += 1

    return total_loss / max(n_valid, 1)


def msa_distance_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    known_mask: torch.Tensor,
    label_smoothing: float = 0.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal cross-entropy loss for binned distance predictions with masking.

    Labels set to -1 indicate invalid pairs (gap positions) and are ignored.
    The focal term ``(1 - p_t)^gamma`` down-weights confident predictions,
    focusing on the minority short-range pairs critical for structure.

    Args:
        logits: (B, N_seq, P, n_bins) predicted logits.
        labels: (B, N_seq, P) ground-truth bin indices; -1 = ignore.
        known_mask: (B, N_seq) bool; True = supervised.
        label_smoothing: Label smoothing epsilon.
        gamma: Focal focusing parameter. gamma=0 = CE.

    Returns:
        loss: Scalar loss (mean over valid supervised pairs).
    """
    B, N_seq, P, n_bins = logits.shape
    total_loss = logits.sum() * 0.0
    n_valid = 0

    for b in range(B):
        for n in range(N_seq):
            if not known_mask[b, n]:
                continue
            lbl = labels[b, n]
            valid = lbl >= 0
            if not valid.any():
                continue
            logp = F.log_softmax(logits[b, n, valid], dim=-1)
            if label_smoothing > 0.0:
                target = F.one_hot(lbl[valid], n_bins).float()
                target = target * (1.0 - label_smoothing) + label_smoothing / n_bins
                ce = -(target * logp).sum(-1)
            else:
                ce = F.nll_loss(logp, lbl[valid], reduction="none")
            if gamma > 0.0:
                pt = logp.gather(-1, lbl[valid, None]).exp().squeeze(-1)
                focal = (1.0 - pt).pow(gamma).detach().clamp_min(1e-8)
                loss = (ce * focal).mean()
            else:
                loss = ce.mean()
            total_loss = total_loss + loss
            n_valid += 1

    return total_loss / max(n_valid, 1)