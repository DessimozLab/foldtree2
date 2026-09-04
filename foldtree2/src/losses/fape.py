"""Minimal geometry loss stack used by FoldTree2 training loops.

This module intentionally contains only the core utilities needed by current
training and notebook workflows:
- Quaternion/rotation conversions
- RT -> coordinate reconstruction
- Pairwise local-frame FAPE
- Delta-coordinate loss
- lDDT and differentiable lDDT losses

The API is kept stable for existing imports in training scripts.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _normalize_quaternion(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternions on the last axis."""
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)


def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternions in (w, x, y, z) format to rotation matrices.

    Args:
        quat: Tensor of shape (..., 4)

    Returns:
        Tensor of shape (..., 3, 3)
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (..., 4), got {tuple(quat.shape)}")

    quat = _normalize_quaternion(quat)
    w, x, y, z = quat.unbind(dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)

    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)

    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def rotation_matrix_to_quaternion(rot_matrices: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions in (w, x, y, z) format.

    Args:
        rot_matrices: Tensor of shape (..., 3, 3)

    Returns:
        Tensor of shape (..., 4)
    """
    if rot_matrices.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (..., 3, 3), got {tuple(rot_matrices.shape)}")

    flat = rot_matrices.reshape(-1, 3, 3)
    quat = torch.zeros((flat.shape[0], 4), dtype=flat.dtype, device=flat.device)

    r00 = flat[:, 0, 0]
    r11 = flat[:, 1, 1]
    r22 = flat[:, 2, 2]
    trace = r00 + r11 + r22

    m1 = trace > 0
    if m1.any():
        s = torch.sqrt(torch.clamp(trace[m1] + 1.0, min=0.0)) * 2.0
        quat[m1, 0] = 0.25 * s
        quat[m1, 1] = (flat[m1, 2, 1] - flat[m1, 1, 2]) / s
        quat[m1, 2] = (flat[m1, 0, 2] - flat[m1, 2, 0]) / s
        quat[m1, 3] = (flat[m1, 1, 0] - flat[m1, 0, 1]) / s

    m2 = (~m1) & (r00 > r11) & (r00 > r22)
    if m2.any():
        s = torch.sqrt(torch.clamp(1.0 + r00[m2] - r11[m2] - r22[m2], min=0.0)) * 2.0
        quat[m2, 0] = (flat[m2, 2, 1] - flat[m2, 1, 2]) / s
        quat[m2, 1] = 0.25 * s
        quat[m2, 2] = (flat[m2, 0, 1] + flat[m2, 1, 0]) / s
        quat[m2, 3] = (flat[m2, 0, 2] + flat[m2, 2, 0]) / s

    m3 = (~m1) & (~m2) & (r11 > r22)
    if m3.any():
        s = torch.sqrt(torch.clamp(1.0 + r11[m3] - r00[m3] - r22[m3], min=0.0)) * 2.0
        quat[m3, 0] = (flat[m3, 0, 2] - flat[m3, 2, 0]) / s
        quat[m3, 1] = (flat[m3, 0, 1] + flat[m3, 1, 0]) / s
        quat[m3, 2] = 0.25 * s
        quat[m3, 3] = (flat[m3, 1, 2] + flat[m3, 2, 1]) / s

    m4 = (~m1) & (~m2) & (~m3)
    if m4.any():
        s = torch.sqrt(torch.clamp(1.0 + r22[m4] - r00[m4] - r11[m4], min=0.0)) * 2.0
        quat[m4, 0] = (flat[m4, 1, 0] - flat[m4, 0, 1]) / s
        quat[m4, 1] = (flat[m4, 0, 2] + flat[m4, 2, 0]) / s
        quat[m4, 2] = (flat[m4, 1, 2] + flat[m4, 2, 1]) / s
        quat[m4, 3] = 0.25 * s

    quat = _normalize_quaternion(quat)
    return quat.reshape(*rot_matrices.shape[:-2], 4)


def reconstruct_positions(
    R: torch.Tensor,
    T: torch.Tensor,
    batch_idx: Optional[torch.Tensor] = None,
    translation_frame: str = "global",
    include_origin: bool = True,
) -> torch.Tensor:
    """Reconstruct chain coordinates from RT predictions.

    Args:
        R: Rotations, shape (N, 3, 3)
        T: Translations, shape (N, 3)
        batch_idx: Optional chain indices per residue, shape (N,)
        translation_frame: "global" for direct cumsum or "local" for composed frames
        include_origin: If True, prepend origin for each chain
    """
    if translation_frame not in {"global", "local"}:
        raise ValueError(f"Unknown translation_frame={translation_frame}")

    def _single(R_s: torch.Tensor, T_s: torch.Tensor) -> torch.Tensor:
        if translation_frame == "global":
            coords = torch.cumsum(T_s, dim=0)
        else:
            curr_pos = torch.zeros(3, dtype=T_s.dtype, device=T_s.device)
            curr_R = torch.eye(3, dtype=T_s.dtype, device=T_s.device)
            out = []
            for i in range(T_s.shape[0]):
                curr_pos = curr_pos + curr_R @ T_s[i]
                out.append(curr_pos.clone())
                curr_R = curr_R @ R_s[i]
            coords = torch.stack(out, dim=0) if out else T_s.new_zeros((0, 3))

        if include_origin:
            origin = torch.zeros((1, 3), dtype=T_s.dtype, device=T_s.device)
            return torch.cat([origin, coords], dim=0)
        return coords

    if batch_idx is None:
        return _single(R, T)

    outputs = []
    for b in torch.unique(batch_idx, sorted=True):
        mask = batch_idx == b
        outputs.append(_single(R[mask], T[mask]))

    if not outputs:
        return torch.zeros((0, 3), dtype=T.dtype, device=T.device)
    return torch.cat(outputs, dim=0)


def integrate_ca_steps(
    steps: Tensor,
    batch_idx: Optional[Tensor] = None,
    anchor: Optional[Tensor] = None,
) -> Tensor:
    """Integrate local CA step vectors into CA-like coordinates.

    ``steps`` is shaped ``(N, 3)``. The first step in each chain is treated as
    the anchor displacement and is ignored by default, so residue 0 starts at
    the origin unless ``anchor`` is provided. Residue ``i`` is reached by
    summing ``steps[1:i+1]``.
    """
    if steps.ndim != 2 or steps.shape[-1] != 3:
        raise ValueError(f"Expected steps with shape (N, 3), got {tuple(steps.shape)}")

    def _single(steps_s: Tensor, anchor_s: Optional[Tensor]) -> Tensor:
        if steps_s.shape[0] == 0:
            return steps_s.new_zeros((0, 3))
        coords = torch.zeros_like(steps_s)
        if anchor_s is not None:
            coords[0] = anchor_s.to(device=steps_s.device, dtype=steps_s.dtype)
        if steps_s.shape[0] > 1:
            coords[1:] = (coords[0].unsqueeze(0) + torch.cumsum(steps_s[1:], dim=0)).to(dtype=coords.dtype)
        return coords

    if batch_idx is None:
        return _single(steps, anchor)

    coords = torch.zeros_like(steps)
    for b in torch.unique(batch_idx, sorted=True):
        idx = torch.where(batch_idx == b)[0]
        if idx.numel() == 0:
            continue
        anchor_b = None
        if anchor is not None:
            anchor_b = anchor[int(b.item())] if anchor.ndim == 2 else anchor
        coords[idx] = _single(steps[idx], anchor_b).to(dtype=coords.dtype)
    return coords


def _validate_ca_frames(frames: Tensor, n_res: int) -> None:
    if frames.ndim != 3 or frames.shape[-2:] != (3, 3) or frames.shape[0] != n_res:
        raise ValueError(f"Expected frames with shape ({n_res}, 3, 3), got {tuple(frames.shape)}")


def ca_local_step_targets(
    true_ca: Tensor,
    frames: Tensor,
    batch_idx: Optional[Tensor] = None,
    frame_offset: str = "prev",
) -> tuple[Tensor, Tensor]:
    """Return CA step targets expressed in residue-local backbone frames.

    ``frames`` are rotation matrices whose columns are local axes in global
    coordinates. With row-vector arithmetic, global -> local is ``step @ R``.
    For the default ``frame_offset="prev"``, step ``i`` uses frame ``i-1``.
    """
    if frame_offset not in {"prev", "current"}:
        raise ValueError(f"Unknown frame_offset={frame_offset}")
    if true_ca.ndim != 2 or true_ca.shape[-1] != 3:
        raise ValueError(f"Expected true_ca with shape (N, 3), got {tuple(true_ca.shape)}")
    _validate_ca_frames(frames, true_ca.shape[0])

    global_steps, mask = _ca_step_targets(true_ca, batch_idx=batch_idx)
    frame_idx = torch.arange(true_ca.shape[0], device=true_ca.device)
    if frame_offset == "prev":
        if batch_idx is None:
            frame_idx = torch.clamp(frame_idx - 1, min=0)
        else:
            for b in torch.unique(batch_idx, sorted=True):
                idx = torch.where(batch_idx == b)[0]
                if idx.numel() > 1:
                    frame_idx[idx[1:]] = idx[:-1]
                if idx.numel() > 0:
                    frame_idx[idx[0]] = idx[0]

    target = torch.zeros_like(true_ca)
    if mask.any():
        target[mask] = torch.einsum("ni,nij->nj", global_steps[mask], frames[frame_idx[mask]]).to(dtype=target.dtype)
    return target, mask


def integrate_local_ca_steps(
    local_steps: Tensor,
    frames: Tensor,
    batch_idx: Optional[Tensor] = None,
    anchor: Optional[Tensor] = None,
    frame_offset: str = "prev",
) -> Tensor:
    """Integrate local CA step vectors using provided residue frames."""
    if frame_offset not in {"prev", "current"}:
        raise ValueError(f"Unknown frame_offset={frame_offset}")
    if local_steps.ndim != 2 or local_steps.shape[-1] != 3:
        raise ValueError(f"Expected local_steps with shape (N, 3), got {tuple(local_steps.shape)}")
    _validate_ca_frames(frames, local_steps.shape[0])

    def _single(steps_s: Tensor, frames_s: Tensor, anchor_s: Optional[Tensor]) -> Tensor:
        if steps_s.shape[0] == 0:
            return steps_s.new_zeros((0, 3))
        coords = torch.zeros_like(steps_s)
        if anchor_s is not None:
            coords[0] = anchor_s.to(device=steps_s.device, dtype=steps_s.dtype)
        for i in range(1, steps_s.shape[0]):
            frame_i = frames_s[i - 1] if frame_offset == "prev" else frames_s[i]
            global_step = torch.matmul(steps_s[i], frame_i.transpose(-1, -2))
            coords[i] = (coords[i - 1] + global_step).to(dtype=coords.dtype)
        return coords

    if batch_idx is None:
        return _single(local_steps, frames, anchor)

    coords = torch.zeros_like(local_steps)
    for b in torch.unique(batch_idx, sorted=True):
        idx = torch.where(batch_idx == b)[0]
        if idx.numel() == 0:
            continue
        anchor_b = None
        if anchor is not None:
            anchor_b = anchor[int(b.item())] if anchor.ndim == 2 else anchor
        coords[idx] = _single(local_steps[idx], frames[idx], anchor_b).to(dtype=coords.dtype)
    return coords


COARSE_BACKBONE_OFFSETS = {
    "ca": (0.0, 0.0, 0.0),
    "c": (1.52, 0.0, 0.0),
    "cb": (-0.60, -0.77, -1.18),
    "n": (-0.53, 1.36, 0.0),
}


def coarse_backbone_atoms_from_ca_frames(
    ca_coords: Tensor,
    frames: Tensor,
    atom_names: Sequence[str] = ("ca", "cb", "n"),
    atom_offsets: Optional[dict[str, Sequence[float]]] = None,
) -> Tensor:
    """Place coarse CA/CB/N atoms from CA origins and residue frames."""
    if ca_coords.ndim != 2 or ca_coords.shape[-1] != 3:
        raise ValueError(f"Expected ca_coords with shape (N, 3), got {tuple(ca_coords.shape)}")
    _validate_ca_frames(frames, ca_coords.shape[0])

    offsets_src = COARSE_BACKBONE_OFFSETS if atom_offsets is None else atom_offsets
    try:
        offsets = [offsets_src[name.lower()] for name in atom_names]
    except KeyError as exc:
        raise ValueError(f"Unknown coarse backbone atom name: {exc.args[0]}") from exc

    local_offsets = torch.tensor(offsets, dtype=ca_coords.dtype, device=ca_coords.device)
    global_offsets = torch.einsum("ak,njk->naj", local_offsets, frames)
    return ca_coords.unsqueeze(1) + global_offsets


def coarse_backbone_fape_loss(
    true_atoms: Tensor,
    pred_atoms: Tensor,
    true_frames: Tensor,
    pred_frames: Tensor,
    true_origins: Tensor,
    pred_origins: Tensor,
    batch: Optional[Tensor] = None,
    d_clamp: float = 10.0,
    reduction: str = "mean",
    pair_sample_size: Optional[int] = None,
) -> Tensor:
    """Frame-aligned point error over a coarse per-residue atom set."""
    if true_atoms.shape != pred_atoms.shape or true_atoms.ndim != 3 or true_atoms.shape[-1] != 3:
        raise ValueError(
            f"Expected matching atom tensors with shape (N, A, 3), got "
            f"true={tuple(true_atoms.shape)} pred={tuple(pred_atoms.shape)}"
        )
    _validate_ca_frames(true_frames, true_atoms.shape[0])
    _validate_ca_frames(pred_frames, pred_atoms.shape[0])
    if true_origins.shape != pred_origins.shape or true_origins.shape != (true_atoms.shape[0], 3):
        raise ValueError(
            f"Expected origins with shape ({true_atoms.shape[0]}, 3), got "
            f"true={tuple(true_origins.shape)} pred={tuple(pred_origins.shape)}"
        )

    def _single(
        true_atoms_s: Tensor,
        pred_atoms_s: Tensor,
        true_frames_s: Tensor,
        pred_frames_s: Tensor,
        true_origins_s: Tensor,
        pred_origins_s: Tensor,
    ) -> Tensor:
        if pair_sample_size is not None and pair_sample_size > 0 and pair_sample_size < true_atoms_s.shape[0] ** 2:
            sample_i = torch.randint(true_atoms_s.shape[0], (pair_sample_size,), device=true_atoms_s.device)
            sample_j = torch.randint(true_atoms_s.shape[0], (pair_sample_size,), device=true_atoms_s.device)
            true_diff = true_atoms_s[sample_j] - true_origins_s[sample_i, None, :]
            pred_diff = pred_atoms_s[sample_j] - pred_origins_s[sample_i, None, :]
            true_local = torch.einsum("kij,kaj->kai", true_frames_s[sample_i].transpose(-1, -2), true_diff)
            pred_local = torch.einsum("kij,kaj->kai", pred_frames_s[sample_i].transpose(-1, -2), pred_diff)
        else:
            true_diff = true_atoms_s.unsqueeze(0) - true_origins_s[:, None, None, :]
            pred_diff = pred_atoms_s.unsqueeze(0) - pred_origins_s[:, None, None, :]
            true_local = torch.einsum("imaj,ijk->imak", true_diff, true_frames_s)
            pred_local = torch.einsum("imaj,ijk->imak", pred_diff, pred_frames_s)
        error = torch.linalg.vector_norm(pred_local - true_local, dim=-1).clamp(max=d_clamp)
        if reduction == "mean":
            return error.mean()
        if reduction == "sum":
            return error.sum()
        if reduction == "none":
            return error
        raise ValueError(f"Unknown reduction: {reduction}")

    if batch is None:
        return _single(true_atoms, pred_atoms, true_frames, pred_frames, true_origins, pred_origins)

    losses = []
    for b in torch.unique(batch, sorted=True):
        idx = (batch == b).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        losses.append(
            _single(
                true_atoms[idx],
                pred_atoms[idx],
                true_frames[idx],
                pred_frames[idx],
                true_origins[idx],
                pred_origins[idx],
            )
        )
    if not losses:
        return torch.tensor(0.0, device=true_atoms.device, dtype=true_atoms.dtype)
    return torch.stack(losses).mean()


def _dihedral_angle(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor, eps: float = 1e-8) -> Tensor:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = F.normalize(b1, dim=-1, eps=eps)

    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1
    v = F.normalize(v, dim=-1, eps=eps)
    w = F.normalize(w, dim=-1, eps=eps)

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)
    return torch.atan2(y, x)


def backbone_dihedrals_from_n_ca_c(
    n_coords: Tensor,
    ca_coords: Tensor,
    c_coords: Tensor,
    batch: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Compute per-residue phi/psi/omega from backbone N/CA/C atom placements.

    Invalid chain endpoints are returned as zero angles and marked false in the
    returned mask.
    """
    if n_coords.shape != ca_coords.shape or n_coords.shape != c_coords.shape:
        raise ValueError(
            f"Expected N/CA/C tensors with matching shape, got "
            f"N={tuple(n_coords.shape)} CA={tuple(ca_coords.shape)} C={tuple(c_coords.shape)}"
        )
    if n_coords.ndim != 2 or n_coords.shape[-1] != 3:
        raise ValueError(f"Expected atom coordinates with shape (N, 3), got {tuple(n_coords.shape)}")

    angles = n_coords.new_zeros((n_coords.shape[0], 3))
    mask = torch.zeros((n_coords.shape[0], 3), dtype=torch.bool, device=n_coords.device)

    def _single(idx: Tensor) -> None:
        if idx.numel() < 2:
            return
        cur = idx[1:]
        prev = idx[:-1]
        angles[cur, 0] = _dihedral_angle(c_coords[prev], n_coords[cur], ca_coords[cur], c_coords[cur])
        mask[cur, 0] = True

        nxt = idx[1:]
        cur = idx[:-1]
        angles[cur, 1] = _dihedral_angle(n_coords[cur], ca_coords[cur], c_coords[cur], n_coords[nxt])
        angles[cur, 2] = _dihedral_angle(ca_coords[cur], c_coords[cur], n_coords[nxt], ca_coords[nxt])
        mask[cur, 1:] = True

    if batch is None:
        _single(torch.arange(n_coords.shape[0], device=n_coords.device))
        return angles, mask

    for b in torch.unique(batch, sorted=True):
        _single((batch == b).nonzero(as_tuple=True)[0])
    return angles, mask


def coarse_backbone_dihedrals_from_ca_frames(
    ca_coords: Tensor,
    frames: Tensor,
    batch: Optional[Tensor] = None,
    n_coords: Optional[Tensor] = None,
    c_coords: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Place coarse N/C atoms from frames and compute phi/psi/omega."""
    atoms = coarse_backbone_atoms_from_ca_frames(ca_coords, frames, atom_names=("n", "ca", "c"))
    n_atom = atoms[:, 0] if n_coords is None else n_coords
    ca_atom = atoms[:, 1]
    c_atom = atoms[:, 2] if c_coords is None else c_coords
    return backbone_dihedrals_from_n_ca_c(n_atom, ca_atom, c_atom, batch=batch)


def _ca_step_targets(true_ca: Tensor, batch_idx: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    """Return target step vectors and a mask for valid consecutive CA edges."""
    if true_ca.ndim != 2 or true_ca.shape[-1] != 3:
        raise ValueError(f"Expected true_ca with shape (N, 3), got {tuple(true_ca.shape)}")

    target = torch.zeros_like(true_ca)
    mask = torch.zeros(true_ca.shape[0], dtype=torch.bool, device=true_ca.device)

    if batch_idx is None:
        if true_ca.shape[0] > 1:
            target[1:] = (true_ca[1:] - true_ca[:-1]).to(dtype=target.dtype)
            mask[1:] = True
        return target, mask

    for b in torch.unique(batch_idx, sorted=True):
        idx = torch.where(batch_idx == b)[0]
        if idx.numel() < 2:
            continue
        target[idx[1:]] = (true_ca[idx[1:]] - true_ca[idx[:-1]]).to(dtype=target.dtype)
        mask[idx[1:]] = True
    return target, mask


def ca_step_loss(
    pred_steps: Tensor,
    true_ca: Tensor,
    batch_idx: Optional[Tensor] = None,
    frames: Optional[Tensor] = None,
    frame_offset: str = "prev",
    plddt: Optional[Tensor] = None,
    plddt_thresh: float = 0.3,
    beta: float = 0.25,
) -> Tensor:
    """Smooth-L1 loss on local consecutive CA displacement vectors."""
    if frames is None:
        target_steps, step_mask = _ca_step_targets(true_ca, batch_idx=batch_idx)
    else:
        target_steps, step_mask = ca_local_step_targets(
            true_ca,
            frames,
            batch_idx=batch_idx,
            frame_offset=frame_offset,
        )
    if plddt is not None:
        good = plddt.squeeze(-1) >= plddt_thresh
        step_mask = step_mask & good
    if step_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_steps.device, dtype=pred_steps.dtype)
    return F.smooth_l1_loss(pred_steps[step_mask], target_steps[step_mask], beta=beta)


def ca_bond_length_loss(
    pred_steps: Tensor,
    batch_idx: Optional[Tensor] = None,
    target_length: float = 3.8,
) -> Tensor:
    """Penalize local CA step lengths away from the canonical CA-CA spacing."""
    if pred_steps.ndim != 2 or pred_steps.shape[-1] != 3:
        raise ValueError(f"Expected pred_steps with shape (N, 3), got {tuple(pred_steps.shape)}")

    mask = torch.ones(pred_steps.shape[0], dtype=torch.bool, device=pred_steps.device)
    if batch_idx is None:
        if mask.numel() > 0:
            mask[0] = False
    else:
        for b in torch.unique(batch_idx, sorted=True):
            idx = torch.where(batch_idx == b)[0]
            if idx.numel() > 0:
                mask[idx[0]] = False

    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_steps.device, dtype=pred_steps.dtype)
    lengths = pred_steps[mask].norm(dim=-1)
    target = torch.full_like(lengths, float(target_length))
    return F.smooth_l1_loss(lengths, target, beta=0.1)


def ca_pairwise_distance_loss(
    pred_ca: Tensor,
    true_ca: Tensor,
    batch_idx: Optional[Tensor] = None,
    min_seq_sep: int = 2,
    max_seq_sep: Optional[int] = 64,
    max_pairs: Optional[int] = 4096,
) -> Tensor:
    """dRMSD-style loss on CA pairwise distances within each chain."""
    if pred_ca.shape != true_ca.shape:
        raise ValueError(f"Shape mismatch: pred_ca={tuple(pred_ca.shape)} true_ca={tuple(true_ca.shape)}")

    def _single(pred_s: Tensor, true_s: Tensor) -> Optional[Tensor]:
        n = pred_s.shape[0]
        if n <= min_seq_sep:
            return None
        pairs = torch.triu_indices(n, n, offset=min_seq_sep, device=pred_s.device)
        if max_seq_sep is not None:
            seq_sep = pairs[1] - pairs[0]
            pairs = pairs[:, seq_sep <= int(max_seq_sep)]
        if pairs.shape[1] == 0:
            return None
        if max_pairs is not None and pairs.shape[1] > max_pairs:
            take = torch.linspace(0, pairs.shape[1] - 1, max_pairs, device=pred_s.device).long()
            pairs = pairs[:, take]
        pred_d = (pred_s[pairs[0]] - pred_s[pairs[1]]).norm(dim=-1)
        true_d = (true_s[pairs[0]] - true_s[pairs[1]]).norm(dim=-1)
        return F.smooth_l1_loss(pred_d, true_d, beta=0.5)

    losses = []
    if batch_idx is None:
        val = _single(pred_ca, true_ca)
        if val is not None:
            losses.append(val)
    else:
        for b in torch.unique(batch_idx, sorted=True):
            idx = torch.where(batch_idx == b)[0]
            val = _single(pred_ca[idx], true_ca[idx])
            if val is not None:
                losses.append(val)

    if not losses:
        return torch.tensor(0.0, device=pred_ca.device, dtype=pred_ca.dtype)
    return torch.stack(losses).mean()


def coarse_ca_loss(
    pred_steps: Tensor,
    true_ca: Tensor,
    batch_idx: Optional[Tensor] = None,
    pred_ca: Optional[Tensor] = None,
    frames: Optional[Tensor] = None,
    frame_offset: str = "prev",
    step_weight: float = 1.0,
    bond_weight: float = 0.1,
    pairwise_weight: float = 0.25,
    pairwise_min_seq_sep: int = 2,
    pairwise_max_seq_sep: Optional[int] = 64,
    pairwise_max_pairs: Optional[int] = 4096,
    plddt: Optional[Tensor] = None,
    plddt_thresh: float = 0.3,
    return_components: bool = False,
):
    """Minimal coarse-CA loss bundle for local step-vector predictors."""
    if pred_ca is None:
        if frames is None:
            pred_ca = integrate_ca_steps(pred_steps, batch_idx=batch_idx)
        else:
            pred_ca = integrate_local_ca_steps(
                pred_steps,
                frames,
                batch_idx=batch_idx,
                frame_offset=frame_offset,
            )

    components = {
        "step": ca_step_loss(
            pred_steps,
            true_ca,
            batch_idx=batch_idx,
            frames=frames,
            frame_offset=frame_offset,
            plddt=plddt,
            plddt_thresh=plddt_thresh,
        ),
        "bond": ca_bond_length_loss(pred_steps, batch_idx=batch_idx),
        "pairwise": ca_pairwise_distance_loss(
            pred_ca,
            true_ca,
            batch_idx=batch_idx,
            min_seq_sep=pairwise_min_seq_sep,
            max_seq_sep=pairwise_max_seq_sep,
            max_pairs=pairwise_max_pairs,
        ),
    }
    total = (
        float(step_weight) * components["step"]
        + float(bond_weight) * components["bond"]
        + float(pairwise_weight) * components["pairwise"]
    )
    if return_components:
        return total, components
    return total


def split_rt_pred(
    rt_pred: Tensor,
    normalize: bool = True,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Split an ``rt_pred`` tensor into quaternion and translation parts.

    Expects the last dimension to be ``[qw, qx, qy, qz, tx, ty, tz]``.
    """
    if rt_pred.shape[-1] != 7:
        raise ValueError(f"Expected rt_pred with last dim 7, got {rt_pred.shape[-1]}")

    quat = rt_pred[..., :4]
    trans = rt_pred[..., 4:]
    if normalize:
        quat = _normalize_quaternion(quat, eps=eps)
    return quat, trans


def normalize_quaternion(q: Tensor, eps: float = 1e-8) -> Tensor:
    """Public quaternion normalization helper."""
    return _normalize_quaternion(q, eps=eps)


def quaternion_geodesic_loss(
    pred_q: Tensor,
    true_q: Tensor,
    reduction: str = "mean",
    eps: float = 1e-8,
    squared: bool = False,
) -> Tensor:
    """SO(3) angular distance in radians with q ~ -q symmetry."""
    pred_q = normalize_quaternion(pred_q, eps=eps)
    true_q = normalize_quaternion(true_q, eps=eps)

    dot = (pred_q * true_q).sum(dim=-1).abs().clamp(max=1.0)
    angle = 2.0 * torch.atan2(torch.sqrt(torch.clamp(1.0 - dot**2, min=0.0)), dot.clamp_min(eps))
    loss = angle**2 if squared else angle

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction: {reduction}")


def quaternion_angle_loss(
    pred_q: Tensor,
    true_q: Tensor,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> Tensor:
    """Angular quaternion distance in radians: ``2 * acos(|<q1, q2>|)``."""
    pred_q = normalize_quaternion(pred_q, eps=eps)
    true_q = normalize_quaternion(true_q, eps=eps)

    dot = (pred_q * true_q).sum(dim=-1).abs().clamp(max=1.0)
    loss = 2.0 * torch.atan2(torch.sqrt(torch.clamp(1.0 - dot**2, min=0.0)), dot.clamp_min(eps))

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction: {reduction}")


def _fape_single_structure(
    true_R: Tensor,
    true_t: Tensor,
    pred_R: Tensor,
    pred_t: Tensor,
    d_clamp: float = 10.0,
    eps: float = 1e-8,
    reduction: str = "mean",
    pair_sample_size: Optional[int] = None,
) -> Tensor:
    """Pairwise local-frame FAPE for a single structure."""
    n = true_t.shape[0]
    if pair_sample_size is not None and pair_sample_size > 0 and pair_sample_size < n ** 2:
        sample_i = torch.randint(n, (pair_sample_size,), device=true_t.device)
        sample_j = torch.randint(n, (pair_sample_size,), device=true_t.device)
        diff_pred = pred_t[sample_j] - pred_t[sample_i]
        diff_true = true_t[sample_j] - true_t[sample_i]
        local_pred = torch.einsum("nij,nj->ni", pred_R[sample_i].transpose(-1, -2), diff_pred)
        local_true = torch.einsum("nij,nj->ni", true_R[sample_i].transpose(-1, -2), diff_true)
    else:
        diff_pred = pred_t.unsqueeze(1) - pred_t.unsqueeze(0)
        diff_true = true_t.unsqueeze(1) - true_t.unsqueeze(0)
        local_pred = torch.einsum("nij,nmj->nmi", pred_R.transpose(-1, -2), diff_pred)
        local_true = torch.einsum("nij,nmj->nmi", true_R.transpose(-1, -2), diff_true)

    error = torch.linalg.vector_norm(local_pred - local_true, dim=-1)
    error = torch.clamp(error, max=d_clamp)

    if reduction == "mean":
        return error.mean()
    if reduction == "sum":
        return error.sum()
    if reduction == "none":
        return error
    raise ValueError(f"Unknown reduction: {reduction}")


def quaternion_fape_loss(
    true_q: Tensor,
    true_t: Tensor,
    pred_q: Tensor,
    pred_t: Tensor,
    batch: Optional[Tensor] = None,
    d_clamp: float = 10.0,
    eps: float = 1e-8,
    reduction: str = "mean",
    pair_sample_size: Optional[int] = None,
) -> Tensor:
    """Frame-aligned point error from quaternion + translation frames."""
    true_q = normalize_quaternion(true_q, eps=eps)
    pred_q = normalize_quaternion(pred_q, eps=eps)

    true_R = quaternion_to_rotation_matrix(true_q)
    pred_R = quaternion_to_rotation_matrix(pred_q)

    if batch is None:
        return _fape_single_structure(true_R, true_t, pred_R, pred_t, d_clamp=d_clamp, eps=eps, reduction=reduction, pair_sample_size=pair_sample_size)

    losses = []
    for b in torch.unique(batch, sorted=True):
        idx = (batch == b).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        losses.append(
            _fape_single_structure(
                true_R[idx],
                true_t[idx],
                pred_R[idx],
                pred_t[idx],
                d_clamp=d_clamp,
                eps=eps,
                reduction="mean",
                pair_sample_size=pair_sample_size,
            )
        )

    if not losses:
        return torch.tensor(0.0, device=true_q.device, dtype=true_q.dtype)

    stacked = torch.stack(losses)
    if reduction == "mean":
        return stacked.mean()
    if reduction == "sum":
        return stacked.sum()
    if reduction == "none":
        return stacked
    raise ValueError(f"Unknown reduction: {reduction}")


def rt_fape_loss(
    true_rt: Tensor,
    pred_rt: Tensor,
    batch: Optional[Tensor] = None,
    d_clamp: float = 10.0,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> Tensor:
    """Convenience FAPE for ``(..., 7)`` RT tensors."""
    true_q, true_t = split_rt_pred(true_rt, normalize=True, eps=eps)
    pred_q, pred_t = split_rt_pred(pred_rt, normalize=True, eps=eps)
    return quaternion_fape_loss(
        true_q=true_q,
        true_t=true_t,
        pred_q=pred_q,
        pred_t=pred_t,
        batch=batch,
        d_clamp=d_clamp,
        eps=eps,
        reduction=reduction,
    )


def lddt_reconstruction_loss(
    pred_q: Tensor,
    pred_t: Tensor,
    true_coords: Tensor,
    batch: Optional[Tensor] = None,
    cutoff: float = 15.0,
    thresholds: Optional[Sequence[float]] = None,
    plddt: Optional[Tensor] = None,
    plddt_thresh: float = 0.3,
) -> Tensor:
    """Differentiable lDDT loss from translation steps.

    ``pred_q`` is accepted for API compatibility but not used.
    """
    del pred_q
    return differentiable_lddt_loss(
        pred_t=pred_t,
        true_coords=true_coords,
        batch=batch,
        cutoff=cutoff,
        thresholds=thresholds,
        plddt=plddt,
        plddt_thresh=plddt_thresh,
    )


def batch_fape_loss(
    true_q: Tensor,
    true_t: Tensor,
    pred_q: Tensor,
    pred_t: Tensor,
    batch: Optional[Tensor] = None,
    d_clamp: float = 10.0,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> Tensor:
    """Batch-aware FAPE alias for compatibility with training scripts."""
    return quaternion_fape_loss(
        true_q=true_q,
        true_t=true_t,
        pred_q=pred_q,
        pred_t=pred_t,
        batch=batch,
        d_clamp=d_clamp,
        eps=eps,
        reduction=reduction,
    )


def batch_delta_loss(
    true_ca: Tensor,
    pred_q: Tensor,
    pred_t: Tensor,
    batch: Optional[Tensor] = None,
    plddt: Optional[Tensor] = None,
    plddt_thresh: float = 0.3,
) -> Tensor:
    """Batch-aware delta displacement loss via quaternion chain reconstruction."""
    if batch is None:
        pred_ca = reconstruct_positions(quaternion_to_rotation_matrix(pred_q), pred_t)[1:]
        return delta_loss(
            true_ca,
            pred_ca,
            plddt=plddt,
            plddt_thresh=plddt_thresh,
        )

    batch_loss = []
    for b in torch.unique(batch, sorted=True):
        mask_b = (batch == b).nonzero(as_tuple=True)[0]
        if mask_b.numel() < 2:
            continue

        true_ca_b = true_ca[mask_b]
        pred_q_b = pred_q[mask_b]
        pred_t_b = pred_t[mask_b]
        pred_ca_b = reconstruct_positions(quaternion_to_rotation_matrix(pred_q_b), pred_t_b)[1:]
        plddt_b = plddt[mask_b] if plddt is not None else None
        batch_loss.append(
            delta_loss(
                true_ca_b,
                pred_ca_b,
                plddt=plddt_b,
                plddt_thresh=plddt_thresh,
            )
        )

    if not batch_loss:
        return torch.tensor(0.0, device=true_ca.device, dtype=true_ca.dtype)
    return torch.stack(batch_loss).mean()


def batch_structure_losses(
    true_q: Tensor,
    true_t: Tensor,
    pred_q: Tensor,
    pred_t: Tensor,
    true_ca: Tensor,
    batch: Optional[Tensor] = None,
    plddt: Optional[Tensor] = None,
    plddt_thresh: float = 0.3,
    d_clamp: float = 10.0,
    eps: float = 1e-8,
    lddt_cutoff: float = 15.0,
    lddt_thresholds: Optional[Sequence[float]] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute FAPE, lDDT, and delta losses with consistent batching semantics."""
    fape_val = quaternion_fape_loss(
        true_q=true_q,
        true_t=true_t,
        pred_q=pred_q,
        pred_t=pred_t,
        batch=batch,
        d_clamp=d_clamp,
        eps=eps,
        reduction="mean",
    )

    lddt_val = lddt_reconstruction_loss(
        pred_q=pred_q,
        pred_t=pred_t,
        true_coords=true_ca,
        batch=batch,
        cutoff=lddt_cutoff,
        thresholds=lddt_thresholds,
        plddt=plddt,
        plddt_thresh=plddt_thresh,
    )

    delta_val = batch_delta_loss(
        true_ca=true_ca,
        pred_q=pred_q,
        pred_t=pred_t,
        batch=batch,
        plddt=plddt,
        plddt_thresh=plddt_thresh,
    )
    return fape_val, lddt_val, delta_val


def fape_loss(
    true_R: torch.Tensor,
    true_t: torch.Tensor,
    pred_R: torch.Tensor,
    pred_t: torch.Tensor,
    batch: torch.Tensor,
    plddt: Optional[torch.Tensor] = None,
    d_clamp: float = 10.0,
    eps: float = 1e-8,
    temperature: float = 0.25,
    reduction: str = "mean",
    soft: bool = False,
) -> torch.Tensor:
    """Frame-aligned point error in local residue frames.

    This is the pairwise local-frame variant used in FoldTree2 training scripts.
    """
    del plddt, reduction  # kept for API compatibility

    if batch is None:
        batch = torch.zeros(true_t.shape[0], dtype=torch.long, device=true_t.device)

    losses = []
    for b in torch.unique(batch, sorted=True):
        idx = torch.where(batch == b)[0]
        if idx.numel() < 2:
            continue

        diff_pred = pred_t[idx].unsqueeze(1) - pred_t[idx].unsqueeze(0)  # (m, m, 3)
        local_pred = torch.einsum("mij,mnj->mni", pred_R[idx].transpose(1, 2), diff_pred)

        diff_true = true_t[idx].unsqueeze(1) - true_t[idx].unsqueeze(0)
        local_true = torch.einsum("mij,mnj->mni", true_R[idx].transpose(1, 2), diff_true)

        if not soft:
            err = torch.sqrt(((local_pred - local_true) ** 2).sum(dim=-1) + eps)
            err = torch.clamp(err, max=d_clamp)
            losses.append(err.mean())
        else:
            dist_sq = ((local_pred.unsqueeze(1) - local_true.unsqueeze(0)) ** 2).sum(dim=-1)
            dist_sq = torch.clamp(dist_sq, max=d_clamp**2)
            align = F.softmax(-dist_sq / temperature, dim=-1)
            losses.append((align * dist_sq).sum(dim=-1).mean())

    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=true_R.device, dtype=true_R.dtype)


def delta_loss(
    coords: torch.Tensor,
    predcoords: torch.Tensor,
    plddt: Optional[torch.Tensor] = None,
    plddt_thresh: float = 0.3,
    batches: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Delta-coordinate loss on consecutive residues.

    Supports input shapes (N, 3) and (B, N, 3).
    """
    del batches  # API compatibility 

    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
        predcoords = predcoords.unsqueeze(0)

    if coords.ndim != 3 or predcoords.ndim != 3:
        raise ValueError(f"Expected coords/predcoords as 2D or 3D, got {coords.shape} and {predcoords.shape}")
    if coords.shape != predcoords.shape:
        raise ValueError(f"Shape mismatch: coords={coords.shape}, predcoords={predcoords.shape}")

    true_d = coords[:, 1:] - coords[:, :-1]
    pred_d = predcoords[:, 1:] - predcoords[:, :-1]
    dist = torch.sqrt(((pred_d - true_d) ** 2).sum(dim=-1) + 1e-6)

    if plddt is not None:
        if plddt.ndim == 3 and plddt.shape[-1] == 1:
            plddt = plddt.squeeze(-1)
        if plddt.ndim == 1:
            plddt = plddt.unsqueeze(0)
        if plddt.ndim == 2 and plddt.shape[0] == coords.shape[1] and plddt.shape[1] == coords.shape[0]:
            plddt = plddt.transpose(0, 1)
        if plddt.ndim != 2:
            raise ValueError(f"plddt must be (B, N), got {plddt.shape}")

        good = plddt >= plddt_thresh
        edge_mask = good[:, 1:] & good[:, :-1]
        denom = edge_mask.float().sum()
        if denom == 0:
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        val = (dist * edge_mask.float()).sum() / denom
    else:
        val = dist.mean()

    return torch.clamp(val, max=10.0)


def compute_lddt_loss(true_coords: torch.Tensor, pred_coords: torch.Tensor, cutoff: float = 15.0) -> torch.Tensor:
    """Simple hard-threshold lDDT loss (1 - score)."""
    if true_coords.ndim == 3 and pred_coords.ndim == 3:
        per_batch = [compute_lddt_loss(true_coords[b], pred_coords[b], cutoff=cutoff) for b in range(true_coords.shape[0])]
        return torch.stack(per_batch).mean() if per_batch else torch.tensor(0.0, device=true_coords.device)

    true_diff = true_coords.unsqueeze(0) - true_coords.unsqueeze(1)
    pred_diff = pred_coords.unsqueeze(0) - pred_coords.unsqueeze(1)

    d_true = torch.sqrt((true_diff ** 2).sum(dim=-1) + 1e-8)
    d_pred = torch.sqrt((pred_diff.clamp(-1000, 1000) ** 2).sum(dim=-1) + 1e-8)

    mask = (d_true < cutoff).float()
    diff = torch.abs(d_true - d_pred)
    valid = (diff < 0.5 * d_true) * mask

    denom = mask.sum()
    if denom <= 0:
        return torch.tensor(0.0, device=true_coords.device, dtype=true_coords.dtype)

    lddt_score = valid.sum() / denom
    return 1.0 - lddt_score


def differentiable_lddt_loss(
    pred_t: torch.Tensor,
    true_coords: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    cutoff: float = 15.0,
    thresholds: Optional[Sequence[float]] = None,
    plddt: Optional[torch.Tensor] = None,
    plddt_thresh: float = 0.3,
) -> torch.Tensor:
    """Differentiable soft-lDDT from predicted translations.

    """

    if thresholds is None:
        thresholds = (0.5, 1.0, 2.0, 4.0)

    def _single(t_b: torch.Tensor, c_b: torch.Tensor) -> torch.Tensor:
        pred_coords = torch.cumsum(t_b, dim=0)

        true_diff = c_b.unsqueeze(0) - c_b.unsqueeze(1)
        pred_diff = pred_coords.unsqueeze(0) - pred_coords.unsqueeze(1)

        d_true = torch.sqrt((true_diff ** 2).sum(dim=-1) + 1e-6)
        d_pred = torch.sqrt((pred_diff.clamp(-500, 500) ** 2).sum(dim=-1) + 1e-6).clamp(1e-6, 1e6)

        n = d_true.shape[0]
        diag = torch.eye(n, dtype=torch.bool, device=d_true.device)
        neighbor = (d_true < cutoff) & (~diag)
        if neighbor.sum() == 0:
            return torch.tensor(0.0, device=t_b.device, dtype=t_b.dtype)

        delta = torch.abs(d_pred - d_true)
        scores = []
        for thr in thresholds:
            s = 1.0 / (1.0 + (delta / thr) ** 2)
            scores.append(s[neighbor].mean())

        return 1.0 - torch.stack(scores).mean()

    if batch is None:
        if plddt is not None:
            keep = plddt.squeeze() >= plddt_thresh
            if keep.sum() < 2:
                return torch.tensor(0.0, device=pred_t.device, dtype=pred_t.dtype)
            return _single(pred_t[keep], true_coords[keep])
        return _single(pred_t, true_coords)

    losses = []
    for b in torch.unique(batch, sorted=True):
        idx = torch.where(batch == b)[0]
        t_b = pred_t[idx]
        c_b = true_coords[idx]

        if plddt is not None:
            keep = plddt[idx].squeeze() >= plddt_thresh
            if keep.sum() < 2:
                continue
            t_b = t_b[keep]
            c_b = c_b[keep]

        if t_b.shape[0] < 2:
            continue
        losses.append(_single(t_b, c_b))

    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=pred_t.device, dtype=pred_t.dtype)


def distogram_loss(
    logits: Tensor,
    coords: Tensor,
    edge_index: Tensor,
    min_bin: float = 2.0,
    max_bin: float = 21.0,
    no_bins: int = 8,
    eps: float = 1e-6,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Compute per-edge distogram CE loss from logits and coordinates.

    Args:
        logits: Predicted logits, shape (E, no_bins).
        coords: Coordinates, shape (N, 3).
        edge_index: Edge pairs, shape (2, E).
    """
    del eps

    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
        dtype=coords.dtype,
    ) ** 2

    idx0, idx1 = edge_index[0], edge_index[1]
    dists_sq = torch.sum((coords[idx0, :] - coords[idx1, :]) ** 2, dim=-1, keepdim=True)
    true_bins = torch.sum(dists_sq > boundaries, dim=-1)

    return F.cross_entropy(
        logits,
        true_bins,
        reduction="none",
        label_smoothing=label_smoothing,
    )


def recon_loss_disto(
    data,
    res,
    edge_index: Tensor,
    plddt: bool = True,
    nclamp: int = 30,
    no_bins: int = 8,
    key: Optional[str] = None,
    plddt_thresh: float = 0.3,
) -> Tensor:
    """Distogram reconstruction loss for decoder logits on provided edges.

    This remains compatible with existing decoder-based training code.
    """
    del nclamp

    if key is None:
        raise ValueError("recon_loss_disto requires `key` to index logits in decoder output.")

    logits = res[key]
    losses = distogram_loss(logits, data["coords"].x, edge_index, no_bins=no_bins)

    if plddt:
        c1 = data["plddt"].x[edge_index[0]].view(-1, 1) > plddt_thresh
        c2 = data["plddt"].x[edge_index[1]].view(-1, 1) > plddt_thresh
        mask = (c1 & c2).squeeze(1)
        losses = losses[mask]

    if losses.numel() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    return losses.mean()


def sample_random_pairs_by_batch(
    batch: Optional[Tensor],
    n_nodes: int,
    pairs_per_graph: int = 2048,
    min_seq_sep: int = 1,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Sample undirected residue pairs within each graph in a batch.

    Returns edge_index with shape (2, E).
    """
    if device is None:
        device = batch.device if batch is not None else torch.device("cpu")

    if n_nodes <= 1 or pairs_per_graph <= 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    if batch is None:
        batch = torch.zeros(n_nodes, dtype=torch.long, device=device)

    pair_chunks = []
    for b in torch.unique(batch, sorted=True):
        idx = torch.where(batch == b)[0]
        n = idx.numel()
        if n <= max(1, min_seq_sep):
            continue

        # Full enumeration for small graphs; random sampling for larger graphs.
        max_unique_pairs = n * (n - 1) // 2
        take = min(pairs_per_graph, max_unique_pairs)

        if n <= 256 and take == max_unique_pairs:
            local_i, local_j = torch.triu_indices(n, n, offset=max(1, min_seq_sep), device=device)
        else:
            pool = max(4 * take, 1024)
            local_i = torch.randint(0, n, (pool,), device=device)
            step = torch.randint(1, n, (pool,), device=device)
            local_j = (local_i + step) % n
            sep_ok = (local_i - local_j).abs() >= max(1, min_seq_sep)
            local_i = local_i[sep_ok]
            local_j = local_j[sep_ok]

            lo = torch.minimum(local_i, local_j)
            hi = torch.maximum(local_i, local_j)
            keys = lo * n + hi
            uniq = torch.unique(keys, sorted=False)
            if uniq.numel() == 0:
                continue
            lo = torch.div(uniq, n, rounding_mode="floor")
            hi = uniq % n
            local_i, local_j = lo, hi

        if local_i.numel() > take:
            perm = torch.randperm(local_i.numel(), device=device)[:take]
            local_i = local_i[perm]
            local_j = local_j[perm]

        pair_chunks.append(torch.stack([idx[local_i], idx[local_j]], dim=0))

    if not pair_chunks:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    return torch.cat(pair_chunks, dim=1)


def distogram_loss_from_coords(
    true_coords: Tensor,
    pred_coords: Tensor,
    batch: Optional[Tensor] = None,
    plddt: Optional[Tensor] = None,
    plddt_thresh: float = 0.3,
    pairs_per_graph: int = 2048,
    min_seq_sep: int = 1,
    no_bins: int = 16,
    min_bin: float = 2.0,
    max_bin: float = 21.0,
    temperature: float = 1.0,
    label_smoothing: float = 0.0,
    eps: float = 1e-8,
) -> Tensor:
    """Batched sampled distogram loss from predicted and true coordinates.

    Random residue pairs are sampled per structure, distances are binned for
    targets, and predicted distances are converted to differentiable bin logits.
    """
    if true_coords is None or pred_coords is None:
        device = pred_coords.device if pred_coords is not None else (true_coords.device if true_coords is not None else torch.device("cpu"))
        dtype = pred_coords.dtype if pred_coords is not None else (true_coords.dtype if true_coords is not None else torch.float32)
        return torch.tensor(0.0, device=device, dtype=dtype)

    n_nodes = int(true_coords.shape[0])
    if n_nodes == 0:
        return torch.tensor(0.0, device=true_coords.device, dtype=true_coords.dtype)

    edge_index = sample_random_pairs_by_batch(
        batch=batch,
        n_nodes=n_nodes,
        pairs_per_graph=pairs_per_graph,
        min_seq_sep=min_seq_sep,
        device=true_coords.device,
    )
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=true_coords.device, dtype=true_coords.dtype)

    src, dst = edge_index[0], edge_index[1]
    true_dist = torch.norm(true_coords[src] - true_coords[dst], dim=-1)
    pred_dist = torch.norm(pred_coords[src] - pred_coords[dst], dim=-1)

    if plddt is not None:
        conf = plddt.squeeze(-1) if plddt.ndim > 1 else plddt
        mask = (conf[src] >= plddt_thresh) & (conf[dst] >= plddt_thresh)
        true_dist = true_dist[mask]
        pred_dist = pred_dist[mask]

    if true_dist.numel() == 0:
        return torch.tensor(0.0, device=true_coords.device, dtype=true_coords.dtype)

    edges = torch.linspace(min_bin, max_bin, no_bins + 1, device=true_coords.device, dtype=true_coords.dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    true_bins = torch.bucketize(true_dist, edges[1:-1]).long()

    sigma = max((max_bin - min_bin) / max(no_bins, 1), eps)
    temp = max(temperature, eps)
    logits = -((pred_dist.unsqueeze(-1) - centers.unsqueeze(0)) ** 2) / (2.0 * (sigma * temp) ** 2)

    return F.cross_entropy(
        logits,
        true_bins,
        reduction="mean",
        label_smoothing=label_smoothing,
    )


def cross_product_loss(
    true_t: Tensor,
    pred_t: Tensor,
    batch: Optional[Tensor] = None,
    mode: str = "neighbor",
    eps: float = 1e-8,
    plddt: Optional[Tensor] = None,
    plddt_thresh: float = 0.5,
) -> Tensor:
    """Cross-product loss on translation vectors.

    Args:
        true_t: Ground-truth translations, shape (N, 3) or (1, N, 3).
        pred_t: Predicted translations, shape matching ``true_t``.
        batch: Optional chain ids, shape (N,).
        mode: Pairing strategy, either ``"neighbor"`` or ``"random"``.
        eps: Numerical stability constant.
        plddt: Optional per-residue confidence scores, shape (N, 1) or (N,).
        plddt_thresh: Confidence threshold for masking low-confidence residues.
    """
    if true_t.ndim == 3:
        if true_t.shape[0] != 1:
            raise ValueError(f"Expected true_t as (N, 3) or (1, N, 3), got {true_t.shape}")
        true_t = true_t.squeeze(0)
    if pred_t.ndim == 3:
        if pred_t.shape[0] != 1:
            raise ValueError(f"Expected pred_t as (N, 3) or (1, N, 3), got {pred_t.shape}")
        pred_t = pred_t.squeeze(0)

    if true_t.shape != pred_t.shape:
        raise ValueError(f"Shape mismatch: true_t={true_t.shape}, pred_t={pred_t.shape}")
    if true_t.ndim != 2 or true_t.shape[-1] != 3:
        raise ValueError(f"Expected true_t/pred_t to have shape (N, 3), got {true_t.shape}")

    if batch is not None:
        batch = batch.squeeze()
        if batch.ndim != 1 or batch.shape[0] != true_t.shape[0]:
            raise ValueError(f"batch must be shape (N,), got {batch.shape}")

    conf_mask: Optional[Tensor] = None
    if plddt is not None:
        if plddt.ndim == 3 and plddt.shape[0] == 1:
            plddt = plddt.squeeze(0)
        if plddt.ndim == 2 and plddt.shape[-1] == 1:
            plddt = plddt.squeeze(-1)
        if plddt.ndim == 2 and plddt.shape[0] == 1:
            plddt = plddt.squeeze(0)
        if plddt.ndim != 1 or plddt.shape[0] != true_t.shape[0]:
            raise ValueError(f"plddt must be shape (N,), got {plddt.shape}")

        conf_mask = (plddt.to(device=true_t.device) >= plddt_thresh)

    def _sample_random_pairs(n_vec: int, device: torch.device) -> tuple[Tensor, Tensor]:
        n_pairs = max(1, n_vec - 1)
        i = torch.randint(0, n_vec, (n_pairs,), device=device)
        j = torch.randint(0, n_vec - 1, (n_pairs,), device=device)
        j = j + (j >= i).to(j.dtype)
        return i, j

    def _single(t_b: Tensor, pair_i: Optional[Tensor] = None, pair_j: Optional[Tensor] = None) -> Tensor:
        if t_b.shape[0] < 3:
            return torch.tensor(0.0, device=t_b.device, dtype=t_b.dtype)

        deltas = t_b[1:] - t_b[:-1]
        if mode == "neighbor":
            v1 = deltas[:-1]
            v2 = deltas[1:]
        elif mode == "random":
            if pair_i is None or pair_j is None:
                pair_i, pair_j = _sample_random_pairs(deltas.shape[0], deltas.device)
            v1 = deltas[pair_i]
            v2 = deltas[pair_j]
        else:
            raise ValueError(f"Unknown mode={mode}. Expected 'neighbor' or 'random'.")

        cross = torch.cross(v1, v2, dim=-1)
        return torch.sqrt((cross ** 2).sum(dim=-1) + eps).mean()

    if batch is None:
        if conf_mask is not None:
            true_t = true_t[conf_mask]
            pred_t = pred_t[conf_mask]
            if true_t.shape[0] < 3:
                return torch.tensor(0.0, device=true_t.device, dtype=true_t.dtype)

        if mode == "random":
            pair_i, pair_j = _sample_random_pairs(true_t.shape[0] - 1, true_t.device)
            return _single(pred_t, pair_i, pair_j) - _single(true_t, pair_i, pair_j)
        return torch.norm(_single(pred_t) - _single(true_t)).mean()

    losses = []
    for b in torch.unique(batch, sorted=True):
        idx = torch.where(batch == b)[0]
        if conf_mask is not None:
            idx = idx[conf_mask[idx]]
        if idx.numel() < 3:
            continue
        if mode == "random":
            pair_i, pair_j = _sample_random_pairs(idx.numel() - 1, true_t.device)
            losses.append(torch.norm(_single(pred_t[idx], pair_i, pair_j) - _single(true_t[idx], pair_i, pair_j)).mean())
        else:
            losses.append(torch.norm(_single(pred_t[idx]) - _single(true_t[idx])).mean())

    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=true_t.device, dtype=true_t.dtype)


__all__ = [
    "split_rt_pred",
    "normalize_quaternion",
    "quaternion_geodesic_loss",
    "quaternion_angle_loss",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "reconstruct_positions",
    "integrate_ca_steps",
    "integrate_local_ca_steps",
    "coarse_backbone_atoms_from_ca_frames",
    "backbone_dihedrals_from_n_ca_c",
    "coarse_backbone_dihedrals_from_ca_frames",
    "coarse_backbone_fape_loss",
    "ca_step_loss",
    "ca_bond_length_loss",
    "ca_pairwise_distance_loss",
    "coarse_ca_loss",
    "quaternion_fape_loss",
    "rt_fape_loss",
    "fape_loss",
    "batch_structure_losses",
    "batch_fape_loss",
    "batch_delta_loss",
    "delta_loss",
    "compute_lddt_loss",
    "lddt_reconstruction_loss",
    "differentiable_lddt_loss",
    "distogram_loss",
    "recon_loss_disto",
    "sample_random_pairs_by_batch",
    "distogram_loss_from_coords",
    "cross_product_loss",
]
