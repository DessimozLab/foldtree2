#!/usr/bin/env python3
"""Test conversions between protein structure representations and evaluate robustness.

Pipeline:
1. PDB backbone atoms (N, CA, C)
2. Local frames (rotation matrices + translations)
3. Quaternion conversion
4. Quaternion -> rotation reconstruction
5. Noise injection in coordinate / RT / quaternion representations
6. FAPE + lDDT-style loss evaluation
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from foldtree2.src.losses.fape import (
    compute_lddt_loss,
    fape_loss,
    quaternion_to_rotation_matrix,
    reconstruct_positions,
    rotation_matrix_to_quaternion,
)
from foldtree2.src.pdbgraph import PDB2PyG


@dataclass
class EvalResult:
    name: str
    fape: float
    lddt_loss: float


def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return quats / quats.norm(dim=-1, keepdim=True).clamp_min(eps)


def _load_backbone_coords(pdb_path: Path, device: torch.device) -> torch.Tensor:
    pdb2pyg = PDB2PyG()
    n_coords = pdb2pyg.extract_pdb_coordinates(str(pdb_path), atom_type="N")
    ca_coords = pdb2pyg.extract_pdb_coordinates(str(pdb_path), atom_type="CA")
    c_coords = pdb2pyg.extract_pdb_coordinates(str(pdb_path), atom_type="C")

    if not (len(n_coords) == len(ca_coords) == len(c_coords)):
        raise ValueError("Backbone atom coordinate lengths do not match (N, CA, C).")

    backbone = torch.stack([n_coords, ca_coords, c_coords], dim=1).to(device)
    return backbone


def _compute_rtq(backbone_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotations, translations = PDB2PyG.compute_local_frame(backbone_coords)
    quats = rotation_matrix_to_quaternion(rotations)
    quats = _normalize_quaternions(quats)
    return rotations, translations, quats


def _random_rotation_matrices(
    n: int,
    sigma_rad: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sigma_rad <= 0:
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)

    axis = torch.randn(n, 3, device=device, dtype=dtype)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    theta = torch.randn(n, 1, device=device, dtype=dtype) * sigma_rad

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros_like(x)

    k = torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )

    identity = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)

    return identity + sin_t * k + (1 - cos_t) * (k @ k)


def _evaluate(
    name: str,
    true_rotations: torch.Tensor,
    true_translations: torch.Tensor,
    pred_rotations: torch.Tensor,
    pred_translations: torch.Tensor,
) -> EvalResult:
    if true_rotations.shape != pred_rotations.shape:
        raise ValueError("Rotation tensor shapes must match for evaluation.")
    if true_translations.shape != pred_translations.shape:
        raise ValueError("Translation tensor shapes must match for evaluation.")

    batch = torch.zeros(true_rotations.shape[0], dtype=torch.long, device=true_rotations.device)
    fape_val = float(
        fape_loss(
            true_R=true_rotations,
            true_t=true_translations,
            pred_R=pred_rotations,
            pred_t=pred_translations,
            batch=batch,
        ).item()
    )

    true_positions = reconstruct_positions(true_rotations, true_translations)
    pred_positions = reconstruct_positions(pred_rotations, pred_translations)
    lddt_val = float(compute_lddt_loss(true_positions, pred_positions).item())

    return EvalResult(name=name, fape=fape_val, lddt_loss=lddt_val)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    pdb_path = Path(args.pdb_path)

    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    backbone = _load_backbone_coords(pdb_path=pdb_path, device=device)
    rotations, translations, quats = _compute_rtq(backbone)

    print(f"Loaded {backbone.shape[0]} residues from {pdb_path}")
    print(f"Backbone tensor shape (N residues, N/CA/C, xyz): {tuple(backbone.shape)}")

    # Roundtrip conversion R -> q -> R
    rotations_from_quat = quaternion_to_rotation_matrix(quats)
    rot_roundtrip_error = torch.norm(rotations - rotations_from_quat, dim=(-2, -1))
    print(f"Mean R->q->R Frobenius error: {rot_roundtrip_error.mean().item():.6e}")
    print(f"Max  R->q->R Frobenius error: {rot_roundtrip_error.max().item():.6e}")

    # Noisy coordinates -> recompute RTQ
    noisy_backbone = backbone + torch.randn_like(backbone) * args.coord_noise
    coord_rotations, coord_translations, _ = _compute_rtq(noisy_backbone)

    # Noisy rotation/translation directly
    rot_noise = _random_rotation_matrices(
        n=rotations.shape[0],
        sigma_rad=args.rot_noise_rad,
        device=device,
        dtype=rotations.dtype,
    )
    noisy_rotations = rot_noise @ rotations
    noisy_translations = translations + torch.randn_like(translations) * args.trans_noise

    # Noisy quaternions -> back to rotation matrices
    noisy_quats = _normalize_quaternions(quats + torch.randn_like(quats) * args.quat_noise)
    quat_noisy_rotations = quaternion_to_rotation_matrix(noisy_quats)

    # Evaluation
    results = [
        _evaluate(
            name="Roundtrip (R->q->R)",
            true_rotations=rotations,
            true_translations=translations,
            pred_rotations=rotations_from_quat,
            pred_translations=translations,
        ),
        _evaluate(
            name=f"Coordinate noise (sigma={args.coord_noise})",
            true_rotations=rotations,
            true_translations=translations,
            pred_rotations=coord_rotations,
            pred_translations=coord_translations,
        ),
        _evaluate(
            name=f"RT noise (rot_sigma={args.rot_noise_rad}, trans_sigma={args.trans_noise})",
            true_rotations=rotations,
            true_translations=translations,
            pred_rotations=noisy_rotations,
            pred_translations=noisy_translations,
        ),
        _evaluate(
            name=f"Quaternion noise (sigma={args.quat_noise})",
            true_rotations=rotations,
            true_translations=translations,
            pred_rotations=quat_noisy_rotations,
            pred_translations=translations,
        ),
    ]

    print("\nLoss summary (lower is better):")
    print("-" * 90)
    print(f"{'Experiment':50s} {'FAPE':>16s} {'lDDT loss':>16s}")
    print("-" * 90)
    for result in results:
        print(f"{result.name:50s} {result.fape:16.6f} {result.lddt_loss:16.6f}")
    print("-" * 90)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test representation conversions and robustness to noise on protein backbone frames."
    )
    parser.add_argument(
        "--pdb-path",
        type=str,
        default="foldtree2/config/1eei.pdb",
        help="Path to input PDB file.",
    )
    parser.add_argument(
        "--coord-noise",
        type=float,
        default=0.25,
        help="Gaussian noise sigma added to backbone coordinates (Angstrom).",
    )
    parser.add_argument(
        "--rot-noise-rad",
        type=float,
        default=0.05,
        help="Std-dev (radians) for random rotation perturbations in RT-space.",
    )
    parser.add_argument(
        "--trans-noise",
        type=float,
        default=0.10,
        help="Gaussian noise sigma added to translation vectors (Angstrom).",
    )
    parser.add_argument(
        "--quat-noise",
        type=float,
        default=0.05,
        help="Gaussian noise sigma added to quaternion components before renormalization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use (e.g. 'cpu', 'cuda').",
    )
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
