#!/usr/bin/env python3
"""Geometry-focused PyTorch Lightning trainer.

Ports the final training-loop logic from the notebook:
foldtree2/notebooks/experiments/test_monodecoders _folding.ipynb

The script intentionally focuses on the geometry loop only.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from foldtree2.src import encoder as ecdr
from foldtree2.src import pdbgraphmk2
from foldtree2.src.losses.fape import (
    coarse_backbone_dihedrals_from_ca_frames,
    coarse_backbone_atoms_from_ca_frames,
    coarse_backbone_fape_loss,
    coarse_ca_loss,
    integrate_local_ca_steps,
    quaternion_to_rotation_matrix,
    reconstruct_positions,
    rotation_matrix_to_quaternion,
)
from foldtree2.src.losses.losses import quaternion_fape_loss, quaternion_geodesic_loss
from foldtree2.src.mono_decoders import Transformer_Geometry_Decoder

try:
    from foldtree2.src.se3_struct_decoder import se3_denoiser

    SE3_AVAILABLE = True
except Exception as exc:  # pragma: no cover - import availability is environment dependent
    se3_denoiser = None
    SE3_AVAILABLE = False
    warnings.warn(f"SE3 decoder import failed: {exc}")

# se3_struct_decoder sets default float64 at import time; restore float32 for this training path.
torch.set_default_dtype(torch.float32)


def ensure_float32_inplace(data):
    for ntype in data.node_types:
        if hasattr(data[ntype], "x") and data[ntype].x is not None and torch.is_floating_point(data[ntype].x):
            data[ntype].x = data[ntype].x.float()

    for etype in data.edge_types:
        if (
            hasattr(data[etype], "edge_attr")
            and data[etype].edge_attr is not None
            and torch.is_floating_point(data[etype].edge_attr)
        ):
            data[etype].edge_attr = data[etype].edge_attr.float()

    return data


def ensure_edge_attrs_inplace(data, edge_dim: int = 1):
    res_dtype = data["res"].x.dtype if ("res" in data.node_types and hasattr(data["res"], "x")) else torch.float32

    for etype in data.edge_types:
        edge_store = data[etype]
        edge_index = edge_store.edge_index
        n_edges = int(edge_index.shape[1])

        has_attr = hasattr(edge_store, "edge_attr") and edge_store.edge_attr is not None
        if (not has_attr) or (edge_store.edge_attr.shape[0] != n_edges):
            edge_store.edge_attr = torch.ones((n_edges, edge_dim), dtype=res_dtype, device=edge_index.device)
            continue

        edge_attr = edge_store.edge_attr
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        if edge_attr.shape[-1] > edge_dim:
            edge_attr = edge_attr[:, :edge_dim]
        elif edge_attr.shape[-1] < edge_dim:
            edge_attr = F.pad(edge_attr, (0, edge_dim - edge_attr.shape[-1]))

        edge_store.edge_attr = edge_attr.to(dtype=res_dtype)

    return data


def node_x(data, key: str, default=None):
    if key in data.node_types and hasattr(data[key], "x") and data[key].x is not None:
        return data[key].x
    return default


def wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def maybe_deg_to_rad_torch(angles: Optional[torch.Tensor], threshold: float = 3.5) -> Optional[torch.Tensor]:
    if angles is None:
        return None
    a = angles.float()
    finite = a[torch.isfinite(a)]
    if finite.numel() == 0:
        return a
    max_abs = float(finite.abs().max().detach().cpu())
    if max_abs > threshold:
        a = torch.deg2rad(a)
    return wrap_to_pi_torch(a)


def periodic_angle_smooth_l1(pred_angles: torch.Tensor, true_angles: torch.Tensor) -> torch.Tensor:
    true_norm = maybe_deg_to_rad_torch(true_angles).to(pred_angles.device, dtype=pred_angles.dtype)
    pred_norm = wrap_to_pi_torch(pred_angles)
    delta = wrap_to_pi_torch(pred_norm - true_norm)
    return F.smooth_l1_loss(delta, torch.zeros_like(delta))


def get_true_geometry(data):
    true_R = node_x(data, "R_true")
    true_t = node_x(data, "t_true")
    true_q = rotation_matrix_to_quaternion(true_R) if true_R is not None else None
    true_angles = maybe_deg_to_rad_torch(node_x(data, "bondangles"))
    true_coords = node_x(data, "coords")
    batch_idx = data["res"].batch if ("res" in data.node_types and hasattr(data["res"], "batch")) else None
    return true_R, true_t, true_q, true_angles, true_coords, batch_idx


def previous_local_rotation_targets(
    rotations: torch.Tensor,
    batch_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-residue previous-frame local rotations and a valid mask.

    For residue i>0 in a chain the target is R[i-1]^T R[i]. Chain starts are
    assigned identity rotations and marked invalid for the auxiliary loss.
    """
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotations with shape [N,3,3], got {tuple(rotations.shape)}")

    rel = torch.eye(3, dtype=rotations.dtype, device=rotations.device).expand(rotations.shape[0], 3, 3).clone()
    mask = torch.zeros(rotations.shape[0], dtype=torch.bool, device=rotations.device)

    if batch_idx is None:
        if rotations.shape[0] > 1:
            rel[1:] = torch.matmul(rotations[:-1].transpose(-1, -2), rotations[1:])
            mask[1:] = True
        return rel, mask

    for b in torch.unique(batch_idx, sorted=True):
        idx = (batch_idx == b).nonzero(as_tuple=True)[0]
        if idx.numel() <= 1:
            continue
        rel[idx[1:]] = torch.matmul(rotations[idx[:-1]].transpose(-1, -2), rotations[idx[1:]])
        mask[idx[1:]] = True
    return rel, mask


def compose_previous_local_rotations(
    local_rotations: torch.Tensor,
    batch_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compose previous-frame local rotations into global frame rotations."""
    if local_rotations.ndim != 3 or local_rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotations with shape [N,3,3], got {tuple(local_rotations.shape)}")

    global_rotations = torch.empty_like(local_rotations)

    def _single(idx: torch.Tensor) -> None:
        if idx.numel() == 0:
            return
        curr = torch.eye(3, dtype=local_rotations.dtype, device=local_rotations.device)
        global_rotations[idx[0]] = curr
        for pos in idx[1:]:
            curr = torch.matmul(curr, local_rotations[pos])
            global_rotations[pos] = curr

    if batch_idx is None:
        _single(torch.arange(local_rotations.shape[0], device=local_rotations.device))
        return global_rotations

    for b in torch.unique(batch_idx, sorted=True):
        _single((batch_idx == b).nonzero(as_tuple=True)[0])
    return global_rotations


def gauge_normalize_frames_to_chain_start(
    rotations: torch.Tensor,
    origins: torch.Tensor,
    batch_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Express absolute frames in each chain's first-residue frame."""
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotations with shape [N,3,3], got {tuple(rotations.shape)}")
    if origins.ndim != 2 or origins.shape[-1] != 3 or origins.shape[0] != rotations.shape[0]:
        raise ValueError(f"Expected origins with shape [{rotations.shape[0]},3], got {tuple(origins.shape)}")

    norm_rot = torch.empty_like(rotations)
    norm_origins = torch.empty_like(origins)

    def _single(idx: torch.Tensor) -> None:
        if idx.numel() == 0:
            return
        base_R_t = rotations[idx[0]].transpose(-1, -2)
        base_origin = origins[idx[0]]
        norm_rot[idx] = torch.matmul(base_R_t, rotations[idx])
        norm_origins[idx] = torch.einsum("ij,nj->ni", base_R_t, origins[idx] - base_origin)

    if batch_idx is None:
        _single(torch.arange(rotations.shape[0], device=rotations.device))
        return norm_rot, norm_origins

    for b in torch.unique(batch_idx, sorted=True):
        _single((batch_idx == b).nonzero(as_tuple=True)[0])
    return norm_rot, norm_origins


def gauge_normalize_points_to_chain_start(
    points: torch.Tensor,
    chain_start_rotations: torch.Tensor,
    chain_start_origins: torch.Tensor,
    batch_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Express points in each chain's first-residue frame."""
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape [N,3], got {tuple(points.shape)}")
    if chain_start_rotations.shape != (points.shape[0], 3, 3):
        raise ValueError(
            f"Expected rotations with shape [{points.shape[0]},3,3], got {tuple(chain_start_rotations.shape)}"
        )
    if chain_start_origins.shape != points.shape:
        raise ValueError(f"Expected origins with shape {tuple(points.shape)}, got {tuple(chain_start_origins.shape)}")

    norm_points = torch.empty_like(points)

    def _single(idx: torch.Tensor) -> None:
        if idx.numel() == 0:
            return
        base_R_t = chain_start_rotations[idx[0]].transpose(-1, -2)
        base_origin = chain_start_origins[idx[0]]
        norm_points[idx] = torch.einsum("ij,nj->ni", base_R_t, points[idx] - base_origin)

    if batch_idx is None:
        _single(torch.arange(points.shape[0], device=points.device))
        return norm_points

    for b in torch.unique(batch_idx, sorted=True):
        _single((batch_idx == b).nonzero(as_tuple=True)[0])
    return norm_points


class FrozenProjectionEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, data):
        x = data["res"].x.float()
        z = self.proj(x)
        return z, torch.zeros((), device=x.device, dtype=x.dtype)


def _load_any_full_model(path: Path) -> nn.Module:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, nn.Module):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("model"), nn.Module):
            return payload["model"]
        if isinstance(payload.get("encoder"), nn.Module):
            return payload["encoder"]
    raise RuntimeError(f"Could not extract nn.Module from {path}")


def load_encoder_any(path: Path, full_model_template: Optional[Path] = None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        model, _, resumed_epoch = ecdr.load_model(str(path))
        return model, {"source": str(path), "mode": "ecdr.load_model", "epoch": resumed_epoch}
    except Exception:
        pass

    payload = torch.load(str(path), map_location="cpu", weights_only=False)

    if isinstance(payload, nn.Module):
        return payload, {"source": str(path), "mode": "torch.load(module)", "epoch": None}

    if isinstance(payload, dict):
        if isinstance(payload.get("model"), nn.Module):
            return payload["model"], {"source": str(path), "mode": "payload[model]", "epoch": payload.get("epoch")}
        if isinstance(payload.get("encoder"), nn.Module):
            return payload["encoder"], {
                "source": str(path),
                "mode": "payload[encoder]",
                "epoch": payload.get("epoch"),
            }

        if len(payload) > 0 and all(torch.is_tensor(v) for v in payload.values()):
            if full_model_template is None or not Path(full_model_template).exists():
                raise RuntimeError("State-dict encoder was found, but no full-model template was provided.")
            template = _load_any_full_model(Path(full_model_template))
            missing, unexpected = template.load_state_dict(payload, strict=False)
            return template, {
                "source": str(path),
                "mode": "state_dict_on_full_template",
                "epoch": None,
                "missing": missing,
                "unexpected": unexpected,
            }

    raise RuntimeError(f"Unsupported encoder checkpoint format in {path}")


class GeometryOnlyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        num_workers: int,
        val_fraction: float = 0.1,
        val_count: Optional[int] = None,
        split_seed: int = 42,
        val_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.val_fraction = float(val_fraction)
        self.val_count = val_count
        self.split_seed = int(split_seed)
        self.struct_dat = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self.struct_dat is None:
            self.struct_dat = pdbgraphmk2.StructureDataset(self.dataset_path)
        if self.train_dataset is not None:
            return

        n_items = len(self.struct_dat)
        if n_items < 2 or (self.val_count is not None and self.val_count <= 0) or self.val_fraction <= 0:
            self.train_dataset = self.struct_dat
            self.val_dataset = None
            return

        if self.val_count is None:
            n_val = int(round(n_items * self.val_fraction))
        else:
            n_val = int(self.val_count)
        n_val = max(1, min(n_val, n_items - 1))

        generator = torch.Generator().manual_seed(self.split_seed)
        indices = torch.randperm(n_items, generator=generator).tolist()
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        self.train_dataset = Subset(self.struct_dat, train_indices)
        self.val_dataset = Subset(self.struct_dat, val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class GeometryFocusedModule(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        transformer_geom_decoder: nn.Module,
        learning_rate: float,
        weight_decay: float,
        lr_scheduler_name: str,
        lr_scheduler_interval: str,
        lr_scheduler_frequency: int,
        lr_scheduler_monitor: str,
        lr_warmup_epochs: int,
        lr_min: float,
        lr_step_size: int,
        lr_gamma: float,
        max_epochs: int,
        clip_grad_norm: float,
        cache_flush_interval: int,
        gc_collect_interval: int,
        use_cuda_ipc_collect: bool,
        use_uncertainty_weighting: bool,
        use_se3: bool,
        se3_decoder: Optional[nn.Module] = None,
        se3_atom_decoder: Optional[nn.Module] = None,
        nan_guard: bool = True,
        use_frame_fape_loss: bool = True,
        use_quat_geodesic_loss: bool = True,
        use_decoder_angle_loss: bool = True,
        use_coarse_ca_loss: bool = False,
        use_coarse_ca_step_loss: bool = True,
        use_coarse_ca_bond_loss: bool = True,
        use_coarse_ca_pairwise_loss: bool = True,
        coarse_ca_weight: float = 1.0,
        coarse_ca_step_weight: float = 1.0,
        coarse_ca_bond_weight: float = 0.1,
        coarse_ca_pairwise_weight: float = 0.25,
        coarse_ca_step_frame: str = "prev",
        coarse_ca_pairwise_max_seq_sep: int = 64,
        coarse_ca_pairwise_max_pairs: int = 4096,
        rotation_target_frame: str = "local",
        use_coarse_backbone_loss: bool = False,
        use_coarse_backbone_atom_loss: bool = True,
        use_coarse_c_loss: bool = False,
        use_coarse_cb_loss: bool = False,
        use_coarse_n_loss: bool = False,
        use_coarse_backbone_fape_loss: bool = True,
        use_coarse_backbone_angle_loss: bool = True,
        coarse_backbone_atom_weight: float = 0.05,
        coarse_c_weight: float = 0.05,
        coarse_cb_weight: float = 0.05,
        coarse_n_weight: float = 0.05,
        coarse_backbone_fape_weight: float = 0.25,
        coarse_backbone_angle_weight: float = 0.0,
        se3_input_source: str = "coarse_ca",
        se3_contact_sketch_top_k: int = 16,
        se3_contact_sketch_threshold: float = 0.0,
        se3_contact_sketch_min_seq_sep: int = 3,
        se3_contact_coord_scale: float = 10.0,
        se3_contact_local_window: int = 1,
        se3_max_nodes: int = 0,
        train_se3_only: bool = False,
        sanitize_nonfinite_grads: bool = False,
        skip_empty_loss_batches: bool = False,
        use_se3_atom_refine: bool = False,
        use_se3_residue_loss: bool = True,
        use_se3_angle_loss: bool = True,
        use_se3_atom_loss: bool = True,
        use_se3_atom_fape_loss: bool = True,
        use_se3_coarse_geometry_losses: bool = False,
        se3_atom_weight: float = 0.05,
        se3_atom_fape_weight: float = 0.25,
        se3_use_codebook_vectors: bool = False,
        se3_use_distance_contacts: bool = False,
        se3_distance_contact_cutoff: float = 8.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.transformer_geom_decoder = transformer_geom_decoder
        self.se3_decoder = se3_decoder
        self.se3_atom_decoder = se3_atom_decoder if se3_atom_decoder is not None else se3_decoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler_name = str(lr_scheduler_name)
        self.lr_scheduler_interval = str(lr_scheduler_interval)
        self.lr_scheduler_frequency = int(lr_scheduler_frequency)
        self.lr_scheduler_monitor = str(lr_scheduler_monitor)
        self.lr_warmup_epochs = int(lr_warmup_epochs)
        self.lr_min = float(lr_min)
        self.lr_step_size = int(lr_step_size)
        self.lr_gamma = float(lr_gamma)
        self.max_epochs = int(max_epochs)
        self.clip_grad_norm = clip_grad_norm
        self.cache_flush_interval = int(cache_flush_interval)
        self.gc_collect_interval = int(gc_collect_interval)
        self.use_cuda_ipc_collect = bool(use_cuda_ipc_collect)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_se3 = use_se3 and (se3_decoder is not None)
        self.nan_guard = nan_guard
        self.use_frame_fape_loss = bool(use_frame_fape_loss)
        self.use_quat_geodesic_loss = bool(use_quat_geodesic_loss)
        self.use_decoder_angle_loss = bool(use_decoder_angle_loss)
        self.use_coarse_ca_loss = bool(use_coarse_ca_loss)
        self.use_coarse_ca_step_loss = bool(use_coarse_ca_step_loss)
        self.use_coarse_ca_bond_loss = bool(use_coarse_ca_bond_loss)
        self.use_coarse_ca_pairwise_loss = bool(use_coarse_ca_pairwise_loss)
        self.coarse_ca_weight = float(coarse_ca_weight)
        self.coarse_ca_step_weight = float(coarse_ca_step_weight)
        self.coarse_ca_bond_weight = float(coarse_ca_bond_weight)
        self.coarse_ca_pairwise_weight = float(coarse_ca_pairwise_weight)
        self.coarse_ca_step_frame = coarse_ca_step_frame
        self.coarse_ca_pairwise_max_seq_sep = int(coarse_ca_pairwise_max_seq_sep)
        self.coarse_ca_pairwise_max_pairs = int(coarse_ca_pairwise_max_pairs)
        if rotation_target_frame not in {"absolute", "local"}:
            raise ValueError(f"Unknown rotation_target_frame={rotation_target_frame}")
        self.rotation_target_frame = rotation_target_frame
        self.use_coarse_backbone_loss = bool(use_coarse_backbone_loss)
        self.use_coarse_backbone_atom_loss = bool(use_coarse_backbone_atom_loss)
        self.use_coarse_c_loss = bool(use_coarse_c_loss)
        self.use_coarse_cb_loss = bool(use_coarse_cb_loss)
        self.use_coarse_n_loss = bool(use_coarse_n_loss)
        self.use_coarse_backbone_fape_loss = bool(use_coarse_backbone_fape_loss)
        self.use_coarse_backbone_angle_loss = bool(use_coarse_backbone_angle_loss)
        self.coarse_backbone_atom_weight = float(coarse_backbone_atom_weight)
        self.coarse_c_weight = float(coarse_c_weight)
        self.coarse_cb_weight = float(coarse_cb_weight)
        self.coarse_n_weight = float(coarse_n_weight)
        self.coarse_backbone_fape_weight = float(coarse_backbone_fape_weight)
        self.coarse_backbone_angle_weight = float(coarse_backbone_angle_weight)
        if se3_input_source not in {"reconstructed_ca", "coarse_ca", "geometry_dot_contacts"}:
            raise ValueError(f"Unknown se3_input_source={se3_input_source}")
        self.se3_input_source = se3_input_source
        self.se3_contact_sketch_top_k = int(se3_contact_sketch_top_k)
        self.se3_contact_sketch_threshold = float(se3_contact_sketch_threshold)
        self.se3_contact_sketch_min_seq_sep = int(se3_contact_sketch_min_seq_sep)
        self.se3_contact_coord_scale = float(se3_contact_coord_scale)
        self.se3_contact_local_window = int(se3_contact_local_window)
        self.se3_max_nodes = int(se3_max_nodes)
        self.train_se3_only = bool(train_se3_only)
        self.sanitize_nonfinite_grads = bool(sanitize_nonfinite_grads)
        self.skip_empty_loss_batches = bool(skip_empty_loss_batches)
        if self.train_se3_only and not self.use_se3:
            raise ValueError("--train-se3-only requires --use-se3 and an available SE3 decoder")
        self.use_se3_atom_refine = bool(use_se3_atom_refine)
        self.use_se3_residue_loss = bool(use_se3_residue_loss)
        self.use_se3_angle_loss = bool(use_se3_angle_loss)
        self.use_se3_atom_loss = bool(use_se3_atom_loss)
        self.use_se3_atom_fape_loss = bool(use_se3_atom_fape_loss)
        self.use_se3_coarse_geometry_losses = bool(use_se3_coarse_geometry_losses)
        self.se3_atom_weight = float(se3_atom_weight)
        self.se3_atom_fape_weight = float(se3_atom_fape_weight)
        self.se3_use_codebook_vectors = bool(se3_use_codebook_vectors)
        self.se3_use_distance_contacts = bool(se3_use_distance_contacts)
        self.se3_distance_contact_cutoff = float(se3_distance_contact_cutoff)

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        if self.train_se3_only:
            for p in self.transformer_geom_decoder.parameters():
                p.requires_grad = False
            self.transformer_geom_decoder.eval()

        self.uncertainty_term_names = [
            "fape_quat",
            "quat_geodesic",
            "angles",
            "coarse_ca",
            "coarse_backbone_atoms",
            "coarse_c",
            "coarse_cb",
            "coarse_n",
            "coarse_backbone_fape",
            "coarse_backbone_angles",
            "fape_quat_se3",
            "quat_geodesic_se3",
            "se3_angles",
            "se3_atom_refine",
            "se3_atom_fape",
        ]
        if self.use_uncertainty_weighting:
            self.kendall_log_vars = torch.nn.ParameterDict(
                {name: torch.nn.Parameter(torch.zeros((), dtype=torch.float32)) for name in self.uncertainty_term_names}
            )

        self.save_hyperparameters(ignore=["encoder", "transformer_geom_decoder", "se3_decoder", "se3_atom_decoder"])

    def configure_optimizers(self):
        train_params = [] if self.train_se3_only else list(self.transformer_geom_decoder.parameters())
        if self.use_se3 and self.se3_decoder is not None:
            train_params += list(self.se3_decoder.parameters())
        if self.use_se3 and self.se3_atom_decoder is not None and self.se3_atom_decoder is not self.se3_decoder:
            train_params += list(self.se3_atom_decoder.parameters())
        if self.use_uncertainty_weighting:
            train_params += list(self.kendall_log_vars.parameters())
        if len(train_params) == 0:
            raise RuntimeError("No trainable parameters selected for optimizer")

        optimizer = torch.optim.AdamW(train_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.lr_scheduler_name == "none":
            return optimizer

        scheduler = None
        if self.lr_scheduler_name == "cosine":
            t_max = max(1, self.max_epochs - self.lr_warmup_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=self.lr_min,
            )
        elif self.lr_scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, self.lr_step_size),
                gamma=self.lr_gamma,
            )
        else:
            raise ValueError(f"Unknown lr_scheduler_name={self.lr_scheduler_name}")

        scheduler_config = {
            "scheduler": scheduler,
            "interval": self.lr_scheduler_interval,
            "frequency": max(1, self.lr_scheduler_frequency),
            "monitor": self.lr_scheduler_monitor,
        }

        if self.lr_scheduler_name == "cosine" and self.lr_warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=max(1, self.lr_warmup_epochs),
            )
            scheduler_config["scheduler"] = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, scheduler],
                milestones=[max(1, self.lr_warmup_epochs)],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def on_train_epoch_start(self):
        self.encoder.eval()
        if self.train_se3_only:
            self.transformer_geom_decoder.eval()

    def on_before_optimizer_step(self, optimizer):
        if not self.sanitize_nonfinite_grads:
            return
        bad_entries = 0
        for param in self.parameters():
            grad = param.grad
            if grad is None:
                continue
            finite = torch.isfinite(grad)
            if finite.all():
                continue
            bad_entries += int((~finite).sum().detach().cpu())
            grad.data = torch.nan_to_num(grad.data, nan=0.0, posinf=0.0, neginf=0.0)
        if bad_entries > 0:
            self.log("train/nonfinite_grad_entries", float(bad_entries), on_step=True, on_epoch=False, prog_bar=False)

    @staticmethod
    def _batch_size_for_logs(data_batch, batch_idx):
        if batch_idx is not None:
            return int(torch.unique(batch_idx).numel())
        return 1

    @staticmethod
    def _flatten_batched_coords(coords_out: torch.Tensor, batch_idx: Optional[torch.Tensor]) -> torch.Tensor:
        if coords_out.ndim == 2:
            return coords_out
        if coords_out.ndim != 3:
            raise RuntimeError(f"Unexpected coords_out ndim={coords_out.ndim}; expected 2 or 3")
        if batch_idx is None:
            return coords_out[0]

        parts = []
        unique_batches = torch.unique(batch_idx, sorted=True)
        for i, b in enumerate(unique_batches):
            n = int((batch_idx == b).sum().item())
            parts.append(coords_out[i, :n])
        return torch.cat(parts, dim=0)

    @staticmethod
    def _step_translations_to_origins(t_steps: torch.Tensor, batch_idx: Optional[torch.Tensor]) -> torch.Tensor:
        """Convert per-residue CA->next-CA steps into per-residue frame origins."""
        if t_steps.ndim != 2 or t_steps.shape[-1] != 3:
            raise ValueError(f"Expected [N,3] step translations, got {tuple(t_steps.shape)}")

        if batch_idx is None:
            origins = torch.zeros_like(t_steps)
            if t_steps.shape[0] > 1:
                origins[1:] = torch.cumsum(t_steps[:-1], dim=0)
            return origins

        origins = torch.zeros_like(t_steps)
        for b in torch.unique(batch_idx, sorted=True):
            idx = (batch_idx == b).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            t_b = t_steps[idx]
            out_b = torch.zeros_like(t_b)
            if t_b.shape[0] > 1:
                out_b[1:] = torch.cumsum(t_b[:-1], dim=0)
            origins[idx] = out_b
        return origins

    @staticmethod
    def _frames_from_ca_only(ca_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ca_coords.ndim != 2 or ca_coords.shape[-1] != 3:
            raise ValueError(f"Expected [N,3] CA coords, got {tuple(ca_coords.shape)}")

        n = ca_coords.shape[0]
        if n == 0:
            raise ValueError("Empty coordinate tensor")

        if n == 1:
            R = torch.eye(3, device=ca_coords.device, dtype=ca_coords.dtype).unsqueeze(0)
            t = torch.zeros((1, 3), device=ca_coords.device, dtype=ca_coords.dtype)
            q = rotation_matrix_to_quaternion(R)
            return R, t, q

        forward = torch.zeros_like(ca_coords)
        forward[:-1] = ca_coords[1:] - ca_coords[:-1]
        forward[-1] = forward[-2]
        forward = forward / torch.clamp(torch.norm(forward, dim=-1, keepdim=True), min=1e-8)

        global_up = torch.tensor([0.0, 0.0, 1.0], device=ca_coords.device, dtype=ca_coords.dtype).expand_as(forward)
        almost_parallel = (torch.abs((forward * global_up).sum(dim=-1)) > 0.95).unsqueeze(-1)
        alt_up = torch.tensor([0.0, 1.0, 0.0], device=ca_coords.device, dtype=ca_coords.dtype).expand_as(forward)
        up = torch.where(almost_parallel, alt_up, global_up)

        right = torch.cross(up, forward, dim=-1)
        right = right / torch.clamp(torch.norm(right, dim=-1, keepdim=True), min=1e-8)

        normal = torch.cross(forward, right, dim=-1)
        normal = normal / torch.clamp(torch.norm(normal, dim=-1, keepdim=True), min=1e-8)

        R = torch.stack([forward, normal, right], dim=-1)

        # For FAPE, translations must be frame origins, not CA->next step vectors.
        t = torch.zeros_like(ca_coords)
        if n > 1:
            steps = ca_coords[1:] - ca_coords[:-1]
            t[1:] = torch.cumsum(steps, dim=0)
        q = rotation_matrix_to_quaternion(R)
        return R, t, q

    def _derive_se3_qt(self, se3_coords: torch.Tensor, batch_idx: Optional[torch.Tensor]):
        if batch_idx is None:
            _, t, q = self._frames_from_ca_only(se3_coords)
            return q, t

        q_parts = []
        t_parts = []
        for b in torch.unique(batch_idx, sorted=True):
            mask = batch_idx == b
            coords_b = se3_coords[mask]
            if coords_b.shape[0] == 0:
                continue
            _, t_b, q_b = self._frames_from_ca_only(coords_b)
            q_parts.append(q_b)
            t_parts.append(t_b)

        if not q_parts:
            raise RuntimeError("Unable to derive SE3 q/t from empty batched coordinates")

        return torch.cat(q_parts, dim=0), torch.cat(t_parts, dim=0)

    @staticmethod
    def _coords_to_steps(coords: torch.Tensor, batch_idx: Optional[torch.Tensor]) -> torch.Tensor:
        steps = torch.zeros_like(coords)
        if batch_idx is None:
            if coords.shape[0] > 1:
                steps[1:] = coords[1:] - coords[:-1]
            return steps
        for b in torch.unique(batch_idx, sorted=True):
            idx = (batch_idx == b).nonzero(as_tuple=True)[0]
            if idx.numel() > 1:
                steps[idx[1:]] = coords[idx[1:]] - coords[idx[:-1]]
        return steps

    def _kendall_weight_terms(self, raw_terms: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        weighted = {}
        for name, loss in raw_terms.items():
            if not self.use_uncertainty_weighting or name not in self.kendall_log_vars:
                weighted[name] = loss
                continue
            s = self.kendall_log_vars[name]
            weighted[name] = torch.exp(-s) * loss + s
        return weighted

    def _get_ft2_token_ids(self, z_local: torch.Tensor) -> Optional[torch.Tensor]:
        vq = getattr(self.encoder, "vector_quantizer", None)
        if vq is None or not hasattr(vq, "discretize_z"):
            return None
        token_ids, _ = vq.discretize_z(z_local)
        return token_ids.to(device=z_local.device, dtype=torch.long)

    @staticmethod
    def _get_codebook_vectors(encoder: nn.Module, token_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if token_ids is None:
            return None
        vq = getattr(encoder, "vector_quantizer", None)
        embeddings = getattr(vq, "embeddings", None)
        weight = getattr(embeddings, "weight", None)
        if weight is None:
            raise RuntimeError("SE3 codebook-vector input requires encoder.vector_quantizer.embeddings.weight")
        return F.embedding(token_ids, weight).detach()

    @staticmethod
    def _get_aa_identity_ids(decoder_out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        aa_logits = decoder_out.get("aa", None)
        if aa_logits is None:
            aa_logits = decoder_out.get("aa_pred", None)
        if aa_logits is None:
            return None
        if aa_logits.ndim != 2:
            raise RuntimeError(f"Expected AA logits with shape [N,20], got {tuple(aa_logits.shape)}")
        return torch.argmax(aa_logits, dim=-1).to(dtype=torch.long)

    def _geometry_dot_contact_sketch(
        self,
        z: torch.Tensor,
        batch_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        z = F.normalize(z.float(), p=2, dim=-1)

        if batch_idx is None:
            graph_indices = [torch.arange(z.shape[0], device=z.device)]
        else:
            graph_indices = [torch.where(batch_idx == i)[0] for i in torch.unique(batch_idx, sorted=True)]

        max_len = max((idx.numel() for idx in graph_indices), default=0)
        sketches = []
        top_k = max(0, self.se3_contact_sketch_top_k)
        min_seq_sep = max(0, self.se3_contact_sketch_min_seq_sep)
        local_window = max(0, self.se3_contact_local_window)

        for idx in graph_indices:
            n = int(idx.numel())
            sketch = torch.zeros((max_len, max_len), dtype=torch.bool, device=z.device)
            if n <= 1:
                sketches.append(sketch)
                continue

            scores = z[idx] @ z[idx].transpose(0, 1)
            pos = torch.arange(n, device=z.device)
            valid = pos[:, None].sub(pos[None, :]).abs() >= min_seq_sep
            valid.fill_diagonal_(False)

            if math.isfinite(self.se3_contact_sketch_threshold):
                contact = scores >= self.se3_contact_sketch_threshold
            else:
                contact = torch.zeros_like(scores, dtype=torch.bool)
            contact &= valid

            if top_k > 0:
                k = min(top_k, max(1, n - 1))
                masked_scores = scores.masked_fill(~valid, -torch.inf)
                row_contact = torch.zeros_like(contact)
                finite_rows = torch.isfinite(masked_scores).any(dim=-1)
                if finite_rows.any():
                    top_idx = torch.topk(masked_scores[finite_rows], k=k, dim=-1).indices
                    row_ids = torch.where(finite_rows)[0].unsqueeze(-1).expand_as(top_idx)
                    row_contact[row_ids, top_idx] = True
                    row_contact &= valid
                contact |= row_contact

            contact = contact | contact.transpose(0, 1)
            if local_window > 0:
                local = pos[:, None].sub(pos[None, :]).abs()
                contact |= (local > 0) & (local <= local_window)
            sketch[:n, :n] = contact
            sketches.append(sketch)

        if len(sketches) == 0:
            return torch.zeros((0, 0, 0), dtype=torch.bool, device=z.device)
        return torch.stack(sketches, dim=0)

    def _prepare_geometry_outputs(self, data_batch):
        """Run the frozen geometry source before the shared loss stack."""
        return self.transformer_geom_decoder(data_batch, contact_pred_index=None)

    def _compute_total_loss(self, data_batch, debug_label: str = ""):
        data_batch = ensure_float32_inplace(data_batch)
        data_batch = ensure_edge_attrs_inplace(data_batch, edge_dim=getattr(self.encoder, "edge_dim", 1))

        with torch.no_grad():
            z_local, _ = self.encoder(data_batch)
            ft2_token_ids = self._get_ft2_token_ids(z_local)
            codebook_vectors = self._get_codebook_vectors(self.encoder, ft2_token_ids)
        data_batch["res"].x = z_local

        out_local = self._prepare_geometry_outputs(data_batch)
        se3_node_features = z_local
        if self.se3_use_codebook_vectors:
            if codebook_vectors is None:
                raise RuntimeError("--se3-use-codebook-vectors requires encoder codebook vectors")
            se3_node_features = torch.cat([z_local, codebook_vectors], dim=-1)
        aa_identity_ids = self._get_aa_identity_ids(out_local)
        true_R, true_t, true_q, true_angles, true_coords, batch_idx = get_true_geometry(data_batch)

        pred_rt = out_local["rt_pred"]
        pred_q = pred_rt[..., :4]
        pred_t = pred_rt[..., 4:]
        pred_ca_steps = out_local.get("ca_step_pred", pred_t)
        pred_R = quaternion_to_rotation_matrix(pred_q)
        pred_q_for_fape = pred_q
        pred_R_for_fape = pred_R
        if self.rotation_target_frame == "local":
            pred_R_for_fape = compose_previous_local_rotations(pred_R, batch_idx=batch_idx)
            pred_q_for_fape = rotation_matrix_to_quaternion(pred_R_for_fape)
        pred_ca_trace = integrate_local_ca_steps(
            pred_ca_steps,
            pred_R_for_fape,
            batch_idx=batch_idx,
            frame_offset="prev",
        )

        raw_terms: Dict[str, torch.Tensor] = {}

        if true_q is not None and true_t is not None and (self.use_frame_fape_loss or self.use_quat_geodesic_loss):
            true_t_origin = self._step_translations_to_origins(true_t, batch_idx=batch_idx)
            pred_t_origin = self._step_translations_to_origins(pred_t, batch_idx=batch_idx)
            true_q_for_fape = true_q
            true_t_for_fape = true_t_origin
            if self.rotation_target_frame == "local":
                true_R_for_fape, true_t_for_fape = gauge_normalize_frames_to_chain_start(
                    true_R,
                    true_t_origin,
                    batch_idx=batch_idx,
                )
                true_q_for_fape = rotation_matrix_to_quaternion(true_R_for_fape)
            if self.use_frame_fape_loss:
                raw_terms["fape_quat"] = quaternion_fape_loss(
                    true_q_for_fape,
                    true_t_for_fape,
                    pred_q_for_fape,
                    pred_t_origin,
                    batch=batch_idx,
                )
            if self.use_quat_geodesic_loss and self.rotation_target_frame == "local":
                true_R_local, local_rot_mask = previous_local_rotation_targets(true_R, batch_idx=batch_idx)
                if local_rot_mask.any():
                    true_q_local = rotation_matrix_to_quaternion(true_R_local)
                    raw_terms["quat_geodesic"] = quaternion_geodesic_loss(
                        pred_q[local_rot_mask],
                        true_q_local[local_rot_mask],
                    )
            elif self.use_quat_geodesic_loss:
                raw_terms["quat_geodesic"] = quaternion_geodesic_loss(pred_q, true_q)

        pred_coords_local = reconstruct_positions(
            pred_R,
            pred_t,
            batch_idx=batch_idx,
            translation_frame="global",
            include_origin=False,
        )
        data_batch["coord_pred"].x = pred_coords_local
        data_batch["coords_pred"].x = pred_coords_local

        if self.use_decoder_angle_loss and out_local.get("angles") is not None and true_angles is not None:
            angle_mask = torch.isfinite(out_local["angles"]) & torch.isfinite(true_angles)
            if angle_mask.any():
                raw_terms["angles"] = periodic_angle_smooth_l1(out_local["angles"][angle_mask], true_angles[angle_mask])

        coarse_atom_names = ("ca", "c", "cb", "n")
        ca_idx, c_idx, cb_idx, n_idx = range(len(coarse_atom_names))
        pred_coarse_atoms = None
        true_coarse_atoms = None
        true_R_atoms = None
        true_ca_atoms = None

        if self.use_coarse_backbone_loss:
            if true_coords is None or true_R is None:
                raise RuntimeError("Coarse backbone loss requires data['coords'].x and data['R_true'].x")
            pred_coarse_atoms = coarse_backbone_atoms_from_ca_frames(
                pred_ca_trace,
                pred_R_for_fape,
                atom_names=coarse_atom_names,
            )

            true_R_atoms = true_R
            true_ca_atoms = true_coords
            if self.rotation_target_frame == "local":
                true_R_atoms, true_ca_atoms = gauge_normalize_frames_to_chain_start(
                    true_R,
                    true_coords,
                    batch_idx=batch_idx,
                )

            true_coarse_atoms = coarse_backbone_atoms_from_ca_frames(
                true_ca_atoms,
                true_R_atoms,
                atom_names=coarse_atom_names,
            )
            true_cb = node_x(data_batch, "cbcoords")
            if true_cb is not None:
                true_cb = true_cb.to(device=true_ca_atoms.device, dtype=true_ca_atoms.dtype)
                if self.rotation_target_frame == "local":
                    true_cb = gauge_normalize_points_to_chain_start(true_cb, true_R, true_coords, batch_idx=batch_idx)
                true_coarse_atoms = true_coarse_atoms.clone()
                true_coarse_atoms[:, cb_idx] = true_cb
            true_n = node_x(data_batch, "ncoords")
            if true_n is not None:
                true_n = true_n.to(device=true_ca_atoms.device, dtype=true_ca_atoms.dtype)
                if self.rotation_target_frame == "local":
                    true_n = gauge_normalize_points_to_chain_start(true_n, true_R, true_coords, batch_idx=batch_idx)
                true_coarse_atoms = true_coarse_atoms.clone()
                true_coarse_atoms[:, n_idx] = true_n
            true_c = node_x(data_batch, "ccoords")
            if true_c is not None:
                true_c = true_c.to(device=true_ca_atoms.device, dtype=true_ca_atoms.dtype)
                if self.rotation_target_frame == "local":
                    true_c = gauge_normalize_points_to_chain_start(true_c, true_R, true_coords, batch_idx=batch_idx)
                true_coarse_atoms = true_coarse_atoms.clone()
                true_coarse_atoms[:, c_idx] = true_c

            data_batch["coarse_ca_pred"].x = pred_coarse_atoms[:, ca_idx]
            data_batch["coarse_c_pred"].x = pred_coarse_atoms[:, c_idx]
            data_batch["coarse_cb_pred"].x = pred_coarse_atoms[:, cb_idx]
            data_batch["coarse_n_pred"].x = pred_coarse_atoms[:, n_idx]

            if self.use_coarse_backbone_atom_loss and self.coarse_backbone_atom_weight > 0:
                raw_terms["coarse_backbone_atoms"] = self.coarse_backbone_atom_weight * F.smooth_l1_loss(
                    pred_coarse_atoms,
                    true_coarse_atoms,
                    beta=0.5,
                )
            if self.use_coarse_c_loss and self.coarse_c_weight > 0:
                raw_terms["coarse_c"] = self.coarse_c_weight * F.smooth_l1_loss(
                    pred_coarse_atoms[:, c_idx],
                    true_coarse_atoms[:, c_idx],
                    beta=0.5,
                )
            if self.use_coarse_cb_loss and self.coarse_cb_weight > 0:
                raw_terms["coarse_cb"] = self.coarse_cb_weight * F.smooth_l1_loss(
                    pred_coarse_atoms[:, cb_idx],
                    true_coarse_atoms[:, cb_idx],
                    beta=0.5,
                )
            if self.use_coarse_n_loss and self.coarse_n_weight > 0:
                raw_terms["coarse_n"] = self.coarse_n_weight * F.smooth_l1_loss(
                    pred_coarse_atoms[:, n_idx],
                    true_coarse_atoms[:, n_idx],
                    beta=0.5,
                )
            if self.use_coarse_backbone_fape_loss and self.coarse_backbone_fape_weight > 0:
                raw_terms["coarse_backbone_fape"] = self.coarse_backbone_fape_weight * coarse_backbone_fape_loss(
                    true_coarse_atoms,
                    pred_coarse_atoms,
                    true_R_atoms,
                    pred_R_for_fape,
                    true_ca_atoms,
                    pred_ca_trace,
                    batch=batch_idx,
                )
            if self.use_coarse_backbone_angle_loss and self.coarse_backbone_angle_weight > 0 and true_angles is not None:
                pred_bb_angles, pred_angle_mask = coarse_backbone_dihedrals_from_ca_frames(
                    pred_ca_trace,
                    pred_R_for_fape,
                    batch=batch_idx,
                )
                true_bb_angles, true_angle_mask = coarse_backbone_dihedrals_from_ca_frames(
                    true_ca_atoms,
                    true_R_atoms,
                    batch=batch_idx,
                    n_coords=true_coarse_atoms[:, n_idx],
                    c_coords=true_coarse_atoms[:, c_idx],
                )
                angle_mask = pred_angle_mask & true_angle_mask
                if angle_mask.any():
                    angle_target = true_angles.to(device=pred_bb_angles.device, dtype=pred_bb_angles.dtype)
                    derived_delta = wrap_to_pi_torch(pred_bb_angles - angle_target)
                    frame_delta = wrap_to_pi_torch(pred_bb_angles - true_bb_angles)
                    raw_terms["coarse_backbone_angles"] = self.coarse_backbone_angle_weight * (
                        F.smooth_l1_loss(derived_delta[angle_mask], torch.zeros_like(derived_delta[angle_mask]))
                        + 0.25 * F.smooth_l1_loss(frame_delta[angle_mask], torch.zeros_like(frame_delta[angle_mask]))
                    )

        if self.use_coarse_ca_loss:
            if true_coords is None:
                raise RuntimeError("Coarse CA loss requires data['coords'].x")
            frames = None
            pred_ca_for_loss = pred_coords_local
            if self.coarse_ca_step_frame == "prev":
                if true_R is None:
                    raise RuntimeError("--coarse-ca-step-frame prev requires data['R_true'].x")
                frames = true_R
                pred_ca_for_loss = None
            elif self.coarse_ca_step_frame != "global":
                raise RuntimeError(f"Unknown coarse CA step frame: {self.coarse_ca_step_frame}")

            ca_step_weight = self.coarse_ca_step_weight if self.use_coarse_ca_step_loss else 0.0
            ca_bond_weight = self.coarse_ca_bond_weight if self.use_coarse_ca_bond_loss else 0.0
            ca_pairwise_weight = self.coarse_ca_pairwise_weight if self.use_coarse_ca_pairwise_loss else 0.0
            if ca_step_weight <= 0 and ca_bond_weight <= 0 and ca_pairwise_weight <= 0:
                raise RuntimeError("--use-coarse-ca-loss is enabled, but all coarse CA component losses are disabled")

            ca_total, _ca_components = coarse_ca_loss(
                pred_ca_steps,
                true_coords,
                batch_idx=batch_idx,
                pred_ca=pred_ca_for_loss,
                frames=frames,
                frame_offset="prev",
                step_weight=ca_step_weight,
                bond_weight=ca_bond_weight,
                pairwise_weight=ca_pairwise_weight,
                pairwise_max_seq_sep=self.coarse_ca_pairwise_max_seq_sep,
                pairwise_max_pairs=self.coarse_ca_pairwise_max_pairs,
                return_components=True,
            )
            raw_terms["coarse_ca"] = ca_total * self.coarse_ca_weight

        se3_skip = False
        if self.use_se3 and self.se3_decoder is not None:
            try:
                if ft2_token_ids is None:
                    raise RuntimeError(
                        "SE3 branch requires FoldTree2 token ids from encoder.vector_quantizer, "
                        "but the active encoder does not expose discretize_z()."
                    )
                se3_edge_attr_dict = None
                if self.se3_input_source == "geometry_dot_contacts":
                    contact_z = out_local.get("se3_contact_z", out_local.get("z", None))
                    if contact_z is None:
                        raise RuntimeError("Geometry dot-contact SE3 input requires a contact embedding")
                    if contact_z.ndim != 2:
                        raise RuntimeError(
                            "Geometry dot-contact SE3 input expects decoder embeddings with shape [N,D]; "
                            f"got {tuple(contact_z.shape)}. Try --transformer-width 3 --transformer-nheads 1."
                        )
                    se3_seed_coords = out_local.get("se3_seed_coords", None)
                    if se3_seed_coords is None:
                        se3_seed_coords = contact_z
                    if se3_seed_coords.ndim != 2 or se3_seed_coords.shape[-1] != 3:
                        raise RuntimeError(
                            "Geometry dot-contact SE3 input requires seed coordinates with shape [N,3]; "
                            f"got {tuple(se3_seed_coords.shape)}"
                        )
                    se3_seed_coords = se3_seed_coords * self.se3_contact_coord_scale
                    se3_seed_coords = se3_seed_coords.detach() if self.train_se3_only else se3_seed_coords
                    se3_edge_attr_dict = {
                        "dot_prod": self._geometry_dot_contact_sketch(se3_seed_coords, batch_idx),
                        # The initial residue pass is intentionally seeded
                        # only by the learned dot-product contact graph.
                        "use_distance_contacts": False,
                        "distance_contact_cutoff": self.se3_distance_contact_cutoff,
                    }
                elif self.se3_input_source == "coarse_ca":
                    se3_seed_coords = pred_ca_trace
                else:
                    se3_seed_coords = pred_coords_local
                if self.train_se3_only and self.se3_input_source != "geometry_dot_contacts":
                    se3_seed_coords = se3_seed_coords.detach()
                if self.se3_max_nodes > 0 and se3_seed_coords.shape[0] > self.se3_max_nodes:
                    zero = torch.zeros((), device=se3_seed_coords.device, dtype=torch.float32)
                    for param in self.parameters():
                        if param.requires_grad:
                            zero = zero + param.float().sum() * 0.0
                    return zero, raw_terms, raw_terms, True, batch_idx
                # Keep the production geometry decoder on the original
                # latent width; only the SE3 pass receives optional codebook
                # vectors concatenated to those features.
                data_batch["se3_seed_coords"].x = se3_seed_coords
                data_batch["res"].x = se3_node_features
                out_s = self.se3_decoder(
                    data_batch,
                    edge_attr_dict=se3_edge_attr_dict,
                    ft2_token_ids=ft2_token_ids,
                    aa_identity_ids=aa_identity_ids,
                    coords_pred=se3_seed_coords,
                )
                if out_s.get("coors_out") is not None:
                    se3_coords_step = self._flatten_batched_coords(out_s["coors_out"], batch_idx)
                    se3_coords_step = se3_coords_step.to(se3_seed_coords.device, dtype=se3_seed_coords.dtype)
                    data_batch["se3_coords_pred"].x = se3_coords_step
                    if self.use_se3_coarse_geometry_losses and self.use_coarse_ca_loss:
                        if true_coords is None:
                            raise RuntimeError("SE3 coarse CA loss requires data['coords'].x")
                        se3_ca_steps = self._coords_to_steps(se3_coords_step, batch_idx)
                        ca_frames = None
                        if self.coarse_ca_step_frame == "prev":
                            if true_R is None:
                                raise RuntimeError("SE3 coarse CA step loss requires data['R_true'].x")
                            ca_frames = true_R
                            frame_idx = torch.arange(se3_ca_steps.shape[0], device=se3_ca_steps.device)
                            if batch_idx is None:
                                frame_idx = torch.clamp(frame_idx - 1, min=0)
                            else:
                                for b in torch.unique(batch_idx, sorted=True):
                                    idx = (batch_idx == b).nonzero(as_tuple=True)[0]
                                    if idx.numel() > 1:
                                        frame_idx[idx[1:]] = idx[:-1]
                                    if idx.numel() > 0:
                                        frame_idx[idx[0]] = idx[0]
                            se3_ca_steps = torch.einsum(
                                "ni,nij->nj", se3_ca_steps, ca_frames[frame_idx]
                            )
                        elif self.coarse_ca_step_frame != "global":
                            raise RuntimeError(f"Unknown coarse CA step frame={self.coarse_ca_step_frame}")
                        ca_step_weight = self.coarse_ca_step_weight if self.use_coarse_ca_step_loss else 0.0
                        ca_bond_weight = self.coarse_ca_bond_weight if self.use_coarse_ca_bond_loss else 0.0
                        ca_pairwise_weight = self.coarse_ca_pairwise_weight if self.use_coarse_ca_pairwise_loss else 0.0
                        if ca_step_weight <= 0 and ca_bond_weight <= 0 and ca_pairwise_weight <= 0:
                            raise RuntimeError("SE3 coarse CA loss has no enabled components")
                        se3_ca_total, _ = coarse_ca_loss(
                            se3_ca_steps,
                            true_coords,
                            batch_idx=batch_idx,
                            pred_ca=se3_coords_step,
                            frames=ca_frames,
                            frame_offset="prev",
                            step_weight=ca_step_weight,
                            bond_weight=ca_bond_weight,
                            pairwise_weight=ca_pairwise_weight,
                            pairwise_max_seq_sep=self.coarse_ca_pairwise_max_seq_sep,
                            pairwise_max_pairs=self.coarse_ca_pairwise_max_pairs,
                            return_components=True,
                        )
                        raw_terms["se3_coarse_ca"] = se3_ca_total * self.coarse_ca_weight
                    if self.use_se3_residue_loss and true_q is not None and true_t is not None:
                        q_se3, t_se3 = self._derive_se3_qt(se3_coords_step, batch_idx)
                        true_t_origin = self._step_translations_to_origins(true_t, batch_idx=batch_idx)
                        raw_terms["fape_quat_se3"] = quaternion_fape_loss(
                            true_q, true_t_origin, q_se3, t_se3, batch=batch_idx
                        )
                        raw_terms["quat_geodesic_se3"] = quaternion_geodesic_loss(q_se3, true_q)

                if self.use_se3_angle_loss and out_s.get("angles") is not None and true_angles is not None:
                    se3_angle_pred = out_s["angles"][..., :3]
                    se3_angle_mask = torch.isfinite(se3_angle_pred) & torch.isfinite(true_angles)
                    if se3_angle_mask.any():
                        raw_terms["se3_angles"] = periodic_angle_smooth_l1(
                            se3_angle_pred[se3_angle_mask],
                            true_angles[se3_angle_mask],
                        )

                if self.use_se3_atom_refine:
                    atom_decoder = self.se3_atom_decoder
                    if atom_decoder is None:
                        raise RuntimeError("--use-se3-atom-refine requires an atom SE3 decoder")
                    if pred_coarse_atoms is None or true_coarse_atoms is None or true_R_atoms is None or true_ca_atoms is None:
                        raise RuntimeError("--use-se3-atom-refine requires --use-coarse-backbone-loss")
                    atom_type_count = max(
                        pred_coarse_atoms.shape[1],
                        int(getattr(atom_decoder, "num_atom_types", pred_coarse_atoms.shape[1])),
                    )
                    # The dataset has direct coordinates for the four coarse
                    # backbone atoms. Seed any additional atom types at CA so
                    # the SE3 decoder can still produce a complete atom axis.
                    pred_atom_seed = pred_coarse_atoms
                    if atom_type_count > pred_coarse_atoms.shape[1]:
                        pred_atom_seed = pred_coarse_atoms[:, :1].expand(-1, atom_type_count, -1).clone()
                        pred_atom_seed[:, :pred_coarse_atoms.shape[1]] = pred_coarse_atoms
                    # The first SE3 pass uses FoldTree2 discrete character
                    # IDs. This second pass is atom-level, so its first four
                    # channels are explicitly the coarse backbone order
                    # (CA, C, CB, N). Extra output slots are seeded as CA and
                    # use the CA atom type until dedicated atom labels exist.
                    atom_type_ids = torch.zeros(
                        (pred_atom_seed.shape[0], atom_type_count),
                        device=pred_atom_seed.device,
                        dtype=torch.long,
                    )
                    # Keep the public atom-type convention CA=0, CB=1, N=2,
                    # C=3 while preserving the existing tensor order
                    # (CA, C, CB, N) used by the coarse losses.
                    canonical_coarse_ids = torch.tensor(
                        [0, 3, 1, 2], device=pred_atom_seed.device, dtype=torch.long
                    )
                    atom_type_ids[:, :pred_coarse_atoms.shape[1]] = canonical_coarse_ids
                    out_atom = atom_decoder(
                        data_batch,
                        edge_attr_dict={
                            **(se3_edge_attr_dict or {}),
                            # Distance contacts are introduced only after the
                            # initial dot-product graph has produced coarse
                            # atom coordinates.
                            "use_distance_contacts": self.se3_use_distance_contacts,
                        },
                        coords_pred_atoms=pred_atom_seed,
                        atom_type_ids=atom_type_ids,
                    )
                    se3_atoms = out_atom.get("coors_out_atoms", None)
                    if se3_atoms is None:
                        raise RuntimeError("SE3 atom refinement did not return coors_out_atoms")
                    se3_atoms = se3_atoms.to(device=true_coarse_atoms.device, dtype=true_coarse_atoms.dtype)
                    data_batch["se3_atoms_pred"].x = se3_atoms
                    supervised_se3_atoms = se3_atoms[:, :pred_coarse_atoms.shape[1]]
                    data_batch["se3_ca_pred"].x = se3_atoms[:, ca_idx]
                    data_batch["se3_c_pred"].x = se3_atoms[:, c_idx]
                    data_batch["se3_cb_pred"].x = se3_atoms[:, cb_idx]
                    data_batch["se3_n_pred"].x = se3_atoms[:, n_idx]
                    if self.use_se3_atom_loss and self.se3_atom_weight > 0:
                        raw_terms["se3_atom_refine"] = self.se3_atom_weight * F.smooth_l1_loss(
                            supervised_se3_atoms,
                            true_coarse_atoms,
                            beta=0.5,
                        )
                    if self.use_se3_atom_fape_loss and self.se3_atom_fape_weight > 0:
                        raw_terms["se3_atom_fape"] = self.se3_atom_fape_weight * coarse_backbone_fape_loss(
                            true_coarse_atoms,
                            supervised_se3_atoms,
                            true_R_atoms,
                            pred_R_for_fape,
                            true_ca_atoms,
                            se3_atoms[:, ca_idx],
                            batch=batch_idx,
                        )
                    if self.use_se3_coarse_geometry_losses:
                        if self.use_coarse_backbone_atom_loss and self.coarse_backbone_atom_weight > 0:
                            raw_terms["se3_coarse_backbone_atoms"] = self.coarse_backbone_atom_weight * F.smooth_l1_loss(
                                supervised_se3_atoms,
                                true_coarse_atoms,
                                beta=0.5,
                            )
                        if self.use_coarse_c_loss and self.coarse_c_weight > 0:
                            raw_terms["se3_coarse_c"] = self.coarse_c_weight * F.smooth_l1_loss(
                                supervised_se3_atoms[:, c_idx],
                                true_coarse_atoms[:, c_idx],
                                beta=0.5,
                            )
                        if self.use_coarse_cb_loss and self.coarse_cb_weight > 0:
                            raw_terms["se3_coarse_cb"] = self.coarse_cb_weight * F.smooth_l1_loss(
                                supervised_se3_atoms[:, cb_idx],
                                true_coarse_atoms[:, cb_idx],
                                beta=0.5,
                            )
                        if self.use_coarse_n_loss and self.coarse_n_weight > 0:
                            raw_terms["se3_coarse_n"] = self.coarse_n_weight * F.smooth_l1_loss(
                                supervised_se3_atoms[:, n_idx],
                                true_coarse_atoms[:, n_idx],
                                beta=0.5,
                            )
                        if self.use_coarse_backbone_fape_loss and self.coarse_backbone_fape_weight > 0:
                            raw_terms["se3_coarse_backbone_fape"] = self.coarse_backbone_fape_weight * coarse_backbone_fape_loss(
                                true_coarse_atoms,
                                supervised_se3_atoms,
                                true_R_atoms,
                                pred_R_for_fape,
                                true_ca_atoms,
                                se3_atoms[:, ca_idx],
                                batch=batch_idx,
                            )
                        if self.use_coarse_backbone_angle_loss and self.coarse_backbone_angle_weight > 0 and true_angles is not None:
                            pred_bb_angles, pred_angle_mask = coarse_backbone_dihedrals_from_ca_frames(
                                se3_atoms[:, ca_idx],
                                pred_R_for_fape,
                                batch=batch_idx,
                                n_coords=se3_atoms[:, n_idx],
                                c_coords=se3_atoms[:, c_idx],
                            )
                            true_bb_angles, true_angle_mask = coarse_backbone_dihedrals_from_ca_frames(
                                true_ca_atoms,
                                true_R_atoms,
                                batch=batch_idx,
                                n_coords=true_coarse_atoms[:, n_idx],
                                c_coords=true_coarse_atoms[:, c_idx],
                            )
                            angle_mask = pred_angle_mask & true_angle_mask
                            if angle_mask.any():
                                angle_target = true_angles.to(device=pred_bb_angles.device, dtype=pred_bb_angles.dtype)
                                derived_delta = wrap_to_pi_torch(pred_bb_angles - angle_target)
                                frame_delta = wrap_to_pi_torch(pred_bb_angles - true_bb_angles)
                                raw_terms["se3_coarse_backbone_angles"] = self.coarse_backbone_angle_weight * (
                                    F.smooth_l1_loss(derived_delta[angle_mask], torch.zeros_like(derived_delta[angle_mask]))
                                    + 0.25 * F.smooth_l1_loss(frame_delta[angle_mask], torch.zeros_like(frame_delta[angle_mask]))
                                )
            except Exception as exc:
                # Fail fast instead of skipping SE3 on one rank only. Silent rank divergence
                # can deadlock distributed runs at optimizer sync boundaries.
                raise RuntimeError(
                    f"SE3 branch failed at {debug_label}: {exc}. "
                    "Reduce SE3 size, lower batch size, or run with --no-use-se3."
                ) from exc

        if len(raw_terms) == 0:
            if self.skip_empty_loss_batches:
                zero = torch.zeros((), device=z_local.device, dtype=torch.float32)
                for param in self.parameters():
                    if param.requires_grad:
                        zero = zero + param.float().sum() * 0.0
                return zero, raw_terms, raw_terms, True, batch_idx
            raise RuntimeError("No valid losses were constructed for this batch.")

        if self.nan_guard:
            bad_terms = [k for k, v in raw_terms.items() if not torch.isfinite(v).all()]
            if bad_terms:
                raise RuntimeError(f"Non-finite raw loss term(s) at {debug_label}: {bad_terms}")

        weighted_terms = self._kendall_weight_terms(raw_terms)
        total = torch.stack([v.float() for v in weighted_terms.values()]).sum()

        if self.nan_guard and (not torch.isfinite(total).all()):
            raise RuntimeError(f"Non-finite total loss at {debug_label}")

        return total, raw_terms, weighted_terms, se3_skip, batch_idx

    def training_step(self, batch, batch_idx):
        total_loss, raw_terms, weighted_terms, se3_skip, data_batch_idx = self._compute_total_loss(
            batch, debug_label=f"epoch={self.current_epoch + 1} step={batch_idx}"
        )
        self._log_loss_terms("train", total_loss, raw_terms, weighted_terms, se3_skip, batch, data_batch_idx)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, raw_terms, weighted_terms, se3_skip, data_batch_idx = self._compute_total_loss(
            batch, debug_label=f"val epoch={self.current_epoch + 1} step={batch_idx}"
        )
        self._log_loss_terms("val", total_loss, raw_terms, weighted_terms, se3_skip, batch, data_batch_idx)
        return total_loss

    def _log_loss_terms(self, stage: str, total_loss, raw_terms, weighted_terms, se3_skip, batch, data_batch_idx):
        batch_size = self._batch_size_for_logs(batch, data_batch_idx)
        on_step = stage == "train"
        self.log(
            f"{stage}/loss",
            total_loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        for name, value in raw_terms.items():
            self.log(
                f"{stage}/raw_{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        for name, value in weighted_terms.items():
            self.log(
                f"{stage}/weighted_{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        self.log(
            f"{stage}/se3_skip",
            float(se3_skip),
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        if self.use_uncertainty_weighting:
            for name, param in self.kendall_log_vars.items():
                sigma = torch.exp(0.5 * param)
                self.log(
                    f"{stage}/sigma_{name}",
                    sigma,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Periodic cache flushing can reduce allocator fragmentation in long runs.
        if self.cache_flush_interval > 0 and ((batch_idx + 1) % self.cache_flush_interval == 0):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if self.use_cuda_ipc_collect:
                    torch.cuda.ipc_collect()

        if self.gc_collect_interval > 0 and ((batch_idx + 1) % self.gc_collect_interval == 0):
            gc.collect()


def _cli_overridden_destinations(parser: argparse.ArgumentParser) -> set[str]:
    option_actions = parser._option_string_actions
    seen = set()
    for token in sys.argv[1:]:
        if token == "--":
            break
        if token.startswith("--") and "=" in token:
            option_name = token.split("=", 1)[0]
            action = option_actions.get(option_name)
            if action is not None:
                seen.add(action.dest)
            continue
        if token.startswith("--"):
            action = option_actions.get(token)
            if action is not None:
                seen.add(action.dest)
            continue
        if token.startswith("-") and len(token) > 1 and not token.startswith("--"):
            for option_name, action in option_actions.items():
                if option_name == token:
                    seen.add(action.dest)
                    break
    return seen


def load_config_into_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if not getattr(args, "config", None):
        return args

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    elif suffix == ".json":
        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh) or {}
    else:
        raise ValueError("Config file must be YAML (.yaml/.yml) or JSON (.json)")

    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} must contain a top-level dictionary/object")

    cli_overrides = _cli_overridden_destinations(parser)
    for key, value in config.items():
        if not hasattr(args, key):
            print(f"Warning: ignoring config key '{key}' because it is not a recognized argument.")
            continue
        if key in cli_overrides:
            print(f"  {key}: {getattr(args, key)} (from CLI, overriding config)")
            continue
        setattr(args, key, value)
        print(f"  {key}: {value} (from config)")
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Geometry-focused Lightning trainer from notebook final loop")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to a YAML or JSON config file. Command-line args override config-file values.",
    )
    parser.add_argument("--dataset", type=str, default="structs_training_mk2.h5", help="Path to HDF5 StructureDataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Optional cap on train batches per epoch for smoke tests/debug runs",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Optional cap on validation batches per epoch for smoke tests/debug runs",
    )
    parser.add_argument("--batch-size", type=int, default=5, help="Micro-batch size")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Validation batch size (defaults to batch size)")
    parser.add_argument(
        "--target-effective-batch-size",
        type=int,
        default=10,
        help="Target effective batch size via gradient accumulation",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument(
        "--allow-hdf5-multiprocessing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow num_workers>0 on HDF5 datasets (can deadlock with h5py-backed datasets)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of dataset held out for validation")
    parser.add_argument("--val-count", type=int, default=None, help="Exact validation item count; overrides val fraction")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for deterministic train/validation split")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay")
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "step"],
        default="none",
        help="Optional learning rate scheduler",
    )
    parser.add_argument(
        "--lr-scheduler-interval",
        choices=["epoch", "step"],
        default="epoch",
        help="How often to step the LR scheduler",
    )
    parser.add_argument(
        "--lr-scheduler-frequency",
        type=int,
        default=1,
        help="Scheduler step frequency for Lightning scheduler config",
    )
    parser.add_argument(
        "--lr-scheduler-monitor",
        type=str,
        default="train/loss_epoch",
        help="Monitor name for schedulers that require one",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        type=int,
        default=0,
        help="Warmup epochs used before cosine decay",
    )
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum LR for cosine decay")
    parser.add_argument("--lr-step-size", type=int, default=10, help="Epoch/step interval for StepLR")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="Decay factor for StepLR")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clip norm")
    parser.add_argument(
        "--cache-flush-interval",
        type=int,
        default=0,
        help="Flush CUDA allocator cache every N train batches (0 disables)",
    )
    parser.add_argument(
        "--gc-collect-interval",
        type=int,
        default=0,
        help="Run Python gc.collect() every N train batches (0 disables)",
    )
    parser.add_argument(
        "--use-cuda-ipc-collect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run torch.cuda.ipc_collect() when cache flushing",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--pretrained-encoder-path",
        type=str,
        default="/home/dmoi/projects/foldtree2/models/notebook/final_30char_mk2_contacts_aa_encoder_epoch_4.pth",
        help="Preferred pretrained encoder checkpoint",
    )
    parser.add_argument(
        "--pretrained-encoder-full-path",
        type=str,
        default="/home/dmoi/projects/foldtree2/models/notebook/final_30char_mk2_contacts_aa_encoder_full_epoch_4.pt",
        help="Fallback full-model checkpoint used for state_dict loading",
    )
    parser.add_argument(
        "--pretrained-geometry-decoder-path",
        type=str,
        default=None,
        help="Frozen production geometry encoder-decoder pair checkpoint for the production SE3 entry point",
    )
    parser.add_argument(
        "--production-coordinate-source",
        choices=["auto", "bottleneck", "z", "trans_pred", "trans_local_pred"],
        default="auto",
        help="3D seed exported by the production geometry decoder",
    )
    parser.add_argument(
        "--production-coordinate-scale",
        type=float,
        default=1.0,
        help="Scale applied to the production decoder's 3D seed before SE3 refinement",
    )
    parser.add_argument(
        "--se3-num-atom-types",
        type=int,
        default=0,
        help="Number of atom types emitted by SE3; 0 uses the encoder vocabulary size",
    )
    parser.add_argument(
        "--fallback-latent-dim",
        type=int,
        default=64,
        help="Latent dim used by fallback FrozenProjectionEncoder",
    )
    parser.add_argument(
        "--use-se3",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable optional SE3 decoder branch",
    )
    parser.add_argument(
        "--se3-input-source",
        choices=["reconstructed_ca", "coarse_ca", "geometry_dot_contacts"],
        default="coarse_ca",
        help="Coordinate/contact seed for the residue-level SE3 branch",
    )
    parser.add_argument(
        "--se3-use-codebook-vectors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Concatenate selected encoder codebook vectors to continuous encoder features for SE3",
    )
    parser.add_argument(
        "--se3-contact-sketch-top-k",
        type=int,
        default=16,
        help="Top-k geometry dot-product contacts per residue when --se3-input-source=geometry_dot_contacts",
    )
    parser.add_argument(
        "--se3-contact-sketch-threshold",
        type=float,
        default=0.0,
        help="Minimum normalized dot product included in the SE3 contact sketch",
    )
    parser.add_argument(
        "--se3-use-distance-contacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add distance-based contacts from the predicted coordinates to the SE3 graph",
    )
    parser.add_argument(
        "--se3-distance-contact-cutoff",
        type=float,
        default=8.0,
        help="Distance-contact cutoff in angstroms",
    )
    parser.add_argument(
        "--se3-contact-sketch-min-seq-sep",
        type=int,
        default=3,
        help="Minimum sequence separation for geometry dot-product contacts",
    )
    parser.add_argument(
        "--se3-contact-coord-scale",
        type=float,
        default=10.0,
        help="Scale applied to 3D geometry contact embeddings before feeding them as SE3 coordinates",
    )
    parser.add_argument(
        "--se3-contact-local-window",
        type=int,
        default=1,
        help="Always include residue pairs within this sequence window in the SE3 dot-contact graph",
    )
    parser.add_argument(
        "--se3-max-nodes",
        type=int,
        default=0,
        help="Skip SE3 batches with more than this many residue/atom nodes (0 disables)",
    )
    parser.add_argument(
        "--train-se3-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the geometry decoder and optimize only the SE3 branch",
    )
    parser.add_argument(
        "--sanitize-nonfinite-grads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace non-finite gradient entries with zero before optimizer.step()",
    )
    parser.add_argument(
        "--skip-empty-loss-batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip batches where all active loss terms lack finite supervision",
    )
    parser.add_argument(
        "--use-se3-atom-refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run an experimental SE3 refinement pass over coarse CA/C/CB/N atom coordinates",
    )
    parser.add_argument(
        "--use-frame-fape-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train on transformer quaternion/frame FAPE loss",
    )
    parser.add_argument(
        "--use-quat-geodesic-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train on transformer quaternion geodesic loss",
    )
    parser.add_argument(
        "--use-decoder-angle-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train on direct angle-head reconstruction loss",
    )
    parser.add_argument(
        "--use-coarse-ca-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the coarse CA objective to the geometry decoder loss stack",
    )
    parser.add_argument(
        "--use-coarse-ca-step-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local/global CA step component inside the coarse CA objective",
    )
    parser.add_argument(
        "--use-coarse-ca-bond-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CA bond-length component inside the coarse CA objective",
    )
    parser.add_argument(
        "--use-coarse-ca-pairwise-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pairwise CA distance component inside the coarse CA objective",
    )
    parser.add_argument(
        "--use-coarse-backbone-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add coarse CA/C/CB/N atom placement and atom-FAPE objectives",
    )
    parser.add_argument(
        "--use-coarse-backbone-atom-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use aggregate smooth-L1 loss over the coarse CA/C/CB/N atom bundle",
    )
    parser.add_argument(
        "--use-coarse-c-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use separate smooth-L1 placement loss for coarse C atoms",
    )
    parser.add_argument(
        "--use-coarse-cb-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use separate smooth-L1 placement loss for coarse CB atoms",
    )
    parser.add_argument(
        "--use-coarse-n-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use separate smooth-L1 placement loss for coarse N atoms",
    )
    parser.add_argument(
        "--use-coarse-backbone-fape-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use atom-FAPE over placed coarse CA/C/CB/N atoms",
    )
    parser.add_argument(
        "--use-coarse-backbone-angle-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use phi/psi/omega loss derived from placed coarse N/CA/C atoms",
    )
    parser.add_argument("--coarse-ca-weight", type=float, default=1.0, help="Outer weight for the coarse CA term")
    parser.add_argument("--coarse-ca-step-weight", type=float, default=1.0, help="Local CA step loss weight")
    parser.add_argument("--coarse-ca-bond-weight", type=float, default=0.1, help="CA bond-length regularizer weight")
    parser.add_argument("--coarse-ca-pairwise-weight", type=float, default=0.25, help="CA pairwise distance loss weight")
    parser.add_argument(
        "--coarse-backbone-atom-weight",
        type=float,
        default=0.05,
        help="Smooth-L1 weight for directly placed coarse CA/CB/N atoms",
    )
    parser.add_argument("--coarse-c-weight", type=float, default=0.05, help="Smooth-L1 weight for placed coarse C atoms")
    parser.add_argument("--coarse-cb-weight", type=float, default=0.05, help="Smooth-L1 weight for placed coarse CB atoms")
    parser.add_argument("--coarse-n-weight", type=float, default=0.05, help="Smooth-L1 weight for placed coarse N atoms")
    parser.add_argument(
        "--coarse-backbone-fape-weight",
        type=float,
        default=0.25,
        help="FAPE weight for placed coarse CA/CB/N atoms",
    )
    parser.add_argument(
        "--coarse-backbone-angle-weight",
        type=float,
        default=0.0,
        help="Optional loss weight for phi/psi/omega derived from coarse N/CA/C atom placements",
    )
    parser.add_argument(
        "--se3-atom-weight",
        type=float,
        default=0.05,
        help="Smooth-L1 loss weight for SE3-refined CA/C/CB/N atom coordinates",
    )
    parser.add_argument(
        "--se3-atom-fape-weight",
        type=float,
        default=0.25,
        help="Atom-FAPE loss weight for SE3-refined CA/C/CB/N atom coordinates",
    )
    parser.add_argument(
        "--coarse-ca-step-frame",
        choices=["global", "prev"],
        default="prev",
        help="Frame for CA step targets: raw global deltas or previous residue local frame",
    )
    parser.add_argument(
        "--coarse-ca-pairwise-max-seq-sep",
        type=int,
        default=64,
        help="Maximum sequence separation for coarse CA pairwise supervision",
    )
    parser.add_argument(
        "--coarse-ca-pairwise-max-pairs",
        type=int,
        default=4096,
        help="Maximum sampled CA pairs per chain for coarse CA pairwise supervision",
    )
    parser.add_argument(
        "--rotation-target-frame",
        choices=["local", "absolute"],
        default="local",
        help="Quaternion target frame: local previous-residue rotations or absolute backbone frames",
    )
    parser.add_argument(
        "--transformer-width",
        type=int,
        default=96,
        help="Width used for transformer geometry hidden channels",
    )
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=2,
        help="Number of transformer geometry layers",
    )
    parser.add_argument(
        "--transformer-nheads",
        type=int,
        default=4,
        help="Number of transformer attention heads",
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.05,
        help="Dropout for transformer geometry decoder",
    )
    parser.add_argument(
        "--rt-hidden",
        type=str,
        default="128,64,32",
        help="Comma-separated translation head hidden sizes",
    )
    parser.add_argument(
        "--rotation-hidden",
        type=str,
        default="128,64,32",
        help="Comma-separated rotation/quaternion head hidden sizes",
    )
    parser.add_argument(
        "--ca-step-hidden",
        type=str,
        default="128,64,32",
        help="Comma-separated CA step head hidden sizes",
    )
    parser.add_argument(
        "--ss-hidden",
        type=str,
        default="64,32,16",
        help="Comma-separated SS head hidden sizes",
    )
    parser.add_argument(
        "--angles-hidden",
        type=str,
        default="64,32,16",
        help="Comma-separated angle head hidden sizes",
    )
    parser.add_argument(
        "--se3-hidden",
        type=int,
        default=4,
        help="SE3 hidden channel size",
    )
    parser.add_argument(
        "--se3-out-channels",
        type=int,
        default=96,
        help="SE3 angle head output channels",
    )
    parser.add_argument(
        "--se3-depth",
        type=int,
        default=4,
        help="SE3 GotenNet depth",
    )
    parser.add_argument(
        "--se3-heads",
        type=int,
        default=5,
        help="SE3 attention heads",
    )
    parser.add_argument(
        "--se3-dim-head",
        type=int,
        default=20,
        help="SE3 attention head dimension",
    )
    for variant in ("residue", "atom"):
        parser.add_argument(
            f"--se3-{variant}-hidden",
            type=int,
            default=None,
            help=f"Override hidden width for the specialized {variant} SE3 decoder",
        )
        parser.add_argument(
            f"--se3-{variant}-depth",
            type=int,
            default=None,
            help=f"Override depth for the specialized {variant} SE3 decoder",
        )
        parser.add_argument(
            f"--se3-{variant}-heads",
            type=int,
            default=None,
            help=f"Override attention heads for the specialized {variant} SE3 decoder",
        )
        parser.add_argument(
            f"--se3-{variant}-dim-head",
            type=int,
            default=None,
            help=f"Override attention head dimension for the specialized {variant} SE3 decoder",
        )
    parser.add_argument(
        "--use-se3-residue-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use residue-level SE3 coordinate frame losses",
    )
    parser.add_argument(
        "--use-se3-angle-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SE3 angle-head reconstruction loss",
    )
    parser.add_argument(
        "--use-se3-atom-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use smooth-L1 loss on SE3-refined CA/C/CB/N atom coordinates",
    )
    parser.add_argument(
        "--use-se3-atom-fape-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use atom-FAPE loss on SE3-refined CA/C/CB/N atom coordinates",
    )
    parser.add_argument(
        "--use-se3-coarse-geometry-losses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the shared coarse CA/backbone objectives to SE3 outputs",
    )
    parser.add_argument(
        "--use-uncertainty-weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Kendall uncertainty weighting for active geometry terms",
    )
    parser.add_argument(
        "--nan-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise runtime errors on non-finite losses",
    )
    parser.add_argument("--accelerator", type=str, default="auto", help="Lightning accelerator")
    parser.add_argument("--devices", type=str, default="auto", help="Lightning devices value")
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Lightning strategy (e.g. auto, ddp, ddp_find_unused_parameters_true)",
    )
    parser.add_argument("--precision", type=str, default="32-true", help="Lightning precision setting")
    parser.add_argument("--log-every-n-steps", type=int, default=10, help="Logging interval")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/notebook_geometry",
        help="Directory for Lightning checkpoints",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=1,
        help="How many best checkpoints to keep (monitoring val/loss when validation is enabled, else train/loss_epoch)",
    )
    parser.add_argument(
        "--enable-epoch-visualizations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save a fixed-sample reconstruction figure after each validation epoch",
    )
    parser.add_argument(
        "--visualization-dir",
        type=str,
        default=None,
        help="Directory for epoch reconstruction figures (defaults to checkpoint-dir/visualizations)",
    )
    parser.add_argument(
        "--visualization-sample-index",
        type=int,
        default=0,
        help="Dataset index used for epoch reconstruction visualizations",
    )
    parser.add_argument(
        "--visualization-max-residues",
        type=int,
        default=0,
        help="Maximum residues plotted in epoch figures; 0 plots the full sample",
    )
    args = parser.parse_args()
    args = load_config_into_args(args, parser)
    return args


def parse_devices(devices_arg: str):
    if isinstance(devices_arg, int):
        return devices_arg
    if devices_arg == "auto":
        return "auto"
    if "," in devices_arg:
        return [int(x.strip()) for x in devices_arg.split(",") if x.strip()]
    try:
        return int(devices_arg)
    except ValueError:
        return devices_arg


def parse_int_list(value: str) -> list:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def infer_strategy(accelerator: str, devices, strategy_arg: str) -> str:
    if strategy_arg != "auto":
        return strategy_arg

    is_multi_gpu = False
    if accelerator in ("gpu", "cuda", "auto"):
        if isinstance(devices, int):
            is_multi_gpu = devices > 1
        elif isinstance(devices, list):
            is_multi_gpu = len(devices) > 1
        elif isinstance(devices, str):
            is_multi_gpu = devices not in ("auto", "1")

    if is_multi_gpu:
        # This model may skip some loss terms/branches at runtime; allow DDP unused params.
        return "ddp_find_unused_parameters_true"

    return "auto"


def build_encoder(args, data_sample, device):
    data_sample = ensure_float32_inplace(data_sample)
    data_sample = ensure_edge_attrs_inplace(data_sample, edge_dim=1)

    encoder = None
    load_errors = []
    candidates = [Path(args.pretrained_encoder_path), Path(args.pretrained_encoder_full_path)]

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            encoder, load_info = load_encoder_any(candidate, full_model_template=Path(args.pretrained_encoder_full_path))
            encoder = encoder.to(device)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False

            with torch.no_grad():
                _z_test, _vq_test = encoder(data_sample)

            print(
                f"Loaded pretrained encoder via {load_info['mode']} "
                f"from {load_info['source']} epoch={load_info.get('epoch')}"
            )
            if "missing" in load_info or "unexpected" in load_info:
                print("missing_keys:", len(load_info.get("missing", [])))
                print("unexpected_keys:", len(load_info.get("unexpected", [])))
            break
        except Exception as exc:
            load_errors.append((str(candidate), str(exc)))
            encoder = None

    if encoder is None:
        for src, err in load_errors:
            print(f"Encoder load failed for {src}: {err}")

        in_dim = int(data_sample["res"].x.shape[-1])
        encoder = FrozenProjectionEncoder(in_dim=in_dim, out_dim=args.fallback_latent_dim).to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        print("Using frozen projection encoder fallback.")

    with torch.no_grad():
        z_probe, vq_probe = encoder(data_sample)
    latent_dim = int(z_probe.shape[-1])
    print("latent_dim:", latent_dim, "vq_probe:", float(vq_probe.detach().cpu()))

    return encoder, latent_dim


def build_se3_decoder(
    input_dim: int,
    data_sample,
    device,
    args,
    se3_num_atom_types: int,
    hidden: Optional[int] = None,
    depth: Optional[int] = None,
    heads: Optional[int] = None,
    dim_head: Optional[int] = None,
    out_channels: Optional[int] = None,
):
    if not SE3_AVAILABLE:
        print("SE3 requested but module is unavailable; continuing without SE3 branch.")
        return None
    se3_device = device if device.type == "cuda" else torch.device("cpu")
    try:
        decoder = se3_denoiser(
            in_channels=input_dim,
            hidden_channels=[hidden if hidden is not None else args.se3_hidden],
            out_channels=out_channels if out_channels is not None else args.se3_out_channels,
            num_embeddings=30,
            commitment_cost=0.25,
            metadata={"edge_types": data_sample.edge_types},
            edge_dim=1,
            depth=depth if depth is not None else args.se3_depth,
            heads=heads if heads is not None else args.se3_heads,
            dim_head=dim_head if dim_head is not None else args.se3_dim_head,
            return_coors=True,
            num_atom_types=se3_num_atom_types,
        ).to(se3_device)
        decoder.device = se3_device
        return decoder
    except Exception as exc:
        print(f"SE3 decoder unavailable at runtime: {exc}")
        return None


def build_decoders(latent_dim: int, data_sample, device, use_se3: bool, args, se3_num_atom_types: int):
    rt_hidden = parse_int_list(args.rt_hidden)
    rotation_hidden = parse_int_list(args.rotation_hidden)
    ca_step_hidden = parse_int_list(args.ca_step_hidden)
    ss_hidden = parse_int_list(args.ss_hidden)
    angles_hidden = parse_int_list(args.angles_hidden)

    transformer_geom_decoder = Transformer_Geometry_Decoder(
        in_channels={"res": latent_dim},
        hidden_channels={("res", "backbone", "res"): [args.transformer_width] * 3},
        concat_positions=True,
        nheads=args.transformer_nheads,
        layers=args.transformer_layers,
        RTdecoder_hidden=rt_hidden,
        rotationdecoder_hidden=rotation_hidden,
        castepdecoder_hidden=ca_step_hidden,
        ssdecoder_hidden=ss_hidden,
        anglesdecoder_hidden=angles_hidden,
        dropout=args.transformer_dropout,
        normalize=True,
        residual=False,
        learn_positions=False,
        output_rt=True,
        output_ca_steps=True,
        output_ss=True,
        output_angles=True,
    ).to(device)

    se3_decoder = None
    if use_se3:
        se3_input_dim = latent_dim * (2 if getattr(args, "se3_use_codebook_vectors", False) else 1)
        se3_decoder = build_se3_decoder(
            se3_input_dim, data_sample, device, args, se3_num_atom_types
        )
        if se3_decoder is not None:
            print(f"SE3 decoder initialized on {se3_decoder.device}.")

    return transformer_geom_decoder, se3_decoder


def main():
    args = parse_args()

    accelerator_norm = str(args.accelerator).strip().lower()
    if accelerator_norm in {"gpu", "cuda", "auto"} and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        print("Set torch float32 matmul precision to 'high' for CUDA Tensor Core performance.")

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # h5py file handles inside Dataset objects are commonly not safe with multi-process workers.
    if (
        dataset_path.suffix.lower() in {".h5", ".hdf5"}
        and args.num_workers > 0
        and not args.allow_hdf5_multiprocessing
    ):
        print(
            "Warning: forcing --num-workers=0 for HDF5 dataset to avoid h5py worker deadlocks. "
            "Use --allow-hdf5-multiprocessing to override."
        )
        args.num_workers = 0

    data_module = GeometryOnlyDataModule(
        dataset_path=str(dataset_path),
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        val_count=args.val_count,
        split_seed=args.split_seed,
    )
    data_module.setup("fit")
    train_size = len(data_module.train_dataset) if data_module.train_dataset is not None else 0
    val_size = len(data_module.val_dataset) if data_module.val_dataset is not None else 0
    print(f"Dataset split: train={train_size} val={val_size} split_seed={args.split_seed}")
    print(
        "Active loss switches: "
        f"frame_fape={args.use_frame_fape_loss} "
        f"quat_geodesic={args.use_quat_geodesic_loss} "
        f"angle_head={args.use_decoder_angle_loss} "
        f"coarse_ca={args.use_coarse_ca_loss} "
        f"coarse_ca_components=(step={args.use_coarse_ca_step_loss}, bond={args.use_coarse_ca_bond_loss}, pairwise={args.use_coarse_ca_pairwise_loss}) "
        f"coarse_backbone={args.use_coarse_backbone_loss} "
        f"coarse_atoms=(bundle={args.use_coarse_backbone_atom_loss}, c={args.use_coarse_c_loss}, cb={args.use_coarse_cb_loss}, n={args.use_coarse_n_loss}) "
        f"coarse_backbone_fape={args.use_coarse_backbone_fape_loss} "
        f"coarse_backbone_angles={args.use_coarse_backbone_angle_loss} "
        f"se3={args.use_se3} "
        f"se3_input={args.se3_input_source} "
        f"train_se3_only={args.train_se3_only}"
    )

    probe_loader = DataLoader(data_module.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    data_sample = next(iter(probe_loader))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    encoder, latent_dim = build_encoder(args, data_sample.to(device), device)
    se3_num_atom_types = max(4, int(args.se3_num_atom_types or getattr(encoder, "num_embeddings", 20)))
    transformer_geom_decoder, se3_decoder = build_decoders(
        latent_dim,
        data_sample,
        device,
        args.use_se3,
        args,
        se3_num_atom_types=se3_num_atom_types,
    )

    module = GeometryFocusedModule(
        encoder=encoder,
        transformer_geom_decoder=transformer_geom_decoder,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_name=args.lr_scheduler,
        lr_scheduler_interval=args.lr_scheduler_interval,
        lr_scheduler_frequency=args.lr_scheduler_frequency,
        lr_scheduler_monitor=args.lr_scheduler_monitor,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_min=args.lr_min,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        max_epochs=args.epochs,
        clip_grad_norm=args.clip_grad,
        cache_flush_interval=args.cache_flush_interval,
        gc_collect_interval=args.gc_collect_interval,
        use_cuda_ipc_collect=args.use_cuda_ipc_collect,
        use_uncertainty_weighting=args.use_uncertainty_weighting,
        use_se3=args.use_se3,
        se3_decoder=se3_decoder,
        nan_guard=args.nan_guard,
        use_frame_fape_loss=args.use_frame_fape_loss,
        use_quat_geodesic_loss=args.use_quat_geodesic_loss,
        use_decoder_angle_loss=args.use_decoder_angle_loss,
        use_coarse_ca_loss=args.use_coarse_ca_loss,
        use_coarse_ca_step_loss=args.use_coarse_ca_step_loss,
        use_coarse_ca_bond_loss=args.use_coarse_ca_bond_loss,
        use_coarse_ca_pairwise_loss=args.use_coarse_ca_pairwise_loss,
        coarse_ca_weight=args.coarse_ca_weight,
        coarse_ca_step_weight=args.coarse_ca_step_weight,
        coarse_ca_bond_weight=args.coarse_ca_bond_weight,
        coarse_ca_pairwise_weight=args.coarse_ca_pairwise_weight,
        coarse_ca_step_frame=args.coarse_ca_step_frame,
        coarse_ca_pairwise_max_seq_sep=args.coarse_ca_pairwise_max_seq_sep,
        coarse_ca_pairwise_max_pairs=args.coarse_ca_pairwise_max_pairs,
        rotation_target_frame=args.rotation_target_frame,
        use_coarse_backbone_loss=args.use_coarse_backbone_loss,
        use_coarse_backbone_atom_loss=args.use_coarse_backbone_atom_loss,
        use_coarse_c_loss=args.use_coarse_c_loss,
        use_coarse_cb_loss=args.use_coarse_cb_loss,
        use_coarse_n_loss=args.use_coarse_n_loss,
        use_coarse_backbone_fape_loss=args.use_coarse_backbone_fape_loss,
        use_coarse_backbone_angle_loss=args.use_coarse_backbone_angle_loss,
        coarse_backbone_atom_weight=args.coarse_backbone_atom_weight,
        coarse_c_weight=args.coarse_c_weight,
        coarse_cb_weight=args.coarse_cb_weight,
        coarse_n_weight=args.coarse_n_weight,
        coarse_backbone_fape_weight=args.coarse_backbone_fape_weight,
        coarse_backbone_angle_weight=args.coarse_backbone_angle_weight,
        se3_input_source=args.se3_input_source,
        se3_contact_sketch_top_k=args.se3_contact_sketch_top_k,
        se3_contact_sketch_threshold=args.se3_contact_sketch_threshold,
        se3_contact_sketch_min_seq_sep=args.se3_contact_sketch_min_seq_sep,
        se3_contact_coord_scale=args.se3_contact_coord_scale,
        se3_contact_local_window=args.se3_contact_local_window,
        se3_max_nodes=args.se3_max_nodes,
        train_se3_only=args.train_se3_only,
        sanitize_nonfinite_grads=args.sanitize_nonfinite_grads,
        skip_empty_loss_batches=args.skip_empty_loss_batches,
        use_se3_atom_refine=args.use_se3_atom_refine,
        use_se3_residue_loss=args.use_se3_residue_loss,
        use_se3_angle_loss=args.use_se3_angle_loss,
        use_se3_atom_loss=args.use_se3_atom_loss,
        use_se3_atom_fape_loss=args.use_se3_atom_fape_loss,
        se3_atom_weight=args.se3_atom_weight,
        se3_atom_fape_weight=args.se3_atom_fape_weight,
    )

    accum_steps = max(1, math.ceil(args.target_effective_batch_size / max(1, args.batch_size)))
    effective_batch_size = args.batch_size * accum_steps
    print(
        f"Training config: micro_batch_size={args.batch_size} accum_steps={accum_steps} "
        f"effective_batch_size={effective_batch_size} epochs={args.epochs}"
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="geometry-{epoch:02d}-{step}",
        monitor="val/loss" if data_module.val_dataset is not None else "train/loss_epoch",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
    )

    parsed_devices = parse_devices(args.devices)
    strategy = infer_strategy(args.accelerator, parsed_devices, args.strategy)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=parsed_devices,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=accum_steps,
        gradient_clip_val=args.clip_grad,
        gradient_clip_algorithm="norm",
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
