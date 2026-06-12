#!/usr/bin/env python3
"""Geometry-focused PyTorch Lightning trainer.

Ports the final training-loop logic from the notebook:
foldtree2/notebooks/experiments/test_monodecoders _folding.ipynb

The script intentionally focuses on the geometry loop only.
"""

from __future__ import annotations

import argparse
import gc
import math
import random
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from foldtree2.src import encoder as ecdr
from foldtree2.src import pdbgraphmk2
from foldtree2.src.losses.fape import (
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
    def __init__(self, dataset_path: str, batch_size: int, num_workers: int):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.struct_dat = None

    def setup(self, stage: Optional[str] = None):
        if self.struct_dat is None:
            self.struct_dat = pdbgraphmk2.StructureDataset(self.dataset_path)

    def train_dataloader(self):
        return DataLoader(
            self.struct_dat,
            batch_size=self.batch_size,
            shuffle=True,
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
        clip_grad_norm: float,
        cache_flush_interval: int,
        gc_collect_interval: int,
        use_cuda_ipc_collect: bool,
        use_uncertainty_weighting: bool,
        use_se3: bool,
        se3_decoder: Optional[nn.Module] = None,
        nan_guard: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.transformer_geom_decoder = transformer_geom_decoder
        self.se3_decoder = se3_decoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.cache_flush_interval = int(cache_flush_interval)
        self.gc_collect_interval = int(gc_collect_interval)
        self.use_cuda_ipc_collect = bool(use_cuda_ipc_collect)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_se3 = use_se3 and (se3_decoder is not None)
        self.nan_guard = nan_guard

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.uncertainty_term_names = [
            "fape_quat",
            "quat_geodesic",
            "angles",
            "fape_quat_se3",
            "quat_geodesic_se3",
            "se3_angles",
        ]
        if self.use_uncertainty_weighting:
            self.kendall_log_vars = torch.nn.ParameterDict(
                {name: torch.nn.Parameter(torch.zeros((), dtype=torch.float32)) for name in self.uncertainty_term_names}
            )

        self.save_hyperparameters(ignore=["encoder", "transformer_geom_decoder", "se3_decoder"])

    def configure_optimizers(self):
        train_params = list(self.transformer_geom_decoder.parameters())
        if self.use_se3 and self.se3_decoder is not None:
            train_params += list(self.se3_decoder.parameters())
        if self.use_uncertainty_weighting:
            train_params += list(self.kendall_log_vars.parameters())

        return torch.optim.AdamW(train_params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def on_train_epoch_start(self):
        self.encoder.eval()

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
    def _get_aa_identity_ids(decoder_out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        aa_logits = decoder_out.get("aa", None)
        if aa_logits is None:
            aa_logits = decoder_out.get("aa_pred", None)
        if aa_logits is None:
            return None
        if aa_logits.ndim != 2:
            raise RuntimeError(f"Expected AA logits with shape [N,20], got {tuple(aa_logits.shape)}")
        return torch.argmax(aa_logits, dim=-1).to(dtype=torch.long)

    def _compute_total_loss(self, data_batch, debug_label: str = ""):
        data_batch = ensure_float32_inplace(data_batch)
        data_batch = ensure_edge_attrs_inplace(data_batch, edge_dim=getattr(self.encoder, "edge_dim", 1))

        with torch.no_grad():
            z_local, _ = self.encoder(data_batch)
            ft2_token_ids = self._get_ft2_token_ids(z_local)
        data_batch["res"].x = z_local

        out_local = self.transformer_geom_decoder(data_batch, contact_pred_index=None)
        aa_identity_ids = self._get_aa_identity_ids(out_local)
        true_R, true_t, true_q, true_angles, true_coords, batch_idx = get_true_geometry(data_batch)

        pred_rt = out_local["rt_pred"]
        pred_q = pred_rt[..., :4]
        pred_t = pred_rt[..., 4:]
        pred_R = quaternion_to_rotation_matrix(pred_q)

        raw_terms: Dict[str, torch.Tensor] = {}

        if true_q is not None and true_t is not None:
            true_t_origin = self._step_translations_to_origins(true_t, batch_idx=batch_idx)
            pred_t_origin = self._step_translations_to_origins(pred_t, batch_idx=batch_idx)
            raw_terms["fape_quat"] = quaternion_fape_loss(true_q, true_t_origin, pred_q, pred_t_origin, batch=batch_idx)
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

        if out_local.get("angles") is not None and true_angles is not None:
            raw_terms["angles"] = periodic_angle_smooth_l1(out_local["angles"], true_angles)

        se3_skip = False
        if self.use_se3 and self.se3_decoder is not None:
            try:
                if ft2_token_ids is None:
                    raise RuntimeError(
                        "SE3 branch requires FoldTree2 token ids from encoder.vector_quantizer, "
                        "but the active encoder does not expose discretize_z()."
                    )
                out_s = self.se3_decoder(
                    data_batch,
                    ft2_token_ids=ft2_token_ids,
                    aa_identity_ids=aa_identity_ids,
                    coords_pred=pred_coords_local,
                )
                if out_s.get("coors_out") is not None and true_q is not None and true_t is not None:
                    se3_coords_step = self._flatten_batched_coords(out_s["coors_out"], batch_idx)
                    se3_coords_step = se3_coords_step.to(pred_coords_local.device, dtype=pred_coords_local.dtype)
                    q_se3, t_se3 = self._derive_se3_qt(se3_coords_step, batch_idx)
                    true_t_origin = self._step_translations_to_origins(true_t, batch_idx=batch_idx)
                    raw_terms["fape_quat_se3"] = quaternion_fape_loss(true_q, true_t_origin, q_se3, t_se3, batch=batch_idx)
                    raw_terms["quat_geodesic_se3"] = quaternion_geodesic_loss(q_se3, true_q)

                if out_s.get("angles") is not None and true_angles is not None:
                    raw_terms["se3_angles"] = periodic_angle_smooth_l1(out_s["angles"][..., :3], true_angles)
            except Exception as exc:
                # Fail fast instead of skipping SE3 on one rank only. Silent rank divergence
                # can deadlock distributed runs at optimizer sync boundaries.
                raise RuntimeError(
                    f"SE3 branch failed at {debug_label}: {exc}. "
                    "Reduce SE3 size, lower batch size, or run with --no-use-se3."
                ) from exc

        if len(raw_terms) == 0:
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

        batch_size = self._batch_size_for_logs(batch, data_batch_idx)
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        for name, value in raw_terms.items():
            self.log(f"train/raw_{name}", value, on_step=False, on_epoch=True, batch_size=batch_size)
        for name, value in weighted_terms.items():
            self.log(f"train/weighted_{name}", value, on_step=False, on_epoch=True, batch_size=batch_size)

        self.log("train/se3_skip", float(se3_skip), on_step=True, on_epoch=True, batch_size=batch_size)

        if self.use_uncertainty_weighting:
            for name, param in self.kendall_log_vars.items():
                sigma = torch.exp(0.5 * param)
                self.log(f"train/sigma_{name}", sigma, on_step=False, on_epoch=True, batch_size=batch_size)

        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Periodic cache flushing can reduce allocator fragmentation in long runs.
        if self.cache_flush_interval > 0 and ((batch_idx + 1) % self.cache_flush_interval == 0):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if self.use_cuda_ipc_collect:
                    torch.cuda.ipc_collect()

        if self.gc_collect_interval > 0 and ((batch_idx + 1) % self.gc_collect_interval == 0):
            gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description="Geometry-focused Lightning trainer from notebook final loop")
    parser.add_argument("--dataset", type=str, default="structs_training_mk2.h5", help="Path to HDF5 StructureDataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=5, help="Micro-batch size")
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
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay")
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
        help="Comma-separated RT head hidden sizes",
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
        help="How many best checkpoints to keep (monitoring train/loss_epoch)",
    )
    return parser.parse_args()


def parse_devices(devices_arg: str):
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


def build_decoders(latent_dim: int, data_sample, device, use_se3: bool, args, se3_num_atom_types: int):
    rt_hidden = parse_int_list(args.rt_hidden)
    ss_hidden = parse_int_list(args.ss_hidden)
    angles_hidden = parse_int_list(args.angles_hidden)

    transformer_geom_decoder = Transformer_Geometry_Decoder(
        in_channels={"res": latent_dim},
        hidden_channels={("res", "backbone", "res"): [args.transformer_width] * 3},
        concat_positions=True,
        nheads=args.transformer_nheads,
        layers=args.transformer_layers,
        RTdecoder_hidden=rt_hidden,
        ssdecoder_hidden=ss_hidden,
        anglesdecoder_hidden=angles_hidden,
        dropout=args.transformer_dropout,
        normalize=True,
        residual=False,
        learn_positions=False,
        output_rt=True,
        output_ss=True,
        output_angles=True,
    ).to(device)

    se3_decoder = None
    if use_se3:
        if not SE3_AVAILABLE:
            print("SE3 requested but module is unavailable; continuing without SE3 branch.")
        else:
            se3_device = device if device.type == "cuda" else torch.device("cpu")
            try:
                se3_decoder = se3_denoiser(
                    in_channels=latent_dim,
                    hidden_channels=[args.se3_hidden],
                    out_channels=args.se3_out_channels,
                    num_embeddings=30,
                    commitment_cost=0.25,
                    metadata={"edge_types": data_sample.edge_types},
                    edge_dim=1,
                    depth=args.se3_depth,
                    heads=args.se3_heads,
                    dim_head=args.se3_dim_head,
                    return_coors=True,
                    num_atom_types=se3_num_atom_types,
                ).to(se3_device)
                se3_decoder.device = se3_device
                print(f"SE3 decoder initialized on {se3_device}.")
            except Exception as exc:
                se3_decoder = None
                print(f"SE3 decoder unavailable at runtime: {exc}")

    return transformer_geom_decoder, se3_decoder


def main():
    args = parse_args()

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
        num_workers=args.num_workers,
    )
    data_module.setup("fit")

    probe_loader = DataLoader(data_module.struct_dat, batch_size=args.batch_size, shuffle=True, num_workers=0)
    data_sample = next(iter(probe_loader))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    encoder, latent_dim = build_encoder(args, data_sample.to(device), device)
    se3_num_atom_types = int(getattr(encoder, "num_embeddings", 20))
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
        clip_grad_norm=args.clip_grad,
        cache_flush_interval=args.cache_flush_interval,
        gc_collect_interval=args.gc_collect_interval,
        use_cuda_ipc_collect=args.use_cuda_ipc_collect,
        use_uncertainty_weighting=args.use_uncertainty_weighting,
        use_se3=args.use_se3,
        se3_decoder=se3_decoder,
        nan_guard=args.nan_guard,
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
        monitor="train/loss_epoch",
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
        callbacks=[checkpoint_callback],
    )

    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
