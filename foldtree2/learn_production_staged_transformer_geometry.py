#!/usr/bin/env python3
"""Train a staged transformer geometry refiner from frozen production outputs."""

from __future__ import annotations

import copy
import inspect
import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foldtree2 import learn_geometry_lightning as base
from foldtree2.learn_production_geometry_se3_lightning import load_production_decoder
from foldtree2.src.losses.fape import (
    coarse_backbone_atoms_from_ca_frames,
    coarse_backbone_dihedrals_from_ca_frames,
    coarse_backbone_fape_loss,
    coarse_ca_loss,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from foldtree2.src.losses.losses import quaternion_fape_loss, quaternion_geodesic_loss


class StagedTransformerRefiner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_atom_types: int,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.05,
        max_step: float = 4.0,
        max_refine_delta: float = 2.0,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.max_step = float(max_step)
        self.max_refine_delta = float(max_refine_delta)
        self.input_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU())
        self.type_embed = nn.Embedding(max(4, int(num_atom_types)), hidden)
        self.coord_proj = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, hidden), nn.GELU())
        self.contact_proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())

        self.stage1 = self._encoder(hidden, heads, layers, dropout)
        self.stage2 = self._encoder(hidden, heads, layers, dropout)
        self.stage3 = self._encoder(hidden, heads, layers, dropout)

        self.step1 = nn.Linear(hidden, 3)
        self.delta2 = nn.Linear(hidden, 3)
        self.delta3 = nn.Linear(hidden, 3)
        self.angle1 = nn.Linear(hidden, 3)
        self.angle2 = nn.Linear(hidden, 3)
        self.angle3 = nn.Linear(hidden, 3)

    @staticmethod
    def _encoder(hidden: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, int(heads)),
            dim_feedforward=hidden * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=max(1, int(layers)))

    @staticmethod
    def _pack(x: torch.Tensor, batch_idx: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if batch_idx is None:
            idx = torch.arange(x.shape[0], device=x.device)
            return x.unsqueeze(0), torch.ones((1, x.shape[0]), dtype=torch.bool, device=x.device), [idx]
        values = torch.unique(batch_idx, sorted=True)
        indices = [(batch_idx == b).nonzero(as_tuple=True)[0] for b in values]
        max_len = max((idx.numel() for idx in indices), default=0)
        packed = x.new_zeros((len(indices), max_len, x.shape[-1]))
        mask = torch.zeros((len(indices), max_len), dtype=torch.bool, device=x.device)
        for i, idx in enumerate(indices):
            packed[i, : idx.numel()] = x[idx]
            mask[i, : idx.numel()] = True
        return packed, mask, indices

    @staticmethod
    def _unpack(x: torch.Tensor, indices: list[torch.Tensor], n_nodes: int) -> torch.Tensor:
        flat = x.new_zeros((n_nodes, x.shape[-1]))
        for i, idx in enumerate(indices):
            flat[idx] = x[i, : idx.numel()]
        return flat

    @staticmethod
    def _coords_from_steps(steps: torch.Tensor, batch_idx: Optional[torch.Tensor]) -> torch.Tensor:
        coords = torch.zeros_like(steps)
        if batch_idx is None:
            if steps.shape[0] > 1:
                coords[1:] = torch.cumsum(steps[:-1].float(), dim=0).to(dtype=coords.dtype)
            return coords
        for b in torch.unique(batch_idx, sorted=True):
            idx = (batch_idx == b).nonzero(as_tuple=True)[0]
            if idx.numel() > 1:
                coords[idx[1:]] = torch.cumsum(steps[idx[:-1]].float(), dim=0).to(dtype=coords.dtype)
        return coords

    @staticmethod
    def _contact_aggregate(x: torch.Tensor, contact: Optional[torch.Tensor], batch_idx: Optional[torch.Tensor]) -> torch.Tensor:
        if contact is None or contact.numel() == 0:
            return torch.zeros_like(x)
        out = torch.zeros_like(x)
        groups = [torch.arange(x.shape[0], device=x.device)] if batch_idx is None else [
            (batch_idx == b).nonzero(as_tuple=True)[0] for b in torch.unique(batch_idx, sorted=True)
        ]
        for gi, idx in enumerate(groups):
            n = idx.numel()
            if n == 0:
                continue
            adj = contact[0 if contact.shape[0] == 1 else gi, :n, :n].to(device=x.device, dtype=x.dtype)
            denom = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out[idx] = adj @ x[idx] / denom
        return out

    def forward(
        self,
        features: torch.Tensor,
        token_ids: torch.Tensor,
        seed_coords: torch.Tensor,
        contact: Optional[torch.Tensor],
        batch_idx: Optional[torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        seed_coords = torch.nan_to_num(seed_coords.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-32.0, 32.0)
        token_ids = token_ids.long().clamp(0, self.type_embed.num_embeddings - 1)

        h0 = self.input_proj(features)
        type_h = self.type_embed(token_ids)
        h_seq, mask, indices = self._pack(h0, batch_idx)
        key_padding = ~mask

        h1 = self.stage1(h_seq, src_key_padding_mask=key_padding)
        h1_flat = self._unpack(h1, indices, features.shape[0])
        step1 = torch.tanh(self.step1(h1_flat)) * self.max_step
        coords1 = self._coords_from_steps(step1, batch_idx)
        angles1 = torch.tanh(self.angle1(h1_flat)) * torch.pi

        contact_h = self.contact_proj(self._contact_aggregate(h1_flat + type_h, contact, batch_idx))
        h2_in = h1_flat + type_h + self.coord_proj(coords1 - seed_coords) + contact_h
        h2_seq, mask, indices = self._pack(h2_in, batch_idx)
        h2 = self.stage2(h2_seq, src_key_padding_mask=~mask)
        h2_flat = self._unpack(h2, indices, features.shape[0])
        coords2 = coords1 + torch.tanh(self.delta2(h2_flat)) * self.max_refine_delta
        angles2 = torch.tanh(self.angle2(h2_flat)) * torch.pi

        contact_h2 = self.contact_proj(self._contact_aggregate(h2_flat + type_h, contact, batch_idx))
        h3_in = h2_flat + type_h + self.coord_proj(coords2 - coords1) + contact_h2
        h3_seq, mask, indices = self._pack(h3_in, batch_idx)
        h3 = self.stage3(h3_seq, src_key_padding_mask=~mask)
        h3_flat = self._unpack(h3, indices, features.shape[0])
        coords3 = coords2 + torch.tanh(self.delta3(h3_flat)) * self.max_refine_delta
        angles3 = torch.tanh(self.angle3(h3_flat)) * torch.pi

        return {
            "stage1": {"coords": coords1, "steps": step1, "angles": angles1, "z": h1_flat},
            "stage2": {"coords": coords2, "steps": self._coords_to_steps(coords2, batch_idx), "angles": angles2, "z": h2_flat},
            "stage3": {"coords": coords3, "steps": self._coords_to_steps(coords3, batch_idx), "angles": angles3, "z": h3_flat},
        }

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


class ProductionStagedTransformerModule(base.GeometryFocusedModule):
    def __init__(
        self,
        *args,
        staged_refiner: StagedTransformerRefiner,
        production_coordinate_source: str = "auto",
        production_coordinate_scale: float = 1.0,
        stage_loss_weights: str = "0.25,0.5,1.0",
        **kwargs,
    ):
        super().__init__(*args, use_se3=False, se3_decoder=None, se3_atom_decoder=None, **kwargs)
        self.staged_refiner = staged_refiner
        self.production_coordinate_source = str(production_coordinate_source)
        self.production_coordinate_scale = float(production_coordinate_scale)
        weights = [float(part) for part in str(stage_loss_weights).split(",") if part.strip()]
        if len(weights) != 3:
            raise ValueError("--stage-loss-weights must contain three comma-separated values")
        self.stage_loss_weights = weights
        self._production_bottleneck = None

        def capture_bottleneck(_module, inputs):
            self._production_bottleneck = inputs[0].detach()

        if hasattr(self.transformer_geom_decoder, "body") and "lin" in self.transformer_geom_decoder.body:
            self._production_bottleneck_hook = self.transformer_geom_decoder.body["lin"].register_forward_pre_hook(
                capture_bottleneck
            )

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.transformer_geom_decoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()
        self.transformer_geom_decoder.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.staged_refiner.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.lr_scheduler_name == "none":
            return optimizer
        if self.lr_scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.max_epochs - self.lr_warmup_epochs),
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.lr_scheduler_interval,
                "frequency": max(1, self.lr_scheduler_frequency),
                "monitor": self.lr_scheduler_monitor,
            },
        }

    def _prepare_production_outputs(self, data_batch):
        with torch.no_grad():
            output = self.transformer_geom_decoder(data_batch, contact_pred_index=None)
        contact_embedding = output.get("z")
        if contact_embedding is None or contact_embedding.ndim != 2:
            raise RuntimeError("Production geometry decoder must return z with shape [N,D]")
        candidates = {
            "bottleneck": self._production_bottleneck,
            "z": contact_embedding,
            "trans_pred": output.get("trans_pred"),
            "trans_local_pred": output.get("trans_local_pred"),
        }
        source = self.production_coordinate_source
        if source == "auto":
            source = next(
                (
                    name for name in ("bottleneck", "z", "trans_pred", "trans_local_pred")
                    if candidates.get(name) is not None
                    and candidates[name].ndim == 2
                    and candidates[name].shape[-1] == 3
                ),
                "bottleneck",
            )
        seed = candidates.get(source)
        if seed is None or seed.ndim != 2 or seed.shape[-1] != 3:
            shapes = {name: (None if value is None else tuple(value.shape)) for name, value in candidates.items()}
            raise RuntimeError(f"Production coordinate source '{source}' is not [N,3]; available={shapes}")
        output = dict(output)
        output["se3_contact_z"] = contact_embedding
        output["seed_coords"] = seed.float() * self.production_coordinate_scale
        return output

    def _stage_frame_outputs(self, coords: torch.Tensor, batch_idx: Optional[torch.Tensor]):
        q_parts = []
        t_parts = []
        groups = [torch.arange(coords.shape[0], device=coords.device)] if batch_idx is None else [
            (batch_idx == b).nonzero(as_tuple=True)[0] for b in torch.unique(batch_idx, sorted=True)
        ]
        for idx in groups:
            _R, t, q = self._frames_from_ca_only(coords[idx])
            q_parts.append(q)
            t_parts.append(t)
        return torch.cat(q_parts, dim=0), torch.cat(t_parts, dim=0)

    def _add_stage_losses(self, raw_terms, name, stage, weight, true_R, true_t, true_q, true_angles, true_coords, batch_idx):
        coords = stage["coords"]
        steps = stage["steps"]
        angles = stage["angles"]
        raw_terms[f"{name}_ca"] = weight * coarse_ca_loss(
            steps,
            true_coords,
            batch_idx=batch_idx,
            pred_ca=coords,
            step_weight=self.coarse_ca_step_weight if self.use_coarse_ca_step_loss else 0.0,
            bond_weight=self.coarse_ca_bond_weight if self.use_coarse_ca_bond_loss else 0.0,
            pairwise_weight=self.coarse_ca_pairwise_weight if self.use_coarse_ca_pairwise_loss else 0.0,
            pairwise_max_seq_sep=self.coarse_ca_pairwise_max_seq_sep,
            pairwise_max_pairs=self.coarse_ca_pairwise_max_pairs,
        )
        if true_q is not None and true_t is not None:
            pred_q, pred_t = self._stage_frame_outputs(coords, batch_idx)
            true_t_origin = self._step_translations_to_origins(true_t, batch_idx=batch_idx)
            raw_terms[f"{name}_fape"] = weight * quaternion_fape_loss(true_q, true_t_origin, pred_q, pred_t, batch=batch_idx)
            raw_terms[f"{name}_quat"] = weight * quaternion_geodesic_loss(pred_q, true_q)
        if true_angles is not None and angles is not None:
            mask = torch.isfinite(angles) & torch.isfinite(true_angles)
            if mask.any():
                raw_terms[f"{name}_angles"] = weight * base.periodic_angle_smooth_l1(angles[mask], true_angles[mask])
        if self.use_coarse_backbone_loss and true_R is not None:
            pred_q, _pred_t = self._stage_frame_outputs(coords, batch_idx)
            pred_R = quaternion_to_rotation_matrix(pred_q)
            pred_atoms = coarse_backbone_atoms_from_ca_frames(coords, pred_R, atom_names=("ca", "c", "cb", "n"))
            true_atoms = coarse_backbone_atoms_from_ca_frames(true_coords, true_R, atom_names=("ca", "c", "cb", "n"))
            true_cb = base.node_x(self._active_batch, "cbcoords")
            true_n = base.node_x(self._active_batch, "ncoords")
            true_c = base.node_x(self._active_batch, "ccoords")
            true_atoms = true_atoms.clone()
            if true_c is not None:
                true_atoms[:, 1] = true_c.to(device=true_atoms.device, dtype=true_atoms.dtype)
            if true_cb is not None:
                true_atoms[:, 2] = true_cb.to(device=true_atoms.device, dtype=true_atoms.dtype)
            if true_n is not None:
                true_atoms[:, 3] = true_n.to(device=true_atoms.device, dtype=true_atoms.dtype)
            raw_terms[f"{name}_backbone_atoms"] = weight * self.coarse_backbone_atom_weight * F.smooth_l1_loss(
                pred_atoms, true_atoms, beta=0.5
            )
            raw_terms[f"{name}_backbone_fape"] = weight * self.coarse_backbone_fape_weight * coarse_backbone_fape_loss(
                true_atoms, pred_atoms, true_R, pred_R, true_coords, coords, batch=batch_idx
            )
            if true_angles is not None and self.coarse_backbone_angle_weight > 0:
                pred_bb, pred_mask = coarse_backbone_dihedrals_from_ca_frames(
                    coords, pred_R, batch=batch_idx, n_coords=pred_atoms[:, 3], c_coords=pred_atoms[:, 1]
                )
                true_bb, true_mask = coarse_backbone_dihedrals_from_ca_frames(
                    true_coords, true_R, batch=batch_idx, n_coords=true_atoms[:, 3], c_coords=true_atoms[:, 1]
                )
                mask = pred_mask & true_mask
                if mask.any():
                    delta = base.wrap_to_pi_torch(pred_bb - true_bb)
                    raw_terms[f"{name}_backbone_angles"] = weight * self.coarse_backbone_angle_weight * F.smooth_l1_loss(
                        delta[mask], torch.zeros_like(delta[mask])
                    )

    def _compute_total_loss(self, data_batch, debug_label: str = ""):
        data_batch = base.ensure_edge_attrs_inplace(base.ensure_float32_inplace(data_batch), edge_dim=getattr(self.encoder, "edge_dim", 1))
        self._active_batch = data_batch
        with torch.no_grad():
            z_local, _ = self.encoder(data_batch)
            ft2_token_ids = self._get_ft2_token_ids(z_local)
            codebook_vectors = self._get_codebook_vectors(self.encoder, ft2_token_ids)
        if ft2_token_ids is None or codebook_vectors is None:
            raise RuntimeError("Staged transformer requires encoder VQ token ids and codebook vectors")

        data_batch["res"].x = z_local
        out = self._prepare_production_outputs(data_batch)
        true_R, true_t, true_q, true_angles, true_coords, batch_idx = base.get_true_geometry(data_batch)
        if true_coords is None:
            raise RuntimeError("Staged transformer training requires data['coords'].x")

        features = torch.cat([z_local.float(), codebook_vectors.float(), out["se3_contact_z"].float()], dim=-1)
        contact = self._geometry_dot_contact_sketch(out["seed_coords"], batch_idx)
        stages = self.staged_refiner(features, ft2_token_ids, out["seed_coords"], contact, batch_idx)

        raw_terms: Dict[str, torch.Tensor] = {}
        for i, stage_name in enumerate(("stage1", "stage2", "stage3")):
            self._add_stage_losses(
                raw_terms,
                stage_name,
                stages[stage_name],
                self.stage_loss_weights[i],
                true_R,
                true_t,
                true_q,
                true_angles,
                true_coords,
                batch_idx,
            )

        if self.nan_guard:
            bad = [name for name, value in raw_terms.items() if not torch.isfinite(value).all()]
            if bad:
                raise RuntimeError(f"Non-finite staged loss term(s) at {debug_label}: {bad}")
        weighted_terms = self._kendall_weight_terms(raw_terms)
        total = torch.stack([value.float() for value in weighted_terms.values()]).sum()
        if self.nan_guard and not torch.isfinite(total).all():
            raise RuntimeError(f"Non-finite staged total loss at {debug_label}")
        return total, raw_terms, weighted_terms, False, batch_idx


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--staged-hidden", type=int, default=128)
    parser.add_argument("--staged-heads", type=int, default=4)
    parser.add_argument("--staged-layers", type=int, default=2)
    parser.add_argument("--staged-dropout", type=float, default=0.05)
    parser.add_argument("--staged-max-step", type=float, default=4.0)
    parser.add_argument("--staged-max-refine-delta", type=float, default=2.0)
    parser.add_argument("--stage-loss-weights", type=str, default="0.25,0.5,1.0")
    staged_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = base.parse_args()
    for key, value in vars(staged_args).items():
        setattr(args, key, value)

    base.pl.seed_everything(args.seed, workers=True)
    data_module = base.GeometryOnlyDataModule(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        val_count=args.val_count,
        split_seed=args.split_seed,
    )
    data_module.setup("fit")
    probe = next(iter(base.DataLoader(data_module.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, latent_dim = base.build_encoder(args, probe.to(device), device)
    production_decoder = load_production_decoder(args.pretrained_geometry_decoder_path, device)
    se3_num_atom_types = max(4, int(args.se3_num_atom_types or getattr(encoder, "num_embeddings", 20)))
    probe_for_decoder = copy.deepcopy(probe).to(device)
    probe_for_decoder = base.ensure_edge_attrs_inplace(base.ensure_float32_inplace(probe_for_decoder), edge_dim=getattr(encoder, "edge_dim", 1))
    with torch.no_grad():
        z_probe, _ = encoder(probe_for_decoder)
        probe_for_decoder["res"].x = z_probe
        prod_probe = production_decoder(probe_for_decoder, contact_pred_index=None)
    prod_z = prod_probe.get("z")
    if prod_z is None or prod_z.ndim != 2:
        raise RuntimeError("Production decoder probe did not return z with shape [N,D]")
    input_dim = latent_dim * 2 + int(prod_z.shape[-1])
    staged_refiner = StagedTransformerRefiner(
        input_dim=input_dim,
        num_atom_types=se3_num_atom_types,
        hidden=args.staged_hidden,
        heads=args.staged_heads,
        layers=args.staged_layers,
        dropout=args.staged_dropout,
        max_step=args.staged_max_step,
        max_refine_delta=args.staged_max_refine_delta,
    ).to(device)

    constructor = inspect.signature(base.GeometryFocusedModule.__init__)
    aliases = {"lr_scheduler_name": "lr_scheduler", "max_epochs": "epochs", "clip_grad_norm": "clip_grad"}
    kwargs = {}
    for name in constructor.parameters:
        if name in {"self", "encoder", "transformer_geom_decoder", "se3_decoder", "se3_atom_decoder", "use_se3"}:
            continue
        source = aliases.get(name, name)
        if hasattr(args, source):
            kwargs[name] = getattr(args, source)
    kwargs["train_se3_only"] = False
    module = ProductionStagedTransformerModule(
        encoder=encoder,
        transformer_geom_decoder=production_decoder,
        staged_refiner=staged_refiner,
        production_coordinate_source=args.production_coordinate_source,
        production_coordinate_scale=args.production_coordinate_scale,
        stage_loss_weights=args.stage_loss_weights,
        **kwargs,
    )

    accum_steps = max(1, math.ceil(args.target_effective_batch_size / max(1, args.batch_size)))
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = base.pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="production-staged-transformer-{epoch:02d}-{step}",
        monitor="val/loss" if data_module.val_dataset is not None else "train/loss_epoch",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
    )
    parsed_devices = base.parse_devices(args.devices)
    strategy = base.infer_strategy(args.accelerator, parsed_devices, args.strategy)
    trainer = base.pl.Trainer(
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
    print(
        "Production staged transformer training: "
        f"hidden={args.staged_hidden} layers={args.staged_layers} heads={args.staged_heads} "
        f"max_step={args.staged_max_step} max_refine_delta={args.staged_max_refine_delta} "
        f"micro_batch_size={args.batch_size} accumulation={accum_steps}"
    )
    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
