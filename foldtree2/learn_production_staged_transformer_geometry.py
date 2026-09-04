#!/usr/bin/env python3
"""Train a staged transformer geometry refiner from frozen production outputs."""

from __future__ import annotations

import copy
import inspect
import argparse
import math
import sys
from pathlib import Path
from typing import Dict

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
from foldtree2.src.se3_struct_decoder import StagedTransformerRefiner


class ProductionStagedTransformerModule(base.GeometryFocusedModule):
    def __init__(
        self,
        *args,
        staged_refiner: StagedTransformerRefiner,
        production_coordinate_source: str = "auto",
        production_coordinate_scale: float = 1.0,
        stage_loss_weights: str = "0.25,0.5,1.0",
        stage_angle_loss: bool = True,
        **kwargs,
    ):
        kwargs.pop("use_se3", None)
        kwargs.pop("se3_decoder", None)
        kwargs.pop("se3_atom_decoder", None)
        super().__init__(*args, use_se3=False, se3_decoder=None, se3_atom_decoder=None, **kwargs)
        self.staged_refiner = staged_refiner
        self.production_coordinate_source = str(production_coordinate_source)
        self.production_coordinate_scale = float(production_coordinate_scale)
        weights = [float(part) for part in str(stage_loss_weights).split(",") if part.strip()]
        if len(weights) != 3:
            raise ValueError("--stage-loss-weights must contain three comma-separated values")
        self.stage_loss_weights = weights
        self.stage_angle_loss = bool(stage_angle_loss)
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
            raw_terms[f"{name}_fape"] = weight * quaternion_fape_loss(
                true_q,
                true_t_origin,
                pred_q,
                pred_t,
                batch=batch_idx,
                pair_sample_size=self.fape_pair_sample_size or None,
            )
            raw_terms[f"{name}_quat"] = weight * quaternion_geodesic_loss(pred_q, true_q)
        if self.stage_angle_loss and true_angles is not None and angles is not None:
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
        contact = self._geometry_dot_contact_sketch(
            out["se3_contact_z"],
            batch_idx,
            contact_temp=getattr(self.transformer_geom_decoder, "contact_temp", None),
            contact_bias=getattr(self.transformer_geom_decoder, "contact_bias", None),
        )
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
    parser.add_argument("--staged-use-mhc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--staged-mhc-streams", type=int, default=4)
    parser.add_argument("--staged-mhc-sinkhorn-iters", type=int, default=5)
    parser.add_argument("--staged-mhc-temperature", type=float, default=1.0)
    parser.add_argument("--staged-mhc-eps", type=float, default=1e-6)
    parser.add_argument("--stage-loss-weights", type=str, default="0.25,0.5,1.0")
    parser.add_argument("--stage-angle-loss", action=argparse.BooleanOptionalAction, default=True)
    staged_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = base.parse_args()
    for key, value in vars(staged_args).items():
        setattr(args, key, value)
    staged_cli = []
    for key, value in vars(staged_args).items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            staged_cli.append(option if value else "--no-" + key.replace("_", "-"))
        else:
            staged_cli.extend([option, str(value)])
    sys.argv = [sys.argv[0], *remaining, *staged_cli]

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
        use_mhc=args.staged_use_mhc,
        mhc_streams=args.staged_mhc_streams,
        mhc_sinkhorn_iters=args.staged_mhc_sinkhorn_iters,
        mhc_temperature=args.staged_mhc_temperature,
        mhc_eps=args.staged_mhc_eps,
    ).to(device).float()

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
        stage_angle_loss=args.stage_angle_loss,
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
