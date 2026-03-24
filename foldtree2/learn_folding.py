"""
Folding Refiner Training Script
================================
Train QuaternionFoldingRefiner on top of a frozen encoder (+ optional frozen
coarse-geometry decoder).  This is Stage 2 in the staged training pipeline:

  Stage 1: encoder + MultiMonoDecoder (learn_lightning.py)
  Stage 2: frozen encoder → QuaternionFoldingRefiner  (this script)
  Stage 3: frozen refiner → SidechainRefiner

Usage
-----
# minimal run
python foldtree2/learn_folding.py \
    --dataset structs_train_final.h5 \
    --encoder-path models/my_encoder.pt \
    --model-name folding_refiner_v1

# with optional coarse-geometry warm-start from a saved decoder
python foldtree2/learn_folding.py \
    --dataset structs_train_final.h5 \
    --encoder-path models/my_encoder.pt \
    --decoder-path models/my_decoder.pt \
    --model-name folding_refiner_v1 \
    --epochs 500 \
    --batch-size 8 \
    --accumulate-grad-batches 4 \
    --mixed-precision \
    --gpus 4 \
    --strategy ddp

# from YAML config
python foldtree2/learn_folding.py --config config_folding_refiner.yaml
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from transformers import (
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foldtree2.src import pdbgraph
from foldtree2.src.folding_refiner import QuaternionFoldingRefiner
from foldtree2.src.losses.fape import (
    fape_loss,
    quaternion_to_rotation_matrix,
)
from foldtree2.src.losses.losses import angles_reconstruction_loss

# ── CLI / YAML argument parsing ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train QuaternionFoldingRefiner (Stage 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file (CLI args override YAML values)")
    p.add_argument("--dataset", type=str, default="structs_train_final.h5",
                   help="Path to HDF5 dataset")
    p.add_argument("--val-dataset", type=str, default=None,
                   help="Optional separate validation HDF5 dataset")
    p.add_argument("--val-split", type=float, default=0.05,
                   help="Fraction of training data to use as validation when "
                        "--val-dataset is not given")
    p.add_argument("--encoder-path", type=str, required=True,
                   help="Path to a saved encoder .pt file")
    p.add_argument("--decoder-path", type=str, default=None,
                   help="Optional path to a saved coarse-geometry decoder .pt "
                        "file whose output is used to warm-start refinement")
    p.add_argument("--model-name", type=str, default="folding_refiner",
                   help="Base name for saved checkpoints")
    p.add_argument("--output-dir", type=str, default="./models",
                   help="Directory to save checkpoints")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing checkpoints")
    p.add_argument("--save-config", type=str, default=None,
                   help="Save resolved config to this YAML file")

    # ── Refiner architecture ─────────────────────────────────────────────────
    p.add_argument("--hidden-channels", type=int, default=256,
                   help="Transformer d_model for the refiner")
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--nlayers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--out-angles", action="store_true",
                   help="Predict backbone bond angles in addition to RT")
    p.add_argument("--max-refinement-steps", type=int, default=4,
                   help="Max unrolling steps during training")
    p.add_argument("--convergence-tol", type=float, default=1e-4)
    p.add_argument("--min-refinement-steps", type=int, default=1)

    # ── Loss weights ─────────────────────────────────────────────────────────
    p.add_argument("--fape-weight", type=float, default=1.0)
    p.add_argument("--rt-weight", type=float, default=0.1,
                   help="L1 loss on quaternion + translation directly")
    p.add_argument("--angles-weight", type=float, default=0.05)
    p.add_argument("--step-discount", type=float, default=0.9,
                   help="Geometric discount applied to per-step losses "
                        "(1.0 = all steps equal, <1.0 = later steps weighted more "
                        "when used as step^(max-step) weighting)")

    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05,
                   help="Fraction of total steps used for LR warm-up")
    p.add_argument("--lr-schedule", type=str, default="cosine",
                   choices=["cosine", "linear", "none"])
    p.add_argument("--clip-grad", action="store_true")
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)

    # ── pLDDT masking ────────────────────────────────────────────────────────
    p.add_argument("--mask-plddt", action="store_true")
    p.add_argument("--plddt-threshold", type=float, default=0.7)

    # ── Hardware ─────────────────────────────────────────────────────────────
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--strategy", type=str, default="auto",
                   choices=["auto", "ddp", "ddp_find_unused_parameters_true"])
    p.add_argument("--seed", type=int, default=42)

    return p


def merge_yaml(args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config into args (CLI values take precedence)."""
    if args.config is None or not YAML_AVAILABLE:
        return args
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    # Only set values that were not explicitly passed on the CLI
    cli_keys = {a.dest for a in build_parser()._actions}
    defaults = vars(build_parser().parse_args([
        "--encoder-path", args.encoder_path or "PLACEHOLDER"
    ]))
    for k, v in cfg.items():
        dest = k.replace("-", "_")
        if dest in cli_keys and vars(args).get(dest) == defaults.get(dest):
            setattr(args, dest, v)
    return args


# ── Data module ──────────────────────────────────────────────────────────────

class FoldingDataModule(pl.LightningDataModule):
    """
    Wraps StructureDataset with an optional train/val split.
    """

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        val_dataset_path: str | None = None,
        val_split: float = 0.05,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.val_dataset_path = val_dataset_path
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        full = pdbgraph.StructureDataset(self.dataset_path)

        if self.val_dataset_path:
            self.train_dataset = full
            self.val_dataset = pdbgraph.StructureDataset(self.val_dataset_path)
        else:
            n_val = max(1, int(len(full) * self.val_split))
            n_train = len(full) - n_val
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full, [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


# ── Lightning module ─────────────────────────────────────────────────────────

class FoldingRefinerModule(pl.LightningModule):
    """
    Stage-2 Lightning module.

    Architecture
    ~~~~~~~~~~~~
    - ``encoder``       : frozen, maps graph → residue embeddings + VQ loss
    - ``coarse_decoder``: frozen (optional), produces ``rt_pred`` warm-start
    - ``refiner``       : **trainable** QuaternionFoldingRefiner

    Loss
    ~~~~
    For each refinement step *s* (multi-step unrolling):

        L_s = fape_weight  * FAPE(R_pred_s, t_pred_s, R_true, t_true)
            + rt_weight    * L1(rt_pred_s, rt_true)
            + angles_weight* angle_loss (if --out-angles)

    Total loss = Σ_s  w_s * L_s
    where w_s = discount^(max_steps - s)  (later steps weighted more when discount<1).
    """

    def __init__(self, encoder, refiner, args, coarse_decoder=None):
        super().__init__()
        self.encoder = encoder
        self.coarse_decoder = coarse_decoder
        self.refiner = refiner
        self.args = args

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        # Freeze coarse decoder if provided
        if self.coarse_decoder is not None:
            for p in self.coarse_decoder.parameters():
                p.requires_grad_(False)
            self.coarse_decoder.eval()

        self.save_hyperparameters(ignore=["encoder", "refiner", "coarse_decoder"])

    # ── helpers ──────────────────────────────────────────────────────────────

    def _batch_size(self, data) -> int:
        batch = data["res"].batch
        return int(batch.max().item()) + 1 if batch is not None else 1

    def _plddt_mask(self, data):
        if not self.args.mask_plddt:
            return None
        plddt = data.get("plddt", None)
        if plddt is None:
            return None
        scores = plddt.x.squeeze(-1)
        return (scores >= self.args.plddt_threshold)

    @staticmethod
    def _rt_true(data):
        """Return (R_true, t_true) from the HeteroData object."""
        R_true = data["R_true"].x  # (N, 3, 3)
        t_true = data["t_true"].x  # (N, 3)
        return R_true, t_true

    @staticmethod
    def _quat_trans_to_R(quat, trans):
        """Convert (N,4) quaternion + (N,3) translation → R (N,3,3), t (N,3)."""
        R = quaternion_to_rotation_matrix(quat)       # (N, 3, 3)
        return R, trans

    # ── core forward ─────────────────────────────────────────────────────────

    def _encode(self, data):
        """Run frozen encoder; inject embeddings into data['res'].x."""
        with torch.no_grad():
            z, _vqloss = self.encoder(data)
        data["res"].x = z
        return data

    def _coarse_init(self, data):
        """
        Optional: run frozen coarse decoder to obtain initial RT estimate.
        Returns a decoder_out dict or None.
        """
        if self.coarse_decoder is None:
            return None
        with torch.no_grad():
            decoder_out = self.coarse_decoder(data, None)
        return decoder_out

    def _compute_step_loss(self, quat, trans, R_true, t_true, batch,
                           angles=None, plddt_mask=None):
        """Loss at a single refinement step."""
        R_pred, t_pred = self._quat_trans_to_R(quat, trans)

        # FAPE
        fape = fape_loss(R_true, t_true, R_pred, t_pred, batch)

        # Direct RT regression (L1 on [quat | trans] concatenated)
        rt_pred = torch.cat([quat, trans], dim=-1)
        # Build true quaternion from R_true for RT supervision
        # We supervise with t_true directly; for the quat part we use the
        # compact representation that's closest to the target rotation.
        # Simple proxy: penalise translation error + a rotation geodesic term.
        t_loss = F.smooth_l1_loss(trans, t_true)
        # Geodesic rotation loss: 1 - |q_pred · q_R_true|
        # We don't have quaternion GT from the dataset (only R_true), so we
        # back out the rotation agreement via the trace of R_pred^T R_true.
        Rt = torch.einsum("bij,bkj->bik", R_pred, R_true)  # (N,3,3)
        trace = Rt.diagonal(dim1=-2, dim2=-1).sum(-1)       # (N,)
        cos_angle = ((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6)
        rot_loss = (1.0 - cos_angle).mean()
        rt_loss = t_loss + rot_loss

        # Angle loss (optional)
        angle_loss = torch.tensor(0.0, device=quat.device)
        if angles is not None and hasattr(self.args, "angles_weight") and self.args.angles_weight > 0:
            angle_loss = angles_reconstruction_loss(
                angles,
                None,  # no direct per-residue GT angles available here
            ) if False else angle_loss  # placeholder – wire up when GT exists

        total = (
            self.args.fape_weight * fape
            + self.args.rt_weight * rt_loss
            + self.args.angles_weight * angle_loss
        )
        return total, fape, rt_loss

    # ── training / validation steps ──────────────────────────────────────────

    def _shared_step(self, data, split: str):
        bs = self._batch_size(data)
        batch_idx = data["res"].batch

        # 1. Encode (frozen)
        data = self._encode(data)

        # 2. Optional coarse warm-start (frozen)
        decoder_out = self._coarse_init(data)

        # 3. Refine (trainable) – multi-step unrolling
        refiner_out = self.refiner(
            data,
            decoder_out=decoder_out,
            use_self_output=True,
            return_history=False,
        )

        R_true, t_true = self._rt_true(data)

        # Per-step losses via history-free shortcut:
        # The refiner already ran max_steps internally; we get the final state.
        # For true multi-step unrolling with per-step supervision we run the
        # refiner step-by-step manually.
        total_loss = torch.tensor(0.0, device=self.device)
        fape_sum = torch.tensor(0.0, device=self.device)
        rt_sum = torch.tensor(0.0, device=self.device)

        max_steps = self.args.max_refinement_steps
        discount = self.args.step_discount

        # Step-by-step unrolling for per-step supervision
        quat = refiner_out["quat_pred"].detach()   # start from final state
        trans = refiner_out["trans_pred"].detach()

        # Re-run step-by-step from the coarse init for proper supervision
        if decoder_out is not None and decoder_out.get("rt_pred") is not None:
            rt0 = decoder_out["rt_pred"]
            quat0 = rt0[..., :4]
            trans0 = rt0[..., 4:]
        elif decoder_out is not None and decoder_out.get("quat_pred") is not None:
            quat0 = decoder_out["quat_pred"]
            trans0 = decoder_out["trans_pred"]
        else:
            quat0 = torch.zeros(R_true.shape[0], 4, device=self.device, dtype=R_true.dtype)
            quat0[:, 0] = 1.0
            trans0 = torch.zeros(R_true.shape[0], 3, device=self.device, dtype=R_true.dtype)

        quat_cur = quat0
        trans_cur = trans0

        for step in range(max_steps):
            # Run a single refinement step (max_refinement_steps=1 override)
            step_out = self.refiner(
                data,
                init_quat=quat_cur.detach(),
                init_trans=trans_cur.detach(),
                use_self_output=True,
                return_history=False,
                # override to run exactly 1 step
                max_refinement_steps=1,
                min_refinement_steps=1,
                convergence_tol=1e10,
            )

            q_s = step_out["quat_pred"]
            t_s = step_out["trans_pred"]
            angles_s = step_out.get("angles")

            step_loss, fape_s, rt_s = self._compute_step_loss(
                q_s, t_s, R_true, t_true, batch_idx, angles=angles_s,
            )

            # Geometric step weighting: later steps carry more weight
            w = discount ** (max_steps - 1 - step)
            total_loss = total_loss + w * step_loss
            fape_sum = fape_sum + w * fape_s
            rt_sum = rt_sum + w * rt_s

            quat_cur = q_s
            trans_cur = t_s

        # Normalise by sum of weights
        weight_sum = sum(discount ** (max_steps - 1 - s) for s in range(max_steps))
        total_loss = total_loss / weight_sum
        fape_sum = fape_sum / weight_sum
        rt_sum = rt_sum / weight_sum

        # Logging
        self.log(f"{split}/loss", total_loss, on_step=(split == "train"),
                 on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"{split}/fape", fape_sum, on_step=False,
                 on_epoch=True, batch_size=bs)
        self.log(f"{split}/rt_loss", rt_sum, on_step=False,
                 on_epoch=True, batch_size=bs)
        self.log(f"{split}/refinement_steps",
                 float(refiner_out["refinement_steps"]),
                 on_step=False, on_epoch=True, batch_size=bs)

        torch.cuda.empty_cache()
        gc.collect()
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    # ── optimiser / scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.refiner.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        if self.args.lr_schedule == "none" or not TRANSFORMERS_AVAILABLE:
            return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(total_steps * self.args.warmup_ratio))

        if self.args.lr_schedule == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, total_steps
            )
        else:  # linear
            scheduler = get_linear_schedule_with_warmup(
                optimizer, warmup_steps, total_steps
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()
    args = merge_yaml(args)

    pl.seed_everything(args.seed, workers=True)

    # ── save resolved config ─────────────────────────────────────────────────
    if args.save_config and YAML_AVAILABLE:
        os.makedirs(os.path.dirname(args.save_config) or ".", exist_ok=True)
        with open(args.save_config, "w") as fh:
            yaml.dump(vars(args), fh, default_flow_style=False)
        print(f"Config saved to {args.save_config}")

    # ── load frozen encoder ──────────────────────────────────────────────────
    print(f"Loading encoder from {args.encoder_path}")
    encoder = torch.load(args.encoder_path, map_location="cpu")
    encoder.eval()

    # ── optional frozen coarse decoder ──────────────────────────────────────
    coarse_decoder = None
    if args.decoder_path:
        print(f"Loading coarse decoder from {args.decoder_path}")
        coarse_decoder = torch.load(args.decoder_path, map_location="cpu")
        coarse_decoder.eval()

    # ── infer encoder output dimension from a data sample ───────────────────
    data_module = FoldingDataModule(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        num_workers=0,  # temp, just for the probe
        val_dataset_path=args.val_dataset,
        val_split=args.val_split,
    )
    data_module.setup()
    probe_loader = DataLoader(data_module.train_dataset, batch_size=1, shuffle=False)
    probe_batch = next(iter(probe_loader))
    with torch.no_grad():
        probe_z, _ = encoder(probe_batch)
    in_channels = probe_z.shape[-1]
    print(f"Encoder output dimension: {in_channels}")
    del probe_loader, probe_batch, probe_z

    # ── build refiner ────────────────────────────────────────────────────────
    # out_channels not used beyond angle head, so we pass the same value.
    refiner = QuaternionFoldingRefiner(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.hidden_channels,
        dropout_p=args.dropout,
        decoder_hidden=args.hidden_channels // 2,
        out_angles=args.out_angles,
        nheads=args.nheads,
        nlayers=args.nlayers,
        max_refinement_steps=args.max_refinement_steps,
        convergence_tol=args.convergence_tol,
        min_refinement_steps=args.min_refinement_steps,
    )
    n_params = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    print(f"Refiner parameters: {n_params:,}")

    # ── Lightning module ─────────────────────────────────────────────────────
    model = FoldingRefinerModule(
        encoder=encoder,
        refiner=refiner,
        args=args,
        coarse_decoder=coarse_decoder,
    )

    # ── callbacks ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"{args.model_name}_{{epoch:03d}}_{{val/loss:.4f}}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ── logger ───────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(save_dir="./runs", name=args.model_name)

    # ── strategy ─────────────────────────────────────────────────────────────
    devices = args.gpus if torch.cuda.is_available() else 1
    strategy = args.strategy if args.gpus > 1 else "auto"

    # ── trainer ──────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        strategy=strategy,
        precision="16-mixed" if args.mixed_precision else 32,
        gradient_clip_val=1.0 if args.clip_grad else 0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True,
    )

    # ── data module (real, with workers) ─────────────────────────────────────
    data_module = FoldingDataModule(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_dataset_path=args.val_dataset,
        val_split=args.val_split,
    )

    # ── print summary ─────────────────────────────────────────────────────────
    print("\nFolding Refiner Training Configuration")
    print(f"  Dataset          : {args.dataset}")
    print(f"  Encoder          : {args.encoder_path}")
    print(f"  Coarse decoder   : {args.decoder_path or 'none (identity init)'}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Grad accumulation: {args.accumulate_grad_batches}")
    print(f"  Refiner layers   : {args.nlayers}  heads: {args.nheads}  d_model: {args.hidden_channels}")
    print(f"  Max refine steps : {args.max_refinement_steps}")
    print(f"  Loss weights     : FAPE={args.fape_weight}  RT={args.rt_weight}  angles={args.angles_weight}")
    print(f"  Mixed precision  : {args.mixed_precision}")
    print(f"  GPUs             : {devices}")
    print()

    trainer.fit(model, datamodule=data_module)

    # ── save final refiner ────────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, f"{args.model_name}_refiner_final.pt")
    torch.save(refiner.state_dict(), final_path)
    print(f"\nFinal refiner weights saved to {final_path}")
    print(f"Best checkpoint : {checkpoint_cb.best_model_path}")
    print(f"Best val/loss   : {checkpoint_cb.best_model_score:.4f}")
    print(f"TensorBoard     : {logger.log_dir}")


if __name__ == "__main__":
    main()
