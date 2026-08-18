#!/usr/bin/env python3
"""Train an SE3 coordinate refiner from a frozen production encoder/decoder pair.

The production geometry decoder supplies two complementary inputs:

* ``z`` is normalized per residue and used for the dot-product contact graph.
* its 3D translation output is used as the initial residue coordinate sketch.

The SE3 module and its atom-level refinement path use the loss stack from
``learn_geometry_lightning.py``.  All options, including YAML/JSON config files,
are shared with that trainer.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

# Make direct ``python foldtree2/<script>.py`` invocation behave like the
# installed-package invocation used by the Alps launcher.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foldtree2 import learn_geometry_lightning as base


class ProductionGeometrySE3Module(base.GeometryFocusedModule):
    """GeometryFocusedModule with a frozen production geometry source."""

    def __init__(self, *args, production_coordinate_source="auto", production_coordinate_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.production_coordinate_source = str(production_coordinate_source)
        self.production_coordinate_scale = float(production_coordinate_scale)

        for parameter in self.transformer_geom_decoder.parameters():
            parameter.requires_grad = False
        self.transformer_geom_decoder.eval()

    def _prepare_geometry_outputs(self, data_batch):
        with torch.no_grad():
            output = self.transformer_geom_decoder(data_batch, contact_pred_index=None)

        contact_embedding = output.get("z")
        if contact_embedding is None or contact_embedding.ndim != 2:
            raise RuntimeError("Production geometry decoder must return z with shape [N,D]")

        candidates = {
            "z": contact_embedding,
            "trans_pred": output.get("trans_pred"),
            "trans_local_pred": output.get("trans_local_pred"),
        }
        source = self.production_coordinate_source
        if source == "auto":
            source = "z" if contact_embedding.shape[-1] == 3 else "trans_pred"
            if candidates.get(source) is None:
                source = "trans_local_pred"

        seed = candidates.get(source)
        if seed is None or seed.ndim != 2 or seed.shape[-1] != 3:
            shapes = {name: (None if value is None else tuple(value.shape)) for name, value in candidates.items()}
            raise RuntimeError(
                f"Production coordinate source '{source}' is not [N,3]; available decoder outputs: {shapes}"
            )

        # The shared SE3 path applies se3_contact_coord_scale. Normalize that
        # factor here so production-coordinate-scale remains an independent flag.
        output = dict(output)
        output["se3_contact_z"] = contact_embedding
        output["se3_seed_coords"] = seed * (self.production_coordinate_scale / max(self.se3_contact_coord_scale, 1e-8))
        return output


def load_production_decoder(path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Production geometry decoder not found: {checkpoint}")

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.nn.Module):
        decoder = loaded
    elif isinstance(loaded, dict):
        decoder = loaded.get("model") or loaded.get("decoder") or loaded.get("geometry_decoder")
        if not isinstance(decoder, torch.nn.Module):
            raise RuntimeError(f"{checkpoint} does not contain a serialized geometry decoder module")
    else:
        raise RuntimeError(f"Unsupported production decoder checkpoint type: {type(loaded)!r}")

    if not hasattr(decoder, "decoders") or "geometry_cnn" not in decoder.decoders:
        raise RuntimeError(f"{checkpoint} is not a MultiMonoDecoder with a geometry_cnn decoder")

    decoder = decoder.to(device)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    print(f"Loaded frozen production geometry decoder from {checkpoint}")
    return decoder


def main():
    args = base.parse_args()
    if not args.pretrained_geometry_decoder_path:
        raise ValueError("--pretrained-geometry-decoder-path is required for this trainer")

    # Reuse the established data split, encoder compatibility checks, SE3
    # construction, trainer setup, and all loss-related command-line options.
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
    print(f"Dataset split: train={len(data_module.train_dataset)} val={len(data_module.val_dataset or [])}")

    probe_loader = base.DataLoader(data_module.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    data_sample = next(iter(probe_loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, latent_dim = base.build_encoder(args, data_sample.to(device), device)
    production_decoder = load_production_decoder(args.pretrained_geometry_decoder_path, device)

    # Build only the SE3 branch from the common factory. The fresh geometry
    # decoder returned by the factory is intentionally discarded.
    _, se3_decoder = base.build_decoders(
        latent_dim,
        data_sample,
        device,
        args.use_se3,
        args,
        se3_num_atom_types=max(4, int(args.se3_num_atom_types or getattr(encoder, "num_embeddings", 20))),
    )

    constructor = inspect.signature(base.GeometryFocusedModule.__init__)
    module_kwargs = {}
    for name in constructor.parameters:
        if name in {"self", "encoder", "transformer_geom_decoder", "se3_decoder"}:
            continue
        if hasattr(args, name):
            module_kwargs[name] = getattr(args, name)
    module_kwargs.update(
        encoder=encoder,
        transformer_geom_decoder=production_decoder,
        se3_decoder=se3_decoder,
        production_coordinate_source=args.production_coordinate_source,
        production_coordinate_scale=args.production_coordinate_scale,
    )
    module = ProductionGeometrySE3Module(**module_kwargs)

    accum_steps = max(1, (args.target_effective_batch_size + args.batch_size - 1) // args.batch_size)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor = "val/loss" if data_module.val_dataset is not None else "train/loss_epoch"
    checkpoint_callback = base.pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="production-se3-{epoch:02d}-{step}",
        monitor=monitor,
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
        f"Production SE3 training: coordinate_source={module.production_coordinate_source} "
        f"coordinate_scale={module.production_coordinate_scale} "
        f"micro_batch_size={args.batch_size} accumulation={accum_steps}"
    )
    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
