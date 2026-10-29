#!/usr/bin/env python
"""Smoke-train the coarse CA step decoder on an HDF5 StructureDataset."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from foldtree2.src.losses import coarse_ca_loss
from foldtree2.src.mono_decoders import MultiMonoDecoder
from foldtree2.src.pdbgraphmk2 import StructureDataset


class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def select_indices(dataset, max_samples, max_residues, seed):
    indices = []
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    for idx in order:
        graph = dataset[idx]
        n_res = graph["res"].x.shape[0]
        if max_residues is not None and n_res > max_residues:
            continue
        indices.append(idx)
        if len(indices) >= max_samples:
            break
    return indices


def canonicalize_ca_targets(ca, batch, frames=None):
    out = torch.zeros_like(ca)
    for graph_id in torch.unique(batch, sorted=True):
        idx = torch.where(batch == graph_id)[0]
        if idx.numel() == 0:
            continue
        centered = ca[idx] - ca[idx[0]].unsqueeze(0)
        if frames is not None:
            first_frame = frames[idx[0]]
            centered = centered @ first_frame
        out[idx] = centered
    return out


def run_epoch(model, loader, optimizer, device, args):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "step": 0.0, "bond": 0.0, "pairwise": 0.0, "batches": 0}

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        plddt = batch["plddt"].x if args.mask_plddt and "plddt" in batch.node_types else None
        true_ca = batch["coords"].x
        frames = None
        pred_ca = out["ca_coords_pred"]
        if args.step_frame == "canonical" or args.canonicalize_targets:
            frames = batch["R_true"].x if "R_true" in batch.node_types else None
            true_ca = canonicalize_ca_targets(true_ca, batch["res"].batch, frames=frames)
            frames = None
        elif args.step_frame == "prev":
            if "R_true" not in batch.node_types:
                raise RuntimeError('--step-frame prev requires batch["R_true"].x')
            frames = batch["R_true"].x
            pred_ca = None
        loss, components = coarse_ca_loss(
            out["ca_steps_pred"],
            true_ca,
            batch_idx=batch["res"].batch,
            pred_ca=pred_ca,
            frames=frames,
            frame_offset="prev",
            step_weight=args.step_weight,
            bond_weight=args.bond_weight,
            pairwise_weight=args.pairwise_weight,
            pairwise_max_seq_sep=args.pairwise_max_seq_sep,
            pairwise_max_pairs=args.pairwise_max_pairs,
            plddt=plddt,
            plddt_thresh=args.plddt_threshold,
            return_components=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        totals["loss"] += float(loss.detach().cpu())
        for key in ("step", "bond", "pairwise"):
            totals[key] += float(components[key].detach().cpu())
        totals["batches"] += 1

    denom = max(totals.pop("batches"), 1)
    return {key: val / denom for key, val in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="notebooks/structs_training_mk2.h5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--max-residues", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-weight", type=float, default=1.0)
    parser.add_argument("--bond-weight", type=float, default=0.1)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-max-seq-sep", type=int, default=64)
    parser.add_argument("--pairwise-max-pairs", type=int, default=4096)
    parser.add_argument("--mask-plddt", action="store_true")
    parser.add_argument(
        "--step-frame",
        choices=["global", "canonical", "prev"],
        default="prev",
        help="Frame used for CA step targets: raw global, first-frame canonicalized, or previous residue frame.",
    )
    parser.add_argument("--canonicalize-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plddt-threshold", type=float, default=0.3)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metrics-output", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset = StructureDataset(args.dataset)
    needed = args.train_samples + args.val_samples
    indices = select_indices(dataset, needed, args.max_residues, args.seed)
    if len(indices) < needed:
        raise RuntimeError(f"Only found {len(indices)} samples matching max_residues={args.max_residues}")

    train_set = IndexedSubset(dataset, indices[: args.train_samples])
    val_set = IndexedSubset(dataset, indices[args.train_samples : needed])
    sample = train_set[0]
    in_dim = sample["res"].x.shape[1]

    model = MultiMonoDecoder(
        {
            "coarse_ca": {
                "in_channels": {"res": in_dim},
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "nheads": args.nheads,
                "dropout": args.dropout,
            }
        }
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    metrics = []
    print(
        f"dataset={args.dataset} train={len(train_set)} val={len(val_set)} "
        f"in_dim={in_dim} device={args.device}"
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, args.device, args)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, args.device, args)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        metrics.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_step={train_metrics['step']:.4f} "
            f"train_pair={train_metrics['pairwise']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_step={val_metrics['step']:.4f} "
            f"val_pair={val_metrics['pairwise']:.4f}"
        )

    if args.metrics_output:
        output_path = Path(args.metrics_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
