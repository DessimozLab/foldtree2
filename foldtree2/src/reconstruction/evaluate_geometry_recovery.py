"""Evaluate how well the MSA Transformer recovers geometry it never saw.

Runs a trained heads-mode checkpoint over OMA families with the geometry
features of *every* known-structure sequence withheld from the encoder input
(`--withhold-ratio 1.0`), then scores the distance-head predictions against
the true CA pair distances.  Uniform guessing over the 16 distance bins gives
6.25% top-1 accuracy, so any accuracy above that is cross-sequence structural
inference.

Usage:
    python -m foldtree2.src.reconstruction.evaluate_geometry_recovery \
        --checkpoint models/msa_reconstruction/run2_geommask/<best>.ckpt \
        --embed-dim 128 --num-layers 4 --num-heads 4
"""

import argparse

import torch
from torch.utils.data import DataLoader

from foldtree2.src.reconstruction.tree_msa_dataset import TreeMsaDataset, msa_collate
from foldtree2.src.reconstruction.train_msa_reconstruction import MsaReconstructionModule

_EDGES = (3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 14., 16., 20., 24., 32.)
# Representative distance for each of the 16 bins (midpoints, clipped at the ends).
_REPS = [2.5] + [(_EDGES[i] + _EDGES[i + 1]) / 2.0 for i in range(len(_EDGES) - 1)] + [32.0]


def evaluate(args, withhold_ratio: float) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TreeMsaDataset(
        cohort_path=args.cohort, families_root=args.families_root,
        mask_ratio=args.mask_ratio, family_ids=args.family_ids,
        alphabet_size=args.alphabet_size,
    )
    # Reproduce the training split so --val-split scores families the model
    # never saw.
    if args.val_split and args.family_ids is None:
        n = len(ds)
        n_train = max(1, int(n * 0.85))
        _, ds = torch.utils.data.random_split(
            ds, [n_train, n - n_train],
            generator=torch.Generator().manual_seed(args.seed),
        )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=msa_collate)

    ckpt = torch.load(args.checkpoint, map_location=device)
    hp = ckpt.get("hyper_parameters", {})
    model = MsaReconstructionModule(
        use_se3=False,
        alphabet_size=hp.get("alphabet_size", args.alphabet_size),
        embed_dim=hp.get("embed_dim", args.embed_dim),
        num_heads=hp.get("num_heads", args.num_heads),
        num_layers=hp.get("num_layers", args.num_layers),
        focal_gamma=args.focal_gamma,
        geometry_withhold_ratio=withhold_ratio,
        propagate_geometry=hp.get("propagate_geometry", False),
        attn_pairs=hp.get("attn_pairs", False),
        coord_head=hp.get("coord_head", False),
        plddt_threshold=args.plddt_threshold,
        max_pairs=args.max_pairs,
        loss_weights={"distance": 1.0},
    )
    state = ckpt["state_dict"]
    # Older checkpoints predate the "no geometry" marker; keep their weights and
    # leave the new parameter at its initial value.
    for k, v in model.state_dict().items():
        state.setdefault(k, v)
    model.load_state_dict(state)
    model.to(device).eval()

    reps = torch.tensor(_REPS, device=device)
    n_pairs = n_correct = 0
    err_sum = 0.0
    n_seq = 0
    bin_correct = torch.zeros(16, dtype=torch.long)
    bin_total = torch.zeros(16, dtype=torch.long)
    pred_hist = torch.zeros(16, dtype=torch.long)

    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model.forward(batch)
            known = batch["known_structure"]
            coords = batch["coords"]
            pairs = out["pairs"]
            if pairs is None or coords is None or known is None:
                continue
            labels = model._pair_labels(
                coords, pairs, model.distance_head.edges.to(device),
                batch.get("plddt"), model.plddt_threshold,
            )
            pred = out["distance_logits"].argmax(dim=-1)
            valid = known.unsqueeze(-1) & (labels >= 0)
            if not valid.any():
                continue
            n_seq += int(known.sum().item())
            lab = labels[valid]
            pr = pred[valid]
            n_pairs += int(lab.numel())
            n_correct += int((pr == lab).sum().item())
            bin_total += torch.bincount(lab.cpu(), minlength=16)
            pred_hist += torch.bincount(pr.cpu(), minlength=16)
            bin_correct += torch.bincount(lab[pr == lab].cpu(), minlength=16)
            true_d = torch.linalg.norm(
                coords[:, :, pairs[:, 0]] - coords[:, :, pairs[:, 1]], dim=-1
            )
            err_sum += float((reps[pred] - true_d).abs()[valid].sum().item())

    present = bin_total > 0
    macro = (bin_correct[present] / bin_total[present]).mean().item()
    return {
        "sequences": n_seq,
        "pairs": n_pairs,
        "top1_acc": n_correct / max(1, n_pairs),
        "macro_acc": macro,
        "majority_acc": (bin_total.max().item() / max(1, n_pairs)),
        "per_bin_recall": {int(b): round(bin_correct[b].item() / bin_total[b].item(), 3)
                           for b in range(16) if bin_total[b].item() > 0},
        "pred_hist": {int(b): round(pred_hist[b].item() / max(1, n_pairs), 3)
                      for b in range(16) if pred_hist[b].item() > 0},
        "mean_abs_dist_err": err_sum / max(1, n_pairs),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cohort", default="configs/oma_benchmark_cohort.json")
    p.add_argument("--families-root", default="families/Information_benchmark/marker_genes")
    p.add_argument("--family-ids", nargs="*", default=None)
    p.add_argument("--mask-ratio", type=float, default=0.25)
    p.add_argument("--alphabet-size", type=int, default=51)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--plddt-threshold", type=float, default=70.0)
    p.add_argument("--max-pairs", type=int, default=40000)
    p.add_argument("--withhold-ratio", type=float, default=1.0,
                   help="Fraction of known-structure rows whose geometry is hidden "
                        "from the encoder. 1.0 = pure inference.")
    p.add_argument("--val-split", action="store_true",
                   help="Score only the 15% of families held out by the training "
                        "split (same --seed as training).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    res = evaluate(args, args.withhold_ratio)
    print(f"withhold_ratio={args.withhold_ratio}")
    print(f"  sequences scored   : {res['sequences']}")
    print(f"  valid pairs        : {res['pairs']}")
    print(f"  top-1 bin acc      : {res['top1_acc']:.4f}   (uniform {1/16:.4f}, majority {res['majority_acc']:.4f})")
    print(f"  macro per-bin acc  : {res['macro_acc']:.4f}")
    print(f"  mean |dist| err    : {res['mean_abs_dist_err']:.2f} A")
    print(f"  predicted bins     : {res['pred_hist']}")
    print(f"  per-bin recall     : {res['per_bin_recall']}")


if __name__ == "__main__":
    main()
