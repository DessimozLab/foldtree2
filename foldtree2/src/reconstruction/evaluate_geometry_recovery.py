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

    model = MsaReconstructionModule(
        use_se3=False,
        alphabet_size=args.alphabet_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        focal_gamma=args.focal_gamma,
        geometry_withhold_ratio=withhold_ratio,
        plddt_threshold=args.plddt_threshold,
        max_pairs=args.max_pairs,
        loss_weights={"distance": 1.0},
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
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
            n_pairs += int(valid.sum().item())
            n_correct += int(((pred == labels) & valid).sum().item())
            true_d = torch.linalg.norm(
                coords[:, :, pairs[:, 0]] - coords[:, :, pairs[:, 1]], dim=-1
            )
            err_sum += float((reps[pred] - true_d).abs()[valid].sum().item())

    return {
        "sequences": n_seq,
        "pairs": n_pairs,
        "top1_acc": n_correct / max(1, n_pairs),
        "mean_abs_dist_err": err_sum / max(1, n_pairs),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cohort", default="configs/oma_benchmark_cohort.json")
    p.add_argument("--families-root", default="families/Information_benchmark/marker_genes")
    p.add_argument("--family-ids", nargs="*", default=None)
    p.add_argument("--mask-ratio", type=float, default=0.25)
    p.add_argument("--alphabet-size", type=int, default=31)
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
    print(f"  sequences scored : {res['sequences']}")
    print(f"  valid pairs      : {res['pairs']}")
    print(f"  top-1 bin acc    : {res['top1_acc']:.4f}   (uniform baseline 0.0625)")
    print(f"  mean |dist| err  : {res['mean_abs_dist_err']:.2f} A")


if __name__ == "__main__":
    main()
