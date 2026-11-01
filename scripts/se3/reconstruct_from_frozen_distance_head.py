#!/usr/bin/env python3
"""Direct coordinate reconstruction using a trained distance head checkpoint.

Loads the frozen best.pt from a PairwiseDistanceHead run and uses its
predictions as differentiable restraints for coordinate optimization,
reporting TM-score/RMSD/lDDT per-chain.  This is a diagnostic: it tests
whether the distance head's predictions are geometrically informative
enough to recover folds in direct optimization (the ``reconstruction``
diagnostic), *before* feeding them into the SE3 refiner.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F

from foldtree2.se3_cache import FrozenDataset, open_bundle
from foldtree2.src.distance.distance_restraints import (
    PairwiseDistanceHead, distribution_restraint_loss, eligible_pairs,
    validate_edges,
)
from foldtree2.train_se3_lightning import calibrated


def load_head(checkpoint_path, device):
    """Load a frozen PairwiseDistanceHead from a best.pt checkpoint."""
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in state and 'model_config' in state:
        model = PairwiseDistanceHead(**state['model_config']).to(device)
        model.load_state_dict(state['model'])
    elif 'model_config' in state:
        model = PairwiseDistanceHead(**state['model_config']).to(device)
        model.load_state_dict(state['model'])
    else:
        raise ValueError('Unknown checkpoint format')
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def predict_restraints(model, entries, device):
    """Run frozen distance head on cache entries, export restraints.

    Returns a list per-entry dicts with 'id', 'edges', 'pairs' (canonical
    eligible pairs), 'probabilities' (softmaxed distribution), and 'mask'.
    """
    exports = {}
    for entry in entries:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in entry.items()}
        pairs = eligible_pairs(inputs['mask']).to(device)
        with torch.no_grad():
            logits = model(inputs, pairs)
            probs = logits.softmax(-1).cpu()
        exports[inputs['id']] = {
            'id': inputs['id'], 'mask': inputs['mask'].cpu(),
            'pairs': pairs.cpu(), 'probabilities': probs,
        }
    return exports


def evaluate_coords(coords, target, mask):
    """Quick TM-score approximation + RMSD + lDDT for diagnostic use."""
    try:
        from tmtools import tm_align
        p = coords[mask].detach().cpu().double().numpy()
        t = target[mask].detach().cpu().double().numpy()
        result = tm_align(p, t, 'A' * len(p), 'A' * len(t))
        tm_score = result.tm_norm_chain2
    except ImportError:
        tm_score = float('nan')
    pc = (coords[mask] - coords[mask].mean(0)).detach().cpu()
    tc = (target[mask] - target[mask].mean(0)).detach().cpu()
    u, _, vh = torch.linalg.svd(pc.T @ tc)
    sign = torch.ones(3)
    sign[-1] = torch.linalg.det(u @ vh)
    aligned = pc @ u @ torch.diag(sign) @ vh
    rmsd = (aligned - tc).square().mean().sqrt().item()
    pd = torch.cdist(pc, pc)
    td = torch.cdist(tc, tc)
    off = ~torch.eye(len(pc), dtype=torch.bool)
    neighbors = (td < 15.) & (td > 0.) & off
    if neighbors.any():
        delta = (pd - td).abs()
        lddt = torch.stack([
            (delta[neighbors] < c).float().mean() for c in (.5, 1., 2., 4.)
        ]).mean().item()
    else:
        lddt = float('nan')
    return {'tm_score': tm_score if not math.isnan(tm_score) else None,
            'ca_rmsd': float(rmsd), 'lddt': float(lddt)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_distance_data_scaling_5120_20261022/comparison/bundle_5120')
    parser.add_argument('--head', required=True,
                        help='Path to trained distance head best.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=8,
                        help='Number of chains for the reconstruction panel')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--lr', type=float, default=.05)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    # Distinguish three cases: local-only (baseline, no restraints), and
    # decoder mode only (which is what our trained head uses).
    modes = ('local_only', 'decoder')

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    manifest = open_bundle(args.cache)

    # Load the frozen distance head.
    model = load_head(args.head, device)
    edges = model.edges  # the bin edges this head was trained with
    print(f"Loaded distance head: hidden={model.hidden} "
          f"context_layers={model.context_layers} "
          f"contact_bias={model.contact_bias} "
          f"kendall_terms={list(model.kendall_terms)}")

    # Select validation chains for the panel.
    rng = torch.Generator().manual_seed(args.seed)
    ids = torch.randperm(len(manifest['val_ids']), generator=rng).tolist()
    ids = [manifest['val_ids'][i] for i in ids[:args.count]]

    dataset = FrozenDataset(args.cache, manifest, 'val')
    panel = []
    for identifier in ids:
        entry = calibrated(dataset[dataset.ids.index(identifier)], manifest['scale'])
        mask = entry['mask']
        index = torch.arange(len(mask))
        separation = index[None] - index[:, None]
        valid = mask[:, None] & mask[None]
        local = {
            'bond_indices': (mask[:-1] & mask[1:]).nonzero().flatten(),
            'excluded_pairs': ((separation >= 2) & valid).nonzero(),
        }
        panel.append({'entry': entry, 'local': local})
    print(f"Panel: {len(panel)} chains: {ids}")

    def generic_local_objective(coords, entry, local):
        """Generic local priors: bond length + excluded volume."""
        mask = entry['mask']
        bonds = local['bond_indices']
        excluded = local['excluded_pairs']
        loss = torch.zeros((), device=coords.device)
        if bonds.any():
            bond_len = (coords[bonds + 1] - coords[bonds]).norm(dim=-1)
            loss = loss + F.smooth_l1_loss(bond_len, torch.full_like(bond_len, 3.8))
        if excluded.any():
            d = (coords[excluded[:, 0]] - coords[excluded[:, 1]]).norm(dim=-1)
            loss = loss + F.relu(3.6 - d).mean()
        return loss

    def objective(coords, entry, local, restraint_pred):
        loss = generic_local_objective(coords, entry, local)
        if restraint_pred is not None:
            pairs = restraint_pred['pairs']
            if pairs is not None and len(pairs) > 0 and pairs.shape[0] > 0:
                restraint = distribution_restraint_loss(
                    coords, pairs, restraint_pred['probabilities'], edges)
                loss = loss + restraint
        return loss

    for mode in modes:
        output = root / mode
        output.mkdir(exist_ok=False)
        parameters = [
            torch.nn.Parameter(sample['entry']['coords'].clone().float().to(device))
            for sample in panel
        ]
        # Predict restraints for decoder mode (frozen head runs once).
        predictions = None
        if mode == 'decoder':
            predictions = predict_restraints(
                model, [s['entry'] for s in panel], device)

        def current_score():
            with torch.no_grad():
                vals = []
                for p, sample in zip(parameters, panel):
                    pred = None if mode == 'local_only' else predictions[sample['entry']['id']]
                    vals.append(float(objective(p, sample['entry'], sample['local'], pred)))
                return sum(vals) / len(vals)

        best, best_step = current_score(), 0
        best_coords = [p.detach().clone() for p in parameters]

        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.)
        print(f"\n=== {mode} ===")
        print(f"  initial objective: {best:.6f}")
        initial_metrics = [
            evaluate_coords(p.detach(), sample['entry']['target'].to(device), sample['entry']['mask'])
            for p, sample in zip(parameters, panel)
        ]

        started = time.monotonic()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            for p, sample in zip(parameters, panel):
                pred = None if mode == 'local_only' else predictions[sample['entry']['id']]
                loss = objective(p, sample['entry'], sample['local'], pred)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite objective')
                (loss / len(panel)).backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            value = current_score()
            if value < best:
                best, best_step = value, step
                best_coords = [p.detach().clone() for p in parameters]
            if step % 100 == 0 or step == 1:
                print(f"  step {step:>5}: objective={value:.6f} norm={norm:.3f}")
            if time.monotonic() - started >= args.seconds:
                break

        final_metrics = [
            evaluate_coords(p.detach(), sample['entry']['target'], sample['entry']['mask'])
            for p, sample in zip(parameters, panel)
        ]
        best_results = [
            evaluate_coords(bc, sample['entry']['target'], sample['entry']['mask'])
            for bc, sample in zip(best_coords, panel)
        ]

        for i, identifier in enumerate(ids):
            print(f"  {identifier}:")
            print(f"    initial:  TM={initial_metrics[i]['tm_score']:.4f} "
                  f"RMSD={initial_metrics[i]['ca_rmsd']:.3f} "
                  f"lDDT={initial_metrics[i]['lddt']:.4f}")
            print(f"    best:     TM={best_results[i]['tm_score']:.4f} "
                  f"RMSD={best_results[i]['ca_rmsd']:.3f} "
                  f"lDDT={best_results[i]['lddt']:.4f}")

        summary = {
            'mode': mode, 'steps': step, 'best_step': best_step,
            'best_objective': float(best),
            'initial': [{'id': ids[i], **initial_metrics[i]}
                        for i in range(len(ids))],
            'best': [{'id': ids[i], **best_results[i]}
                     for i in range(len(ids))],
            'mean_initial_tm': sum(m['tm_score'] or 0 for m in initial_metrics) / len(initial_metrics),
            'mean_best_tm': sum(m['tm_score'] or 0 for m in best_results) / len(best_results),
            'mean_best_rmsd': sum(m['ca_rmsd'] for m in best_results) / len(best_results),
            'mean_best_lddt': sum(m['lddt'] for m in best_results) / len(best_results),
            'elapsed_seconds': time.monotonic() - started,
        }
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(f"  Mean best TM={summary['mean_best_tm']:.4f} "
              f"RMSD={summary['mean_best_rmsd']:.3f} "
              f"lDDT={summary['mean_best_lddt']:.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()