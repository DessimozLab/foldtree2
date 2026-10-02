#!/usr/bin/env python3
"""Evaluate the selected encoder/decoder pair on a reproducible reserved split."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', required=True)
    parser.add_argument('--decoder', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--val-seed', type=int, default=7)
    parser.add_argument('--val-split', type=float, default=.1)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    import torch
    from torch_geometric.loader import DataLoader
    from foldtree2.src.pdbgraph import StructureDataset
    from foldtree2.src.training_protocol import held_out_split
    torch.set_num_threads(args.threads)
    dataset = StructureDataset(args.dataset)
    _, validation = held_out_split(dataset, args.val_split, args.val_seed)
    encoder = torch.load(args.encoder, map_location=args.device, weights_only=False).eval()
    decoder = torch.load(args.decoder, map_location=args.device, weights_only=False).eval()
    encoder.device = torch.device(args.device)
    correct, total = 0, 0
    counts = torch.zeros(encoder.num_embeddings, dtype=torch.long)
    with torch.no_grad():
        for batch in DataLoader(validation, batch_size=1, num_workers=0):
            batch = batch.to(args.device)
            z, _ = encoder(batch)
            tokens = encoder.vector_quantizer.discretize_z(z)[0].flatten().cpu()
            counts += torch.bincount(tokens, minlength=len(counts))
            batch['res'].x = z
            output = decoder(batch, None)
            if not torch.isfinite(output['aa']).all():
                raise ValueError('Nonfinite AA predictions in selected production pair')
            target = batch['AA'].x.argmax(dim=1)
            correct += int((output['aa'].argmax(dim=1) == target).sum())
            total += len(target)
    result = {'encoder': args.encoder, 'decoder': args.decoder, 'dataset': args.dataset,
              'split_protocol': 'notebook_permutation_prefix_rounded_v1',
              'val_seed': args.val_seed, 'val_split': args.val_split, 'n_structures': len(validation),
              'n_residues': total, 'aa_accuracy': correct / total,
              'observed_states': int((counts > 0).sum()), 'state_counts': counts.tolist()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
