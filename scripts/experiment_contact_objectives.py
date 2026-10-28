"""Small matched loss/head ablation on the frozen development cache."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from foldtree2.se3_cache import digest, open_bundle
from foldtree2.src.distance_restraints import PairwiseDistanceHead, distance_labels
from foldtree2.train_distance_restraints import evaluate, load_records
from scripts.se3.diagnose_se3_reconstruction import atomic_checkpoint, atomic_json


def contact_logit(logits, contact_bins):
    return (torch.logsumexp(logits[:, :contact_bins], -1)
            - torch.logsumexp(logits[:, contact_bins:], -1))


def balanced_contact_loss(logits, labels, contact_bins):
    scores = contact_logit(logits, contact_bins)
    positive = labels < contact_bins
    terms = []
    if positive.any():
        terms.append(F.softplus(-scores[positive]).mean())
    if (~positive).any():
        terms.append(F.softplus(scores[~positive]).mean())
    return torch.stack(terms).mean()


def experiment_cases(args):
    if not getattr(args, 'cases', None):
        return [('baseline', 0, 0.), ('balanced', 0, args.contact_weight),
                ('sphere4', 4, args.contact_weight), ('sphere8', 8, args.contact_weight),
                ('sphere16', 16, args.contact_weight)]
    cases = []
    for specification in args.cases:
        if specification == 'baseline':
            if any(existing[0] == 'baseline' for existing in cases):
                raise ValueError('Duplicate contact objective case: baseline')
            cases.append(('baseline', 0, 0.))
            continue
        try:
            dim_string, weight_string = specification.split(':')
            dim, weight = int(dim_string), float(weight_string)
        except (ValueError, TypeError) as exc:
            raise ValueError('Each case must have the form DIM:WEIGHT') from exc
        if dim < 1 or not math.isfinite(weight) or weight <= 0:
            raise ValueError('Sphere dimensions and loss weights must be positive and finite')
        name = f'sphere{dim}_w{weight:g}'.replace('.', 'p')
        if any(existing[0] == name for existing in cases):
            raise ValueError('Duplicate contact objective case: ' + name)
        cases.append((name, dim, weight))
    return cases


def run(args):
    cases = experiment_cases(args)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    manifest = open_bundle(args.cache)
    records = {s: load_records(args.cache, manifest, s, getattr(args, s + '_count'))
               for s in ('train', 'val')}
    source_paths = [__file__, 'foldtree2/src/distance_restraints.py',
                    'foldtree2/train_distance_restraints.py', 'foldtree2/se3_cache.py']
    identity = {str(p): digest(p) for p in source_paths}
    atomic_json(root / 'provenance.json', {
        'args': vars(args), 'sources': identity, 'bundle_sha256': manifest['bundle_sha256'],
        'selection': {s: [e['id'] for e, _ in r] for s, r in records.items()},
        'scope': 'Single-seed development screen; not independent production validation',
        'selection_rule': 'Fixed final epoch for comparison; no validation checkpoint selection',
        'contact_definition': 'CA distance <8A; residue row separation >=12',
        'sampling': 'Uniform pairs for distance CE; extra equal positive/negative pairs for contact loss',
        'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None})
    for index, p in enumerate(source_paths):
        (root / f'source-{index}-{Path(p).name}').write_bytes(Path(p).read_bytes())
    # Label pools are training-only. Validation retains every eligible pair.
    pools, histogram = [], torch.ones(10, dtype=torch.float64)
    for entry, pairs in records['train']:
        labels, _ = distance_labels(entry['target'], pairs, torch.tensor((4.,6.,8.,10.,12.,16.,20.,24.,32.)))
        histogram += torch.bincount(labels, minlength=10)
        pools.append(((labels < 3).nonzero().flatten(), (labels >= 3).nonzero().flatten()))
    prior = (histogram / histogram.sum()).float().to(device)
    summaries = {}
    for name, dim, weight in cases:
        torch.manual_seed(args.seed)
        model = PairwiseDistanceHead(
            manifest['feature_dim'], manifest['decoder_enrichment']['contact_dim'],
            hidden=args.hidden, context_layers=2, context_heads=8,
            dropout=.1, spherical_dim=dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.05)
        # Projection initialization must not shift the shared dropout RNG stream.
        torch.manual_seed(args.seed + 2)
        uniform_rng = torch.Generator().manual_seed(args.seed)
        contact_rng = torch.Generator().manual_seed(args.seed + 1)
        case = root / name
        case.mkdir()
        def event(kind, **values):
            record = {'event': kind, 'time': time.time(), **values}
            with (case / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps({'case': name, **record}), flush=True)
        event('ready', epoch=0, updates=0, train_count=len(records['train']),
              val_count=len(records['val']), spherical_dim=dim, contact_weight=weight)
        initial, _ = evaluate(model, records['val'], device, prior)
        atomic_json(case / 'initial.json', initial)
        history = []
        start = time.monotonic()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses, ce_losses, contact_losses = [], [], []
            order = torch.randperm(len(records['train']), generator=uniform_rng)
            for batch_index, index in enumerate(order.tolist(), 1):
                entry, pairs = records['train'][index]
                uniform = torch.randperm(len(pairs), generator=uniform_rng)[:args.pairs]
                chosen = [uniform]
                if weight:
                    for pool in pools[index]:
                        if len(pool):
                            chosen.append(pool[torch.randint(len(pool), (args.pairs // 2,), generator=contact_rng)])
                sampled = pairs[torch.cat(chosen)].to(device)
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in entry.items()}
                labels, _ = distance_labels(inputs['target'], sampled, model.edges)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs, sampled)
                ce = F.cross_entropy(logits[:len(uniform)], labels[:len(uniform)])
                contact = (balanced_contact_loss(logits[len(uniform):], labels[len(uniform):],
                                                model.contact_bins) if weight else ce.new_zeros(()))
                loss = ce + weight * contact
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                ce_losses.append(float(ce.detach()))
                contact_losses.append(float(contact.detach()))
                if batch_index % 256 == 0:
                    event('update', epoch=epoch, updates=(epoch - 1) * len(order) + batch_index,
                          loss=losses[-1], distance_ce=ce_losses[-1], contact_loss=contact_losses[-1])
            report, _ = evaluate(model, records['val'], device, prior)
            row = {'epoch': epoch, 'loss': sum(losses)/len(losses),
                   'distance_ce': sum(ce_losses)/len(ce_losses),
                   'contact_loss': sum(contact_losses)/len(contact_losses),
                   'validation': report['aggregate']}
            history.append(row)
            # Preserve the latest completed epoch even if a later epoch is interrupted.
            atomic_checkpoint(case / 'last.pt', {'config': model.model_config(),
                              'state_dict': model.state_dict(), 'epoch': epoch})
            atomic_json(case / 'history.json', history)
            print(json.dumps({'case': name, **row}), flush=True)
        saved = torch.load(case / 'last.pt', map_location=device, weights_only=False)
        reloaded = PairwiseDistanceHead(**saved['config']).to(device)
        reloaded.load_state_dict(saved['state_dict'])
        final, _ = evaluate(reloaded, records['val'], device, prior)
        if final['aggregate'] != report['aggregate']:
            raise RuntimeError('Checkpoint reload changed validation metrics')
        atomic_json(case / 'final_validation.json', final)
        summaries[name] = {'epochs': args.epochs, 'elapsed_seconds': time.monotonic()-start,
                           'spherical_dim': dim, 'contact_weight': weight,
                           'reload_exact': True, 'validation': final['aggregate']}
        atomic_json(root / 'summary.json', summaries)
        event('complete', epoch=args.epochs, updates=args.epochs * len(records['train']))
    if identity != {str(p): digest(p) for p in source_paths}:
        raise RuntimeError('Experiment sources changed during execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_distance_data_scaling_5120_20261022/comparison/bundle_5120')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--train-count', type=int, default=128)
    parser.add_argument('--val-count', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--pairs', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--contact-weight', type=float, default=1.)
    parser.add_argument('--cases', nargs='+', metavar='DIM:WEIGHT',
                        help='Run selected cases: baseline or spherical DIM:WEIGHT')
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())
