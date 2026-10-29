"""Matched frozen-feature distance probes, not coordinate reconstruction."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F

from foldtree2.se3_cache import FrozenDataset, digest, open_bundle
from scripts.se3.diagnose_se3_reconstruction import atomic_checkpoint, atomic_json, select_panel


def sample_pairs(entry, generator, count=256):
    mask = entry['mask']
    idx = torch.arange(len(mask))
    eligible = mask[:, None] & mask[None] & (idx[None] - idx[:, None] >= 12)
    pairs = eligible.nonzero()
    if len(pairs) < count:
        raise ValueError(f"Insufficient eligible long-range pairs: {entry['id']}")
    return pairs[torch.randperm(len(pairs), generator=generator)[:count]]


def pair_features(entry, pairs, representation):
    i, j = pairs.T
    if representation == 'continuous':
        features = entry['features']
    elif representation == 'codebook':
        features = entry['codebook']
    elif representation == 'sequence':
        features = torch.zeros(len(entry['mask']), entry['features'].shape[1])
    else:
        raise ValueError('Unknown representation')
    # Pad codebook inputs so all probes have identical parameter counts.
    features = F.pad(features, (0, entry['features'].shape[1] - features.shape[1]))
    separation = (j - i).float().log1p()[:, None]
    return torch.cat([features[i] + features[j],
                      (features[i] - features[j]).abs(),
                      features[i] * features[j], separation], dim=-1)


def report(prediction, target, contact_logits=None):
    predicted_contacts = prediction < 8. if contact_logits is None else contact_logits > 0.
    true_contacts = target < 8.
    shared = int((predicted_contacts & true_contacts).sum())
    npred, ntrue = int(predicted_contacts.sum()), int(true_contacts.sum())
    return {
        'distance_mae_angstrom': float((prediction - target).abs().mean()),
        'predicted_contacts': npred, 'true_contacts': ntrue, 'pairs': len(target),
        'precision': shared / npred if npred else None,
        'recall': shared / ntrue if ntrue else None,
    }


def run(args):
    if min(args.steps, args.seconds, args.train_count, args.val_count, args.pairs) < 1:
        raise ValueError('Positive probe bounds required')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)

    def event(kind, **values):
        text = json.dumps({'event': kind, 'time': time.time(), **values}, allow_nan=False)
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(text + '\n')
        print(text, flush=True)

    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        manifest = open_bundle(args.cache)
        splits = {}
        selection = {}
        generator = torch.Generator().manual_seed(42)
        for split, count in [('train', args.train_count), ('val', args.val_count)]:
            dataset = FrozenDataset(args.cache, manifest, split)
            candidates = {}
            excluded = []
            for index, identifier in enumerate(dataset.ids):
                if 64 <= manifest['files'][identifier]['length'] <= 128:
                    entry = dataset[index]
                    if float(entry['mask'].float().mean()) >= .9:
                        candidates[identifier] = entry
                    else:
                        excluded.append(identifier)
            ids = select_panel(
                {**manifest, 'train_ids': list(candidates), 'val_ids': []}, count)
            splits[split] = [(candidates[i], sample_pairs(candidates[i], generator, args.pairs))
                             for i in ids]
            selection[split] = {'ids': ids, 'excluded_low_confidence': excluded}
        if set(selection['train']['ids']) & set(selection['val']['ids']):
            raise ValueError('Probe splits overlap')
        atomic_json(root / 'provenance.json', {
            'protocol': 'se3-pairwise-information-v1', 'args': vars(args),
            'selection': selection, 'bundle_sha256': manifest['bundle_sha256'],
            'source_sha256': digest(__file__), 'torch': torch.__version__,
            'dependencies': {name: digest(name) for name in (
                'foldtree2/se3_cache.py', 'scripts/se3/diagnose_se3_reconstruction.py')},
            'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'scope': 'reused development holdout; no independence from production pretraining',
            'inputs': 'symmetric residue feature sum, absolute difference, product; log sequence separation',
            'selection_metric': 'training loss only; evaluate final fixed-budget weights',
            'objective': 'SmoothL1(distance/10) + 0.1 * binary contact cross entropy',
            'contacts': 'CA distance below 8 Angstrom; row separation at least 12',
        })
        (root / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        for split, records in splits.items():
            atomic_json(root / (split + '_pairs.json'), [
                {'id': e['id'], 'pairs': pairs.tolist()} for e, pairs in records])
        results = {}
        for representation in ('sequence', 'codebook', 'continuous'):
            data = {}
            for split, records in splits.items():
                data[split] = (
                    torch.cat([pair_features(e, p, representation) for e, p in records]).to(device),
                    torch.cat([(e['target'][p[:, 0]] - e['target'][p[:, 1]]).norm(dim=-1)
                               for e, p in records]).to(device),
                )
            x, target = data['train']
            mean, scale = x.mean(0), x.std(0).clamp_min(1e-4)
            data = {s: ((a - mean) / scale, b) for s, (a, b) in data.items()}
            torch.manual_seed(42)
            model = torch.nn.Sequential(
                torch.nn.Linear(x.shape[1], 64), torch.nn.GELU(),
                torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 2)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            sampling = torch.Generator().manual_seed(42)
            started = time.monotonic()
            event('ready', representation=representation, train_pairs=len(target))
            for step in range(1, args.steps + 1):
                indices = torch.randint(len(target), (min(1024, len(target)),),
                                        generator=sampling).to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(data['train'][0][indices])
                pred = F.softplus(output[:, 0])
                loss = F.smooth_l1_loss(pred / 10., target[indices] / 10.)
                loss = loss + .1 * F.binary_cross_entropy_with_logits(
                    output[:, 1], (target[indices] < 8.).float())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite probe loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                if step % 100 == 0:
                    event('progress', representation=representation, step=step,
                          loss=float(loss.detach()), gradient_norm=float(norm))
                if time.monotonic() - started >= args.seconds:
                    break
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite probe parameters')
            with torch.no_grad():
                result = {}
                per_chain = {}
                for split, (a, b) in data.items():
                    output = model(a)
                    prediction = F.softplus(output[:, 0])
                    result[split] = report(prediction, b, output[:, 1])
                    result[split]['constant_train_mean_distance_mae'] = float(
                        (b - target.mean()).abs().mean())
                    per_chain[split] = [
                        {'id': entry['id'], **report(
                            prediction[j * args.pairs:(j + 1) * args.pairs],
                            b[j * args.pairs:(j + 1) * args.pairs],
                            output[j * args.pairs:(j + 1) * args.pairs, 1])}
                        for j, (entry, _) in enumerate(splits[split])]
            result.update(steps=step, seconds=time.monotonic() - started)
            results[representation] = result
            atomic_json(root / (representation + '_per_chain.json'), per_chain)
            atomic_checkpoint(root / (representation + '.pt'), {
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'input_mean': mean, 'input_scale': scale, 'step': step,
                'representation': representation, 'resumable': False,
            })
            atomic_json(root / 'summary.json', results)
            event('case_complete', representation=representation, result=result)
        event('complete', representations=list(results), production_forward_calls=0)
    except Exception as exc:
        event('failure', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_clariden/bundle_4096')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', required=True, choices=['cpu', 'cuda:0', 'cuda:1'])
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--train-count', type=int, default=64)
    parser.add_argument('--val-count', type=int, default=16)
    parser.add_argument('--pairs', type=int, default=256)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
