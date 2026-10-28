"""Nested representative-count comparison with fixed original and expanded holdouts."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

from foldtree2.se3_cache import FrozenDataset, digest, identity_hash, open_bundle
from foldtree2.train_distance_restraints import sources as training_sources
from scripts.diagnose_se3_reconstruction import atomic_json
from scripts.enrich_se3_decoder_cache import enrich, enrichment_sources


COUNTS = (256, 1024, 2048)


def choose_splits(source, manifest, previous, train_count, val_count):
    """Retain monitored IDs; append deterministically without moving IDs across splits."""
    if previous['input_identity'] != manifest['input_identity']:
        raise ValueError('Previous and expanded cache production identities differ')
    selected, exclusions = {}, {}
    for split, count in (('train', train_count), ('val', val_count)):
        retained = previous[split + '_ids']
        if count < len(retained) or not set(retained) <= set(manifest[split + '_ids']):
            raise ValueError('Expansion must retain previous IDs in their original split')
        dataset = FrozenDataset(source, manifest, split)
        usable, excluded = [], []
        for index, identifier in enumerate(dataset.ids):
            entry = dataset[index]
            if float(entry['mask'].float().mean()) < .9:
                excluded.append({'id': identifier, 'reason': 'valid_fraction_below_0.9'})
                continue
            mask = entry['mask']
            valid_indices = mask.nonzero().flatten()
            if len(valid_indices) < 2 or int(valid_indices[-1] - valid_indices[0]) < 12:
                excluded.append({'id': identifier, 'reason': 'no_nonlocal_pairs'})
                continue
            usable.append(identifier)
        if not set(retained) <= set(usable):
            raise ValueError('Previous panel no longer passes expanded eligibility')
        retained_set = set(retained)
        remaining = sorted((i for i in usable if i not in retained_set),
                           key=lambda i: (manifest['files'][i]['length'], i))
        # Fixed seed, independent of targets, spreads lengths without choosing easy folds.
        generator = torch.Generator().manual_seed(42 if split == 'train' else 43)
        remaining = [remaining[i] for i in torch.randperm(len(remaining), generator=generator).tolist()]
        ids = list(retained) + remaining[:count - len(retained)]
        if len(ids) != count:
            raise ValueError(f'Insufficient eligible {split} representatives: {len(usable)}')
        selected[split] = ids
        exclusions[split] = excluded
    if set(selected['train']) & set(selected['val']):
        raise ValueError('Expanded splits overlap')
    return selected, exclusions


def write_subset(source, manifest, root, train_ids, val_ids, metadata):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'entries').mkdir()
    if (len(set(train_ids + val_ids)) != len(train_ids + val_ids)
            or not set(train_ids) <= set(manifest['train_ids'])
            or not set(val_ids) <= set(manifest['val_ids'])):
        raise ValueError('Subset must preserve distinct original split IDs')
    files = {}
    for identifier in train_ids + val_ids:
        record = manifest['files'][identifier]
        shutil.copyfile(Path(source) / 'entries' / record['name'], root / 'entries' / record['name'])
        files[identifier] = copy.deepcopy(record)
    result = copy.deepcopy(manifest)
    result.pop('bundle_sha256')
    result.update(train_ids=list(train_ids), val_ids=list(val_ids), files=files,
                  parent_bundle_sha256=manifest['bundle_sha256'], data_scaling=metadata)
    result['scope'] = ('AFDB cluster representatives per dataset origin; original split preserved; '
                       'no cluster-membership/homology audit available; production pretraining overlap unverified')
    result['bundle_sha256'] = identity_hash(result)
    atomic_json(root / 'manifest.json', result)
    return open_bundle(root)


def sources():
    return {**training_sources(), **enrichment_sources(),
            'scripts/scale_distance_data.py': digest(__file__),
            'scripts/extend_distance_data.py': digest('scripts/extend_distance_data.py')}


def subset_report(report, ids):
    selected = [r for r in report['per_chain'] if r['id'] in set(ids)]
    if len(selected) != len(ids):
        raise ValueError('Evaluation report missing expected holdout IDs')
    keys = ('nll', 'capped_distance_mae_angstrom', 'top_L_precision', 'contact_brier')
    return {'chains': len(selected),
            **{key: sum(r['metrics'][key] for r in selected) / len(selected) for key in keys}}


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    identity = sources()
    try:
        parent = open_bundle(args.source)
        previous = open_bundle(args.previous)
        counts = tuple(getattr(args, 'counts', COUNTS))
        if counts == COUNTS and (len(previous['train_ids']) != 256 or len(previous['val_ids']) != 64):
            raise ValueError('Expected the original 256-training/64-validation panel')
        original_val = previous.get('data_scaling', {}).get('original_val_ids', previous['val_ids'])
        if len(original_val) != 64 or not set(original_val) <= set(previous['val_ids']):
            raise ValueError('Expected the original 64 validation IDs')
        selected, excluded = choose_splits(args.source, parent, previous, max(counts), 256)
        additional_val = [i for i in selected['val'] if i not in set(original_val)]
        metadata = {
            'seed': 42, 'previous_bundle_sha256': previous['bundle_sha256'],
            'original_val_ids': original_val, 'train_counts': list(counts),
            'validation': 'fixed256 including original64; all now monitored development IDs',
            'eligibility': '>=90% valid residues and at least one valid row-separation>=12 pair',
        }
        atomic_json(root / 'selection.json', {
            'train_ids': selected['train'], 'val_ids': selected['val'],
            'exclusions': excluded, **metadata})
        atomic_json(root / 'experiment.json', {
            'sources': identity, 'parent_bundle_sha256': parent['bundle_sha256'],
            'metadata': metadata, 'hidden': 256, 'epochs': args.epochs,
            'seconds_per_case': args.seconds, 'pairs_per_chain': 512,
            'objective': 'unchanged population CE; AdamW1e-3 weight_decay.01 clip1 seed42',
            'comparison': 'nested training counts; shared epochs and shared optimizer updates; fixed holdout',
            'length_caveat': 'Original256 are short chains; expanded cohorts contain longer representatives too',
            'scope': 'representative-level refiner holdout, not verified family or production-pretraining independence'})
        for name in identity:
            path = root / 'sources' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(name).read_bytes())
        raw = root / 'raw_bundle'
        write_subset(args.source, parent, raw, selected['train'], selected['val'], metadata)
        print(json.dumps({'event': 'selection_complete', 'train': max(counts), 'val': 256}), flush=True)
        enriched = root / 'enriched'
        enrich(SimpleNamespace(cache=str(raw), output=str(enriched), dataset=args.dataset, device='cpu'))
        full = open_bundle(enriched)
        summaries = {}
        for count in counts:
            if identity != sources():
                raise ValueError('Data-scaling source changed')
            cache = root / f'bundle_{count}'
            write_subset(enriched, full, cache, selected['train'][:count], selected['val'], metadata)
            output = root / f'train_{count}'
            print(json.dumps({'event': 'launch', 'train_count': count, 'time': time.time()}), flush=True)
            with (root / f'train_{count}.log').open('x') as stream:
                subprocess.run([
                    sys.executable, '-u', '-m', 'foldtree2.train_distance_restraints',
                    '--cache', str(cache), '--output', str(output), '--device', args.device,
                    '--hidden', '256', '--epochs', str(args.epochs), '--seconds', str(args.seconds),
                    '--pairs', '512', '--modes', 'decoder'],
                    check=True, timeout=args.seconds + 900, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
            completion = json.loads((output / 'decoder/completion.json').read_text())
            if completion['stop_reason'] == 'epochs' and completion['updates'] != count * args.epochs:
                raise ValueError('Incomplete fixed-epoch scaling case')
            report = json.loads((output / 'decoder/final_validation.json').read_text())
            if ({r['id'] for r in report['per_chain']} != set(selected['val'])
                    or completion['validation']['chains'] != 256):
                raise ValueError('Scaling validation membership changed')
            curves = []
            history = {row['epoch']: row for row in
                       json.loads((output / 'decoder/history.json').read_text())}
            for path in sorted((output / 'decoder').glob('validation-*.json')):
                epoch = int(path.stem.split('-')[-1])
                val = json.loads(path.read_text())
                curves.append({'epoch': epoch, 'updates': history[epoch]['updates'],
                               'elapsed_seconds': history[epoch]['elapsed_seconds'],
                               'original64': subset_report(val, original_val),
                               'additional192': subset_report(val, additional_val)})
            atomic_json(output / 'holdout_curves.json', curves)
            summaries[str(count)] = {
                'updates': completion['updates'], 'stop_reason': completion['stop_reason'],
                'all256': completion['validation'],
                'original64': subset_report(report, original_val),
                'additional192': subset_report(report, additional_val)}
            atomic_json(root / 'progress.json', summaries)
            print(json.dumps({'event': 'case_complete', 'train_count': count,
                              'updates': completion['updates']}), flush=True)
        if identity != sources():
            raise ValueError('Scaling source changed')
        atomic_json(root / 'completion.json', summaries)
    except Exception as exc:
        atomic_json(root / 'failure.json', {'error': repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/se3_clariden/bundle_4096')
    parser.add_argument('--previous', default='runs/se3_decoder_enrichment_v3_20261021')
    parser.add_argument('--dataset', default='foldtree2/notebooks/structs_training_mk2.h5')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0', 'cuda:1'], default='cuda:1')
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--counts', type=int, nargs='+', default=list(COUNTS))
    parser.add_argument('--seconds', type=float, default=7200.)
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError('Positive experiment bounds required')
    if args.counts != sorted(set(args.counts)) or min(args.counts) < 1:
        raise ValueError('Counts must be positive and strictly increasing')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run(args)


if __name__ == '__main__':
    main()
