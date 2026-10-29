"""Hash-verified sequential decoder-fusion comparison on one local GPU."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

from foldtree2.se3_cache import digest, open_bundle
from foldtree2.src.models.decoder_feature_fusion import MODES
from foldtree2.train_se3_decoder_fusion import fusion_sources
from scripts.se3.diagnose_se3_reconstruction import atomic_json


def prepare(cache, root, epochs):
    manifest = open_bundle(cache)
    if 'decoder_enrichment' not in manifest or epochs < 1:
        raise ValueError('Expected enriched bundle and positive epoch count')
    if len(manifest['train_ids']) % 16:
        raise ValueError('Training count must divide into global batches')
    root.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for mode in MODES:
        config = {
            'cache': str(Path(cache).resolve()), 'output': str((root / mode).resolve()),
            'mode': mode, 'epochs': epochs, 'global_batch': 16, 'learning_rate': .0005,
            'train_count': len(manifest['train_ids']), 'val_count': len(manifest['val_ids']),
            'model': {'hidden': 64, 'depth': 4, 'heads': 4, 'input_normalization': 'layer'},
        }
        path = root / (mode + '.yaml')
        path.write_text(yaml.safe_dump(config))
        hashes[mode] = digest(path)
    sources = {**fusion_sources(), 'scripts/se3/run_se3_decoder_fusion.py': digest(__file__)}
    for name in sources:
        path = root / 'sources' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(name).read_bytes())
    atomic_json(root / 'experiment.json', {
        'configs': hashes, 'sources': sources, 'bundle_sha256': manifest['bundle_sha256'],
        'timeout_seconds_per_case': 2700, 'modes': list(MODES),
        'selection': 'unchanged validation FAPE; retain final fixed-budget metrics as well',
        'soft_pairs': 'separate confidence-weighted nonlocal contact-feature aggregation branch',
    })


def run(root, device):
    experiment = json.loads((root / 'experiment.json').read_text())
    try:
        for mode in MODES:
            for name, expected in experiment['sources'].items():
                if digest(name) != expected:
                    raise ValueError('Staged fusion source changed: ' + name)
            path = root / (mode + '.yaml')
            if digest(path) != experiment['configs'][mode]:
                raise ValueError('Staged fusion config changed')
            config = yaml.safe_load(path.read_text())
            if open_bundle(config['cache'])['bundle_sha256'] != experiment['bundle_sha256']:
                raise ValueError('Fusion bundle changed')
            print(json.dumps({'event': 'launch', 'mode': mode, 'time': time.time()}), flush=True)
            with (root / (mode + '.log')).open('x') as log:
                subprocess.run(
                    [sys.executable, '-u', '-m', 'foldtree2.train_se3_decoder_fusion',
                     '--config', str(path), '--device', 'cuda:0'],
                    check=True, timeout=2700, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': device,
                         'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
            output = Path(config['output'])
            completion = json.loads((output / 'completion.json').read_text())
            if completion['updates'] != config['epochs'] * config['train_count'] // 16:
                raise ValueError('Incomplete matched fusion trial')
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            if events[-1]['event'] != 'complete' or any(e['event'] == 'failure' for e in events):
                raise ValueError('Fusion trial has no clean completion')
            selected = []
            for epoch in range(1, config['epochs'] + 1):
                rows = json.loads((output / f'validation-{epoch:03d}.json').read_text())
                provenance = json.loads((output / 'provenance.json').read_text())
                if len(rows) != config['val_count'] or {r['id'] for r in rows} != set(provenance['val_ids']):
                    raise ValueError('Fusion validation ID/count mismatch')
                selected.append(sum(r['metrics']['fape'] for r in rows) / len(rows))
            if selected.index(min(selected)) + 1 != completion['best_epoch']:
                raise ValueError('Fusion checkpoint selection mismatch')
            print(json.dumps({'event': 'case_complete', 'mode': mode, 'time': time.time()}), flush=True)
        atomic_json(root / 'queue_complete.json', {'modes': list(MODES)})
    except Exception as exc:
        atomic_json(root / 'queue_failure.json', {'error': repr(exc), 'time': time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--device', choices=['0', '1'], default='0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    if args.prepare_only:
        prepare(args.cache, root, args.epochs)
    else:
        run(root, args.device)


if __name__ == '__main__':
    main()