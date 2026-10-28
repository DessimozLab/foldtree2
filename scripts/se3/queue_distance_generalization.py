"""Finish width-128 control, cancel remaining memorization, compare held-out widths."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from foldtree2.se3_cache import digest, open_bundle
from foldtree2.train_distance_restraints import sources
from scripts.se3.diagnose_se3_reconstruction import atomic_json


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    identity = {**sources(), __file__: digest(__file__)}
    manifest = open_bundle(args.cache)
    if len(manifest['train_ids']) != 256 or len(manifest['val_ids']) != 64:
        raise ValueError('Expected exact 256/64 development bundle')
    atomic_json(root / 'experiment.json', {
        'sources': identity, 'bundle_sha256': manifest['bundle_sha256'],
        'widths': [64, 128, 256], 'epochs': 128, 'updates_per_case': 32768,
        'seconds_per_case': 3600, 'seed': 42, 'pairs_per_chain': 512,
        'selection': 'final fixed budget; report training and validation curves, no validation selection',
        'scope': 'monitored development holdout; no production pretraining independence claim'})
    for name in identity:
        target = root / 'sources' / Path(name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    try:
        previous = Path('runs/se3_distance_capacity_20261022')
        deadline = time.monotonic() + 1800
        print('Waiting for width128 completion before cancelling remaining memorization', flush=True)
        while not (previous / 'width_128/completion.json').exists():
            if (previous / 'failure.json').exists() or time.monotonic() >= deadline:
                raise RuntimeError('Width128 failed or did not complete within wait ceiling')
            time.sleep(1)
        completion = json.loads((previous / 'width_128/completion.json').read_text())
        if completion['stop_reason'] != 'steps' or completion['updates'] != 40000:
            raise ValueError('Width128 did not finish the expected budget')
        subprocess.run(['systemctl', '--user', 'stop',
                        'foldtree2-se3-distance-capacity-20261022.service'], check=True)
        atomic_json(root / 'memorization_cancellation.json', {
            'reason': 'User requested cancelling width256 memorization after width128',
            'width128_complete': True,
            'width256_started': (previous / 'width_256').exists(),
            'time': time.time()})
        for width in (64, 128, 256):
            if identity != {**sources(), __file__: digest(__file__)}:
                raise ValueError('Queued generalization source changed')
            if open_bundle(args.cache)['bundle_sha256'] != manifest['bundle_sha256']:
                raise ValueError('Queued bundle changed')
            print(json.dumps({'event': 'launch', 'width': width, 'time': time.time()}), flush=True)
            with (root / f'width_{width}.log').open('x') as stream:
                subprocess.run([
                    sys.executable, '-u', '-m', 'foldtree2.train_distance_restraints',
                    '--cache', args.cache, '--output', str(root / f'width_{width}'),
                    '--device', 'cuda:1', '--hidden', str(width), '--epochs', '128',
                    '--seconds', '3600', '--pairs', '512', '--modes', 'decoder'],
                    check=True, timeout=3900, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
            result = json.loads((root / f'width_{width}/decoder/completion.json').read_text())
            if result['stop_reason'] == 'epochs' and result['updates'] != 32768:
                raise ValueError('Incomplete epoch-budget case')
            print(json.dumps({'event': 'case_complete', 'width': width,
                              'updates': result['updates'], 'stop_reason': result['stop_reason']}), flush=True)
        atomic_json(root / 'completion.json', {'complete': True, 'widths': [64, 128, 256]})
    except Exception as exc:
        atomic_json(root / 'failure.json', {'error': repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_decoder_enrichment_v3_20261021')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
