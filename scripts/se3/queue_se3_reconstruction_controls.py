"""Wait for the original local controls, then run bounded follow-up controls."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    root = Path('runs/se3_reconstruction_20261020')
    if args.gpu == 0:
        service = 'foldtree2-se3-coordinates-20261020.service'
        parents = ['coordinates_seed', 'coordinates_native_noise']
        cases = [
            ('coordinates_control', ['--mode', 'coordinates', '--loss', 'control',
                                     '--learning-rate', '.05']),
            ('model_native_noise', ['--mode', 'model', '--initialization', 'native-noise',
                                    '--heldout-count', '16', '--learning-rate', '.0005']),
        ]
    else:
        service = 'foldtree2-se3-memorization-20261020.service'
        parents = ['model_one', 'model_panel']
        cases = [
            ('model_panel_transfer', ['--mode', 'model', '--heldout-count', '16',
                                     '--learning-rate', '.0005', '--evaluate-checkpoint',
                                     str(root / 'model_panel' / 'best.pt')]),
            ('model_control', ['--mode', 'model', '--loss', 'control',
                               '--heldout-count', '16', '--learning-rate', '.0005']),
            ('model_local_contacts', ['--mode', 'model', '--contacts', 'local-only',
                                     '--heldout-count', '16', '--learning-rate', '.0005']),
        ]
    deadline = time.monotonic() + 4200
    print(json.dumps({'event': 'waiting', 'parent_service': service, 'cases': cases}), flush=True)
    while True:
        state = subprocess.run(
            ['systemctl', '--user', 'show', service, '--property=ActiveState', '--value'],
            check=True, capture_output=True, text=True).stdout.strip()
        if state not in ('active', 'activating', 'deactivating'):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Parent control queue did not finish within 70 minutes')
        time.sleep(10)
    result = subprocess.run(
        ['systemctl', '--user', 'show', service, '--property=Result', '--value'],
        check=True, capture_output=True, text=True).stdout.strip()
    if result != 'success':
        raise RuntimeError(f'Parent service did not succeed: {result}')
    for name in parents:
        summary = json.loads((root / name / 'summary.json').read_text())
        if summary['steps'] < 1:
            raise ValueError(f'Parent has no completed updates: {name}')
        events = [json.loads(line) for line in (root / name / 'events.jsonl').read_text().splitlines()]
        if events[-1]['event'] != 'complete' or any(e['event'] == 'failure' for e in events):
            raise RuntimeError(f'Parent has no clean completion: {name}')
    environment = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    for name, options in cases:
        print(json.dumps({'event': 'launch', 'case': name, 'gpu': args.gpu}), flush=True)
        subprocess.run(
            [sys.executable, '-u', '-m', 'scripts.se3.diagnose_se3_reconstruction',
             '--device', f'cuda:{args.gpu}', '--output', str(root / name), *options],
            check=True, env=environment,
        )
    print(json.dumps({'event': 'queue_complete', 'gpu': args.gpu}), flush=True)


if __name__ == '__main__':
    main()
