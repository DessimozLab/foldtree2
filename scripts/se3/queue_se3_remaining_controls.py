"""Dependency-gated nested-panel, model-conditioning and information controls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('runs/se3_reconstruction_20261020')


def wait_for_parent(gpu, timeout=9000):
    service = f'foldtree2-se3-controls-gpu{gpu}-20261020.service'
    deadline = time.monotonic() + timeout
    print(json.dumps({'event': 'waiting', 'service': service}), flush=True)
    while True:
        state = subprocess.run(
            ['systemctl', '--user', 'show', service, '--property=ActiveState', '--value'],
            check=True, capture_output=True, text=True).stdout.strip()
        if state not in ('active', 'activating', 'deactivating'):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Earlier GPU control queue exceeded waiting limit')
        time.sleep(10)
    result = subprocess.run(
        ['systemctl', '--user', 'show', service, '--property=Result', '--value'],
        check=True, capture_output=True, text=True).stdout.strip()
    if result != 'success':
        raise RuntimeError(f'Earlier control queue failed: {result}')
    parents = (['coordinates_control', 'model_native_noise'] if gpu == 0 else
               ['model_panel_transfer', 'model_control', 'model_local_contacts'])
    for name in parents:
        events = [json.loads(line) for line in
                  (ROOT / name / 'events.jsonl').read_text().splitlines()]
        if events[-1]['event'] != 'complete' or any(e['event'] == 'failure' for e in events):
            raise RuntimeError(f'No clean completion for earlier control: {name}')


def commands(gpu):
    count = 2 if gpu == 0 else 4
    common = ['--device', f'cuda:{gpu}']
    runner = [sys.executable, '-u', '-m', 'scripts.se3.diagnose_se3_reconstruction']
    ladder = [*runner, *common, '--mode', 'model', '--count', str(count),
              '--panel-size', '8', '--heldout-count', '16', '--learning-rate', '.0005',
              '--output', str(ROOT / f'ladder_{count}')]
    cases = [ladder]
    for n, source in [(count, f'ladder_{count}'),
                      (1, 'model_one') if gpu == 0 else (8, 'model_panel')]:
        cases.append([
            *runner, *common, '--mode', 'model', '--count', str(n), '--panel-size', '8',
            '--heldout-count', '16', '--learning-rate', '.0005',
            '--evaluate-checkpoint', str(ROOT / source / 'best.pt'), '--audit-model',
            '--output', str(ROOT / f'audit_{n}'),
        ])
    if gpu == 0:
        cases.append([
            sys.executable, '-u', '-m', 'scripts.se3.probe_se3_information', *common,
            '--output', str(ROOT / 'information_probes'),
        ])
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    wait_for_parent(args.gpu)
    for command in commands(args.gpu):
        print(json.dumps({'event': 'launch', 'command': command}), flush=True)
        subprocess.run(command, check=True)
    print(json.dumps({'event': 'queue_complete', 'gpu': args.gpu}), flush=True)


if __name__ == '__main__':
    main()
