"""Submit one bounded training-data growth trial with the retained holdout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from scripts.se3.se3_budget import reserve

ROOT = Path('/capstor/store/cscs/swissai/a0117/foldtree2/se3')
STAGE = 'growth_fp64_8192'


def check_growth(previous, current):
    identity = dict(current)
    expected = identity.pop('bundle_sha256')
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != expected:
        raise ValueError('Growth manifest identity mismatch')
    if (len(current['train_ids']), len(current['val_ids'])) != (8192, 512):
        raise ValueError('Unexpected growth split sizes')
    if (current['train_ids'][:4096] != previous['train_ids']
            or current['val_ids'] != previous['val_ids']):
        raise ValueError('Growth must preserve training prefix and entire validation split')
    if len(set(current['train_ids'] + current['val_ids'])) != 8704:
        raise ValueError('Duplicate or overlapping growth IDs')
    for key in ('input_identity', 'scale', 'calibration_sha256', 'calibration_ids', 'feature_dim'):
        if current[key] != previous[key]:
            raise ValueError('Growth preprocessing differs: ' + key)
    for identifier in previous['files']:
        if current['files'][identifier] != previous['files'][identifier]:
            raise ValueError('Retained frozen entry changed: ' + identifier)


def command(code):
    code = Path(code).resolve()
    return ['sbatch', '--account=a0117', '--partition=normal', '--nodes=2', '--time=120',
            '--ntasks-per-node=1', '--gpus-per-task=4', '--cpus-per-task=8', '--parsable',
            '--job-name=ft2-se3-growth8192', f'--output={code}/logs/{STAGE}-%j.out',
            str(code / 'alps/se3/train.sbatch'), '--config',
            'configs/clariden_se3/growth_fp64_8192.yaml']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--available-credit', type=float, required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    previous = json.loads((ROOT / 'cache/bundle_4096/manifest.json').read_text())
    current = json.loads(Path('/iopsstor/scratch/cscs/dmoi/se3/bundle_growth_8192/manifest.json').read_text())
    check_growth(previous, current)
    for stage in ('fp64_debug_one', 'fp64_debug_two'):
        gate = json.loads((ROOT / 'artifacts' / f'{stage}-gate.json').read_text())
        if not gate['passed'] or gate['maximum_parameter_buffer_difference'] != 0.0:
            raise ValueError('Precision stability gate failed')
    prior = [json.loads(x) for x in (ROOT / 'runs/medium_fp64_trial/events.jsonl').read_text().splitlines()]
    if (len([x for x in prior if x['event'] == 'train_epoch']) != 16
            or not any(x['event'] == 'update' and x['global_step'] == 4096 for x in prior)):
        raise ValueError('Previous learning trial incomplete')
    if (ROOT / 'runs' / STAGE).exists():
        raise ValueError('Growth output exists; use an explicitly bounded recovery')
    os.chdir(code)
    (code / 'logs').mkdir(exist_ok=True)
    os.environ.update(TRAIN_WORKDIR=str(code), EDF_PATH=str(code / 'alps/se3/extended-pinned.toml'),
                      TRAIN_VENV=str(ROOT / 'environments/028cc752968f218de4e5837f75b2e7a8a004162a'))
    cmd = command(code)
    preflight = subprocess.run(cmd[:1] + ['--test-only'] + cmd[1:], capture_output=True, text=True)
    if preflight.returncode:
        raise RuntimeError('Slurm preflight rejected growth: ' + preflight.stderr.strip())
    print(preflight.stderr, flush=True)
    reserve(code / 'configs/clariden_se3/growth_budget.json', ROOT / 'budget.jsonl', STAGE,
            args.available_credit, cmd, args.submit)
    if args.submit:
        receipt = json.loads(Path(str(ROOT / 'budget.jsonl') + '.' + STAGE + '.receipt.json').read_text())
        job = receipt['stdout'].strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected Slurm receipt')
        (ROOT / 'artifacts' / f'{STAGE}-submission.json').write_text(json.dumps({
            'job_id': job, 'code_revision': code.name, 'bundle_sha256': current['bundle_sha256'],
            'command': cmd, 'credit_basis': 'User-confirmed sufficient credit; authorized ceiling 256 GPUh',
        }, indent=2) + '\n')


if __name__ == '__main__':
    main()
