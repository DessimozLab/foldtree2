"""Submit the authorized bounded medium learning trial after stability checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from scripts.se3_budget import reserve


ROOT = Path('/capstor/store/cscs/swissai/a0117/foldtree2/se3')
STAGE = 'medium_fp64_trial'


def command(code):
    code = Path(code).resolve()
    return [
        'sbatch', '--account=a0117', '--partition=normal', '--nodes=2', '--time=120',
        '--ntasks-per-node=1', '--gpus-per-task=4', '--cpus-per-task=8', '--parsable',
        '--job-name=ft2-se3-medium-fp64', f'--output={code}/logs/{STAGE}-%j.out',
        str(code / 'alps/se3/train.sbatch'), '--config',
        'configs/clariden_se3/medium_fp64_trial.yaml',
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--available-credit', type=float, required=True,
                        help='User-confirmed available credit; not a Slurm fair-share value')
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    for stage in ('fp64_debug_one', 'fp64_debug_two'):
        gate = json.loads((ROOT / 'artifacts' / f'{stage}-gate.json').read_text())
        # exact_resume denotes the verifier's same-topology mode. The two-node
        # gate compares topologies and instead reports the measured difference.
        if (not gate['passed'] or gate['maximum_parameter_buffer_difference'] != 0.0
                or (stage.endswith('_one') and not gate['exact_resume'])):
            raise ValueError(f'{stage} stability/recovery gate has not passed')
    if (ROOT / 'runs' / STAGE).exists():
        raise ValueError('Trial output already exists; recovery requires an explicit resume decision')
    os.chdir(code)
    (code / 'logs').mkdir(exist_ok=True)
    os.environ.update(
        TRAIN_WORKDIR=str(code), EDF_PATH=str(code / 'alps/se3/extended-pinned.toml'),
        TRAIN_VENV=str(ROOT / 'environments/028cc752968f218de4e5837f75b2e7a8a004162a'))
    cmd = command(code)
    preflight = subprocess.run(cmd[:1] + ['--test-only'] + cmd[1:], capture_output=True, text=True)
    if preflight.returncode:
        raise RuntimeError('Slurm preflight rejected trial: ' + preflight.stderr.strip())
    print(preflight.stderr, flush=True)
    reserve(code / 'configs/clariden_se3/long_trial_budget.json', ROOT / 'budget.jsonl',
            STAGE, args.available_credit, cmd, args.submit)
    if args.submit:
        receipt = json.loads(Path(str(ROOT / 'budget.jsonl') + '.' + STAGE + '.receipt.json').read_text())
        job = receipt['stdout'].strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected Slurm receipt')
        (ROOT / 'artifacts' / f'{STAGE}-submission.json').write_text(json.dumps({
            'job_id': job, 'code_revision': code.name, 'stage': STAGE, 'command': cmd,
            'credit_basis': 'User confirmed sufficient project credit; 256 GPUh authorized rollout ceiling',
        }, indent=2) + '\n')


if __name__ == '__main__':
    main()
