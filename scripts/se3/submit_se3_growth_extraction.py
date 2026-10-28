"""Reserve and submit the bounded Clariden frozen-cache preparation job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from scripts.se3.se3_budget import reserve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--available-credit', required=True, type=float)
    parser.add_argument('--stage', choices=['growth_extract_clariden', 'growth_extract_h5pyfix', 'growth_extract_runtime_audit', 'growth_extract_ieee'],
                        default='growth_extract_clariden')
    args = parser.parse_args()
    root = Path('/capstor/store/cscs/swissai/a0117/foldtree2/se3')
    code = Path(__file__).resolve().parents[1]
    if args.stage != 'growth_extract_clariden':
        prior_artifact = {'growth_extract_h5pyfix': 'growth-extraction-submission.json',
                          'growth_extract_runtime_audit': 'growth_extract_h5pyfix-submission.json',
                          'growth_extract_ieee': 'growth_extract_runtime_audit-submission.json'}[args.stage]
        previous = json.loads((root / 'artifacts' / prior_artifact).read_text())
        state = subprocess.check_output(['sacct', '-X', '-j', previous['job_id'],
            '--format=State', '-n', '-P'], text=True).strip()
        if state != 'FAILED':
            raise ValueError('Dependency correction requires the preceding failed preparation')
    if not (root / 'inputs/growth_graph_inputs/manifest.json').exists():
        raise ValueError('Graph export has not been transferred')
    if (root / 'cache/bundle_growth_8192').exists():
        raise ValueError('Growth cache already exists; inspect preparation before retrying')
    os.chdir(code)
    os.environ.update(TRAIN_WORKDIR=str(code), EDF_PATH=str(code / 'alps/se3/extended-pinned.toml'),
        TRAIN_VENV=str(root / 'environments/028cc752968f218de4e5837f75b2e7a8a004162a'))
    command = ['sbatch', '--account=a0117', '--partition=normal', '--nodes=1', '--time=30',
               '--ntasks-per-node=1', '--gpus-per-task=4', '--cpus-per-task=8', '--parsable',
               f'--output={code}/logs/growth-extraction-%j.out', str(code / 'alps/se3/extract_growth.sbatch')]
    preflight = subprocess.run(command[:1] + ['--test-only'] + command[1:], capture_output=True, text=True)
    if preflight.returncode:
        raise RuntimeError('Slurm preflight failed: ' + preflight.stderr.strip())
    print(preflight.stderr, flush=True)
    reserve(code / 'configs/clariden_se3/growth_budget.json', root / 'budget.jsonl',
            args.stage, args.available_credit, command, args.submit)
    if args.submit:
        receipt = json.loads((root / ('budget.jsonl.' + args.stage + '.receipt.json')).read_text())
        job = receipt['stdout'].strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected Slurm receipt')
        artifact = 'growth-extraction-submission.json' if args.stage == 'growth_extract_clariden' else args.stage + '-submission.json'
        (root / 'artifacts' / artifact).write_text(json.dumps({
            'job_id': job, 'code_revision': code.name, 'command': command,
            'credit_basis': 'User-confirmed sufficient credit; authorized 256 GPUh ceiling',
        }, indent=2) + '\n')


if __name__ == '__main__':
    main()
