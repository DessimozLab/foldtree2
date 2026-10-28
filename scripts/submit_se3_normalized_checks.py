"""Submit one bounded normalized-model check; no automatic retry or scaling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from scripts.se3_budget import reserve


def command(code, stage):
    code = Path(code).resolve()
    root = Path('/capstor/store/cscs/swissai/a0117/foldtree2/se3')
    train = 'bash "$TRAIN_WORKDIR/alps/se3/train.sbatch"'
    config = f'configs/clariden_se3/{stage}.yaml'
    run = root / 'runs' / stage
    checkpoint = run / 'last.ckpt'
    body = f'set -eu\n{train} --config {config}\n'
    body += f'cp {checkpoint} {run}/update-016.ckpt\n'
    body += f'{train} --config {config} --resume-from {checkpoint} --stop-after-updates 32\n'
    control_name = stage.rsplit('_', 1)[0] + '_control'
    if stage.endswith('_one'):
        body += f'{train} --config configs/clariden_se3/{control_name}.yaml --stop-after-updates 32\n'
        control = root / 'runs' / control_name
        exact = ' --exact'
        nodes, minutes = 1, 30
    else:
        control = root / 'runs' / control_name
        exact = ''
        nodes, minutes = 2, 15
    # Postchecks run inside the image; the ARM venv cannot run in the host shell.
    check = f'''set -eu
cd "$TRAIN_WORKDIR"
source "$TRAIN_VENV/bin/activate"
python -m scripts.verify_se3_normalized_debug {run} {control}{exact} --output {root}/artifacts/{stage}-gate.json
'''
    body += 'srun --nodes=1 --ntasks=1 --environment="$EDF_PATH" --network=disable_rdzv_get --mpi=none --cpu-bind=none --kill-on-bad-exit=1 '
    body += "bash -c '" + check + "'\n"
    return ['sbatch', '--account=a0117', '--partition=debug', f'--nodes={nodes}', f'--time={minutes}',
            '--ntasks-per-node=1', '--gpus-per-task=4', '--cpus-per-task=8', '--parsable',
            f'--job-name=ft2-se3-{stage}', f'--output={code}/logs/{stage}-%j.out', '--wrap', body]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['normalized_debug_one','normalized_debug_two','fp64_debug_one','fp64_debug_two'])
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    code = Path(__file__).resolve().parents[1]
    root = Path('/capstor/store/cscs/swissai/a0117/foldtree2/se3')
    if args.stage.endswith('_two'):
        first = args.stage.rsplit('_',1)[0] + '_one'
        gate = json.loads((root / 'artifacts' / (first+'-gate.json')).read_text())
        if not gate['passed'] or not gate['exact_resume']:
            raise ValueError('One-node normalized resume gate has not passed')
    os.chdir(code)
    os.environ.update(TRAIN_WORKDIR=str(code), EDF_PATH=str(code/'alps/se3/extended-pinned.toml'),
                      TRAIN_VENV=str(root/'environments/028cc752968f218de4e5837f75b2e7a8a004162a'))
    cmd = command(code, args.stage)
    # Validate Slurm resource arguments before reserving or submitting a job.
    result = subprocess.run(cmd[:1] + ['--test-only'] + cmd[1:], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Slurm preflight rejected check: ' + result.stderr.strip())
    print(result.stderr, flush=True)
    reserve(code/'configs/clariden_se3/budget.json', root/'budget.jsonl', args.stage, 256, cmd, args.submit)
    if args.submit:
        receipt = json.loads(Path(str(root/'budget.jsonl')+'.'+args.stage+'.receipt.json').read_text())
        job = receipt['stdout'].strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected Slurm receipt')
        (root/'artifacts'/f'{args.stage}-submission.json').write_text(json.dumps({
            'job_id':job, 'code_revision':code.name, 'stage':args.stage, 'command':cmd}, indent=2)+'\n')
