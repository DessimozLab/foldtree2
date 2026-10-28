#!/usr/bin/env python3
"""Resume the stopped TCS sweep with Snakemake keep-going, preserving its gates."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def keepgoing(batch):
    lines = batch.splitlines(keepends=True)
    selected = [i for i, line in enumerate(lines) if '/bin/snakemake ' in line]
    if len(selected) != 1:
        raise ValueError('Expected exactly one Snakemake invocation')
    i = selected[0]
    if '-k' in lines[i].split() or '--keep-going' in lines[i].split():
        raise ValueError('Keep-going already enabled; inspect existing submission')
    lines[i] = lines[i].replace('/bin/snakemake ', '/bin/snakemake -k ', 1)
    return ''.join(lines)


def resume(root, previous_job):
    state = json.loads((root / 'supervision.json').read_text())
    if (state.get('status') != 'needs_attention' or state.get('phase') != 'full'
            or state['jobs'].get('full') != previous_job):
        raise ValueError('Expected the identified stopped full sweep')
    status = subprocess.check_output(['sacct', '-X', '-j', previous_job, '-n', '-P',
                                      '--format=JobIDRaw,State'], text=True)
    if [previous_job, 'FAILED'] not in [x.split('|')[:2] for x in status.splitlines()]:
        raise ValueError('Previous allocation must be confirmed failed')
    batch = root / 'full.sbatch'
    old = batch.read_text()
    new = keepgoing(old)
    provenance = root / 'operational-source.json'
    hashes = json.loads(provenance.read_text())
    if hashes['full.sbatch'] != hashlib.sha256(old.encode()).hexdigest():
        raise ValueError('Launcher provenance changed')
    operations = root / 'ops-20261020-keepgoing'
    operations.mkdir(exist_ok=False)
    for name in ('full.sbatch', 'supervision.json', 'operational-source.json', 'callback-state.json'):
        if (root / name).exists():
            shutil.copy2(root / name, operations / name)
    candidate = operations / 'full-keepgoing.sbatch'
    candidate.write_text(new)
    subprocess.run(['bash', '-n', str(candidate)], check=True)
    subprocess.run(['sbatch', '--test-only', str(candidate)], check=True)
    batch.write_text(new)
    hashes['full.sbatch'] = hashlib.sha256(new.encode()).hexdigest()
    provenance.write_text(json.dumps(hashes, indent=2) + '\n')
    state['jobs'].pop('full')
    state.setdefault('previous_jobs', []).append({'phase': 'full', 'job_id': previous_job,
        'state': 'FAILED', 'reason': 'User requested -k to execute independent tasks despite scoring failures'})
    state.update(status='active', slurm_state='READY_FOR_SUBMISSION', accounting=[],
                 keep_going=True)
    state.pop('reason', None)
    (root / 'supervision.json').write_text(json.dumps(state, indent=2) + '\n')
    # Existing activation retains project-quota checks and bounded infrastructure retries.
    receipt = {'previous_job': previous_job, 'keep_going': True,
               'launcher_sha256': hashes['full.sbatch'], 'state': 'submitting'}
    target = operations / 'receipt.json'
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    activation = subprocess.check_output(['sbatch', '--parsable', str(root / 'advance.sbatch')],
                                        text=True).strip().split(';')[0]
    if not activation.isdigit():
        raise ValueError('Unexpected activation receipt')
    receipt.update(activation_job=activation, state='queued')
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('--previous-job', required=True)
    args = parser.parse_args()
    with (args.deployment / '.advance.lock').open('a') as advance:
        fcntl.flock(advance, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (args.deployment / '.supervision.lock').open('a') as supervisor:
            fcntl.flock(supervisor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(resume(args.deployment, args.previous_job)))
