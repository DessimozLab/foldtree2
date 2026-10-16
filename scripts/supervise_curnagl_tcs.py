#!/usr/bin/env python3
"""Bounded Slurm supervision: retry infrastructure failures, gate full run on pilot."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

RETRYABLE = {'TIMEOUT', 'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL'}
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'REQUEUED', 'RESIZING', 'SUSPENDED'}


def execute(argv, timeout=35):
    return subprocess.check_output(argv, text=True, timeout=timeout).strip()


def save(path, state):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(path)


def submit(root, phase):
    value = execute(['sbatch', '--parsable', str(root / (phase + '.sbatch'))])
    job_id = value.split(';')[0]
    if not job_id.isdigit():
        raise ValueError('Unexpected Slurm job identifier')
    return job_id


def accounting(job_id):
    queued = execute(['squeue', '-h', '--me', '-o', '%i|%T|%r'])
    for row in queued.splitlines():
        ident, status, reason = row.split('|', 2)
        if ident == job_id:
            return ('DEPENDENCY_FAILED' if reason == 'DependencyNeverSatisfied' else status), []
    rows = execute(['sacct', '-j', job_id, '-n', '-P', '--format=JobIDRaw,State,ExitCode,MaxRSS,Elapsed']).splitlines()
    parsed = [r.split('|') for r in rows if r.strip()]
    own = next((r for r in parsed if r[0] == job_id), None)
    return (own[1].split()[0].rstrip('+') if own else 'ACCOUNTING_PENDING'), parsed


def rss_mb(value):
    if not value:
        return None
    scale = {'K': 1 / 1024, 'M': 1, 'G': 1024, 'T': 1024**2}
    return float(value[:-1]) * scale[value[-1]] if value[-1] in scale else float(value) / 1024


def _tick(root):
    path = root / 'supervision.json'
    if path.exists():
        state = json.loads(path.read_text())
    else:
        first = 'rebuild' if (root / 'rebuild.sbatch').is_file() else 'inventory'
        state = {'phase': first, 'jobs': {}, 'attempts': {}, 'status': 'active'}
    state['heartbeat_utc'] = datetime.now(timezone.utc).isoformat()
    if state['status'] != 'active':
        return state
    free = shutil_disk_free(root)
    state['free_bytes'] = free
    phase = state['phase']
    if phase not in state['jobs']:
        if free < 20 * 1024**3:
            state.update(status='needs_attention', reason='Less than 20 GiB free; refusing new submission')
        else:
            state['jobs'][phase] = submit(root, phase)
            state['attempts'][phase] = 0
        save(path, state)
        return state
    job_id = state['jobs'][phase]
    status, rows = accounting(job_id)
    state['slurm_state'] = status
    state['accounting'] = rows
    if status in ACTIVE or status == 'ACCOUNTING_PENDING':
        save(path, state)
        return state
    if status in RETRYABLE and state['attempts'][phase] < 2 and free >= 20 * 1024**3:
        state.setdefault('previous_jobs', []).append({'phase': phase, 'job_id': job_id, 'state': status})
        state['jobs'][phase] = submit(root, phase)
        state['attempts'][phase] += 1
    elif status != 'COMPLETED' or any(r[2] != '0:0' for r in rows if r[0] in {job_id, job_id + '.batch'}):
        state.update(status='needs_attention', reason=f'{phase} job {job_id}: {status}; no scientific/OOM retries')
    elif phase == 'rebuild':
        cfg_path = root / 'rebuild-result-location.json'
        result_path = Path(json.loads(cfg_path.read_text())['summary'])
        result = json.loads(result_path.read_text())
        if result['passed'] < 1:
            raise ValueError('No independently verified rebuilt CATH families')
        state['cath_rebuild'] = {'passed': result['passed'], 'failed': result['failed'], 'summary': str(result_path)}
        state['phase'] = 'inventory'
    elif phase == 'inventory':
        for name in ('plan.json', 'pilot.json'):
            if not (root / name).is_file():
                raise ValueError('Inventory completed without frozen manifests')
        if not (root / 'runtime-check-passed.json').is_file():
            state['runtime_gate'] = 'waiting_for_runtime_check'
            state['phase'] = 'runtime-check'
        else:
            state['runtime_gate'] = 'passed'
            state['phase'] = 'pilot'
    elif phase == 'runtime-check':
        marker = json.loads((root / 'runtime-check-passed.json').read_text())
        if marker['bundles'] != 5 or marker['methods'] != 13:
            raise ValueError('Runtime check did not validate all production methods')
        state['runtime_gate'] = 'passed'
        state['phase'] = 'pilot'
    elif phase in {'pilot', 'full'}:
        plan_path = root / ('pilot.json' if phase == 'pilot' else 'plan.json')
        plan = json.loads(plan_path.read_text())
        marker = Path(plan['config']['output_dir']) / 'validated.json'
        data = json.loads(marker.read_text())
        import hashlib
        expected = len(plan['families']) * len(plan['methods'])
        if (data['plan_sha256'] != hashlib.sha256(plan_path.read_bytes()).hexdigest()
                or data['scores'] != expected or (phase == 'pilot' and expected != 26)):
            raise ValueError('Completion marker does not pass full-coverage gate')
        if phase == 'pilot':
            peaks = [rss_mb(r[3]) for r in rows if r[0].endswith('.batch') and r[3]]
            if not peaks:
                state['slurm_state'] = 'WAITING_FOR_PEAK_MEMORY_ACCOUNTING'
            elif max(peaks) > 24000:
                state.update(status='needs_attention', reason='Pilot peak memory exceeds 24 GB full per-rule budget')
            else:
                state['pilot_peak_rss_mb'] = max(peaks)
                if state.get('stop_after_pilot'):
                    state.update(status='pilot_complete', reason='Full sweep requires explicit release')
                else:
                    state['phase'] = 'full'
        else:
            state.update(status='complete', expected_scores=expected)
    save(path, state)
    return state


def shutil_disk_free(root):
    import shutil
    physical = shutil.disk_usage(root).free
    if not str(root).startswith('/work/FAC/'):
        return physical
    gpfs = Path('/usr/lpp/mmfs/bin')
    if all(os.access(gpfs / name, os.X_OK) for name in ('mmlsattr', 'mmlsquota')):
        attributes = execute([str(gpfs / 'mmlsattr'), '-L', str(root)], timeout=120)
        fileset = next((line.split(':', 1)[1].strip() for line in attributes.splitlines()
                        if line.startswith('fileset name:')), None)
        if not fileset:
            raise RuntimeError('Cannot identify project fileset; refusing unattended submission')
        quota = execute([str(gpfs / 'mmlsquota'), '-Y', '-j', fileset,
                         '--block-size', 'K', 'work'], timeout=120)
        header = None
        for line in quota.splitlines():
            fields = line.split(':')
            if fields[:3] == ['mmlsquota', 'fileset', 'HEADER']:
                header = fields
            elif header and fields[:3] == ['mmlsquota', 'fileset', '0']:
                data = dict(zip(header, fields))
                if data['filesystemName'] == 'work' and data['name'] == fileset:
                    available = (int(data['blockQuota']) - int(data['blockUsage'])
                                 - int(data['blockInDoubt'])) * 1024
                    if int(data['filesUsage']) + int(data['filesInDoubt']) >= int(data['filesQuota']):
                        return 0
                    return min(physical, max(0, available))
        raise RuntimeError('Cannot verify GPFS project quota; refusing unattended submission')
    quota = execute(['quotacheck', '--show-paths'], timeout=120)
    candidates = []
    for line in quota.splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0].startswith('/work/'):
            project = Path(fields[0])
            if project == root or project in root.parents:
                # quotacheck reports work quotas in T; use conservative decimal TB.
                candidates.append((len(project.parts), int(float(fields[3]) * 1000**4)))
    if not candidates:
        raise RuntimeError('Cannot verify project quota; refusing unattended submission')
    return min(physical, max(candidates)[1])


def tick(root):
    import fcntl
    with (root / '.supervision.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _tick(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('--remote', action='store_true', help='Invoke saved UNIL helper from workstation')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    if args.interval < 60:
        parser.error('Monitoring interval must be at least 60 seconds')
    while True:
        try:
            if args.remote:
                # Operational supervision stays outside the frozen inference source.
                script = args.deployment / 'supervisor.py'
                command = shlex.join(['python3', str(script), '--deployment', str(args.deployment), '--once'])
                # Helper consumes saved passwords privately and stops on auth/MFA rejection.
                output = subprocess.check_output(['/home/dmoi/.local/bin/unil-access', 'ssh', '--', command],
                                                 text=True, timeout=90)
                print(output, flush=True)
                state = json.loads(next(line for line in reversed(output.splitlines()) if line.startswith('{')))
            else:
                state = tick(args.deployment)
                print(json.dumps(state), flush=True)
            if args.once or state['status'] != 'active':
                return
        except Exception as error:
            # Do not reconnect repeatedly through credential, host-key, or scientific failures.
            if isinstance(error, subprocess.CalledProcessError) and error.output:
                print(error.output, flush=True)
            print(json.dumps({'status': 'needs_attention', 'reason': str(error)}), flush=True)
            raise
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
