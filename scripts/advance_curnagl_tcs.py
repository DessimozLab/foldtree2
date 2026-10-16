#!/usr/bin/env python3
"""Advance one gated TCS deployment using short, dependency-driven Slurm callbacks."""
import argparse
import fcntl
import json
from pathlib import Path

from supervise_curnagl_tcs import ACTIVE, execute, save, tick


def advance(root):
    with (root / '.advance.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for _ in range(6):
            before = json.loads((root / 'supervision.json').read_text())
            state = tick(root)
            if state['status'] != 'active':
                return state
            phase = state['phase']
            job = state['jobs'].get(phase)
            if phase != before['phase'] or job != before['jobs'].get(phase):
                continue
            if job is None:
                raise ValueError('Active phase has no Slurm job')
            break
        else:
            raise ValueError('Too many immediate supervision transitions')

        path = root / 'callback-state.json'
        record = json.loads(path.read_text()) if path.exists() else {'history': []}
        waiting = state.get('slurm_state') not in ACTIVE
        key = f'{phase}:{job}'
        waits = record.get('accounting_waits', 0) + 1 if waiting and record.get('key') == key else int(waiting)
        if waits > 10:
            reason = 'Accounting/memory gate unresolved after ten callbacks; inspect Slurm'
            state.update(status='needs_attention', reason=reason)
            save(root / 'supervision.json', state)
            raise RuntimeError(reason)
        command = ['sbatch', '--parsable', '--begin=now+2minutes']
        if not waiting:
            command.append(f'--dependency=afterany:{job}')
        command.append(str(root / 'advance.sbatch'))
        callback = execute(command).split(';')[0]
        if not callback.isdigit():
            raise ValueError('Unexpected Slurm callback identifier')
        record.update(key=key, accounting_waits=waits, callback_job=callback)
        record['history'].append({'phase': phase, 'job': job, 'callback_job': callback})
        save(path, record)
        return {**state, 'callback_job': callback}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(advance(args.deployment.resolve())), flush=True)
