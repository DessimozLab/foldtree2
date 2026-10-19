#!/usr/bin/env python3
"""Bind a Slurm step to one hardware thread per physical core before exec."""
import argparse
import json
import os
from pathlib import Path


def physical_cpu_groups(cpus, sysfs=Path('/sys/devices/system/cpu')):
    groups = {}
    for cpu in sorted(cpus):
        topology = sysfs / f'cpu{cpu}' / 'topology'
        key = (int((topology / 'physical_package_id').read_text()),
               int((topology / 'core_id').read_text()))
        groups.setdefault(key, []).append(cpu)
    return groups


def bind(threads):
    if threads < 1:
        raise ValueError('Thread count must be positive')
    before = sorted(os.sched_getaffinity(0))
    groups = physical_cpu_groups(before)
    record = {'slurm_job': os.environ.get('SLURM_JOB_ID'),
              'slurm_step': os.environ.get('SLURM_STEP_ID'),
              'hostname': os.uname().nodename, 'affinity_before': before,
              'physical_cores_available': len(groups), 'threads': threads}
    if len(groups) < threads:
        print(json.dumps(record), flush=True)
        raise RuntimeError(f'Step exposes {len(groups)} physical cores for {threads} threads')
    selected = [members[0] for members in groups.values()][:threads]
    os.sched_setaffinity(0, selected)
    record['affinity_pinned'] = sorted(os.sched_getaffinity(0))
    if record['affinity_pinned'] != sorted(selected):
        raise RuntimeError('CPU affinity did not match the requested physical cores')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = str(threads)
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', required=True, type=int)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    bind(args.threads)
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if command:
        os.execvp(command[0], command)


if __name__ == '__main__':
    main()
