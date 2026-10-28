#!/usr/bin/env python3
"""Run repeated concurrent production RAxML searches inside a Slurm allocation."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--step-cpus', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--max-taxa', type=int, default=8,
                        help='Bound tool verification cost using deposited alignment rows; no benchmark scores')
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID') or args.repeats < 1 or args.max_taxa < 4
            or args.threads < 1 or args.concurrency < 1 or args.step_cpus < args.threads):
        parser.error('Requires a Slurm allocation and positive repeats')
    root = args.deployment.resolve()
    output = root / 'affinity-checks' / os.environ['SLURM_JOB_ID']
    output.mkdir(parents=True, exist_ok=False)
    worker = Path(__file__).with_name('curnagl_physical_step.py')
    binary = root / 'foldtree2/foldtree2/raxml-ng/raxml-ng'
    pilot = json.loads((root / 'pilot.json').read_text())
    cases = []
    for family in pilot['families']:
        for method, spec in pilot['methods'].items():
            kind = spec['kind']
            if kind not in {'AA', '3Di', 'FT2'}:
                continue
            directory = Path(pilot['config']['output_dir']) / 'families' / family / method
            if not (directory / 'completed.json').is_file():
                raise ValueError(f'Pilot method is not completed: {directory}')
            if kind == 'FT2':
                entry = spec['entry']
                matrix = root / 'foldtree2' / entry['directory'] / entry['raxml']
                model = f'MULTI{entry["size"]}_GTR{{{matrix}}}+I'
            else:
                matrix = root / 'foldtree2/foldtree2/config/3diphy_submats/Q.3Di.AF'
                model = 'LG+G+I' if kind == 'AA' else f'PROTGTR{{{matrix}}}+G+I'
            original = directory / 'alignment.fasta'
            sampled = output / (family + '-' + method + '.fasta')
            records = []
            for line in original.read_text().splitlines(keepends=True):
                if line.startswith('>'):
                    records.append([])
                if not records:
                    raise ValueError(f'Invalid FASTA: {original}')
                records[-1].append(line)
            sampled.write_text(''.join(''.join(row) for row in records[:args.max_taxa]))
            cases.append((family + '-' + method, sampled, model))
            (sampled.with_suffix('.source.json')).write_text(json.dumps({
                'source': str(original), 'source_sha256': sha(original),
                'selected_taxa': min(len(records), args.max_taxa),
                'purpose': 'bounded launch verification; no benchmark scoring'}) + '\n')
    failing = root / 'results/full/families/CATH__3.20.20.330/AA/alignment.fasta'
    cases.insert(0, ('previously-failing-AA', failing, 'LG+G+I'))
    checks = []

    def search(task):
        repeat, (name, alignment, model) = task
        prefix = output / f'{repeat}-{name}'
        command = ['srun', '--exclusive', '--exact', '--ntasks=1',
                   f'--cpus-per-task={args.step_cpus}', '--hint=nomultithread',
                   '--threads-per-core=1', '--cpu-bind=cores', '--mem=4000M',
                   sys.executable, str(worker), '--threads', str(args.threads), '--', str(binary),
                   '--search', '--msa', str(alignment), '--model', model,
                   '--tree', 'rand{10},pars{10}', '--seed', '42', '--threads', str(args.threads),
                   '--workers', '1', '--prefix', str(prefix)]
        started = time.monotonic()
        log = Path(str(prefix) + '.console.log')
        with log.open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=stream)
        text = log.read_text()
        record = {'name': name, 'repeat': repeat, 'command': command,
                  'alignment_sha256': sha(alignment), 'returncode': result.returncode,
                  'elapsed_seconds': time.monotonic() - started, 'log': str(log),
                  'passed': result.returncode == 0 and 'oversubscription detected' not in text
                            and Path(str(prefix) + '.raxml.bestTree').is_file()
                            and Path(str(prefix) + '.raxml.bestModel').is_file()}
        records = [line for line in text.splitlines() if line.startswith('{')]
        record['affinity'] = json.loads(records[0]) if records else None
        record['passed'] = record['passed'] and record['affinity'] is not None
        return record

    tasks = [(repeat, case) for repeat in range(args.repeats) for case in cases]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for record in pool.map(search, tasks):
            checks.append(record)
            print(json.dumps(record), flush=True)
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'hostname': os.uname().nodename,
              'binary_sha256': sha(binary), 'worker_sha256': sha(worker),
              'probe_sha256': sha(__file__), 'step_cpus': args.step_cpus,
              'concurrency': args.concurrency, 'threads': args.threads, 'checks': checks,
              'passed': len(checks) == len(tasks) and all(c['passed'] for c in checks)}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise RuntimeError(f'RAxML affinity check failed; inspect {output}')


if __name__ == '__main__':
    main()
