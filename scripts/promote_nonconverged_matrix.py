#!/usr/bin/env python3
"""Explicitly accept validated, nonconverged matrices and resume benchmarks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from benchmark_common import ROOT, digest, write_json
from prepare_production_alphabets import validate_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--accept-nonconverged', action='store_true', required=True)
    parser.add_argument('--reason', required=True)
    parser.add_argument('--benchmark-outdir', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if not args.reason.strip() or not 1 <= args.threads <= 8:
        parser.error('A reason and 1–8 threads are required')
    entries = json.loads((ROOT / 'configs/production_models.yaml').read_text())['models']
    entry = next(e for e in entries if e['size'] == args.size)
    staging = args.staging.resolve()
    staged = validate_bundle({**entry, 'directory': str(staging)})
    production = ROOT / entry['directory']
    for key in ['encoder', 'decoder']:
        if digest(production / entry[key]) != staged['sha256'][key]:
            raise ValueError('Staged weights do not match production; refuse promotion')
    stem = Path(entry['encoder']).stem
    metrics_name = stem + '_metrics.json'
    metrics = json.loads((staging / metrics_name).read_text())
    if (metrics.get('matrix_size') != args.size
            or metrics.get('counting_method') != 'headerless_all_pair_occurrences_v2'
            or metrics.get('convergence', {}).get('is_converged') is not False
            or metrics.get('nonzero_pairs') != args.size ** 2):
        raise ValueError('Only corrected, fully observed nonconverged matrices can be accepted')
    acceptance = {
        'status': 'accepted_nonconverged', 'authorization': 'explicit_user_approval',
        'approved_at_utc': datetime.now(timezone.utc).isoformat(), 'reason': args.reason,
        'size': args.size, 'source_staging': str(staging),
        'artifact_sha256': staged['sha256'], 'metrics_sha256': digest(staging / metrics_name),
        'convergence': metrics['convergence'],
        'limitation': 'Did not meet the sustained EMA convergence criterion; benchmark results must disclose this.'}
    names = [entry['mafft'], entry['raxml'], metrics_name, stem + '_convergence_acceptance.json']
    existing = [production / name for name in names if (production / name).exists()]
    if existing:
        archive = production / 'previous_matrices' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive.mkdir(parents=True)
        for path in existing:
            shutil.copy2(path, archive / path.name)
    for name in [entry['mafft'], entry['raxml'], metrics_name]:
        shutil.copy2(staging / name, production / name)
    write_json(production / (stem + '_convergence_acceptance.json'), acceptance)
    report = validate_bundle(entry, require_convergence=True)
    print(json.dumps(report, indent=2), flush=True)
    if args.benchmark_outdir:
        out = args.benchmark_outdir.resolve()
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'matrix_acceptance.json', report)
        subprocess.run([sys.executable, '-u', ROOT / 'scripts/run_local_benchmark_queue.py',
            '--benchmark-only', '--sizes', str(args.size), '--threads', str(args.threads),
            '--alignment-root', ROOT / 'runs/local_benchmarks/notebook_frobenius_20260930/experiments',
            '--reuse-native-root', ROOT / 'runs/local_benchmarks/missing_alphabet_analysis_20261001',
            '--outdir', out], cwd=ROOT, check=True)
        queue = json.loads((out / 'queue_status.json').read_text())
        if queue['status'] != 'completed':
            raise RuntimeError('Family benchmark queue has failures; inspect queue_status.json before aggregation')
        for mode in ['native', 'controlled']:
            subprocess.run([sys.executable, '-u', ROOT / 'scripts/benchmark_species_supermatrix.py',
                '--sizes', '10', '20', '30', '40', '50', '--column-mode', mode,
                '--experiment-root', ROOT / 'runs/local_benchmarks/notebook_frobenius_20260930/experiments',
                '--native-root', out, '--outdir', out / ('species_supermatrix_' + mode),
                '--threads', str(args.threads)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
