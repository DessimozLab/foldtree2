#!/usr/bin/env python3
"""Persistent sequential workstation queue; failures do not stall other models.

Run in the activated foldtree2 environment. All encoding here is CPU-only,
leaving GPU 0 training untouched. Status and per-job logs survive the caller.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_common import (AA_ORDER, MULTI_ORDER, ROOT, infer_ml, read_fasta, write_json,
                              begin_stage, complete_stage, digest, tool_version)
from prepare_production_alphabets import validate_bundle
from ancestral_state_uncertainty import reconstruct
from three_di_ml import three_di
from benchmark_cohort import native_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--training-pid', type=int)
    parser.add_argument('--training-dir', type=Path)
    parser.add_argument('--sizes', type=int, nargs='+', default=[20, 30, 40, 10, 50])
    parser.add_argument('--benchmark-only', action='store_true', help='Use already validated production bundles; no training or matrix rebuilds')
    parser.add_argument('--trees-and-ancestral-only', action='store_true',
                        help='Prepare native FT2 alignments and run full tree/ASR jobs; skip wider information analyses')
    parser.add_argument('--alignment-root', type=Path, help='Existing per-size family experiment root to reuse validated alignments')
    parser.add_argument('--reuse-native-root', type=Path, help='Prior successful native/3di output root; reuse only hash-verified stages')
    parser.add_argument('--max-families', type=int, help='Bound the benchmark scope; omit for a full sweep')
    parser.add_argument('--wait-for-service', help='Wait for an existing user service to finish before starting CPU work')
    parser.add_argument('--families', type=Path, default=ROOT / 'families/Information_benchmark/marker_genes')
    parser.add_argument('--cohort', type=Path, help='Frozen OMA cohort; automatic for the standard notebook input root')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error('Workstation queue permits 1–8 CPU threads')
    if set(args.sizes) - {10, 20, 30, 40, 50}:
        parser.error('Sizes must be 10, 20, 30, 40 or 50')
    if args.max_families is not None and args.max_families < 1:
        parser.error('--max-families must be positive')
    if not args.benchmark_only and (args.training_pid is None or args.training_dir is None):
        parser.error('The full preparation queue requires --training-pid and --training-dir')
    if args.trees_and_ancestral_only and not args.benchmark_only:
        parser.error('--trees-and-ancestral-only requires --benchmark-only')
    os.chdir(ROOT)
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    statefile = out / 'queue_status.json'
    state = json.loads(statefile.read_text()) if statefile.exists() else {'jobs': {}, 'scope': 'implemented_core_benchmarks'}
    protocol = {'sizes': args.sizes, 'benchmark_only': args.benchmark_only,
                'max_families': args.max_families, 'families': str(args.families.resolve()), 'threads': args.threads}
    if args.trees_and_ancestral_only or args.alignment_root or args.reuse_native_root:
        protocol.update(trees_and_ancestral_only=args.trees_and_ancestral_only,
                        alignment_root=str(args.alignment_root.resolve()) if args.alignment_root else None,
                        reuse_native_root=str(args.reuse_native_root.resolve()) if args.reuse_native_root else None)
    if 'protocol' in state and state['protocol'] != protocol:
        raise ValueError('Queue scope changed; use a new output directory')
    state['protocol'] = protocol
    if args.trees_and_ancestral_only:
        state['scope'] = 'native_tree_ancestral_sweep'
    if args.wait_for_service:
        while True:
            result = subprocess.run(['systemctl', '--user', 'show', args.wait_for_service,
                                     '--property=ActiveState', '--value'], check=True, capture_output=True, text=True)
            if result.stdout.strip() not in {'active', 'activating', 'deactivating', 'reloading'}:
                break
            state['waiting_for'] = args.wait_for_service
            write_json(statefile, state)
            time.sleep(30)
        state.pop('waiting_for', None)
        write_json(statefile, state)
    entries = {e['size']: e for e in json.loads((ROOT / 'configs/production_models.yaml').read_text())['models']}
    from benchmark_cohort import select_families
    families, cohort_path = select_families(args.families, args.cohort, args.max_families)
    if cohort_path:
        write_json(out / 'cohort.json', {'manifest': str(cohort_path), 'sha256': digest(cohort_path),
                                       'family_ids': [family.name for family in families]})
    if not families:
        raise ValueError('No benchmark families')
    state['n_families'] = len(families)
    write_json(statefile, state)
    env = {**os.environ, 'OMP_NUM_THREADS': str(args.threads), 'MKL_NUM_THREADS': str(args.threads),
           'OPENBLAS_NUM_THREADS': str(args.threads), 'MPLCONFIGDIR': '/tmp/foldtree2-matplotlib'}
    os.environ.update(env)

    def job(name, action):
        if state['jobs'].get(name, {}).get('status') == 'completed':
            return True
        state['jobs'][name] = {'status': 'running', 'started': time.time()}
        write_json(statefile, state)
        print(f'RUN {name}', flush=True)
        try:
            artifacts = action()
        except Exception as error:
            state['jobs'][name].update(status='failed', error=repr(error), finished=time.time())
            print(f'FAILED {name}: {error}', flush=True)
            write_json(statefile, state)
            return False
        state['jobs'][name].update(status='completed', finished=time.time())
        if isinstance(artifacts, (str, dict)):
            state['jobs'][name]['artifacts'] = artifacts
        write_json(statefile, state)
        print(f'DONE {name}', flush=True)
        return True

    def command(name, arguments):
        with (out / (name + '.log')).open('a') as stream:
            subprocess.run([sys.executable, '-u', *map(str, arguments)], env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    def matrices(size):
        stage = ROOT / 'runs/production_alphabets' if size in (20, 30, 40) else args.training_dir.resolve()
        command(f'matrices_{size}', ['scripts/prepare_production_alphabets.py', '--sizes', size,
            '--stages', 'matrices', 'validate', '--rebuild-matrices', '--reuse-staged-encoding',
            '--device', 'cpu', '--threads', args.threads, '--outdir', stage,
            '--matrix-convergence-patience', 1])
        validate_bundle(entries[size], require_convergence=True)

    def native_ft2(size, family, base):
        validate_bundle(entries[size], require_convergence=True)
        aln = base / family.name / 'ft2.aligned.fasta'
        scope = 'pilot_native' if base.parent.name == 'pilot' else 'native'
        destination = out / scope / f'FT2_{size}' / family.name
        if args.reuse_native_root and scope == 'native':
            candidate = native_directory(args.reuse_native_root, f'FT2_{size}', family.name)
            prior = candidate / 'ancestral' / 'provenance.json'
            if (candidate / 'completed.json').exists() and prior.exists():
                signature = json.loads(prior.read_text())
                old_alignments = [Path(p) for p in signature['inputs'] if Path(p).name == 'ft2.aligned.fasta']
                if len(old_alignments) == 1 and old_alignments[0].is_file() and digest(old_alignments[0]) == digest(aln):
                    # Keep original input paths: ASR provenance includes absolute paths.
                    aln = old_alignments[0]
                    destination = candidate
        destination.mkdir(parents=True, exist_ok=True)
        matrix = ROOT / entries[size]['directory'] / entries[size]['raxml']
        tree, fitted = cached_ml(aln, f'MULTI{size}_GTR{{{matrix}}}+I', destination, digest(matrix))
        reconstruct(aln, tree, fitted, destination / 'ml.raxml.log', MULTI_ORDER[:size],
                    destination / 'ancestral', family.name, f'FT2_{size}', args.threads)
        return str(destination)

    def cached_ml(alignment, model, destination, matrix_hash=None):
        signature = {'alignment_sha256': digest(alignment), 'model': model,
                     'matrix_sha256': matrix_hash, 'threads': args.threads,
                     'seed': 42, 'starts_each': 10, 'raxml': tool_version('raxml-ng')}
        tree = destination / 'ml.raxml.bestTree'
        fitted = destination / 'ml.raxml.bestModel'
        if begin_stage(destination, signature):
            tree, fitted = infer_ml(alignment, model, destination / 'ml', args.threads)
            complete_stage(destination, signature, [tree, fitted, destination / 'ml.raxml.log'])
        return tree, fitted

    # First rebuild each retained model independently, then pilot its encoding,
    # information/gain and native ML+ASR before allowing its full sweep.
    ready = []
    for size in [s for s in args.sizes if args.benchmark_only or s in (20, 30, 40)]:
        action = (lambda size=size: validate_bundle(entries[size], require_convergence=True)) if args.benchmark_only else (lambda size=size: matrices(size))
        name = f'validate_{size}' if args.benchmark_only else f'matrices_{size}'
        if job(name, action):
            ready.append(size)

    def ft2_benchmarks(size):
        pilot = out / 'pilot' / str(size)
        full = (args.alignment_root.resolve() if args.alignment_root else out / 'experiments') / str(size)
        common = ['scripts/production_alphabet_experiments.py', '--size', size,
                  '--families', args.families, '--device', 'cpu', '--threads', args.threads]
        if cohort_path:
            common.extend(['--cohort', cohort_path])
        if not job(f'pilot_{size}', lambda: command(f'pilot_{size}', [*common, '--outdir', pilot, '--max-families', 1])):
            return
        if not job(f'pilot_native_{size}', lambda: native_ft2(size, families[0], pilot)):
            return
        limit = ['--max-families', args.max_families] if args.max_families is not None else []
        if job(f'experiments_{size}', lambda: command(f'experiments_{size}', [*common, '--outdir', full, *limit])):
            for family in families:
                job(f'native_{size}_{family.name}', lambda family=family: native_ft2(size, family, full))

    for size in ready:
        ft2_benchmarks(size)

    # Each baseline uses its own alignment and inferred ML tree, not an FT2 tree.
    for family in families:
        def baseline_three(family=family):
            labels = read_fasta(family / 'sequences.aligned.fa', aligned=True)
            cohort = {name.split('|')[0] for name in labels}
            paths = [family / 'structs' / (name + '.pdb') for name in sorted(cohort)]
            three = out / '3di' / family.name
            if args.reuse_native_root:
                candidate = native_directory(args.reuse_native_root, '3Di', family.name)
                if (candidate / 'completed.json').exists():
                    three = candidate
            three_di(paths, three, threads=args.threads)
            reconstruct(three / '3di.aligned.fasta', three / 'ml.raxml.bestTree',
                three / 'ml.raxml.bestModel', three / 'ml.raxml.log', AA_ORDER,
                three / 'ancestral', family.name, '3Di_foldmason', args.threads)
            return str(three)

        def baseline_aa(family=family):
            aa = out / 'native' / 'AA' / family.name
            if args.reuse_native_root:
                candidate = native_directory(args.reuse_native_root, 'AA', family.name)
                if (candidate / 'completed.json').exists():
                    aa = candidate
            aa.mkdir(parents=True, exist_ok=True)
            tree, fitted = cached_ml(family / 'sequences.aligned.fa', 'LG+G+I', aa)
            reconstruct(family / 'sequences.aligned.fa', tree, fitted, aa / 'ml.raxml.log', AA_ORDER,
                        aa / 'ancestral', family.name, 'AA', args.threads)
            return str(aa)
        job(f'3Di_{family.name}', baseline_three)
        job(f'AA_{family.name}', baseline_aa)

    # Missing models are produced by the unchanged GPU training supervisor.
    for size in [s for s in args.sizes if not args.benchmark_only and s in (10, 50)]:
        entry = entries[size]
        directory = ROOT / entry['directory']
        while not (all((directory / entry[key]).is_file() for key in ('encoder', 'decoder'))
                   and (directory / 'best_pair_metrics.json').is_file()):
            # Check selected-pair validation explicitly before consuming checkpoints.
            try:
                os.kill(args.training_pid, 0)
            except ProcessLookupError:
                break
            state['waiting_for'] = f'training_{size}'
            write_json(statefile, state)
            time.sleep(30)
        state.pop('waiting_for', None)
        if job(f'matrices_{size}', lambda size=size: matrices(size)):
            ready.append(size)
            ft2_benchmarks(size)
    if not args.trees_and_ancestral_only and ready:
        experiment_root = args.alignment_root.resolve() if args.alignment_root else out / 'experiments'
        limit = ['--max-families', args.max_families] if args.max_families else []
        job('joint_OMA_comparison', lambda: command('joint_OMA_comparison', [
            'scripts/compare_oma_representations.py', '--cohort', cohort_path,
            '--experiment-root', experiment_root, '--native-root', out, '--sizes', *ready,
            '--outdir', out / 'joint_comparison', '--threads', args.threads, *limit]))
    state['finished'] = time.time()
    state['status'] = 'completed' if all(j['status'] == 'completed' for j in state['jobs'].values()) else 'finished_with_failures'
    state['remaining_scope'] = ['extended notebook analyses', 'paired family aggregation and plotting']
    if args.trees_and_ancestral_only:
        state['remaining_scope'].append('joint_OMA_comparison must run after the native sweep')
    write_json(statefile, state)


if __name__ == '__main__':
    main()
