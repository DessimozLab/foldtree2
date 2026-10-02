#!/usr/bin/env python3
"""Joint AA/3Di/FT2 information and gain benchmarks on the frozen OMA cohort.

Controlled gain uses FoldMason's residue-corresponding columns on the OMA AA
reference topology. Native gain and ancestral reconstruction retain each
strategy's own alignment and fitted tree; their columns are never index-paired.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from alphabet_information_benchmark import analyze, read_common_families
from benchmark_cohort import DEFAULT_COHORT, load, native_directory, resolve_path, verify_stage
from benchmark_common import (AA_ORDER, ROOT, begin_stage, complete_stage, digest, project_tokens,
                              read_fasta, run, tool_version, write_fasta, write_json)
from phylogenetic_information_gain import (DatasetSpec, analyze_alphabet_phylogenetic_info,
    compute_summary_stats, parse_raxml_sitelh, run_raxml_sitelh)
from prepare_production_alphabets import validate_bundle


def controlled_alignments(family, frame, native_root):
    """All representations share columns only after verifying residue identity."""
    ident = family['family']
    three = native_directory(native_root, '3Di', ident)
    aa = read_fasta(three / 'aa.aligned.fasta', aligned=True)
    tokens = read_fasta(three / '3di.aligned.fasta', aligned=True)
    labels = {t['accession']: t['label'] for t in family['taxa']}
    if set(aa) != set(labels) or set(tokens) != set(labels):
        raise ValueError(f'FoldMason/common-cohort taxa mismatch: {ident}')
    reference = read_fasta(resolve_path(family['aa_alignment']), aligned=True)
    for accession, label in labels.items():
        if aa[accession].replace('-', '') != reference[label].replace('-', ''):
            raise ValueError(f'FoldMason/OMA residue correspondence mismatch: {ident}/{accession}')
        if len(aa[accession]) != len(tokens[accession]) or any((x == '-') != (y == '-') for x, y in zip(aa[accession], tokens[accession])):
            raise ValueError('FoldMason AA/3Di columns do not correspond')
    result = {'AA': {labels[k]: v for k, v in aa.items()}, '3Di': {labels[k]: v for k, v in tokens.items()}}
    for name, rows in frame[(frame.family == ident) & frame.models.str.startswith('FT2_')].groupby('models'):
        sequences = {row.id: row.seq for row in rows.itertuples()}
        projected = project_tokens(aa, sequences)
        result[name] = {labels[k]: v for k, v in projected.items()}
    return result


def shared_occupancy(records, supports, max_missing=.3):
    masks = []
    for name, alignment in records.items():
        msa = np.array([list(seq) for _, seq in sorted(alignment.items())])
        n_valid = np.isin(msa, supports[name]).sum(axis=0)
        masks.append((n_valid >= 4) & ((1 - n_valid / len(msa)) <= max_missing))
    return np.logical_and.reduce(masks)


def gain_family(family, frame, alignments, supports, native_root, entries, outdir, threads, native=False):
    ident = family['family']
    projected = controlled_alignments(family, frame, native_root) if not native else None
    mask = shared_occupancy(projected, supports) if projected is not None else None
    protocol = 'native_tree_native_alignment' if native else 'FoldMason_common_columns_OMA_AA_topology'
    rows = []
    for name in sorted(supports):
        destination = outdir / protocol / ident / name
        destination.mkdir(parents=True, exist_ok=True)
        if native:
            directory = native_directory(native_root, name, ident)
            native_signature = verify_stage(directory)
            alignment = directory / '3di.aligned.fasta' if name == '3Di' else resolve_path(family['aa_alignment']) if name == 'AA' else alignments[(ident, name)]
            tree, model = directory / 'ml.raxml.bestTree', directory / 'ml.raxml.bestModel'
            records = read_fasta(alignment, aligned=True)
            if name != '3Di' and native_signature['alignment_sha256'] != digest(alignment):
                raise ValueError(f'Native fitted tree/model does not match alignment: {ident}/{name}')
        else:
            alignment = destination / 'aligned.fasta'
            records = projected[name]
            if alignment.exists():
                if read_fasta(alignment, aligned=True) != records:
                    raise ValueError(f'Controlled alignment changed; use a new output directory: {alignment}')
            else:
                write_fasta(alignment, records)
            tree = resolve_path(family['reference_tree'])
            if name == 'AA':
                model = 'LG+G+I'
            elif name == '3Di':
                model = f'PROTGTR{{{ROOT / "foldtree2/config/3diphy_submats/Q.3Di.AF"}}}+G+I'
            else:
                entry = entries[int(name.removeprefix('FT2_'))]
                model = f'MULTI{entry["size"]}_GTR{{{ROOT / entry["directory"] / entry["raxml"]}}}+I'
        model_hash = digest(model) if native else digest(ROOT / 'foldtree2/config/3diphy_submats/Q.3Di.AF') if name == '3Di' else None
        if not native and name.startswith('FT2_'):
            entry = entries[int(name.removeprefix('FT2_'))]
            model_hash = digest(ROOT / entry['directory'] / entry['raxml'])
        signature = {'protocol': protocol, 'family': ident, 'representation': name,
            'alignment_sha256': digest(alignment), 'tree_sha256': digest(tree), 'model': str(model),
            'model_sha256': model_hash, 'states': supports[name], 'threads': threads,
            'raxml': tool_version('raxml-ng'), 'common_occupancy_mask': mask.astype(int).tolist() if mask is not None else None}
        summary_file = destination / 'summary.json'
        if begin_stage(destination, signature):
            if native:
                run(['raxml-ng', '--sitelh', '--redo', '--msa', alignment, '--tree', tree,
                     '--model', model, '--opt-model', 'off', '--opt-branches', 'off',
                     '--threads', threads, '--prefix', destination / 'sites'], destination / 'sites.command.log')
                sitelh = destination / 'sites.raxml.siteLH'
            else:
                sitelh = run_raxml_sitelh(alignment, tree, model, destination / 'sites', 'raxml-ng', threads)
            spec = DatasetSpec(name, name, alignment, tree, str(model), sitelh, None)
            table = analyze_alphabet_phylogenetic_info(spec, parse_raxml_sitelh(sitelh), set(supports[name]), '-', .3)
            if mask is not None:
                table = table[table.column_index.map(lambda i: bool(mask[i]))]
            else:
                table = table[(table.n_valid_taxa >= 4) & (table.n_valid_taxa / len(records) >= .7)]
            if table.empty:
                raise ValueError(f'No eligible gain columns: {ident}/{name}')
            output = destination / 'site_gain.csv'
            table.to_csv(output, index=False)
            summary = {'family': ident, 'model': name, 'protocol': protocol, **compute_summary_stats(table, name)}
            write_json(summary_file, summary)
            complete_stage(destination, signature, [output, summary_file, sitelh])
        rows.append(json.loads(summary_file.read_text()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=DEFAULT_COHORT)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[20, 30, 40])
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--stages', nargs='+', choices=['information', 'controlled-gain', 'native-gain'],
                        default=['information', 'controlled-gain', 'native-gain'])
    parser.add_argument('--orders', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-families', type=int)
    parser.add_argument('--wait-for-service')
    args = parser.parse_args()
    if not 1 <= args.threads <= 8 or min(args.orders) < 0 or args.folds < 2 or (args.max_families is not None and args.max_families < 1):
        parser.error('Invalid thread, order, fold, or family limit')
    if len(set(args.sizes)) != len(args.sizes) or set(args.sizes) - {10, 20, 30, 40, 50}:
        parser.error('Select unique production alphabet sizes from 10, 20, 30, 40, 50')
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.wait_for_service:
        while True:
            state = subprocess.run(['systemctl', '--user', 'show', args.wait_for_service, '-p', 'ActiveState', '--value'],
                                   capture_output=True, text=True, check=True).stdout.strip()
            if state not in {'active', 'activating', 'deactivating', 'reloading'}:
                break
            write_json(args.outdir / 'status.json', {'status': 'waiting', 'service': args.wait_for_service})
            time.sleep(30)
    write_json(args.outdir / 'status.json', {'status': 'validating_inputs', 'required_baselines': ['AA', '3Di']})
    cohort = load(args.cohort)
    entries = {e['size']: e for e in json.loads((ROOT / 'configs/production_models.yaml').read_text())['models']}
    for size in args.sizes:
        validate_bundle(entries[size], require_convergence=True)
    directories = [args.experiment_root / str(size) for size in args.sizes]
    frame, alignments, supports = read_common_families(directories, args.native_root, cohort, args.max_families)
    signature = {'protocol': 'joint_OMA_AA_3Di_FT2_v1', 'cohort_sha256': digest(args.cohort),
        'models': sorted(supports), 'sizes': args.sizes, 'max_families': args.max_families, 'stages': args.stages,
        'orders': args.orders, 'folds': args.folds, 'seed': args.seed, 'threads': args.threads,
        'experiment_provenance': {str(p.resolve()): digest(p / 'provenance.json') for p in directories}}
    provenance = args.outdir / 'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text()) != signature:
        raise ValueError('Comparison protocol changed; use a new output directory')
    write_json(provenance, signature)
    write_json(args.outdir / 'status.json', {'status': 'running', **signature})
    if 'information' in args.stages:
        info = args.outdir / 'information'
        info.mkdir(exist_ok=True)
        frame.to_csv(info / 'sequences.csv', index=False)
        analyze(frame, supports.copy(), alignments, info, args.orders, args.folds, args.seed,
                {'entropy', 'mdl', 'mi', 'position', 'kmer'}, True)
        write_json(info / 'completed.json', signature)
    failures = []
    families = sorted(cohort['families'], key=lambda f: f['family'])
    if args.max_families:
        families = families[:args.max_families]
    for stage in [s for s in args.stages if s != 'information']:
        aggregate = []
        for family in families:
            try:
                aggregate.extend(gain_family(family, frame, alignments, supports, args.native_root,
                    entries, args.outdir, args.threads, native=stage == 'native-gain'))
            except Exception as error:
                failures.append({'stage': stage, 'family': family['family'], 'error': repr(error)})
            if aggregate:
                pd.DataFrame(aggregate).to_csv(args.outdir / (stage + '.csv'), index=False)
            write_json(args.outdir / 'status.json', {'status': 'running', 'stage': stage,
                        'n_family_representation_results': len(aggregate), 'failures': failures})
    report = {'status': 'completed' if not failures else 'finished_with_failures', **signature,
              'n_families': len(families), 'failures': failures}
    write_json(args.outdir / 'status.json', report)
    if failures:
        raise RuntimeError(f'{len(failures)} family/stage failures; see status.json')
    write_json(args.outdir / 'completed.json', report)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--outdir' in sys.argv:
            directory = Path(sys.argv[sys.argv.index('--outdir') + 1])
            status = directory / 'status.json'
            record = json.loads(status.read_text()) if status.exists() else {}
            record.update(status='failed', error=repr(error))
            write_json(status, record)
        raise
