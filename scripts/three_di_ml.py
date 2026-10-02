#!/usr/bin/env python3
"""Canonical native 3Di alignment and empirical-model ML inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_common import (AA_ORDER, ROOT, begin_stage, complete_stage, digest,
    infer_ml, read_fasta, read_paml, run, tool_version, write_fasta, write_json)


def canonical_ids(records, paths):
    result = {}
    for ident, seq in records.items():
        matches = [p.stem for p in paths if ident in {p.stem, p.name}]
        if not matches:
            for p in paths:
                chains = {line[21:22] for line in p.read_text().splitlines() if line.startswith('ATOM')}
                if len(chains) == 1 and ident in {p.stem + '_' + next(iter(chains)), p.name + '_' + next(iter(chains))}:
                    matches.append(p.stem)
        if len(matches) != 1 or matches[0] in result:
            raise ValueError(f'Ambiguous chain/identifier from structural aligner: {ident}')
        result[matches[0]] = seq.upper()
    if set(result) != {p.stem for p in paths}:
        raise ValueError('Structural aligner changed the requested taxon cohort')
    return result


def three_di(structures, outdir, route='foldmason', threads=8, seed=42, starts=10,
             matrix=None, score_matrix=None):
    paths = sorted(Path(p).resolve() for p in structures)
    matrix = Path(matrix or ROOT / 'foldtree2/config/3diphy_submats/Q.3Di.AF').resolve()
    outdir = Path(outdir).resolve()
    if len(paths) < 4 or len({p.stem for p in paths}) != len(paths):
        raise ValueError('Need at least four uniquely named structures')
    for path in paths:
        if len({line[21:22] for line in path.read_text().splitlines() if line.startswith('ATOM')}) != 1:
            raise ValueError(f'Expected an explicitly selected single protein chain: {path}')
    read_paml(matrix, 20)
    tools = {'raxml-ng': tool_version('raxml-ng'),
             route: tool_version('foldmason' if route == 'foldmason' else 'foldseek')}
    signature = {'protocol': 'native_3di_ml_v1', 'route': route, 'threads': threads,
        'seed': seed, 'starts_each': starts, 'matrix_sha256': digest(matrix),
        'structures': {str(p): digest(p) for p in paths}, 'tools': tools,
        'scoring_matrix': digest(score_matrix) if score_matrix else None}
    if not begin_stage(outdir, signature):
        return outdir
    log = outdir / 'alignment.log'
    if route == 'foldmason':
        run(['foldmason', 'easy-msa', *paths, outdir / 'alignment', outdir / 'tmp',
             '--threads', threads], log)
        aa = canonical_ids(read_fasta(outdir / 'alignment_aa.fa', aligned=True), paths)
        three = canonical_ids(read_fasta(outdir / 'alignment_3di.fa', aligned=True), paths)
    else:
        if not score_matrix:
            raise ValueError('Foldseek/MAFFT requires --score-matrix: alignment scores, NOT Q.3Di.AF')
        run(['foldseek', 'createdb', *paths, outdir / 'db'], log)
        run(['foldseek', 'convert2fasta', outdir / 'db', outdir / 'aa.raw.fa'], log)
        run(['foldseek', 'convert2fasta', outdir / 'db_ss', outdir / '3di.raw.fa'], log)
        aa_raw = canonical_ids(read_fasta(outdir / 'aa.raw.fa'), paths)
        three_raw = canonical_ids(read_fasta(outdir / '3di.raw.fa'), paths)
        write_fasta(outdir / '3di.raw.fa', three_raw)
        run(['mafft', '--amino', '--thread', threads, '--localpair', '--maxiterate', '1000',
             '--aamatrix', score_matrix, outdir / '3di.raw.fa'], log, outdir / '3di.aligned.raw.fa')
        three = read_fasta(outdir / '3di.aligned.raw.fa', aligned=True)
        aa = {}
        for ident, seq in three.items():
            if len(seq.replace('-', '')) != len(aa_raw[ident]):
                raise ValueError(f'AA/3Di residue count differs: {ident}')
            iterator = iter(aa_raw[ident])
            aa[ident] = ''.join('-' if c == '-' else next(iterator) for c in seq)
    if set(aa) != set(three):
        raise ValueError('AA/3Di taxa differ')
    residue_maps = {}
    for ident in aa:
        if len(aa[ident]) != len(three[ident]) or any((a == '-') != (b == '-') for a, b in zip(aa[ident], three[ident])):
            raise ValueError(f'AA/3Di column correspondence differs: {ident}')
        unknown = set(three[ident]) - set(AA_ORDER + '-X?')
        if unknown:
            raise ValueError(f'Unknown 3Di symbols: {unknown}')
        index = 0
        positions = []
        for char in aa[ident]:
            positions.append(None if char == '-' else index)
            index += char != '-'
        residue_maps[ident] = positions
    write_fasta(outdir / 'aa.aligned.fasta', aa)
    write_fasta(outdir / '3di.aligned.fasta', three)
    write_fasta(outdir / '3di.unaligned.fasta', {k: v.replace('-', '') for k, v in three.items()})
    write_json(outdir / 'residue_maps.json', residue_maps)
    model = f'PROTGTR{{{matrix}}}+G+I'
    tree, fitted = infer_ml(outdir / '3di.aligned.fasta', model, outdir / 'ml', threads, seed, starts)
    run(['raxml-ng', '--sitelh', '--redo', '--msa', outdir / '3di.aligned.fasta',
         '--tree', tree, '--model', fitted, '--opt-model', 'off', '--opt-branches', 'off',
         '--threads', threads, '--prefix', outdir / 'sites'], outdir / 'sites.command.log')
    complete_stage(outdir, signature, [outdir / p for p in ['aa.aligned.fasta', '3di.aligned.fasta',
        '3di.unaligned.fasta', 'residue_maps.json', 'ml.raxml.bestTree', 'ml.raxml.bestModel', 'sites.raxml.siteLH']],
        representation='3Di', state_order=AA_ORDER, model=model, n_taxa=len(aa))
    return outdir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structures', nargs='+', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--route', choices=['foldmason', 'foldseek'], default='foldmason')
    parser.add_argument('--matrix', type=Path)
    parser.add_argument('--score-matrix', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--starts', type=int, default=10)
    args = parser.parse_args()
    if args.threads < 1 or args.starts < 1:
        parser.error('Threads and starts must be positive')
    print(json.dumps({'outdir': str(three_di(**vars(args)))}))


if __name__ == '__main__':
    main()
