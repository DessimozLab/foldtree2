#!/usr/bin/env python3
"""Standalone information-content experiments from the alphabet benchmark notebook.

Reads completed family outputs from production_alphabet_experiments.py; no
model inference or notebook execution is needed. Includes AA baselines,
permutation controls, k-mer family discrimination, backoff Markov entropy rates,
cross-family MDL proxies, Henikoff-weighted alignment entropy, and AA/token MI.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

import alphabet_information_metrics as metrics
from phylogenetic_information_gain import read_alignment_file

AA = list('ACDEFGHIKLMNPQRSTVWY')
RAXML_SYMBOLS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,/:;<=>@[\\]^_{|}~'


def read_families(directories, family_ids=None):
    rows, alignments, supports = [], {}, {'AA': AA}
    seen_aa = {}
    for directory in directories:
        provenance = json.loads((directory / 'provenance.json').read_text())
        size = provenance['size']
        representation = f'FT2_{size}'
        if representation in supports:
            raise ValueError(f'Duplicate experiment input for {representation}')
        supports[representation] = list(RAXML_SYMBOLS[:size])
        replacements = {'"': chr(248), '#': chr(247), '>': chr(249), '=': chr(250), '<': chr(251),
                        '-': chr(252), ' ': chr(253), '\r': chr(254), '\n': chr(255)}
        alphabet = sorted(replacements.get(chr(i + 1), chr(i + 1)) for i in range(size))
        mapping = {ord(char): RAXML_SYMBOLS[i] for i, char in enumerate(alphabet)}
        for completed in sorted(directory.glob('*/results.json')):
            family = completed.parent
            if family_ids is not None and family.name not in family_ids:
                continue
            aa_records = {r.record_id.split()[0].split('|')[0]: r.seq for r in read_alignment_file(family / 'aa.aligned.fasta')}
            tokens, current = {}, None
            for line in (family / 'encoded.hex').read_text().splitlines():
                if line.startswith('>'):
                    current = line[1:].strip()
                    tokens[current] = ''
                elif current is not None:
                    tokens[current] += ''.join(mapping[int(token, 16)] for token in line.split())
            for ident, aa_aligned in aa_records.items():
                sequence = tokens[ident]
                amino_acids = aa_aligned.replace('-', '')
                if len(sequence) != len(amino_acids):
                    raise ValueError(f'Residue correspondence mismatch: {family.name}/{ident}')
                rows.append({'family': family.name, 'id': ident, 'models': representation,
                             'seq': sequence, 'aa': amino_acids})
                key = (family.name, ident)
                if key in seen_aa and seen_aa[key] != amino_acids:
                    raise ValueError(f'AA data differ across alphabet inputs: {key}')
                if key not in seen_aa:
                    rows.append({'family': family.name, 'id': ident, 'models': 'AA',
                                 'seq': amino_acids, 'aa': amino_acids})
                    seen_aa[key] = amino_acids
            alignments[(family.name, representation)] = family / 'ft2.aligned.fasta'
            alignments[(family.name, 'AA')] = family / 'aa.aligned.fasta'
    if not rows:
        raise ValueError('No completed family experiments found')
    return pd.DataFrame(rows), alignments, supports


def read_common_families(directories, native_root, cohort, limit=None):
    """AA/FT2/3Di reader with identical frozen OMA families and reference taxa."""
    from benchmark_cohort import native_directory, resolve_path, verify_stage
    from benchmark_common import AA_ORDER, ROOT, digest, read_fasta
    entries = {e['size']: e for e in json.loads((ROOT / 'configs/production_models.yaml').read_text())['models']}
    for directory in directories:
        provenance = json.loads((directory / 'provenance.json').read_text())
        entry = entries[provenance['size']]
        for key in ('encoder', 'mafft', 'raxml'):
            if provenance[key + '_sha256'] != digest(ROOT / entry['directory'] / entry[key]):
                raise ValueError(f'Experiment {key} differs from production: {directory}')
    families = sorted(cohort['families'], key=lambda f: f['family'])
    if limit:
        families = families[:limit]
    selected = {f['family'] for f in families}
    frame, alignments, supports = read_families(directories, selected)
    expected_models = set(supports)
    for name in expected_models:
        observed = set(frame.loc[frame.models == name, 'family'])
        if observed != selected:
            raise ValueError(f'Incomplete common cohort for {name}: missing {sorted(selected-observed)}')
    extra = []
    for family in families:
        ident = family['family']
        reference = read_fasta(resolve_path(family['aa_alignment']), aligned=True)
        labels = {t['accession']: t['label'] for t in family['taxa']}
        aa = {acc: reference[label].replace('-', '') for acc, label in labels.items()}
        for name in expected_models:
            rows = frame[(frame.family == ident) & (frame.models == name)]
            if set(rows.id) != set(aa) or len(rows) != len(aa):
                raise ValueError(f'Common OMA taxon mismatch: {ident}/{name}')
            if any(row.aa != aa[row.id] for row in rows.itertuples()):
                raise ValueError(f'AA reference mismatch: {ident}/{name}')
        native = native_directory(native_root, '3Di', ident)
        signature = verify_stage(native)
        if signature['matrix_sha256'] != digest(ROOT / 'foldtree2/config/3diphy_submats/Q.3Di.AF'):
            raise ValueError(f'3Di does not use the canonical published matrix: {ident}')
        expected_structures = {str(resolve_path(t['pdb']).resolve()): family['inputs'][t['pdb']] for t in family['taxa']}
        if signature['structures'] != expected_structures:
            raise ValueError(f'3Di does not use the frozen OMA structures: {ident}')
        native_aa = read_fasta(native / 'aa.aligned.fasta', aligned=True)
        tokens = read_fasta(native / '3di.unaligned.fasta')
        if set(tokens) != set(aa) or set(native_aa) != set(aa):
            raise ValueError(f'3Di taxon mismatch: {ident}')
        for acc in aa:
            if native_aa[acc].replace('-', '') != aa[acc] or len(tokens[acc]) != len(aa[acc]):
                raise ValueError(f'3Di/OMA residue correspondence mismatch: {ident}/{acc}')
            extra.append({'family': ident, 'id': acc, 'models': '3Di', 'seq': tokens[acc], 'aa': aa[acc]})
        alignments[(ident, '3Di')] = native / '3di.aligned.fasta'
    supports['3Di'] = list(AA_ORDER)
    frame = pd.concat([frame, pd.DataFrame(extra)], ignore_index=True)
    frame = frame.sort_values(['models', 'family', 'id']).reset_index(drop=True)
    if frame.duplicated(['models', 'family', 'id']).any():
        raise ValueError('Duplicate representation/family/taxon in common cohort')
    return frame, alignments, supports


def analyze(frame, supports, alignments, outdir, orders, folds, seed, analyses, controls):
    if controls:
        rng = np.random.default_rng(seed)
        shuffled = frame.copy()
        def shuffle_row(row):
            seq = list(row.seq)
            indices = [i for i, c in enumerate(seq) if c in supports[row.models]]
            values = rng.permutation([seq[i] for i in indices])
            for i, value in zip(indices, values):
                seq[i] = value
            return ''.join(seq)
        shuffled['seq'] = shuffled.apply(shuffle_row, axis=1)
        shuffled['models'] += '__within_shuffle'
        supports.update({name + '__within_shuffle': states for name, states in list(supports.items())})
        frame = pd.concat([frame, shuffled], ignore_index=True)
    usage, rate_rows, mdl_rows, mi_rows, position_rows, column_mi_rows = [], [], [], [], [], []
    for name, model_frame in frame.groupby('models', sort=True):
        states = supports[name]
        for family, group in model_frame.groupby('family', sort=True):
            sequences = group.seq.tolist()
            counts = Counter(c for seq in sequences for c in seq if c in states)
            n = sum(counts.values())
            entropy = -sum((count / n) * np.log2(count / n) for count in counts.values())
            usage.append({'model': name, 'family': family, 'alphabet_size': len(states), 'n_tokens': n,
                          'observed_states': len(counts), 'entropy_bits': entropy, 'effective_states': 2 ** entropy,
                          'n_missing_tokens': sum(map(len, sequences)) - n})
            if 'entropy' in analyses and len(sequences) >= 2:
                split = np.random.default_rng(seed).permutation(len(sequences))
                n_test = min(max(1, round(.2 * len(sequences))), len(sequences) - 1)
                train = [sequences[i] for i in split[n_test:]]
                test = [sequences[i] for i in split[:n_test]]
                counts_by_order = metrics.train_markov_counts(train, states, {c: i for i, c in enumerate(states)}, max(orders))
                probability = metrics.make_prob_fn(*counts_by_order, states)
                for order in orders:
                    events = [(tuple(seq[max(0, i-order):i]), c) for seq in test for i, c in enumerate(seq)
                              if c in states and all(x in states for x in seq[max(0, i-order):i])]
                    bits = sum(-np.log2(probability(context, c)) for context, c in events)
                    tokens = len(events)
                    if not tokens:
                        continue
                    rate_rows.append({'model': name, 'family': family, 'order': order, 'heldout_tokens': tokens,
                                      'entropy_rate_bits': bits / tokens})
            if 'mi' in analyses and (name.startswith('FT2_') or name.split('__')[0] == '3Di'):
                result = metrics.compute_positionwise_stats_fixed_support(sequences, group.aa.tolist(), states,
                                                                           AA, list(product(states, repeat=2)))
                mi_rows.append({'model': name, 'family': family, **result})
            if 'position' in analyses and (family, name) in alignments:
                records = read_alignment_file(alignments[(family, name)])
                msa = np.array([list(record.seq) for record in records])
                weights = metrics.compute_henikoff_weights(msa, set(states))
                for column in range(msa.shape[1]):
                    values = msa[:, column]
                    valid = np.isin(values, states)
                    gap_fraction = float(np.mean(values == '-'))
                    if gap_fraction > .3 or not valid.any():
                        continue
                    freqs = {c: float(weights[values == c].sum()) for c in states}
                    position_rows.append({'model': name, 'family': family, 'column': column,
                                          'gap_fraction': gap_fraction,
                                          'entropy_bits': metrics.entropy_from_weighted_freqs(freqs)})
                if name.startswith('FT2_') or name == '3Di':
                    aa_records = read_alignment_file(alignments[(family, 'AA')])
                    token_by_id = dict(zip(group.id, group.seq))
                    aa_msa = np.array([list(record.seq) for record in aa_records])
                    projected = np.array([list(metrics.project_ft2_onto_aa_gaps(
                        record.seq, token_by_id[record.record_id.split('|')[0]])) for record in aa_records])
                    aa_weights = metrics.compute_henikoff_weights(aa_msa, set(AA))
                    for column in range(aa_msa.shape[1]):
                        valid = np.isin(aa_msa[:, column], AA) & np.isin(projected[:, column], states)
                        gap_fraction = float(np.mean(aa_msa[:, column] == '-'))
                        if gap_fraction > .3 or not valid.any():
                            continue
                        joint = Counter()
                        # Convert relative Henikoff weights to effective counts
                        # so smoothing is comparable across alphabet sizes.
                        scale = valid.sum() / aa_weights[valid].sum()
                        for i in np.flatnonzero(valid):
                            joint[(projected[i, column], aa_msa[i, column])] += float(aa_weights[i] * scale)
                        column_mi_rows.append({'model': name, 'family': family, 'column': column,
                                               'n_valid_taxa': int(valid.sum()), 'gap_fraction': gap_fraction,
                                               'mi_bits': metrics.mi_from_joint_fixed_support(joint, states, AA)})
        if 'mdl' in analyses:
            families = np.array(sorted(model_frame.family.unique()))
            np.random.default_rng(seed).shuffle(families)
            if len(families) < 2:
                print(f'MDL skipped for {name}: cross-family CV needs at least two families')
            else:
                for fold, test_families in enumerate(np.array_split(families, min(folds, len(families)))):
                    is_test = model_frame.family.isin(test_families)
                    train = model_frame.loc[~is_test, 'seq'].tolist()
                    test = model_frame.loc[is_test, 'seq'].tolist()
                    for order in orders:
                        probs, stats = metrics.train_ngram_model(train, states, order)
                        bits, tokens, bpt = metrics.encode_bits(test, states, order, probs)
                        header = metrics.model_cost_proxy(states, order, stats, sum(map(len, train)))
                        mdl_rows.append({'model': name, 'fold': fold, 'order': order, 'test_families': len(test_families),
                                         'bits': bits, 'tokens': tokens, 'bits_per_token': bpt,
                                         'model_cost_proxy_bits': header,
                                         'mdl_proxy_bits_per_token': (bits + header) / tokens if tokens else None})
        print(f'Completed information metrics for {name}', flush=True)
    for filename, rows in [('alphabet_usage.csv', usage), ('entropy_rates.csv', rate_rows), ('mdl.csv', mdl_rows),
                           ('aa_token_mi.csv', mi_rows), ('position_entropy.csv', position_rows)]:
        if rows:
            pd.DataFrame(rows).to_csv(outdir / filename, index=False)
    if column_mi_rows:
        pd.DataFrame(column_mi_rows).to_csv(outdir / 'column_equivalent_mi.csv', index=False)
    if 'kmer' in analyses:
        rows = []
        for weighted in (False, True):
            result = metrics.run_kmer_fold_discrimination(frame, n_folds=folds, seed=seed, use_weighted=weighted, supports=supports)
            for name, by_k in result['k_mer_discrimination'].items():
                for k, values in by_k.items():
                    rows.append({'model': name, 'k': k, 'weighted': weighted, **values})
        pd.DataFrame(rows).to_csv(outdir / 'kmer_discrimination.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dirs', type=Path, nargs='+', required=True,
                        help='Completed per-size production experiment directories')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--analyses', nargs='+', choices=['entropy', 'mdl', 'mi', 'position', 'kmer'],
                        default=['entropy', 'mdl', 'mi', 'position', 'kmer'])
    parser.add_argument('--orders', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--permutation-controls', action='store_true')
    parser.add_argument('--native-root', type=Path, help='Include mandatory 3Di from a native sweep, using the common OMA cohort')
    parser.add_argument('--legacy-ft2-only', action='store_true', help='Explicitly allow an incomplete legacy AA/FT2-only analysis')
    parser.add_argument('--cohort', type=Path, help='Frozen common cohort; defaults to configs/oma_benchmark_cohort.json with --native-root')
    parser.add_argument('--max-families', type=int)
    args = parser.parse_args()
    if min(args.orders) < 0 or args.folds < 2:
        parser.error('Orders must be nonnegative and folds at least two')
    if args.max_families is not None and args.max_families < 1:
        parser.error('--max-families must be positive')
    if not args.native_root and not args.legacy_ft2_only:
        parser.error('3Di is required: supply --native-root; use --legacy-ft2-only only for explicit legacy analyses')
    if args.native_root:
        from benchmark_cohort import DEFAULT_COHORT, load
        args.cohort = args.cohort or DEFAULT_COHORT
        frame, alignments, supports = read_common_families(args.experiment_dirs, args.native_root, load(args.cohort), args.max_families)
    else:
        frame, alignments, supports = read_families(args.experiment_dirs)
    args.outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.outdir / 'sequences.csv', index=False)
    analyze(frame, supports, alignments, args.outdir, args.orders, args.folds, args.seed,
            set(args.analyses), args.permutation_controls)
    provenance = {'inputs': {str(path.resolve()): json.loads((path / 'provenance.json').read_text()) for path in args.experiment_dirs},
                  'analyses': args.analyses, 'orders': args.orders, 'folds': args.folds,
                  'seed': args.seed, 'permutation_controls': args.permutation_controls,
                  'n_sequences': len(frame), 'n_families': int(frame.family.nunique())}
    provenance['legacy_ft2_only'] = args.legacy_ft2_only
    if args.native_root:
        from benchmark_common import digest
        provenance.update(cohort_sha256=digest(args.cohort), native_root=str(args.native_root.resolve()),
                          required_representations=sorted(supports), max_families=args.max_families)
    (args.outdir / 'completed.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
