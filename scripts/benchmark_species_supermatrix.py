#!/usr/bin/env python3
"""Notebook-derived concatenated OMA gain benchmark on one fixed species topology.

Native mode reuses existing strategy alignments: columns are NOT index-paired.
Controlled mode projects all representations to corresponding FoldMason columns.
Both use the same species/family cohort, pad missing species, and fit nuisance
parameters per representation without searching for a new tree topology.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
import subprocess
import time

import numpy as np
import pandas as pd

from alphabet_information_benchmark import read_common_families
from benchmark_cohort import DEFAULT_COHORT, load, native_directory, resolve_path
from benchmark_common import (ROOT, begin_stage, complete_stage, digest, read_fasta,
                              tool_version, write_fasta, write_json)
from compare_oma_representations import controlled_alignments, shared_occupancy
from phylogenetic_information_gain import (DatasetSpec, analyze_alphabet_phylogenetic_info,
    compute_summary_stats, parse_raxml_sitelh, run_raxml_sitelh, compute_pairwise_mi)
from prepare_production_alphabets import validate_bundle


def species_labels(family):
    labels = {}
    for taxon in family['taxa']:
        parts = taxon['label'].split('|')
        if len(parts) != 2 or not parts[1]:
            raise ValueError(f'Invalid OMA species label: {taxon["label"]}')
        species = parts[1] + '_A'  # Match the notebook's AA ASTRAL tree labels.
        if species in labels.values():
            raise ValueError(f'Multiple copies in {family["family"]}/{species}; explicit ortholog selection required')
        labels[taxon['accession']] = species
    return labels


def concatenate(blocks, taxa):
    """Deterministic block order; every species contributes the same block width."""
    taxa = sorted(taxa)
    sequences = {taxon: [] for taxon in taxa}
    partitions, offset = [], 0
    for family, records in sorted(blocks.items()):
        if not records or not set(records) <= set(taxa):
            raise ValueError(f'Unknown species or empty family: {family}')
        lengths = {len(seq) for seq in records.values()}
        if len(lengths) != 1 or next(iter(lengths)) == 0:
            raise ValueError(f'Invalid alignment width: {family}')
        width = next(iter(lengths))
        for taxon in taxa:
            sequences[taxon].append(records.get(taxon, '-' * width))
        partitions.append({'family': family, 'start': offset, 'end': offset + width,
                           'n_columns': width, 'present_species': len(records)})
        offset += width
    result = {taxon: ''.join(parts) for taxon, parts in sequences.items()}
    if any(set(sequence) <= {'-'} for sequence in result.values()):
        raise ValueError('At least one species has no observed data in this cohort')
    return result, pd.DataFrame(partitions)


def paired_family_comparison(tables, draws=1000, seed=42):
    """Paired family bootstrap; no pairing of native alignment columns or nodes."""
    rows = []
    for left, right in combinations(sorted(tables), 2):
        a, b = tables[left].set_index('family'), tables[right].set_index('family')
        for metric in ['phylo_gain_mean', 'phylo_gain_sum', 'phylo_gain_norm_mean', 'h_tip_mean',
                       'phylo_gain_bits_per_tip_mean', 'normalized_tip_entropy_mean']:
            if metric not in a or metric not in b:
                continue
            pair = pd.concat([a[metric].rename('a'), b[metric].rename('b')], axis=1).dropna()
            if pair.empty:
                continue
            delta = (pair.a - pair.b).to_numpy()
            rng = np.random.default_rng(seed)
            bootstrap = np.array([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(draws)])
            low, high = np.quantile(bootstrap, [.025, .975])
            rows.append({'left': left, 'right': right, 'metric': metric, 'n_common_families': len(pair),
                         'mean_difference_left_minus_right': delta.mean(), 'lower_095': low, 'upper_095': high})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[10, 20, 30, 40])
    parser.add_argument('--cohort', type=Path, default=DEFAULT_COHORT)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True, help='Native AA/3Di sweep root, following reused artifact paths')
    parser.add_argument('--species-tree', type=Path, default=ROOT / 'configs/oma_fixed_species_tree.nwk')
    parser.add_argument('--column-mode', choices=['native', 'controlled'], default='native')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-families', type=int, help='Smoke-test subset; omit for the frozen full cohort')
    parser.add_argument('--max-missing', type=float, default=.3)
    parser.add_argument('--bootstrap-draws', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true', help='Validate inputs and write supermatrices; do not evaluate likelihoods')
    parser.add_argument('--wait-for-service', help='Wait for another user service before CPU-intensive benchmarking')
    parser.add_argument('--prune-tree-to-cohort', action='store_true', help='Explicitly prune unused species for a subset smoke test; never add or rename tips')
    args = parser.parse_args()
    if not 1 <= args.threads <= 8 or not 0 <= args.max_missing < 1 or args.bootstrap_draws < 1:
        parser.error('Invalid thread, missingness or bootstrap limit')
    if not args.sizes or len(set(args.sizes)) != len(args.sizes) or set(args.sizes) - {10, 20, 30, 40, 50}:
        parser.error('Select unique supported alphabet sizes')
    if args.max_families is not None and args.max_families < 1:
        parser.error('--max-families must be positive')
    if args.wait_for_service:
        args.outdir.mkdir(parents=True, exist_ok=True)
        while subprocess.run(['systemctl', '--user', 'show', args.wait_for_service, '-p', 'ActiveState', '--value'],
                             capture_output=True, text=True, check=True).stdout.strip() in {'active', 'activating', 'deactivating', 'reloading'}:
            write_json(args.outdir / 'status.json', {'status': 'waiting', 'service': args.wait_for_service})
            time.sleep(30)
    cohort = load(args.cohort)
    families = sorted(cohort['families'], key=lambda f: f['family'])
    if args.max_families:
        families = families[:args.max_families]
    entries = {e['size']: e for e in json.loads((ROOT / 'configs/production_models.yaml').read_text())['models']}
    bundle_reports = {str(size): validate_bundle(entries[size], require_convergence=True) for size in args.sizes}
    frame, alignments, supports = read_common_families([args.experiment_root / str(s) for s in args.sizes],
                                                    args.native_root, cohort, args.max_families)
    blocks = {name: {} for name in supports}
    source_hashes = {}
    for family in families:
        ident = family['family']
        labels = species_labels(family)
        if args.column_mode == 'controlled':
            projections = controlled_alignments(family, frame, args.native_root)
            records = {name: {labels[key.split('|')[0]]: value for key, value in rows.items()}
                       for name, rows in projections.items()}
        else:
            records = {}
            for name in supports:
                path = (resolve_path(family['aa_alignment']) if name == 'AA' else
                        native_directory(args.native_root, '3Di', ident) / '3di.aligned.fasta'
                        if name == '3Di' else alignments[(ident, name)])
                source_hashes[str(path.resolve())] = digest(path)
                source = read_fasta(path, aligned=True)
                if {key.split('|')[0] for key in source} != set(labels):
                    raise ValueError(f'Family/representation taxon mismatch: {ident}/{name}')
                records[name] = {labels[key.split('|')[0]]: seq for key, seq in source.items()}
        for name in supports:
            blocks[name][ident] = records[name]
    taxa = sorted({species for family in families for species in species_labels(family).values()})
    from ete3 import Tree
    species_tree = args.species_tree.resolve()
    tree = Tree(str(species_tree), format=1)
    if args.prune_tree_to_cohort and set(taxa) < set(tree.get_leaf_names()):
        tree.prune(taxa, preserve_branch_length=True)
    if len(tree.get_leaf_names()) != len(set(tree.get_leaf_names())) or set(tree.get_leaf_names()) != set(taxa):
        raise ValueError('Fixed species-tree tips must exactly match concatenated species labels (SPECIES_A)')
    matrices, partitions = {}, {}
    for name in supports:
        matrices[name], partitions[name] = concatenate(blocks[name], taxa)
    controlled_mask = shared_occupancy(matrices, supports, args.max_missing) if args.column_mode == 'controlled' else None
    signature = {'protocol': 'OMA_concatenated_fixed_species_topology_v1', 'column_mode': args.column_mode,
                 'species_tree_sha256': digest(species_tree), 'cohort_sha256': digest(args.cohort),
                 'family_ids': [f['family'] for f in families], 'models': sorted(supports), 'taxa': taxa,
                 'max_missing': args.max_missing, 'seed': args.seed, 'bootstrap_draws': args.bootstrap_draws,
                 'threads': args.threads, 'fixed_topology_fit_model_and_branches': True,
                 'prune_tree_to_cohort': args.prune_tree_to_cohort,
                 'source_alignment_sha256': source_hashes, 'production_bundles': bundle_reports,
                 'supermatrix_sha256': {name: __import__('hashlib').sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
                                        for name, rows in matrices.items()}, 'raxml': tool_version('raxml-ng')}
    out = args.outdir.resolve()
    needs_run = begin_stage(out, signature)
    if not needs_run:
        print('Complete validated cached supermatrix comparison:', out)
        return
    outputs = []
    if args.prune_tree_to_cohort:
        species_tree = out / 'fixed_species_tree.nwk'
        tree.write(format=1, outfile=str(species_tree))
        outputs.append(species_tree)
    summaries, family_tables = [], {}
    write_json(out / 'status.json', {'status': 'preparing_supermatrices', 'models': sorted(supports)})
    for name in sorted(supports):
        destination = out / name
        destination.mkdir(exist_ok=True)
        alignment = destination / 'supermatrix.fasta'
        if alignment.exists() and read_fasta(alignment, aligned=True) != matrices[name]:
            raise ValueError('Existing supermatrix changed; use a new output directory')
        if not alignment.exists():
            write_fasta(alignment, matrices[name])
        partitions[name].to_csv(destination / 'family_blocks.csv', index=False)
        outputs.extend([alignment, destination / 'family_blocks.csv'])
        if args.prepare_only:
            continue
        model = ('LG+G+I' if name == 'AA' else
                 f'PROTGTR{{{ROOT / "foldtree2/config/3diphy_submats/Q.3Di.AF"}}}+G+I' if name == '3Di' else
                 f'MULTI{entries[int(name[4:])]["size"]}_GTR{{{ROOT / entries[int(name[4:])]["directory"] / entries[int(name[4:])]["raxml"]}}}+I')
        local_signature = {**signature, 'representation': name, 'alignment_sha256': digest(alignment),
                           'model': model, 'published_3di_matrix_sha256': digest(ROOT / 'foldtree2/config/3diphy_submats/Q.3Di.AF')}
        if begin_stage(destination, local_signature):
            write_json(out / 'status.json', {'status': 'evaluating_fixed_topology', 'model': name})
            site_lh = run_raxml_sitelh(alignment, species_tree, model, destination / 'sites', 'raxml-ng', args.threads)
            fitted_tree = Tree(str(destination / 'sites_evaluate.raxml.bestTree'), format=1)
            if tree.robinson_foulds(fitted_tree, unrooted_trees=True)[0] != 0:
                raise ValueError('Evaluation changed the fixed species topology')
            spec = DatasetSpec(name, name, alignment, species_tree, model, site_lh, None)
            table = analyze_alphabet_phylogenetic_info(spec, parse_raxml_sitelh(site_lh), set(supports[name]), '-', args.max_missing)
            if table.empty:
                raise ValueError(f'No eligible supermatrix columns for {name}')
            eligible = table.n_valid_taxa >= 4
            if controlled_mask is not None:
                eligible &= table.column_index.map(lambda i: bool(controlled_mask[i]))
            table = table[eligible].copy()
            if table.empty:
                raise ValueError(f'No eligible columns under the selected occupancy protocol: {name}')
            bounds = partitions[name]
            indices = np.searchsorted(bounds.end.to_numpy(), table.column_index.to_numpy(), side='right')
            table['family'] = bounds.family.to_numpy()[indices]
            table['family_column_index'] = table.column_index.to_numpy() - bounds.start.to_numpy()[indices]
            table['normalized_tip_entropy'] = table.entropy_tip / np.log2(len(supports[name]))
            table['phylo_gain_bits'] = table.phylo_gain / np.log(2)
            table['phylo_gain_bits_per_tip'] = table.phylo_gain_bits / table.n_valid_taxa
            table.drop(columns=['alignment_column']).to_csv(destination / 'site_metrics.csv.gz', index=False)
            local = []
            for family in families:
                subset = table[table.family == family['family']]
                local.append(dict(family=family['family'], **compute_summary_stats(subset, name),
                                  phylo_gain_bits_per_tip_mean=float(subset.phylo_gain_bits_per_tip.mean()),
                                  normalized_tip_entropy_mean=float(subset.normalized_tip_entropy.mean())))
            pd.DataFrame(local).to_csv(destination / 'family_metrics.csv', index=False)
            write_json(destination / 'summary.json', {**compute_summary_stats(table, name),
                       'n_families': len(families), 'n_species': len(taxa), 'alphabet_size': len(supports[name]),
                       'total_alignment_columns': len(next(iter(matrices[name].values()))),
                       'phylo_gain_bits_per_tip_mean': float(table.phylo_gain_bits_per_tip.mean()),
                       'normalized_tip_entropy_mean': float(table.normalized_tip_entropy.mean()),
                       'n_families_with_eligible_columns': int(table.family.nunique()),
                       'gain_units': 'natural log; gain_bits also reported per site'})
            complete_stage(destination, local_signature, [destination / 'site_metrics.csv.gz', destination / 'family_metrics.csv',
                destination / 'summary.json', site_lh, destination / 'sites_evaluate.raxml.bestTree',
                destination / 'sites_evaluate.raxml.bestModel'])
        summaries.append(json.loads((destination / 'summary.json').read_text()))
        family_tables[name] = pd.read_csv(destination / 'family_metrics.csv', dtype={'family': str})
        outputs.extend([destination / 'completed.json', destination / 'summary.json'])
    if args.prepare_only:
        write_json(out / 'status.json', {'status': 'prepared_only', 'models': sorted(supports), 'n_families': len(families)})
        return
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    paired_family_comparison(family_tables, args.bootstrap_draws, args.seed).to_csv(out / 'paired_family_comparisons.csv', index=False)
    if args.column_mode == 'controlled':
        cross = []
        for left, right in combinations(sorted(supports), 2):
            a, b = matrices[left], matrices[right]
            for site in np.flatnonzero(controlled_mask):
                mi, nmi, valid = compute_pairwise_mi(''.join(a[t][site] for t in taxa), ''.join(b[t][site] for t in taxa), set(supports[left]), set(supports[right]), '-')
                cross.append({'left': left, 'right': right, 'column_index': site, 'mi_bits': mi, 'nmi': nmi, 'n_valid_joint': valid})
        pd.DataFrame(cross).to_csv(out / 'column_cross_mi.csv.gz', index=False)
        outputs.append(out / 'column_cross_mi.csv.gz')
    outputs.extend([out / 'summary.csv', out / 'paired_family_comparisons.csv'])
    complete_stage(out, signature, outputs)
    write_json(out / 'status.json', {'status': 'completed', 'models': sorted(supports), 'n_families': len(families)})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--outdir' in sys.argv:
            directory = Path(sys.argv[sys.argv.index('--outdir') + 1])
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / 'status.json', {'status': 'failed', 'error': repr(error)})
        raise
