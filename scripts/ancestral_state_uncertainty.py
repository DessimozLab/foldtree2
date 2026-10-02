#!/usr/bin/env python3
"""Stream node/site uncertainty from a strategy's OWN alignment and fitted tree."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from benchmark_common import (begin_stage, complete_stage, digest, read_fasta, run,
    stationary_from_log, tool_version, write_json)
from alphabet_information_metrics import compute_henikoff_weights
from foldtree2.src.ancestral import run_ancestral, ancestral_states_to_fasta


def uncertainty(probabilities, prior):
    p, pi = np.asarray(probabilities, float), np.asarray(prior, float)
    if p.shape != pi.shape or p.ndim != 1 or len(p) < 2:
        raise ValueError('Posterior and prior must have the same full alphabet support')
    if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=max(2e-4, len(p) * 5e-6 + 1e-8)):
        raise ValueError('Nonfinite, negative, or unnormalized posterior probabilities')
    if not np.isfinite(pi).all() or (pi <= 0).any() or not np.isclose(pi.sum(), 1, atol=2e-5):
        raise ValueError('Invalid stationary prior')
    p, pi = p / p.sum(), pi / pi.sum()
    positive = p > 0
    h = float(-np.sum(p[positive] * np.log2(p[positive])))
    ordered = np.sort(p)
    return {'max_probability': float(ordered[-1]), 'top_two_margin': float(ordered[-1] - ordered[-2]),
        'entropy_bits': h, 'normalized_entropy': h / np.log2(len(p)), 'effective_states': 2 ** h,
        'kl_from_stationary_bits': float(np.sum(p[positive] * np.log2(p[positive] / pi[positive]))),
        'confidence_090': int(ordered[-1] >= .9), 'confidence_095': int(ordered[-1] >= .95)}


def column_entropy(chars, states, weights=None):
    counts = np.zeros(len(states))
    mapping = {s: i for i, s in enumerate(states)}
    for i, char in enumerate(chars):
        if char in mapping:
            counts[mapping[char]] += 1 if weights is None else weights[i]
    if counts.sum() == 0:
        return None
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def posterior_rows(path, states):
    with Path(path).open() as stream:
        header = None
        for line in stream:
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split()
            if header is None:
                header = fields
                if header[:2] != ['Node', 'Site']:
                    raise ValueError(f'Unknown ancestral-probability header: {header}')
                columns = []
                for state in states:
                    candidates = [state, 'p_' + state, 'P(' + state + ')']
                    matches = [header.index(c) for c in candidates if c in header]
                    if len(matches) != 1:
                        raise ValueError(f'Posterior table lacks an unambiguous column for state {state!r}')
                    columns.append(matches[0])
                continue
            if len(fields) != len(header):
                raise ValueError('Malformed ancestral probability row')
            yield fields[0], int(fields[1]) - 1, np.asarray([float(fields[i]) for i in columns])
        if header is None:
            raise ValueError('Empty posterior table')


def rooted_nodes(tree_path, outdir, outgroups):
    from ete3 import Tree
    tree = Tree(str(tree_path), format=1)
    all_taxa = set(tree.get_leaf_names())
    def partition(node):
        parts = [tuple(sorted(c.get_leaf_names())) for c in node.children]
        if not node.is_root():
            parts.append(tuple(sorted(all_taxa - set(node.get_leaf_names()))))
        return tuple(sorted(parts))
    original_names = {partition(node): node.name for node in tree.traverse() if not node.is_leaf()}
    if outgroups:
        if not set(outgroups) <= set(tree.get_leaf_names()):
            raise ValueError('Specified outgroup taxa missing from strategy tree')
        node = tree & outgroups[0] if len(outgroups) == 1 else tree.get_common_ancestor(outgroups)
        if node.is_root() or set(node.get_leaf_names()) != set(outgroups):
            raise ValueError('Outgroup is not a proper monophyletic clade')
        tree.set_outgroup(node)
        rooting = 'specified_outgroup'
    else:
        source = outdir / 'mad_input.nwk'
        shutil.copy2(tree_path, source)
        run(['mad', '-t', source], outdir / 'rooting.log')
        rooted = Path(str(source) + '.rooted')
        if not rooted.exists():
            raise FileNotFoundError('MAD did not produce a rooted tree')
        tree = Tree(str(rooted), format=1)
        rooting = 'MAD_inferred'
    # MAD discards internal labels. Restore by the incident unrooted clades,
    # not descendant sets, which change when the root moves.
    for node in tree.traverse():
        if not node.is_leaf():
            node.name = original_names.get(partition(node), '')
    if set(tree.get_leaf_names()) != set(Tree(str(tree_path), format=1).get_leaf_names()):
        raise ValueError('Rooting changed tree taxa')
    tree.write(format=1, outfile=str(outdir / 'rooted.nwk'))
    max_depth = max(tree.get_distance(tip) for tip in tree.iter_leaves())
    if max_depth <= 0:
        raise ValueError('Cannot normalize root distances on a zero-length tree')
    nodes = {}
    for node in tree.traverse():
        if node.is_leaf() or not node.name:
            continue  # A synthetic root has no reconstructed posterior.
        descendants = sorted(node.get_leaf_names())
        stable = hashlib.sha256('\0'.join(descendants).encode()).hexdigest()[:20]
        depth = tree.get_distance(node)
        nodes[node.name] = {'node_id': stable, 'root_distance': depth,
            'normalized_root_depth': depth / max_depth,
            'nearest_tip_distance': min(node.get_distance(tip) for tip in tree.iter_leaves()),
            'descendant_tip_count': len(descendants), 'descendants': descendants}
    return tree, nodes, rooting


def reconstruct(alignment, tree, model, fitted_log, states, outdir, family='unknown',
                strategy='unknown', threads=8, outgroups=None, min_free_gb=10):
    alignment, tree, model, fitted_log = map(lambda p: Path(p).resolve(), (alignment, tree, model, fitted_log))
    outdir = Path(outdir).resolve()
    records = read_fasta(alignment, aligned=True)
    from ete3 import Tree
    parsed = Tree(str(tree), format=1)
    if set(records) != set(parsed.get_leaf_names()):
        raise ValueError('Strategy alignment and strategy tree have different taxa')
    if len(set(states)) != len(states):
        raise ValueError('Duplicate state labels')
    prior = stationary_from_log(fitted_log, len(states))
    length = len(next(iter(records.values())))
    expected_bytes = max(1, len(records) - 2) * length * (len(states) * 20 + 1200)
    disk = shutil.disk_usage(outdir.parent if outdir.parent.exists() else alignment.parent)
    if disk.free < expected_bytes + min_free_gb * 1024**3:
        raise RuntimeError(f'Insufficient disk: require {expected_bytes} estimated uncompressed bytes plus {min_free_gb} GiB reserve')
    signature = {'protocol': 'own_alignment_own_tree_marginal_uncertainty_v2', 'family': family,
        'strategy': strategy, 'states': list(states), 'outgroups': sorted(outgroups or []),
        'inputs': {str(p): digest(p) for p in (alignment, tree, model, fitted_log)},
        'raxml': tool_version('raxml-ng'), 'mad_sha256': digest(shutil.which('mad')) if not outgroups else None}
    if not begin_stage(outdir, signature):
        return outdir
    for index, node in enumerate(parsed.traverse()):
        if not node.is_leaf():
            node.name = f'ASR{index}'
    labelled = outdir / 'input.labelled.nwk'
    parsed.write(format=1, outfile=str(labelled))
    native = run_ancestral(alignment, labelled, model, outdir / 'asr',
        overwrite=True, freeze_fitted=True, threads=threads,
        runner=lambda command: run(command, outdir / 'ancestral.command.log'))
    consensus = Path(ancestral_states_to_fasta(native['states'], outdir / 'ancestral.fasta'))
    asr_tree = native['tree']
    _, nodes, rooting = rooted_nodes(asr_tree, outdir, outgroups)
    taxa = sorted(records)
    msa = np.array([list(records[t]) for t in taxa])
    weights = compute_henikoff_weights(msa, set(states))
    global_entropy = [column_entropy(col, states) for col in msa.T]
    global_weighted = [column_entropy(col, states, weights) for col in msa.T]
    valid = np.isin(msa, list(states))
    descendants = {name: [taxa.index(t) for t in data['descendants']] for name, data in nodes.items()}
    output = outdir / 'node_site_uncertainty.csv.gz'
    total, seen_nodes = 0, set()
    last_site = {name: -1 for name in nodes}
    sums = {k: 0. for k in uncertainty(np.full(len(states), 1 / len(states)), prior)}
    metadata = ['family', 'strategy', 'node', 'node_id', 'site', 'rooting', 'root_distance',
        'normalized_root_depth', 'nearest_tip_distance', 'descendant_tip_count', 'tip_entropy_bits',
        'weighted_tip_entropy_bits', 'normalized_tip_entropy', 'descendant_tip_entropy_bits',
        'weighted_descendant_tip_entropy_bits', 'n_valid_tips', 'gap_fraction', 'passes_occupancy']
    with gzip.open(output, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=metadata + list(sums) + ['p_' + s for s in states])
        writer.writeheader()
        for name, site, probability in posterior_rows(outdir / 'asr.raxml.ancestralProbs', states):
            if name not in nodes or not 0 <= site < length:
                raise ValueError(f'Unknown ancestral node/site: {name}/{site}')
            if site != last_site[name] + 1:
                raise ValueError('Duplicate, missing, or unordered posterior sites')
            last_site[name] = site
            data = nodes[name]
            subset = descendants[name]
            h = global_entropy[site]
            metrics = uncertainty(probability, prior)
            occupancy = int(valid[:, site].sum())
            row = {'family': family, 'strategy': strategy, 'node': name, 'site': site,
                'rooting': rooting, **{k: v for k, v in data.items() if k != 'descendants'},
                'tip_entropy_bits': h, 'weighted_tip_entropy_bits': global_weighted[site],
                'normalized_tip_entropy': h / np.log2(len(states)) if h is not None else None,
                'descendant_tip_entropy_bits': column_entropy(msa[subset, site], states),
                'weighted_descendant_tip_entropy_bits': column_entropy(msa[subset, site], states, weights[subset]),
                'n_valid_tips': occupancy, 'gap_fraction': 1 - occupancy / len(taxa),
                'passes_occupancy': int(occupancy >= 4 and 1 - occupancy / len(taxa) <= .3),
                **metrics, **{'p_' + s: float(p) for s, p in zip(states, probability)}}
            writer.writerow(row)
            for key, value in metrics.items():
                sums[key] += value
            seen_nodes.add(name)
            total += 1
    if not total or total != len(seen_nodes) * length or seen_nodes != set(nodes):
        raise ValueError('Incomplete node/site posterior coverage')
    # Verify uncompressed output first, then compress without discarding its only copy prematurely.
    raw = outdir / 'asr.raxml.ancestralProbs'
    with raw.open('rb') as source, gzip.open(str(raw) + '.gz', 'wb') as target:
        shutil.copyfileobj(source, target)
    raw.unlink()  # Reproducible raw table retained losslessly in .gz.
    write_json(outdir / 'summary.json', {'family': family, 'strategy': strategy, 'rooting': rooting,
        'n_node_sites': total, 'n_nodes': len(nodes), 'n_sites': length, 'state_order': list(states),
        'stationary_frequencies': prior.tolist(), 'estimated_uncompressed_bytes': expected_bytes,
        **{key: value / total for key, value in sums.items()}})
    complete_stage(outdir, signature, [output, outdir / 'summary.json', outdir / 'rooted.nwk',
        Path(str(raw) + '.gz'), consensus, native['states'], native['tree']], n_node_sites=total)
    return outdir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True, help='This strategy\'s fitted bestModel file')
    parser.add_argument('--fitted-log', type=Path, required=True)
    parser.add_argument('--states', required=True, help='Exact engine/matrix state order')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--family', default='unknown')
    parser.add_argument('--strategy', default='unknown')
    parser.add_argument('--outgroups', nargs='+')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--min-free-gb', type=float, default=10)
    args = parser.parse_args()
    print(json.dumps({'outdir': str(reconstruct(**vars(args)))}))


if __name__ == '__main__':
    main()
