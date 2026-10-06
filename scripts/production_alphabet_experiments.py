#!/usr/bin/env python3
"""Run notebook-derived alphabet descriptions and fixed-AA-tree phylogenetic gain.

Each structural representation has its own MAFFT alignment. Gain is compared
on the same per-family AA reference topology, not by pairing unrelated columns.
AA/token mutual information uses residue correspondence before alignment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from phylogenetic_information_gain import (DatasetSpec, analyze_alphabet_phylogenetic_info,
    compute_summary_stats, parse_raxml_sitelh, run_raxml_sitelh, read_alignment_file)


def notebook_functions():
    """Return the standalone numerical methods extracted from the notebook."""
    import alphabet_information_metrics as metrics
    names = {'safe_log2', 'train_markov_counts', 'make_prob_fn', 'train_ngram_model', 'encode_bits', 'model_cost_proxy',
             'mi_from_joint_fixed_support', 'shannon_entropy_from_counts_fixed_support'}
    return {name: getattr(metrics, name) for name in names}


def run_stdout(command, path):
    with path.open('w') as stream:
        subprocess.run(list(map(str, command)), stdout=stream, check=True)


def descriptions(sequences, aa_sequences, alphabet, functions, seed=7):
    # Split by protein, before fitting; never concatenate across protein boundaries.
    indices = np.random.default_rng(seed).permutation(len(sequences))
    n_test = min(max(1, round(len(sequences) * .2)), len(sequences) - 1)
    test = [sequences[i] for i in indices[:n_test]]
    train = [sequences[i] for i in indices[n_test:]]
    counts = Counter(''.join(sequences))
    total = sum(counts.values())
    entropy = -sum((n / total) * np.log2(n / total) for n in counts.values())
    summary = {'n_sequences': len(sequences), 'n_tokens': total, 'n_train_sequences': len(train),
               'n_test_sequences': len(test), 'observed_states': len(counts), 'alphabet_size': len(alphabet),
               'entropy_bits': float(entropy), 'effective_states': float(2 ** entropy),
               'uniform_bits_per_token': float(np.log2(len(alphabet))), 'split_seed': seed}
    if any(len(seq) != len(aa) for seq, aa in zip(sequences, aa_sequences)):
        raise ValueError('AA and structural tokens do not correspond residue by residue')
    aa_states = list('ACDEFGHIKLMNPQRSTVWY')
    joint = Counter((x, y) for seq, aa in zip(sequences, aa_sequences) for x, y in zip(seq, aa) if y in aa_states)
    summary['aa_token_mi_bits'] = functions['mi_from_joint_fixed_support'](joint, alphabet, aa_states)
    markov = functions['train_markov_counts'](train, alphabet, {c: i for i, c in enumerate(alphabet)}, 3)
    probability = functions['make_prob_fn'](*markov, alphabet)
    rows = []
    for order in range(4):
        probs, stats = functions['train_ngram_model'](train, alphabet, order)
        bits, n, bpt = functions['encode_bits'](test, alphabet, order, probs)
        header = functions['model_cost_proxy'](alphabet, order, stats, sum(map(len, train)))
        backoff_bits = 0.0
        n_backoff = 0
        for seq in test:
            for i, symbol in enumerate(seq):
                context = tuple(seq[max(0, i - order):i])
                backoff_bits -= np.log2(probability(context, symbol))
                n_backoff += 1
        rows.append({'order': order, 'heldout_bits': bits, 'heldout_tokens': n,
                     'heldout_bits_per_token': bpt, 'model_cost_proxy_bits': header,
                     'mdl_proxy_bits_per_token': (bits + header) / n if n else None,
                     'backoff_entropy_rate_bits': backoff_bits / n_backoff})
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--families', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-families', type=int)
    parser.add_argument('--cohort', type=Path, help='Frozen OMA cohort; used automatically for the notebook family root')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    from foldtree2.src.pdbgraphmk2 import PDB2PyG
    from ete3 import Tree
    entry = next(x for x in json.loads((ROOT / 'configs/production_models.yaml').read_text())['models'] if x['size'] == args.size)
    from prepare_production_alphabets import validate_bundle
    bundle_report = validate_bundle(entry, require_convergence=True)
    directory = ROOT / entry['directory']
    model = torch.load(directory / entry['encoder'], map_location=args.device, weights_only=False)
    model.eval()
    model.device = torch.device(args.device)
    converter = PDB2PyG(aapropcsv=str(ROOT / 'foldtree2/config/aaindex1.csv'))
    reverse_aa = {v: k for k, v in converter.aaindex.items()}
    replacements = {'"': chr(248), '#': chr(247), '>': chr(249), '=': chr(250), '<': chr(251),
                    '-': chr(252), ' ': chr(253), '\r': chr(254), '\n': chr(255)}
    alphabet = sorted(replacements.get(chr(i + 1), chr(i + 1)) for i in range(args.size))
    symbols = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,/:;<=>@[\\]^_{|}~'
    mapping = {char: symbols[i] for i, char in enumerate(alphabet)}
    functions = notebook_functions()
    args.outdir.mkdir(parents=True, exist_ok=True)
    signature = {'encoder_sha256': hashlib.sha256((directory / entry['encoder']).read_bytes()).hexdigest(),
                 'raxml_sha256': hashlib.sha256((directory / entry['raxml']).read_bytes()).hexdigest(),
                 'mafft_sha256': hashlib.sha256((directory / entry['mafft']).read_bytes()).hexdigest(),
                 'size': args.size, 'phylo_protocol': 'fixed_per_family_AA_topology_optimized_model_branches_v2',
                 'description_protocol': 'notebook_markov_and_mdl_protein_holdout',
                 'max_families': args.max_families}
    if bundle_report.get('matrix_acceptance') == 'accepted_nonconverged':
        signature['matrix_acceptance'] = bundle_report['convergence_acceptance']
    provenance = args.outdir / 'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text()) != signature:
        raise ValueError('Output provenance differs; use a new output directory')
    provenance.write_text(json.dumps(signature, indent=2) + '\n')
    from benchmark_cohort import select_families
    families, cohort_path = select_families(args.families, args.cohort, args.max_families)
    if cohort_path:
        cohort_record = {'manifest': str(cohort_path), 'sha256': hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
                         'family_ids': [family.name for family in families]}
        record = args.outdir / 'cohort.json'
        if record.exists() and json.loads(record.read_text()) != cohort_record:
            raise ValueError('Experiment cohort changed; use a new output directory')
        record.write_text(json.dumps(cohort_record, indent=2) + '\n')
    if not families:
        raise ValueError('No families with at least four structures')
    summaries, markov_rows, phylo_rows = [], [], []
    for family in families:
        output = args.outdir / family.name
        output.mkdir(exist_ok=True)
        done = output / 'results.json'
        if done.exists():
            result = json.loads(done.read_text())
        else:
            aa_records = {r.record_id.split()[0]: r.seq for r in read_alignment_file(family / 'sequences.aligned.fa')}
            accession_labels = {label.split('|')[0]: label for label in aa_records}
            if len(accession_labels) != len(aa_records):
                raise ValueError(f'Duplicate AA accessions in {family.name}')
            pdbs = sorted((family / 'structs').glob('*.pdb'))
            missing = set(accession_labels) - {pdb.stem for pdb in pdbs}
            if missing:
                raise ValueError(f'Missing reference AA structures in {family.name}: {missing}')
            # Several families have an extra PDB absent from the reference AA
            # alignment/tree. Use the reference sample for both representations.
            excluded = [pdb.name for pdb in pdbs if pdb.stem not in accession_labels]
            pdbs = [pdb for pdb in pdbs if pdb.stem in accession_labels]
            sequences, aas, ids = [], [], []
            with torch.no_grad():
                for pdb in pdbs:
                    graph = converter.struct2pyg(str(pdb)).to(args.device)
                    z, _ = model(graph)
                    tokens = model.vector_quantizer.discretize_z(z)[0].flatten().tolist()
                    seq = ''.join(replacements.get(chr(int(t) + 1), chr(int(t) + 1)) for t in tokens)
                    aa = ''.join(reverse_aa[int(t)] for t in graph['AA'].x.argmax(dim=1).tolist())
                    sequences.append(seq)
                    aas.append(aa)
                    ids.append(pdb.stem)
            summary, markov = descriptions(sequences, aas, alphabet, functions)
            summary['excluded_nonreference_pdbs'] = excluded
            with (output / 'encoded.hex').open('w') as stream:
                for ident, seq in zip(ids, sequences):
                    stream.write(f'>{ident}\n' + ' '.join(f'{ord(c):02x}' for c in seq) + '\n')
            run_stdout(['hex2maffttext', output / 'encoded.hex'], output / 'encoded.ASCII')
            run_stdout(['mafft', '--text', '--thread', args.threads, '--localpair', '--maxiterate', '1000',
                        '--textmatrix', directory / entry['mafft'], output / 'encoded.ASCII'], output / 'aligned.ASCII')
            run_stdout(['maffttext2hex', output / 'aligned.ASCII'], output / 'aligned.hex')
            # Parse hex token rows without treating control characters as whitespace.
            alignment, current = {}, None
            for line in (output / 'aligned.hex').read_text().splitlines():
                if line.startswith('>'):
                    current = line[1:].strip()
                    alignment[current] = ''
                elif current is not None:
                    alignment[current] += ''.join('-' if t == '--' else mapping[chr(int(t, 16))] for t in line.split())
            tree = family / 'raxml_lg_tree.raxml.bestTree'
            aa_alignment = output / 'aa.aligned.fasta'
            # Notebook AA labels carry a |species suffix; PDB filenames are accessions.
            if set(alignment) != set(accession_labels):
                raise ValueError(f'AA/FT2 accessions differ in family {family.name}')
            alignment = {accession_labels[ident]: seq for ident, seq in alignment.items()}
            ft2_alignment = output / 'ft2.aligned.fasta'
            ft2_alignment.write_text(''.join(f'>{ident}\n{seq}\n' for ident, seq in alignment.items()))
            aa_alignment.write_text(''.join(f'>{ident}\n{seq}\n' for ident, seq in aa_records.items()))
            taxa = set(Tree(str(tree)).get_leaf_names())
            if taxa != set(alignment) or taxa != set(aa_records):
                raise ValueError(f'Tree/AA/FT2 taxa differ in family {family.name}')
            phylo = []
            for name, path, states, substitution in [
                ('AA', aa_alignment, set('ACDEFGHIKLMNPQRSTVWY'), 'LG+G+I'),
                ('FT2', ft2_alignment, set(mapping.values()), f"MULTI{args.size}_GTR{{{directory / entry['raxml']}}}+I")]:
                sitelh = Path(f'{output / name}.raxml.siteLH')
                if not sitelh.exists():
                    sitelh = run_raxml_sitelh(path, tree, substitution, output / name, 'raxml-ng', args.threads)
                spec = DatasetSpec(name, name.lower(), path, tree, substitution, sitelh, None)
                frame = analyze_alphabet_phylogenetic_info(spec, parse_raxml_sitelh(sitelh), states, '-', .3)
                if frame.empty:
                    raise ValueError(f'No columns pass occupancy filtering: {family.name}/{name}')
                frame.to_csv(output / f'phylo_info_{name}.csv', index=False)
                phylo.append(compute_summary_stats(frame, name))
            result = {'description': summary, 'markov': markov, 'phylo': phylo}
            done.write_text(json.dumps(result, indent=2) + '\n')
        summaries.append({'family': family.name, 'size': args.size, **result['description']})
        markov_rows.extend({'family': family.name, 'size': args.size, **row} for row in result['markov'])
        phylo_rows.extend({'family': family.name, 'size': args.size, **row} for row in result['phylo'])
        pd.DataFrame(summaries).to_csv(args.outdir / 'alphabet_description.csv', index=False)
        pd.DataFrame(markov_rows).to_csv(args.outdir / 'entropy_rate_mdl.csv', index=False)
        pd.DataFrame(phylo_rows).to_csv(args.outdir / 'phylogenetic_gain.csv', index=False)
        print(f"{args.size} states: completed {len(summaries)}/{len(families)} families", flush=True)
    (args.outdir / 'completed.json').write_text(json.dumps({'families': len(families), **signature}, indent=2) + '\n')


if __name__ == '__main__':
    main()
