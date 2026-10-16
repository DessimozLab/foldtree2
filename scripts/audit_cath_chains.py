#!/usr/bin/env python3
"""Diagnostic-only audit of every chain in selected excluded CATH families."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import re
import sqlite3

from Bio.PDB.PDBExceptions import PDBException

from rebuild_cath_families import (canonical_taxid, digest, download,
    historical_records, observed_sequence, write_json)


def alignment_metrics(reference, observed):
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner(mode='local', match_score=2, mismatch_score=-3,
                             open_gap_score=-8, extend_gap_score=-.5)
    alignments = aligner.align(reference, observed)
    mapping = [None] * len(observed)
    matches = pairs = 0
    score = 0.
    try:
        alignment = alignments[0]
    except IndexError:
        alignment = None
    if alignment is not None:
        score = float(alignment.score)
        for rr, oo in zip(*alignment.aligned):
            for i, j in zip(range(*map(int, rr)), range(*map(int, oo))):
                mapping[j] = i
                pairs += 1
                matches += reference[i] == observed[j]
    return {'score': score, 'matches': matches, 'aligned_pairs': pairs,
        'identity': matches / pairs if pairs else 0.,
        'observed_coverage': pairs / len(observed),
        'reference_coverage': pairs / len(reference),
        'observed_length': len(observed), 'reference_length': len(reference),
        'mismatches': pairs - matches, 'unmapped_observed_residues': len(observed)-pairs,
        'reference_positions': mapping,
        'passes_sequence_gate': matches >= 20 and pairs / len(observed) >= .8
            and (matches / pairs if pairs else 0.) >= .95}


def cached_file(cache, accession, historical, fetch):
    if not re.fullmatch(r'[A-Z0-9]{6,10}(?:-\d+)?', accession):
        return None
    suffix = '.unisave.txt' if historical else '.json'
    path = Path(cache) / (accession + suffix)
    if not path.exists() and fetch:
        import fcntl
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists():
                url = (f'https://rest.uniprot.org/unisave/{accession}?format=txt' if historical
                       else f'https://rest.uniprot.org/uniprotkb/{accession}.json')
                download(url, path)
    return path if path.exists() else None


def audit_family(task):
    family, cache, taxonomy, fetch = task
    family = Path(family)
    metadata = family / 'sequence_dataset.csv'
    if not metadata.is_file():
        metadata = family / 'finalset.csv'
    rows = list(csv.DictReader(metadata.open())) if metadata.is_file() else []
    results = []
    with sqlite3.connect(f'file:{taxonomy}?mode=ro', uri=True) as db:
        for pdb in sorted((family / 'structs').glob('*.pdb')):
            accession = pdb.stem
            result = {'family': family.name, 'accession': accession,
                      'pdb': str(pdb.resolve()), 'pdb_sha256': digest(pdb),
                      'errors': [], 'sources': {}, 'pdb_candidates_unverified': []}
            try:
                observed = observed_sequence(pdb)
                result['observed_sequence'] = observed
                candidates = []
                matched_rows = [r for r in rows if r.get('query') == accession or r.get('Entry') == accession]
                result['metadata_rows'] = len(matched_rows)
                if metadata.is_file():
                    result['sources'][str(metadata)] = digest(metadata)
                for row in matched_rows:
                    candidates.append({'sequence': row.get('Sequence', '').upper(),
                        'taxids': [row['Organism ID']] if row.get('Organism ID') else [],
                        'source': str(metadata), 'kind': 'original metadata',
                        'lineage': row.get('Taxonomic lineage (Ids)', ''), 'versions': []})
                # Retrieve new references only when original sequence evidence fails.
                needs_fetch = fetch and not any(c['sequence'] and
                    alignment_metrics(c['sequence'], observed)['passes_sequence_gate'] for c in candidates)
                visited = set()
                def current(identifier):
                    if identifier in visited or len(visited) >= 16:
                        return
                    visited.add(identifier)
                    try:
                        path = cached_file(cache, identifier, False, needs_fetch)
                        if path is None:
                            return
                        result['sources'][str(path)] = digest(path)
                        entry = json.loads(path.read_text())
                        if entry.get('sequence'):
                            candidates.append({'sequence': entry['sequence']['value'],
                                'taxids': [str(entry.get('organism', {}).get('taxonId', ''))],
                                'source': str(path), 'kind': 'current UniProt', 'versions': [],
                                'reference_accession': entry.get('primaryAccession')})
                        for target in entry.get('inactiveReason', {}).get('mergeDemergeTo', []):
                            current(target)
                    except (ValueError, KeyError, FileNotFoundError) as error:
                        result['errors'].append({'stage': 'current reference', 'error': str(error)})
                current(accession)
                try:
                    archive = cached_file(cache, accession, True, needs_fetch)
                    if archive:
                        result['sources'][str(archive)] = digest(archive)
                        unique = {}
                        with archive.open() as stream:
                            for record in historical_records(stream):
                                if accession not in record['accessions']:
                                    continue
                                key = (record['sequence'], tuple(record['taxids']))
                                unique.setdefault(key, {'sequence': record['sequence'],
                                    'taxids': record['taxids'], 'source': str(archive),
                                    'kind': 'historical UniProt', 'versions': []})['versions'].append(record['entry_version'])
                        candidates.extend(unique.values())
                        # Mapping hints only: these may cover different chains/versions.
                        result['pdb_candidates_unverified'] = sorted(set(re.findall(
                            r'^DR   PDB; ([A-Za-z0-9]{4});', archive.read_text(), re.M)))
                except (ValueError, KeyError, FileNotFoundError) as error:
                    result['errors'].append({'stage': 'historical reference', 'error': str(error)})
                ranked = []
                for candidate in candidates:
                    sequence = candidate['sequence']
                    if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWYXBZUO'):
                        continue
                    candidate['metrics'] = alignment_metrics(sequence, observed)
                    candidate['canonical_taxids'] = []
                    for taxid in candidate['taxids']:
                        try:
                            candidate['canonical_taxids'].append(canonical_taxid(db, int(taxid)))
                        except (ValueError, TypeError) as error:
                            candidate.setdefault('taxonomy_errors', []).append(str(error))
                    ranked.append(candidate)
                ranked.sort(key=lambda c: (c['metrics']['passes_sequence_gate'],
                    c['metrics']['matches'], c['metrics']['score']), reverse=True)
                passing = [c for c in ranked if c['metrics']['passes_sequence_gate']]
                taxa = sorted({t for c in passing for t in c['canonical_taxids']})
                result.update(reference_candidates=len(ranked), best_references=ranked[:5],
                    sequence_gate_passes=bool(passing), verified_reference_taxids=taxa,
                    status='sequence_verified' if passing else 'sequence_gate_failed',
                    observed_sequence_source='unchanged PDB residues')
                if not ranked:
                    result['status'] = 'no_reference'
                # Retain provenance hints that survived PDBFixer, without guessing a PDB ID.
                result['pdb_header_evidence'] = [line for line in pdb.read_text().splitlines()
                    if line.startswith(('HEADER', 'COMPND', 'SOURCE', 'DBREF', 'TITLE'))]
            except (ValueError, KeyError, PDBException, FileNotFoundError) as error:
                result.update(status='audit_error')
                result['errors'].append({'stage': 'chain audit', 'error': str(error)})
            results.append(result)
    return {'family': family.name, 'chains': results, 'inclusion_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True, help='Run-local reference cache, never a frozen cache')
    parser.add_argument('--taxonomy-db', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--fetch-missing', action='store_true')
    args = parser.parse_args()
    if args.out.exists() or args.workers < 1:
        parser.error('Use a new output directory and positive worker count')
    if args.fetch_missing and args.cache.resolve().parent != args.out.resolve().parent:
        parser.error('Writable cache must be a sibling of the new output directory in the isolated audit run')
    database = args.taxonomy_db.resolve()
    if digest(database) != json.loads(database.with_name('snapshot.json').read_text())['sqlite_sha256']:
        raise ValueError('Taxonomy snapshot changed')
    selection = json.loads(args.summary.read_text())
    families = [r['family'] for r in selection['results'] if r.get('reason', '').endswith(
        ': 0 verified historical identities; unresolved/ambiguous')]
    args.out.mkdir(parents=True)
    tasks = [(str(Path(selection['source_root']) / family), str(args.cache.resolve()),
              str(database), args.fetch_missing) for family in families]
    chains = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(audit_family, tasks):
            write_json(args.out / 'families' / (result['family']+'.json'), result)
            chains.extend(result['chains'])
            print(json.dumps({'family': result['family'], 'chains': len(result['chains'])}), flush=True)
    with (args.out / 'chains.csv').open('w', newline='') as stream:
        fields = ['family', 'accession', 'status', 'identity', 'observed_coverage', 'matches',
                  'observed_length', 'reference_length', 'pdb_candidates', 'errors']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for chain in chains:
            best = chain.get('best_references', [])
            metrics = best[0]['metrics'] if best else {}
            writer.writerow(dict(family=chain['family'], accession=chain['accession'], status=chain['status'],
                **{k: metrics.get(k) for k in fields[3:8]},
                pdb_candidates=';'.join(chain['pdb_candidates_unverified']), errors=json.dumps(chain['errors'])))
    write_json(args.out / 'summary.json', {'families': len(families), 'chains': len(chains),
        'chain_statuses': dict(Counter(c['status'] for c in chains)),
        'input_summary_sha256': digest(args.summary), 'taxonomy_sha256': digest(database),
        'inclusion_changed': False, 'thresholds': {'identity': .95, 'observed_coverage': .8, 'matches': 20}})


if __name__ == '__main__':
    main()
