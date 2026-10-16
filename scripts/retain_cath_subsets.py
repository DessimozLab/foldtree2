#!/usr/bin/env python3
"""Build a NEW explicit CATH cohort with validated subsets and exclusion statistics."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics
from types import SimpleNamespace

from audit_cath_chains import audit_family
from curnagl_tcs import family_record
from rebuild_cath_families import canonical_lineage, digest, validate_rebuild, write_json


def stats(values):
    values = list(values)
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'std_population': statistics.pstdev(values) if values else None,
            'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def select_chain(chain, db):
    if chain['status'] not in {'sequence_verified', 'sequence_gate_failed'}:
        return None, None, chain['status']
    candidates = [r for r in chain.get('best_references', []) if r['metrics']['passes_sequence_gate']]
    if not candidates:
        return None, None, 'sequence_gate_failed'
    # Original unique metadata supplies ancestral taxonomy, matching the legacy TCS protocol.
    original = [r for r in candidates if r['kind'] == 'original metadata' and r.get('lineage')]
    if chain.get('metadata_rows') == 1 and original:
        reference = original[0]
        try:
            raw = reference['lineage'].split(',')[-1]
            match = re.fullmatch(r'\s*(\d+)(?:\s*\([^)]*\))?\s*', raw)
            if not match:
                raise ValueError('Malformed original taxonomy')
            lineage = canonical_lineage(db, int(match[1]))
            return reference, lineage, None
        except ValueError:
            pass
    taxa = chain.get('verified_reference_taxids', [])
    if len(taxa) != 1:
        return None, None, 'missing_or_conflicting_reference_taxonomy'
    choices = [r for r in candidates if taxa[0] in r.get('canonical_taxids', [])]
    if not choices:
        return None, None, 'no_retained_reference_with_verified_taxonomy'
    reference = choices[0]
    try:
        parent = db.execute('SELECT parent FROM species WHERE taxid=?', (taxa[0],)).fetchone()
        if not parent:
            raise ValueError('Absent reference taxonomy')
        lineage = canonical_lineage(db, int(parent[0]))
    except ValueError:
        return None, None, 'unresolved_reference_taxonomy'
    return reference, lineage, None


def process_family(task):
    old, source, rebuilt, audit_root, cache, database, destination, reports = task
    family = old['family']
    source = Path(source) / family
    destination = Path(destination) / family
    originals = {p.stem: p for p in (source / 'structs').glob('*.pdb')}
    report = {'family': family, 'original': len(originals), 'retained': 0,
              'excluded': 0, 'excluded_taxa': {}, 'previous_status': old['status']}
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as db:
        if old['status'] == 'passed':
            previous = Path(rebuilt) / family
            evidence = validate_rebuild(previous)
            if set(originals) != set(evidence['proteins']):
                raise ValueError('Original cohort changed since full-family validation')
            destination.symlink_to(previous.resolve(), target_is_directory=True)
            report.update(retained=len(originals), status='retained_unchanged',
                distinct_lineages=len({p['lineage'] for p in evidence['proteins'].values()}))
        else:
            audit = Path(audit_root) / (family+'.json')
            if audit.exists():
                data = json.loads(audit.read_text())
            else:
                audit = Path(reports) / 'additional_audits' / (family+'.json')
                data = audit_family((str(source), cache, database, False))
                write_json(audit, data)
            if {c['accession'] for c in data['chains']} != set(originals):
                raise ValueError('Audit does not cover the current original cohort')
            proteins, rows, excluded_metrics = {}, [], []
            for chain in data['chains']:
                accession = chain['accession']
                if digest(originals[accession]) != chain['pdb_sha256']:
                    raise ValueError('Original experimental chain changed since audit')
                for path, sha in chain['sources'].items():
                    if digest(path) != sha:
                        raise ValueError('Reference source changed since audit')
                reference, lineage, reason = select_chain(chain, db)
                if reason:
                    report['excluded_taxa'][accession] = reason
                    best = chain.get('best_references', [])
                    if best:
                        excluded_metrics.append(best[0]['metrics'])
                    continue
                proteins[accession] = {'source_pdb': str(originals[accession].resolve()),
                    'source_pdb_sha256': chain['pdb_sha256'],
                    'reference_sequence': reference['sequence'], 'observed_sequence': chain['observed_sequence'],
                    'lineage': lineage, 'mapping': reference['metrics'],
                    'reference_source': reference['source'],
                    'reference_sha256': chain['sources'][reference['source']],
                    'reference_evidence': chain['sources']}
                rows.append({'query': accession, 'Entry': accession,
                    'Sequence': chain['observed_sequence'], 'Sequence_source': 'observed_pdb_residues',
                    'UniProt_sequence': reference['sequence'], 'Taxonomic lineage (Ids)': lineage})
            report.update(retained=len(rows), excluded=len(report['excluded_taxa']),
                distinct_lineages=len({r['Taxonomic lineage (Ids)'] for r in rows}),
                exclusion_reasons=dict(Counter(report['excluded_taxa'].values())),
                excluded_metrics={k: stats(m[k] for m in excluded_metrics)
                    for k in ('identity', 'observed_coverage', 'observed_length')})
            if len(rows) < 4:
                report['status'] = 'family_excluded_fewer_than_four_validated_tips'
            else:
                destination.mkdir()
                (destination/'structs').mkdir()
                for accession in proteins:
                    (destination/'structs'/(accession+'.pdb')).symlink_to(originals[accession].resolve())
                for filename in ('sequence_dataset.csv','finalset.csv'):
                    with (destination/filename).open('w', newline='') as stream:
                        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                        writer.writeheader();writer.writerows(rows)
                (destination/'identifiers.txt').write_text('\n'.join(proteins)+'\n')
                (destination/'sequences.fasta').write_text(''.join(f'>{r["query"]}\n{r["Sequence"]}\n' for r in rows))
                write_json(destination/'rebuild_provenance.json', {
                    'schema':'cath_validated_subset_v1', 'family':family,
                    'original_taxa':sorted(originals), 'excluded_taxa': report['excluded_taxa'],
                    'all_existing_structure_taxa_retained':not report['excluded_taxa'],
                    'metadata_sha256':digest(destination/'finalset.csv'),
                    'audit_source':str(audit.resolve()), 'audit_sha256':digest(audit),
                    'taxonomy_db':database, 'proteins':proteins, 'structures_modified':False,
                    'thresholds':{'min_identity':.95,'min_coverage':.8,'min_matches':20}})
                report['status'] = 'retained_validated_subset'
        if destination.exists():
            baseline=SimpleNamespace(SAFE=re.compile(r'^[A-Za-z0-9_.-]+$'),digest=digest)
            # Exercise the benchmark's real family gate, including evidence and taxonomy.
            family_record(destination,'CATH',Path(family),db,baseline)
    report['excluded_fraction'] = report['excluded']/report['original'] if report['original'] else 0.
    write_json(Path(reports)/'families'/(family+'.json'),report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True,help='Completed audit results directory')
    parser.add_argument('--cache',type=Path,required=True,help='Read-only public reference cache')
    parser.add_argument('--taxonomy-db',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.out.exists() or args.workers<1:
        parser.error('Choose a new output directory and positive worker count')
    cfg=json.loads(args.summary.read_text())
    database=args.taxonomy_db.resolve()
    if digest(database)!=json.loads(database.with_name('snapshot.json').read_text())['sqlite_sha256']:
        raise ValueError('Taxonomy snapshot changed')
    cohort=args.out/'CATH_data';cohort.mkdir(parents=True)
    tasks=[(r,cfg['source_root'],cfg['output_root'],str(args.audit/'families'),str(args.cache),
            str(database),str(cohort),str(args.out)) for r in cfg['results']]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process_family,tasks):
            results.append(result)
            print(json.dumps({k:result[k] for k in ('family','status','retained','excluded')}),flush=True)
    with (args.out/'family_exclusions.csv').open('w',newline='') as stream:
        fields=['family','status','original','retained','excluded','excluded_fraction','distinct_lineages','exclusion_reasons']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in results:
            writer.writerow({k:json.dumps(r.get(k,{})) if k=='exclusion_reasons' else r.get(k) for k in fields})
    retained=[r for r in results if r['status'].startswith('retained')]
    affected=[r for r in results if r['previous_status']!='passed']
    summary={'families_total':len(results),'families_retained':len(retained),
        'families_recovered':sum(r['status']=='retained_validated_subset' for r in results),
        'families_excluded':len(results)-len(retained),
        'protein_tips_retained':sum(r['retained'] for r in retained),
        'chain_validation_exclusions':sum(r['excluded'] for r in results),
        'tips_omitted_from_final_cohort':sum(r['original'] for r in results)-sum(r['retained'] for r in retained),
        'excluded_per_family_all':stats(r['excluded'] for r in results),
        'excluded_per_family_previously_failed':stats(r['excluded'] for r in affected),
        'excluded_fraction_previously_failed':stats(r['excluded_fraction'] for r in affected),
        'retained_sizes':stats(r['retained'] for r in retained),
        'family_exclusions':[r for r in results if r not in retained],
        'exclusion_reasons':dict(sum((Counter(r.get('exclusion_reasons',{})) for r in results),Counter())),
        'distinct_lineages':stats(r['distinct_lineages'] for r in retained),
        'families_single_ancestral_lineage':sum(r['distinct_lineages']<2 for r in retained),
        'input_summary_sha256':digest(args.summary),'taxonomy_sha256':digest(database),
        'minimum_distinct_protein_tips':4,'population_std':True,'originals_modified':False}
    write_json(args.out/'summary.json',summary)


if __name__=='__main__':
    main()
