#!/usr/bin/env python3
"""Rebuild experimental CATH cohorts with observed sequences and audited reference mappings."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request

SAFE = re.compile(r'^[A-Za-z0-9_.-]+$')
TAXDUMP = 'https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz'


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        temporary = Path(stream.name)
    temporary.replace(path)


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'FoldTree2-CATH-rebuild/1'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(
                    dir=destination.parent, delete=False) as stream:
                for block in iter(lambda: response.read(1024**2), b''):
                    stream.write(block)
                temporary = Path(stream.name)
            temporary.replace(destination)
            return
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
        time.sleep(2**attempt)


def taxonomy(directory):
    """Build a NEW standard ETE3 database from a checksummed official NCBI dump."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    db = directory / 'taxa.sqlite'
    marker = directory / 'snapshot.json'
    if marker.is_file():
        if digest(db) != json.loads(marker.read_text())['sqlite_sha256']:
            raise ValueError('Taxonomy snapshot changed')
        return db
    if db.exists():
        raise ValueError('Incomplete taxonomy build; use a new taxonomy directory')
    dump = directory / 'taxdump.tar.gz'
    checksum = directory / 'taxdump.tar.gz.md5'
    download(TAXDUMP + '.md5', checksum)
    if not dump.exists():
        download(TAXDUMP, dump)
    expected = checksum.read_text().split()[0]
    if digest(dump, 'md5') != expected:
        raise ValueError('NCBI taxonomy archive checksum mismatch')
    from ete3.ncbi_taxonomy.ncbiquery import update_db
    # ETE3 writes/removes its temporary tables in CWD. Confine these to a new directory.
    build = Path(tempfile.mkdtemp(prefix='ete-build-', dir=directory))
    previous = Path.cwd()
    try:
        os.chdir(build)
        update_db(str(db), str(dump))
    finally:
        os.chdir(previous)
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
        if conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Taxonomy database integrity failure')
        taxa = conn.execute('SELECT COUNT(*) FROM species').fetchone()[0]
    write_json(marker, {'source': TAXDUMP, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
                       'taxdump_sha256': digest(dump), 'taxdump_md5': expected,
                       'sqlite_sha256': digest(db), 'taxa': taxa, 'shared_database_modified': False})
    return db


def canonical_taxid(db, taxid):
    seen = set()
    while taxid not in seen:
        seen.add(taxid)
        if db.execute('SELECT 1 FROM species WHERE taxid=?', (taxid,)).fetchone():
            return taxid
        merged = db.execute('SELECT taxid_new FROM merged WHERE taxid_old=?', (taxid,)).fetchone()
        if not merged:
            raise ValueError(f'Unresolved taxonomy ID: {taxid}')
        taxid = int(merged[0])
    raise ValueError('Taxonomy merge cycle')


def canonical_lineage(db, taxid):
    taxid = canonical_taxid(db, taxid)
    nodes, seen = [], set()
    while taxid != 1:
        if taxid in seen:
            raise ValueError('Taxonomy parent cycle')
        seen.add(taxid)
        row = db.execute('SELECT parent, rank FROM species WHERE taxid=?', (taxid,)).fetchone()
        if not row:
            raise ValueError('Taxonomy parent absent')
        nodes.append(f'{taxid} ({row[1]})')
        taxid = int(row[0])
    if not nodes:
        raise ValueError('Root-only lineage')
    # Baseline uses ancestral lineage, not an invented species ID; omit root 1.
    return ', '.join(reversed(nodes))


def observed_sequence(path):
    from scripts.curnagl.curnagl_tcs import pdb_sequence
    return pdb_sequence(path)


def map_sequence(reference, observed, min_identity=.95, min_coverage=.8, min_matches=20):
    from Bio.Align import PairwiseAligner
    if not reference or set(reference) - set('ACDEFGHIKLMNPQRSTVWYXBZUO'):
        raise ValueError('Missing/invalid UniProt reference sequence')
    aligner = PairwiseAligner(mode='local', match_score=2, mismatch_score=-3,
                             open_gap_score=-8, extend_gap_score=-.5)
    alignments = aligner.align(reference, observed)
    if not alignments:
        raise ValueError('Unverified chain identity: no local alignment')
    alignment = alignments[0]
    mapping = [None] * len(observed)
    matches = pairs = 0
    for ref_range, obs_range in zip(*alignment.aligned):
        for i, j in zip(range(*map(int, ref_range)), range(*map(int, obs_range))):
            mapping[j] = i
            pairs += 1
            matches += reference[i] == observed[j]
    identity = matches / pairs if pairs else 0
    coverage = pairs / len(observed)
    if matches < min_matches or identity < min_identity or coverage < min_coverage:
        raise ValueError(f'Unverified chain identity: identity={identity:.4f}, observed_coverage={coverage:.4f}, matches={matches}')
    return {'reference_positions': mapping, 'identity': identity, 'observed_coverage': coverage,
            'matches': matches, 'aligned_pairs': pairs, 'observed_length': len(observed),
            'reference_length': len(reference)}


def uniprot(accession, cache, observed=None, thresholds=None, seen=(), historical=False):
    if not re.fullmatch(r'[A-Z0-9]{6,10}(?:-\d+)?', accession):
        raise ValueError('Unsafe UniProt accession')
    if accession in seen or len(seen) >= 8:
        raise ValueError('UniProt successor cycle/depth limit')
    if historical:
        return archived_uniprot(accession, cache, observed, thresholds)
    path = Path(cache) / (accession + '.json')
    if not path.exists():
        # File-level lock prevents duplicate requests for shared family members.
        import fcntl
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists():
                download(f'https://rest.uniprot.org/uniprotkb/{accession}.json', path)
    entry = json.loads(path.read_text())
    if entry.get('entryType') == 'Inactive':
        successors = entry.get('inactiveReason', {}).get('mergeDemergeTo', [])
        if not successors:
            row = archived_uniprot(accession, cache, observed, thresholds)
        else:
            candidates = []
            for target in sorted(set(successors)):
                try:
                    candidate = uniprot(target, cache, observed, thresholds, seen + (accession,))
                    if observed is not None:
                        map_sequence(candidate['Sequence'], observed, **(thresholds or {}))
                    candidates.append(candidate)
                except ValueError:
                    continue
            if not candidates:
                row = archived_uniprot(accession, cache, observed, thresholds)
            elif len(candidates) != 1:
                raise ValueError(f'Inactive UniProt {accession}: {len(candidates)} verified successors; identity is ambiguous/unresolved')
            else:
                row = candidates[0]
        row = dict(row)
        row.update(query=accession, Entry=accession)
        row['reference_evidence'] = dict(row.get('reference_evidence', {}), **{str(path): digest(path)})
        return row
    organism = entry['organism']
    return {'query': accession, 'Entry': accession, 'UniProt_primary_accession': entry['primaryAccession'],
            'Sequence': entry['sequence']['value'], 'Organism': organism['scientificName'],
            'Organism ID': str(organism['taxonId']), 'reference_source': str(path),
            'reference_sha256': digest(path)}


def historical_records(stream):
    """Read identity fields without parsing unrelated, version-sensitive annotations.

    FT/RM and other annotation syntax changed across UniProt releases. Recovery
    needs only AC, OS, OX, DT and SQ; require complete records and exact SQ length.
    """
    record = None
    sequence_parts = []
    sequence_length = None
    for number, raw in enumerate(stream, 1):
        line = raw.rstrip('\r\n')
        code = line[:2]
        value = line[5:].strip()
        if code == 'ID':
            if record is not None:
                raise ValueError('Historical record format: missing record terminator')
            record = {'accessions': [], 'taxids': [], 'organism': [], 'entry_version': ''}
            sequence_parts, sequence_length = [], None
        elif code == '//':
            if record is None or sequence_length is None or not record['accessions']:
                raise ValueError('Historical record format: incomplete identity/sequence fields')
            sequence = ''.join(sequence_parts)
            if len(sequence) != sequence_length or not sequence:
                raise ValueError('Historical record format: SQ sequence length mismatch')
            record['sequence'] = sequence
            record['organism'] = ' '.join(record['organism'])
            record['taxids'] = sorted(set(record['taxids']))
            yield record
            record = None
        elif record is None:
            if line.strip():
                raise ValueError(f'Historical record format: unexpected text outside entry at line {number}')
        elif code == 'AC':
            record['accessions'].extend(part.strip() for part in value.split(';') if part.strip())
        elif code == 'OS':
            record['organism'].append(value)
        elif code == 'OX':
            match = re.search(r'NCBI_TaxID=([\d, ]+)', value)
            if not match:
                raise ValueError('Historical record format: invalid taxonomy field')
            record['taxids'].extend(x.strip() for x in match[1].split(',') if x.strip())
        elif code == 'DT':
            match = re.search(r'entry version (\d+)', value)
            if match:
                record['entry_version'] = match[1]
        elif code == 'SQ':
            match = re.match(r'SEQUENCE\s+(\d+)\s+AA;', value)
            if not match or sequence_length is not None:
                raise ValueError('Historical record format: invalid/repeated SQ header')
            sequence_length = int(match[1])
        elif code == '  ' and sequence_length is not None:
            sequence = re.sub(r'[\s\d]', '', line)
            if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWYXBZUO'):
                raise ValueError('Historical record format: invalid sequence residues')
            sequence_parts.append(sequence)
        # Other fields are annotations, not inputs to chain identity or taxonomy.
    if record is not None:
        raise ValueError('Historical record format: truncated final entry')


def archived_uniprot(accession, cache, observed, thresholds):
    """Recover an accession-specific historical reference, never infer taxonomy from a homolog."""
    path = Path(cache) / (accession + '.unisave.txt')
    path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            download(f'https://rest.uniprot.org/unisave/{accession}?format=txt', path)
    candidates = []
    with path.open() as stream:
        for record in historical_records(stream):
            if accession not in record['accessions']:
                continue
            taxids = record['taxids']
            if len(taxids) != 1:
                continue
            sequence = record['sequence']
            if observed is not None:
                try:
                    map_sequence(sequence, observed, **(thresholds or {}))
                except ValueError:
                    continue
            candidates.append({'query': accession, 'Entry': accession,
                'UniProt_primary_accession': record['accessions'][0],
                'Sequence': sequence, 'Organism': record['organism'],
                'Organism ID': str(taxids[0]), 'reference_source': str(path),
                'reference_sha256': digest(path), 'Reference_kind': 'UniSave historical',
                'Reference_version': record['entry_version']})
    # Versions may repeat identical sequence/taxonomy; contradictory identities must not be guessed.
    identities = {(r['Sequence'], r['Organism ID']) for r in candidates}
    if len(identities) != 1:
        raise ValueError(f'UniSave {accession}: {len(identities)} verified historical identities; unresolved/ambiguous')
    return candidates[0]


def rebuild_family(task):
    source, destination, database, cache, thresholds = task
    source, destination = Path(source), Path(destination)
    name = source.name
    report = destination.parent / 'reports' / (name + '.json')
    ready = destination / name
    marker = ready / 'rebuild_provenance.json'
    try:
        if marker.exists():
            validate_rebuild(ready)
            return {'family': name, 'status': 'passed', 'taxa': len(json.loads(marker.read_text())['proteins']), 'cached': True}
        if ready.exists():
            raise ValueError('Incomplete output family; do not overwrite it')
        paths = sorted((source / 'structs').glob('*.pdb'))
        if len(paths) < 4:
            raise ValueError('Fewer than four existing experimental chains; no substitution with AlphaFold')
        metadata = source / 'sequence_dataset.csv'
        if not metadata.is_file():
            metadata = source / 'finalset.csv'
        original_rows = []
        if metadata.is_file():
            with metadata.open(newline='') as stream:
                original_rows = list(csv.DictReader(stream))
        rows, proteins = [], {}
        with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as db:
            for pdb in paths:
                accession = pdb.stem
                if not SAFE.fullmatch(accession):
                    raise ValueError('Unsafe chain identifier')
                observed = observed_sequence(pdb)
                candidates = [r for r in original_rows if r.get('query') == accession or r.get('Entry') == accession]
                row = dict(candidates[0]) if len(candidates) == 1 else uniprot(accession, cache, observed, thresholds)
                reference = row.get('Sequence', '').upper()
                try:
                    mapping = map_sequence(reference, observed, **thresholds)
                except ValueError:
                    # Refresh only a suspect/obsolete reference, never change the structure's identity.
                    row = uniprot(accession, cache, observed, thresholds)
                    reference = row['Sequence'].upper()
                    try:
                        mapping = map_sequence(reference, observed, **thresholds)
                    except ValueError:
                        row = uniprot(accession, cache, observed, thresholds, historical=True)
                        reference = row['Sequence'].upper()
                        mapping = map_sequence(reference, observed, **thresholds)
                if row.get('Organism ID'):
                    own_taxid = canonical_taxid(db, int(row['Organism ID']))
                    parent = db.execute('SELECT parent FROM species WHERE taxid=?', (own_taxid,)).fetchone()[0]
                    lineage = canonical_lineage(db, int(parent))
                else:
                    raw = row.get('Taxonomic lineage (Ids)', '').split(',')[-1]
                    match = re.fullmatch(r'\s*(\d+)(?:\s*\([^)]*\))?\s*', raw)
                    if not match:
                        raise ValueError('Missing reference taxonomy; cannot invent a lineage')
                    lineage = canonical_lineage(db, int(match[1]))
                row.update(query=accession, Entry=accession, Sequence=observed,
                           Sequence_source='observed_pdb_residues', UniProt_sequence=reference)
                row['Taxonomic lineage (Ids)'] = lineage
                rows.append(row)
                proteins[accession] = {'source_pdb': str(pdb.resolve()), 'source_pdb_sha256': digest(pdb),
                    'reference_sequence': reference, 'observed_sequence': observed,
                    'lineage': lineage, 'mapping': mapping,
                    'reference_source': row.get('reference_source', str(metadata.resolve())),
                    'reference_sha256': row.get('reference_sha256') or digest(metadata),
                    'reference_evidence': row.get('reference_evidence', {})}
        ready.mkdir(parents=True)
        (ready / 'structs').mkdir()
        for pdb in paths:
            (ready / 'structs' / pdb.name).symlink_to(pdb.resolve())
        fields = sorted(set().union(*(r.keys() for r in rows)))
        for file_name in ('sequence_dataset.csv', 'finalset.csv'):
            with (ready / file_name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        (ready / 'identifiers.txt').write_text('\n'.join(p.stem for p in paths) + '\n')
        (ready / 'sequences.fasta').write_text(''.join(f'>{r["query"]}\n{r["Sequence"]}\n' for r in rows))
        write_json(marker, {'schema': 'cath_observed_sequence_rebuild_v1', 'family': name,
                   'source_metadata': str(metadata.resolve()) if metadata.is_file() else None,
                   'source_metadata_sha256': digest(metadata) if metadata.is_file() else None,
                   'metadata_sha256': digest(ready / 'finalset.csv'), 'taxonomy_db': str(database),
                   'thresholds': thresholds, 'proteins': proteins, 'structures_modified': False,
                   'all_existing_structure_taxa_retained': True})
        validate_rebuild(ready)
        result = {'family': name, 'status': 'passed', 'taxa': len(rows)}
    except (ValueError, FileNotFoundError, KeyError, urllib.error.HTTPError) as error:
        if isinstance(error, urllib.error.HTTPError) and error.code not in {404, 410}:
            raise
        result = {'family': name, 'status': 'failed', 'reason': str(error),
                  'failed_accession': locals().get('accession')}
    write_json(report, result)
    return result


def validate_rebuild(directory):
    """Require independent reference-mapping evidence, not just overwritten CSV sequences."""
    directory = Path(directory)
    data = json.loads((directory / 'rebuild_provenance.json').read_text())
    subset = data['schema'] == 'cath_validated_subset_v1'
    if not subset and (data['schema'] != 'cath_observed_sequence_rebuild_v1' or not data['all_existing_structure_taxa_retained']):
        raise ValueError('Unsupported/incomplete CATH rebuild evidence')
    if subset:
        source_names = set(data['original_taxa'])
        retained = set(data['proteins'])
        excluded = set(data['excluded_taxa'])
        if retained & excluded or retained | excluded != source_names or len(retained) < 4:
            raise ValueError('Invalid explicit CATH subset partition')
        if digest(data['audit_source']) != data['audit_sha256']:
            raise ValueError('CATH subset audit evidence changed')
    if digest(directory / 'finalset.csv') != data['metadata_sha256']:
        raise ValueError('Rebuilt CATH metadata changed')
    paths = sorted((directory / 'structs').glob('*.pdb'))
    if {p.stem for p in paths} != set(data['proteins']):
        raise ValueError('Rebuild evidence does not cover all chains')
    for pdb in paths:
        entry = data['proteins'][pdb.stem]
        if digest(pdb) != entry['source_pdb_sha256'] or digest(entry['reference_source']) != entry['reference_sha256']:
            raise ValueError('Experimental chain or independent reference changed')
        for evidence, sha256 in entry.get('reference_evidence', {}).items():
            if digest(evidence) != sha256:
                raise ValueError('UniProt accession-resolution evidence changed')
        mapping = entry['mapping']
        reference, observed = entry['reference_sequence'], entry['observed_sequence']
        indices = mapping['reference_positions']
        pairs = [(i, j) for j, i in enumerate(indices) if i is not None]
        if len(indices) != len(observed) or any(not 0 <= i < len(reference) for i, _ in pairs):
            raise ValueError('Invalid independent residue mapping')
        if any(a[0] >= b[0] for a, b in zip(pairs, pairs[1:])):
            raise ValueError('Non-monotonic residue mapping')
        matches = sum(reference[i] == observed[j] for i, j in pairs)
        thresholds = data['thresholds']
        if (not pairs or matches < thresholds['min_matches']
                or matches / len(pairs) < thresholds['min_identity']
                or len(pairs) / len(observed) < thresholds['min_coverage']):
            raise ValueError('Independent chain-identity evidence fails')
    return data


def rebuild(source, destination, database, workers, thresholds):
    source, destination, database = map(lambda p: Path(p).resolve(), (source, destination, database))
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Rebuild must not overlap the original CATH root')
    destination.mkdir(parents=True, exist_ok=True)
    families = sorted(p for p in source.iterdir() if p.is_dir() and re.fullmatch(r'\d+(?:\.\d+){3}', p.name))
    tasks = [(str(p), str(destination), str(database), str(destination.parent / 'uniprot-cache'), thresholds) for p in families]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(rebuild_family, tasks):
            results.append(result)
            print(json.dumps(result), flush=True)
    summary = {'families': len(results), 'passed': sum(r['status'] == 'passed' for r in results),
               'failed': sum(r['status'] == 'failed' for r in results), 'results': results,
               'taxonomy_sha256': digest(database), 'source_root': str(source), 'output_root': str(destination),
               'thresholds': thresholds, 'originals_modified': False}
    write_json(destination.parent / 'rebuild_summary.json', summary)
    if summary['passed'] == 0:
        raise ValueError('No CATH families passed independent reconstruction checks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--taxonomy-dir', type=Path, required=True)
    parser.add_argument('--taxonomy-db', type=Path, help='Reuse an existing verified snapshot without modifying it')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--min-identity', type=float, default=.95)
    parser.add_argument('--min-coverage', type=float, default=.8)
    parser.add_argument('--min-matches', type=int, default=20)
    args = parser.parse_args()
    if args.workers < 1 or not 0 < args.min_identity <= 1 or not 0 < args.min_coverage <= 1 or args.min_matches < 1:
        parser.error('Invalid rebuild resources/identity gates')
    if args.taxonomy_db:
        database = args.taxonomy_db.resolve()
        snapshot = json.loads(database.with_name('snapshot.json').read_text())
        if digest(database) != snapshot['sqlite_sha256']:
            raise ValueError('Reused taxonomy snapshot changed')
    else:
        database = taxonomy(args.taxonomy_dir)
    rebuild(args.source, args.out, database, args.workers,
            {'min_identity': args.min_identity, 'min_coverage': args.min_coverage, 'min_matches': args.min_matches})


if __name__ == '__main__':
    main()
