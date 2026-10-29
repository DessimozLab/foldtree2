import csv
import json
from io import StringIO
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from rebuild_cath_families import canonical_lineage, map_sequence, rebuild_family, validate_rebuild, uniprot, historical_records


def historical_fixture(annotation=''):
    return ('ID   TEST_TEST               Reviewed;        30 AA.\n'
        'AC   A0A023T3X2; P12345;\nOS   Test organism\nOS   (strain X).\n'
        'OX   NCBI_TaxID=3 {ECO:0000313|EMBL:TEST};\n'
        'DT   02-SEP-2026, entry version 118.\n' + annotation +
        'SQ   SEQUENCE   30 AA;  3000 MW;  0000000000000000 CRC64;\n'
        '     AAAAAAAAAA AAAAAAAAAA AAAAAAAAAA\n//\n')


@pytest.mark.parametrize('annotation', [
    'FT   DOMAIN         1     30       Legacy annotation.\n',
    'FT   VARIANT         1\nFT                   /note="A -> V"\n',
    'RM   New reference metadata.\n',
])
def test_historical_annotations_do_not_break_identity_reader(annotation):
    records = list(historical_records(StringIO(historical_fixture(annotation))))
    assert records[0]['accessions'] == ['A0A023T3X2', 'P12345']
    assert records[0]['taxids'] == ['3']
    assert records[0]['sequence'] == 'A'*30
    assert records[0]['organism'] == 'Test organism (strain X).'
    assert records[0]['entry_version'] == '118'


@pytest.mark.parametrize('transform', [
    lambda text: text.replace('30 AA;', '31 AA;'),
    lambda text: text.replace('//\n', ''),
    lambda text: text.replace('AAAAAAAAAA\n//', 'AAAAAA*AAA\n//'),
])
def test_historical_reader_rejects_corrupted_sequence_records(transform):
    with pytest.raises(ValueError, match='Historical record format'):
        list(historical_records(StringIO(transform(historical_fixture()))))


def test_historical_reader_keeps_record_boundaries():
    assert len(list(historical_records(StringIO(historical_fixture()*2)))) == 2


def test_inactive_successor_retains_original_id_and_hash_bound_chain(tmp_path):
    inactive = tmp_path / 'A0A023T3X2.json'
    inactive.write_text(json.dumps({'entryType': 'Inactive', 'inactiveReason': {'mergeDemergeTo': ['A0ACD6B8R4']}}))
    successor = tmp_path / 'A0ACD6B8R4.json'
    successor.write_text(json.dumps({'primaryAccession': 'A0ACD6B8R4', 'sequence': {'value': 'A'*30},
        'organism': {'scientificName': 'Test organism', 'taxonId': 3}}))
    result = uniprot('A0A023T3X2', tmp_path, 'A'*30)
    assert result['Entry'] == 'A0A023T3X2'
    assert result['UniProt_primary_accession'] == 'A0ACD6B8R4'
    assert str(inactive) in result['reference_evidence']


def test_ambiguous_successor_is_not_selected_by_order(tmp_path):
    (tmp_path / 'A0A023T3X2.json').write_text(json.dumps({'entryType': 'Inactive',
        'inactiveReason': {'mergeDemergeTo': ['A0ACD6B8R4', 'A0ACD6B8R5']}}))
    for accession in ('A0ACD6B8R4', 'A0ACD6B8R5'):
        (tmp_path / (accession+'.json')).write_text(json.dumps({'primaryAccession': accession,
            'sequence': {'value': 'A'*30}, 'organism': {'scientificName': 'Test', 'taxonId': 3}}))
    with pytest.raises(ValueError, match='ambiguous'):
        uniprot('A0A023T3X2', tmp_path, 'A'*30)


def test_deleted_entry_uses_verified_historical_sequence_and_taxonomy(tmp_path):
    (tmp_path / 'A0A023T3X2.json').write_text(json.dumps({'entryType': 'Inactive',
        'inactiveReason': {'inactiveReasonType': 'DELETED'}}))
    (tmp_path / 'A0A023T3X2.unisave.txt').write_text(
        'ID   TEST_TEST               Unreviewed;        30 AA.\n'
        'AC   A0A023T3X2;\nOS   Test organism.\nOX   NCBI_TaxID=3;\n'
        'SQ   SEQUENCE   30 AA;  3000 MW;  0000000000000000 CRC64;\n'
        '     AAAAAAAAAA AAAAAAAAAA AAAAAAAAAA\n//\n')
    result = uniprot('A0A023T3X2', tmp_path, 'A'*30)
    assert result['Sequence'] == 'A'*30
    assert result['Organism ID'] == '3'
    assert result['Reference_kind'] == 'UniSave historical'


def test_tags_and_truncations_are_verified_without_exact_substring():
    reference = 'MIKTLDSNILREVGTLSRAVNSINDIKYKELKLQKGQFTFLTRICENPGINLVELSNMLKVDKATTTKAIQKLIKAGYVDKKQDKFDKRGYNLTPTDKSLEVYELIIEEENRSIEICFDNFTDEEKQVVTKLLEKMSKNVENEWFKVKR'
    observed = 'RGSHML' + reference[1:-2]
    assert observed not in reference
    mapping = map_sequence(reference, observed)
    assert mapping['identity'] >= .95
    assert mapping['observed_coverage'] >= .8
    assert len(mapping['reference_positions']) == len(observed)
    with pytest.raises(ValueError, match='Unverified chain identity'):
        map_sequence(reference, 'W' * 100)
    with pytest.raises(ValueError, match='no local alignment'):
        map_sequence('A' * 30, 'W' * 30)


def test_rebuild_does_not_report_disk_failure_as_an_excluded_family(tmp_path, monkeypatch):
    import rebuild_cath_families as rebuild
    def failed_validation(path):
        raise OSError(28, 'No space left on device')
    source = tmp_path / 'original/1.2.3.4'
    ready = tmp_path / 'rebuilt/CATH_data/1.2.3.4'
    ready.mkdir(parents=True)
    (ready / 'rebuild_provenance.json').write_text('{}')
    monkeypatch.setattr(rebuild, 'validate_rebuild', failed_validation)
    with pytest.raises(OSError, match='No space left'):
        rebuild.rebuild_family((source, ready.parent, tmp_path / 'taxonomy.sqlite', tmp_path / 'cache', {}))
    assert not (ready.parent.parent / 'reports/1.2.3.4.json').exists()


def test_taxonomy_merges_are_resolved_and_root_is_not_invented(tmp_path):
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE species (taxid INTEGER, parent INTEGER, rank TEXT)')
        db.execute('CREATE TABLE merged (taxid_old INTEGER, taxid_new INTEGER)')
        db.executemany('INSERT INTO species VALUES (?,?,?)', [(1,1,'no rank'), (2,1,'domain'), (3,2,'genus')])
        db.execute('INSERT INTO merged VALUES (30,3)')
        assert canonical_lineage(db, 30) == '2 (domain), 3 (genus)'
        with pytest.raises(ValueError, match='Unresolved taxonomy'):
            canonical_lineage(db, 300)


def test_rebuild_keeps_chains_and_requires_independent_evidence(tmp_path):
    source = tmp_path / 'original/1.2.3.4'
    (source / 'structs').mkdir(parents=True)
    rows = []
    for name in ('a', 'b', 'c', 'd'):
        atoms = ''.join(f'ATOM  {i:5d}  CA  ALA A{i:4d}    {float(i):8.3f}{0.:8.3f}{0.:8.3f}  1.00 80.00           C  \n' for i in range(1, 31))
        (source / 'structs' / (name + '.pdb')).write_text(atoms + 'END\n')
        rows.append({'query': name, 'Entry': name, 'Sequence': 'M' + 'A' * 30 + 'K',
                     'Taxonomic lineage (Ids)': '2 (domain), 3 (genus)'})
    with (source / 'sequence_dataset.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    database = tmp_path / 'taxonomy.sqlite'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE species (taxid INTEGER, parent INTEGER, rank TEXT)')
        db.execute('CREATE TABLE merged (taxid_old INTEGER, taxid_new INTEGER)')
        db.executemany('INSERT INTO species VALUES (?,?,?)', [(1,1,'no rank'), (2,1,'domain'), (3,2,'genus')])
    destination = tmp_path / 'rebuilt/CATH_data'
    thresholds = {'min_identity': .95, 'min_coverage': .8, 'min_matches': 20}
    result = rebuild_family((str(source), str(destination), str(database), str(tmp_path / 'cache'), thresholds))
    assert result['status'] == 'passed'
    ready = destination / source.name
    assert (ready / 'structs/a.pdb').is_symlink()
    data = validate_rebuild(ready)
    assert data['structures_modified'] is False
    assert set(data['proteins']) == {'a', 'b', 'c', 'd'}
    with (ready / 'finalset.csv').open() as stream:
        rebuilt = list(csv.DictReader(stream))
    assert rebuilt[0]['Sequence'] == 'A' * 30
    assert rebuilt[0]['UniProt_sequence'] == 'M' + 'A' * 30 + 'K'
    # The preserved reference file is independently hash-verified, not blindly trusted.
    (source / 'sequence_dataset.csv').write_text('corrupted reference')
    with pytest.raises(ValueError, match='independent reference changed'):
        validate_rebuild(ready)
