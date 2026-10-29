import csv
import gzip
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import scripts.cath.generate_cath_families as generation


def taxonomy(path):
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE species (taxid INTEGER,parent INTEGER,rank TEXT)')
        db.execute('CREATE TABLE merged (taxid_old INTEGER,taxid_new INTEGER)')
        db.executemany('INSERT INTO species VALUES (?,?,?)',[(1,1,'root'),(2,1,'domain'),
            *[(g,2,'genus') for g in (3,4,5,6)],*[(g*10,g,'species') for g in (3,4,5,6)],
            (99,2,'species')])
        db.execute('INSERT INTO merged VALUES (300,30)')


def test_genus_uses_canonical_rank_not_species_or_name(tmp_path):
    path=tmp_path/'taxa.sqlite';taxonomy(path)
    with sqlite3.connect(path) as db:
        assert generation.genus_of(db,300)==3
        assert generation.genus_of(db,30)==generation.genus_of(db,3)
        with pytest.raises(ValueError,match='No classified'):
            generation.genus_of(db,99)


def test_plan_deduplicates_genus_and_keeps_chain_pairing(tmp_path,monkeypatch):
    database=tmp_path/'taxa.sqlite';taxonomy(database)
    source=tmp_path/'source';source.mkdir()
    cath_rows=[];tax_rows=[];domain_rows=[]
    for i,(genus,resolution) in enumerate([(3,2.),(3,1.),(4,1.),(5,1.),(6,1.)]):
        pdb=f'{i+1:04d}';domain=pdb+'A00';accession=f'P{i+1:05d}'
        domain_rows.append(f'{domain} 1 2 3 4 0 0 0 0 30 {resolution}')
        cath_rows.append({'PDB':pdb,'CHAIN':'A','SP_PRIMARY':accession,'CATH_ID':domain})
        tax_rows.append({'PDB':pdb,'CHAIN':'A','TAX_ID':genus*10,'SCIENTIFIC_NAME':'Test'})
    (source/'cath-domain-list.txt').write_text('\n'.join(domain_rows)+'\n')
    for filename,rows in [('pdb_chain_cath_uniprot.csv.gz',cath_rows),('pdb_chain_taxonomy.csv.gz',tax_rows)]:
        with gzip.open(source/filename,'wt',newline='') as stream:
            stream.write('# snapshot\n');writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    def download(url,target):
        target.write_bytes((source/target.name).read_bytes())
    monkeypatch.setattr(generation,'download',download)
    root=tmp_path/'run';root.mkdir()
    result=generation.plan(root,database.resolve())
    family=result['families'][0]
    assert len(family['genera'])==4
    assert family['genera']['3'][0]['accession']=='P00002'
    assert family['genera']['3'][0]['pdb']=='0002'
    assert family['genera']['3'][0]['chain']=='A'
    assert result['exclusions']['within_genus_alternative_chains']==1
    # Generate a family with both alternatives present; still select only one genus-3 tip.
    from Bio.PDB import PDBParser
    template=tmp_path/'template.pdb'
    template.write_text(''.join(f'ATOM  {i:5d}  CA  ALA A{i:4d}    {float(i):8.3f}{0.:8.3f}{0.:8.3f}  1.00 80.00           C  \n' for i in range(1,31))+'END\n')
    def experimental(path,candidate):
        return PDBParser(QUIET=True).get_structure('test',str(template))[0]['A'],'A'*30
    def cached(root,pdb):
        path=root/(pdb+'.cif');path.write_text('fixture observed chain');return path
    monkeypatch.setattr(generation,'experimental_chain',experimental)
    monkeypatch.setattr(generation,'cached_structure',cached)
    (root/'CATH_data').mkdir()
    generated=generation.generate_family((str(root),family))
    assert generated['retained']==4
    proof,_=generation.validate_generated(root/'CATH_data/1.2.3.4')
    assert len({p['candidate']['genus'] for p in proof['proteins'].values()})==4
    assert 'P00001' not in proof['proteins']
    assert proof['missing_residues_filled'] is False


@pytest.mark.parametrize('error', [
    OSError(28, 'No space left on device'),
    MemoryError('Allocation failed'),
    RuntimeError('50 GiB structure cache budget reached'),
])
def test_infrastructure_failures_are_not_scientific_exclusions(tmp_path, monkeypatch, error):
    family = {'family': '1.2.3.4', 'genera_available': 4,
              'genera': {'3': [{'accession': 'P00001', 'pdb': '0001'}]}}
    def failed_download(root, pdb):
        raise error
    monkeypatch.setattr(generation, 'cached_structure', failed_download)
    with pytest.raises(type(error)) as caught:
        generation.generate_family((str(tmp_path), family))
    assert caught.value is error
    assert not (tmp_path / 'reports/1.2.3.4.json').exists()


@pytest.mark.parametrize('code', [404, 410, 429, 503])
def test_only_missing_http_structures_are_candidate_exclusions(tmp_path, monkeypatch, code):
    import urllib.error
    family = {'family': '1.2.3.4', 'genera_available': 4,
              'genera': {'3': [{'accession': 'P00001', 'pdb': '0001'}]}}
    def failed_download(root, pdb):
        raise urllib.error.HTTPError('https://example.invalid', code, 'fixture', {}, None)
    monkeypatch.setattr(generation, 'cached_structure', failed_download)
    if code in {404, 410}:
        result = generation.generate_family((str(tmp_path), family))
        assert result['status'] == 'fewer_than_four_successful_genera'
        assert len(result['candidate_failures']) == 1
    else:
        with pytest.raises(urllib.error.HTTPError):
            generation.generate_family((str(tmp_path), family))
        assert not (tmp_path / 'reports/1.2.3.4.json').exists()


def test_full_structure_cache_stops_before_download_or_reservation(tmp_path, monkeypatch):
    cache = tmp_path / 'structures'
    cache.mkdir()
    budget = {'reserved_or_used_bytes': 50 * 1024**3}
    (cache / 'storage.json').write_text(json.dumps(budget))
    def unexpected_download(*args, **kwargs):
        pytest.fail('No download is allowed beyond the cache budget')
    monkeypatch.setattr(generation.urllib.request, 'urlopen', unexpected_download)
    with pytest.raises(RuntimeError, match='50 GiB structure cache budget'):
        generation.cached_structure(tmp_path, '1abc')
    assert json.loads((cache / 'storage.json').read_text()) == budget


def test_chain_alias_does_not_mutate_deposited_model(tmp_path, monkeypatch):
    from Bio.PDB import PDBParser
    from types import SimpleNamespace
    import warnings
    template = tmp_path / 'two-chains.pdb'
    template.write_text(''.join(
        f'ATOM  {offset+i:5d}  CA  ALA {chain}{i:4d}    {float(i):8.3f}{0.:8.3f}{0.:8.3f}  1.00 80.00           C  \n'
        for chain, offset in [('A', 0), ('C', 30)] for i in range(1, 31)) + 'END\n')
    structure = PDBParser(QUIET=True).get_structure('test', str(template))
    monkeypatch.setattr('Bio.PDB.MMCIFParser', lambda **kwargs:
                        SimpleNamespace(get_structure=lambda *args: structure))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        chain, sequence = generation.experimental_chain(template, {'pdb': 'test', 'chain': 'C'})
    assert chain.id == 'A' and chain.parent is None and sequence == 'A' * 30
    assert structure[0]['C'].id == 'C'
    assert len(structure[0]['A']) == 30
