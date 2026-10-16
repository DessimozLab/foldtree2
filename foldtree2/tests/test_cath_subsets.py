import csv
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from audit_cath_chains import audit_family
from retain_cath_subsets import process_family, stats
from rebuild_cath_families import validate_rebuild


def test_exclusion_statistics_include_zeros_and_use_population_std():
    assert stats([0,2]) == {'n':2,'mean':1,'std_population':1,'median':1,'min':0,'max':2}


def test_validated_subset_retains_four_tips_and_records_removed_chain(tmp_path):
    source=tmp_path/'original/1.2.3.4';(source/'structs').mkdir(parents=True)
    rows=[]
    for name in 'abcde':
        (source/'structs'/(name+'.pdb')).write_text(''.join(
            f'ATOM  {i:5d}  CA  ALA A{i:4d}    {float(i):8.3f}{0.:8.3f}{0.:8.3f}  1.00 80.00           C  \n'
            for i in range(1,31))+'END\n')
        rows.append({'query':name,'Entry':name,'Sequence':'W'*30 if name=='a' else 'A'*30,
                     'Taxonomic lineage (Ids)':'2 (domain), 3 (genus)'})
    with (source/'sequence_dataset.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dbfile=tmp_path/'taxa.sqlite'
    with sqlite3.connect(dbfile) as db:
        db.execute('CREATE TABLE species (taxid INTEGER,parent INTEGER,rank TEXT)')
        db.execute('CREATE TABLE merged (taxid_old INTEGER,taxid_new INTEGER)')
        db.executemany('INSERT INTO species VALUES (?,?,?)',[(1,1,'root'),(2,1,'domain'),(3,2,'genus')])
    audit=audit_family((str(source),str(tmp_path/'cache'),str(dbfile),False))
    audits=tmp_path/'audit';audits.mkdir();(audits/'1.2.3.4.json').write_text(json.dumps(audit))
    output=tmp_path/'subset/CATH_data';output.mkdir(parents=True)
    result=process_family(({'family':'1.2.3.4','status':'failed'},str(source.parent),
        str(tmp_path/'rebuilt'),str(audits),str(tmp_path/'cache'),str(dbfile),str(output),str(output.parent)))
    assert result['retained']==4 and result['excluded']==1
    assert result['status']=='retained_validated_subset'
    proof=validate_rebuild(output/'1.2.3.4')
    assert proof['excluded_taxa']=={'a':'sequence_gate_failed'}
    assert set(proof['proteins'])==set('bcde')
    proof['excluded_taxa']={}
    (output/'1.2.3.4/rebuild_provenance.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='subset partition'):
        validate_rebuild(output/'1.2.3.4')
