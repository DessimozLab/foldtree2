import csv
from pathlib import Path
import sqlite3
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from audit_cath_chains import alignment_metrics, audit_family


def test_metrics_report_failures_without_lowering_gate():
    good = alignment_metrics('M'+'A'*30+'K', 'A'*30)
    assert good['passes_sequence_gate']
    assert good['matches'] == 30
    bad = alignment_metrics('A'*30, 'W'*30)
    assert not bad['passes_sequence_gate']
    assert bad['identity'] == 0
    assert bad['observed_coverage'] == 0


def test_audit_visits_all_chains_and_keeps_inputs_unchanged(tmp_path):
    family = tmp_path / '1.2.3.4'
    (family / 'structs').mkdir(parents=True)
    rows = []
    for name in ('a','b','c','d'):
        pdb = family / 'structs' / (name+'.pdb')
        pdb.write_text(''.join(f'ATOM  {i:5d}  CA  ALA A{i:4d}    {float(i):8.3f}{0.:8.3f}{0.:8.3f}  1.00 80.00           C  \n' for i in range(1,31))+'END\n')
        rows.append({'query':name, 'Entry':name, 'Sequence': 'W'*30 if name=='a' else 'A'*30,
                     'Taxonomic lineage (Ids)':'3 (genus)'})
    metadata = family/'sequence_dataset.csv'
    with metadata.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    before={p:p.read_bytes() for p in family.rglob('*') if p.is_file()}
    database = tmp_path/'taxa.sqlite'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE species (taxid INTEGER, parent INTEGER, rank TEXT)')
        db.execute('CREATE TABLE merged (taxid_old INTEGER, taxid_new INTEGER)')
    result=audit_family((str(family),str(tmp_path/'cache'),str(database),False))
    assert len(result['chains'])==4
    assert result['chains'][0]['status']=='sequence_gate_failed'
    assert all(c['status']=='sequence_verified' for c in result['chains'][1:])
    assert result['inclusion_changed'] is False
    assert all(p.read_bytes()==data for p,data in before.items())


def test_audit_does_not_turn_infrastructure_errors_into_chain_exclusions(tmp_path, monkeypatch):
    import audit_cath_chains as audit
    family = tmp_path / '1.2.3.4'
    (family / 'structs').mkdir(parents=True)
    (family / 'structs/a.pdb').write_text('fixture')
    database = tmp_path / 'taxa.sqlite'
    sqlite3.connect(database).close()
    monkeypatch.setattr(audit, 'observed_sequence', lambda path: 'A' * 30)
    def failed_reference(*args):
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(audit, 'cached_file', failed_reference)
    with pytest.raises(OSError, match='No space left'):
        audit.audit_family((str(family), str(tmp_path / 'cache'), str(database), True))
