"""Common OMA inputs, native projections, missing states and matched folds."""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import benchmark_cohort as cohort
import alphabet_information_metrics as metrics
from benchmark_common import read_fasta, write_fasta
from compare_oma_representations import controlled_alignments, shared_occupancy


@pytest.mark.parametrize('representation,key', [('AA', 'AA_1'), ('3Di', '3Di_1'), ('FT2_20', 'native_20_1')])
def test_native_reuse_follows_inherited_artifact_paths(tmp_path, representation, key):
    root = tmp_path / 'sweep'
    root.mkdir()
    inherited = tmp_path / 'earlier_pilot' / representation / '1'
    (root / 'queue_status.json').write_text(json.dumps({'jobs': {
        key: {'status': 'completed', 'artifacts': str(inherited)}}}))
    assert cohort.native_directory(root, representation, '1') == inherited


def test_native_reuse_follows_chain_and_rejects_cycles(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir()
    new.mkdir()
    artifact = old / 'native' / 'FT2_20' / '1'
    (old / 'queue_status.json').write_text(json.dumps({'jobs': {
        'native_20_1': {'artifacts': str(artifact)}}}))
    (new / 'queue_status.json').write_text(json.dumps({'jobs': {}, 'protocol': {
        'reuse_native_root': str(old)}}))
    assert cohort.native_directory(new, 'FT2_20', '1') == artifact
    (old / 'queue_status.json').write_text(json.dumps({'jobs': {}, 'protocol': {
        'reuse_native_root': str(new)}}))
    with pytest.raises(ValueError, match='Cycle'):
        cohort.native_directory(new, 'FT2_20', '1')


def fixture_family(tmp_path):
    root = tmp_path / 'oma'
    markers = root / 'marker_genes'
    markers.mkdir(parents=True)
    (markers / 'OMAGroup_1.fa').write_text('>OMA1\nAC\n')
    (markers / 'OMAGroup_2.fa').write_text('>OMA2\nAC\n')
    family = root / '1'
    (family / 'structs').mkdir(parents=True)
    aa = {name + '|SPEC': 'A-C' for name in 'ABCD'}
    write_fasta(family / 'sequences.aligned.fa', aa)
    (family / 'raxml_lg_tree.raxml.bestTree').write_text('(A|SPEC:1,B|SPEC:1,(C|SPEC:1,D|SPEC:1):1);')
    for name in 'ABCDE':
        (family / 'structs' / (name + '.pdb')).write_text('fixture structure ' + name)
    return root


def test_freeze_is_oma_defined_and_records_taxon_exclusions(tmp_path):
    root = fixture_family(tmp_path)
    path = tmp_path / 'cohort.json'
    record = cohort.freeze(root, path)
    assert record['requested_family_ids'] == ['1', '2']
    assert [f['family'] for f in record['families']] == ['1']
    assert record['families'][0]['excluded_nonreference_pdbs'] == ['E.pdb']
    assert record['exclusions'][0]['family'] == '2'
    assert cohort.load(path) == record
    families, selected = cohort.select_families(root, path)
    assert families == [root / '1']
    (root / '1/structs/A.pdb').write_text('changed structure')
    with pytest.raises(ValueError, match='changed'):
        cohort.load(path)


def test_projection_has_common_columns_but_does_not_invent_residues(tmp_path):
    root = fixture_family(tmp_path)
    record = cohort.freeze(root, tmp_path / 'cohort.json')['families'][0]
    native = tmp_path / 'native/3di/1'
    native.mkdir(parents=True)
    write_fasta(native / 'aa.aligned.fasta', {n: 'A-C' for n in 'ABCD'})
    write_fasta(native / '3di.aligned.fasta', {n: 'D-W' for n in 'ABCD'})
    frame = pd.DataFrame([{'family': '1', 'models': 'FT2_20', 'id': n, 'seq': '01'} for n in 'ABCD'])
    projected = controlled_alignments(record, frame, tmp_path / 'native')
    assert set(projected) == {'AA', '3Di', 'FT2_20'}
    assert projected['FT2_20']['A|SPEC'] == '0-1'
    supports = {'AA': list('AC'), '3Di': list('DW'), 'FT2_20': list('01')}
    np.testing.assert_array_equal(shared_occupancy(projected, supports), [True, False, True])
    write_fasta(native / 'aa.aligned.fasta', {n: 'A-D' for n in 'ABCD'})
    with pytest.raises(ValueError, match='correspondence'):
        controlled_alignments(record, frame, tmp_path / 'native')


def test_unknown_characters_are_not_bridged_in_contexts():
    unigrams, contexts, transitions = metrics.train_markov_counts(['AXA'], ['A'], {'A': 0}, 1)
    assert unigrams['A'] == 2
    assert not contexts[1] and not transitions[1]
    probabilities, _ = metrics.train_ngram_model(['AA'], ['A'], 1)
    _, n_encoded, _ = metrics.encode_bits(['AXA'], ['A'], 1, probabilities)
    assert n_encoded == 0


def test_kmer_folds_are_identical_across_representations(monkeypatch):
    actual = metrics.make_stratified_folds
    observed = []
    def capture(indices, n_folds, rng):
        folds = actual(indices, n_folds, rng)
        observed.append(folds)
        return folds
    monkeypatch.setattr(metrics, 'make_stratified_folds', capture)
    rows = [{'models': model, 'family': family, 'id': ident, 'seq': 'ACAC'}
            for model in ['AA', '3Di'] for family in ['2', '1'] for ident in ['D', 'C', 'B', 'A']]
    metrics.run_kmer_fold_discrimination(pd.DataFrame(rows), n_folds=2, seed=42)
    assert observed[0] == observed[1]
