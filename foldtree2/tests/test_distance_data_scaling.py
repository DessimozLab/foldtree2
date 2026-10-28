import json
import sys
from types import SimpleNamespace

import pytest
import torch

from foldtree2.se3_cache import digest, identity_hash, open_bundle
from scripts.se3.scale_distance_data import choose_splits, subset_report, write_subset
from scripts.se3 import scale_distance_data
from scripts.se3.extend_distance_data import select_additions


def bundle(tmp_path):
    root = tmp_path / 'parent'
    (root / 'entries').mkdir(parents=True)
    manifest = {'train_ids': ['t0', 't1', 't2', 'bad'], 'val_ids': ['v0', 'v1', 'v2'],
                'input_identity': {'synthetic': True}, 'files': {}, 'feature_dim': 8}
    for identifier in manifest['train_ids'] + manifest['val_ids']:
        mask = torch.ones(32, dtype=torch.bool)
        if identifier == 'bad':
            mask[3:] = False
        entry = {'coords': torch.zeros(32, 3), 'mask': mask}
        name = identifier + '.pt'
        torch.save({'id': identifier, 'identity': manifest['input_identity'], 'entry': entry},
                   root / 'entries' / name)
        manifest['files'][identifier] = {'name': name, 'length': 32,
                                         'sha256': digest(root / 'entries' / name)}
    manifest['bundle_sha256'] = identity_hash(manifest)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root, open_bundle(root)


def test_expansion_retains_nested_originals_and_fixed_holdout(tmp_path):
    root, manifest = bundle(tmp_path)
    previous = {'input_identity': manifest['input_identity'], 'train_ids': ['t1'], 'val_ids': ['v2']}
    selected, excluded = choose_splits(root, manifest, previous, 3, 3)
    assert selected['train'][0] == 't1' and selected['val'][0] == 'v2'
    assert set(selected['train']) == {'t0', 't1', 't2'}
    assert not set(selected['train']) & set(selected['val'])
    assert excluded['train'] == [{'id': 'bad', 'reason': 'valid_fraction_below_0.9'}]
    assert choose_splits(root, manifest, previous, 3, 3)[0] == selected
    smaller, _ = choose_splits(root, manifest, previous, 2, 3)
    assert smaller['train'] == selected['train'][:2]
    assert smaller['val'] == selected['val']
    with pytest.raises(ValueError, match='Insufficient'):
        choose_splits(root, manifest, previous, 4, 3)
    with pytest.raises(ValueError, match='original split'):
        choose_splits(root, manifest, {**previous, 'train_ids': ['v0']}, 3, 3)


def test_subset_copies_hashes_preserves_parent_and_rejects_split_movement(tmp_path):
    root, manifest = bundle(tmp_path)
    result = write_subset(root, manifest, tmp_path / 'subset', ['t1', 't0'], ['v2'], {'seed': 42})
    assert result['train_ids'] == ['t1', 't0']
    assert result['parent_bundle_sha256'] == manifest['bundle_sha256']
    assert result['input_identity'] == manifest['input_identity']
    for identifier in result['train_ids'] + result['val_ids']:
        assert result['files'][identifier] == manifest['files'][identifier]
    assert open_bundle(root) == manifest
    with pytest.raises(ValueError, match='original split'):
        write_subset(root, manifest, tmp_path / 'bad', ['v0'], ['t0'], {})


def test_holdout_reports_separate_original_and_added_ids():
    def row(identifier, value):
        return {'id': identifier, 'metrics': {
            k: value for k in ('nll', 'capped_distance_mae_angstrom', 'top_L_precision', 'contact_brier')}}
    report = {'per_chain': [row('old', 1.), row('new', 3.)]}
    assert subset_report(report, ['old'])['nll'] == 1.
    assert subset_report(report, ['new'])['nll'] == 3.
    with pytest.raises(ValueError, match='missing'):
        subset_report(report, ['missing'])


@pytest.mark.parametrize('seconds', ['nan', 'inf', '0', '-1'])
def test_invalid_runtime_bounds_rejected(monkeypatch, seconds):
    monkeypatch.setattr(sys, 'argv', ['scale_distance_data', '--output', 'unused',
                                    '--seconds', seconds])
    with pytest.raises(ValueError, match='bounds'):
        scale_distance_data.main()


def test_orchestration_records_input_failure_and_does_not_train(tmp_path, monkeypatch):
    root, _ = bundle(tmp_path)
    output = tmp_path / 'experiment'
    monkeypatch.setattr(scale_distance_data, 'sources', lambda: {})
    def unexpected(*args, **kwargs):
        pytest.fail('Training must not start with an invalid original panel')
    monkeypatch.setattr(scale_distance_data.subprocess, 'run', unexpected)
    with pytest.raises(ValueError, match='original 256'):
        scale_distance_data.run(SimpleNamespace(
            source=str(root), previous=str(root), output=str(output)))
    assert 'original 256' in json.loads((output / 'failure.json').read_text())['error']


def test_fresh_additions_exclude_prior_ids_and_apply_matching_eligibility():
    def nodes(length=64, confidence=1.):
        return {'node': {'res': {'x': torch.zeros(length, 20)},
                         'coords': {'x': torch.zeros(length, 3).numpy()},
                         'plddt': {'x': torch.full((length,), confidence).numpy()}}}
    structures = {'old_train': nodes(), 'old_val': nodes(), 'fresh_a': nodes(),
                  'fresh_b': nodes(256), 'low_confidence': nodes(confidence=.1),
                  'short': nodes(63), 'long': nodes(257)}
    data = SimpleNamespace(structlist=list(structures), h5dataset={'structs': structures})
    selected, _ = select_additions(data, {'old_train', 'old_val'}, 2)
    assert set(selected) == {'fresh_a', 'fresh_b'}
    assert select_additions(data, {'old_train', 'old_val'}, 2)[0] == selected
    data.structlist.reverse()
    assert select_additions(data, {'old_train', 'old_val'}, 2)[0] == selected
    with pytest.raises(ValueError, match='Insufficient eligible'):
        select_additions(data, {'old_train', 'old_val'}, 3)