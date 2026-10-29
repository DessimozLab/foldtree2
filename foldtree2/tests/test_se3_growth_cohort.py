import pytest
from copy import deepcopy
from foldtree2.se3_cache import identity_hash
from scripts.se3.prepare_se3_growth_cohort import growth_splits
from scripts.se3.submit_se3_growth_trial import check_growth


def test_growth_preserves_holdout_and_excludes_used_ids(monkeypatch):
    retained = {'train_ids': ['train'], 'val_ids': ['val'], 'excluded_prior_ids': ['earlier']}
    def select(dataset, excluded, count, val_count, max_residues, seed):
        assert excluded == {'train', 'val', 'earlier'}
        assert (count, val_count, max_residues, seed) == (2, 0, 256, 42)
        return ['new1', 'new2'], [], [{'id': 'oversized', 'reason': 'length'}]
    monkeypatch.setattr('scripts.se3.prepare_se3_growth_cohort.fresh_splits', select)
    train, val, skipped = growth_splits(None, retained, 3)
    assert train == ['train', 'new1', 'new2']
    assert val == ['val']
    assert skipped[0]['reason'] == 'length'
    assert retained['train_ids'] == ['train']


@pytest.mark.parametrize('retained,count', [
    ({'train_ids': ['same'], 'val_ids': ['same'], 'excluded_prior_ids': []}, 2),
    ({'train_ids': ['train'], 'val_ids': ['val'], 'excluded_prior_ids': []}, 1),
])
def test_growth_rejects_overlapping_or_non_growing_splits(retained, count):
    with pytest.raises(ValueError):
        growth_splits(None, retained, count)


def manifests():
    previous = {'train_ids': [f't{i}' for i in range(4096)],
                'val_ids': [f'v{i}' for i in range(512)], 'input_identity': {'production': 'frozen'},
                'scale': 10., 'calibration_sha256': 'cal', 'calibration_ids': ['cal1'],
                'feature_dim': 48, 'files': {f't{i}': {'sha256': str(i)} for i in range(4096)}}
    current = deepcopy(previous)
    current['train_ids'] += [f'n{i}' for i in range(4096)]
    current['bundle_sha256'] = identity_hash(current)
    return previous, current


def test_growth_manifest_accepts_retained_inputs():
    previous, current = manifests()
    check_growth(previous, current)


@pytest.mark.parametrize('mutation', ['holdout', 'entry', 'identity'])
def test_growth_manifest_rejects_changed_retained_inputs(mutation):
    previous, current = manifests()
    if mutation == 'holdout':
        current['val_ids'][0] = 'replacement'
    elif mutation == 'entry':
        current['files']['t0']['sha256'] = 'changed'
    else:
        current['input_identity']['production'] = 'changed'
    current.pop('bundle_sha256')
    current['bundle_sha256'] = identity_hash(current)
    with pytest.raises(ValueError):
        check_growth(previous, current)
