import copy
import json
from types import SimpleNamespace

import pytest
import torch

from foldtree2.se3_cache import digest
from foldtree2.src.distance_restraints import DISTANCE_EDGES, eligible_pairs
from scripts.se3.reconstruct_distance_restraints import (
    MODES, reconstruction_objective, run, sources, validate_export,
)


def sample():
    generator = torch.Generator().manual_seed(31)
    mask = torch.ones(32, dtype=torch.bool)
    index = torch.arange(32)
    pairs = eligible_pairs(mask)
    entry = {'id': 'val', 'mask': mask,
             'coords': torch.randn(32, 3, generator=generator).cumsum(0) * 3.,
             'target': torch.randn(32, 3, generator=generator).cumsum(0) * 3.8,
             'angles': torch.zeros(32, 3)}
    prediction = {'id': 'val', 'mask': mask, 'pairs': pairs,
                  'probabilities': torch.randn(len(pairs), 10, generator=generator).softmax(-1)}
    local = {'bond_indices': torch.arange(31),
             'excluded_pairs': (index[None] - index[:, None] >= 2).nonzero()}
    return entry, prediction, local


def test_export_checks_exact_validation_identity_and_pair_coverage():
    entry, prediction, _ = sample()
    manifest = {'bundle_sha256': 'bundle', 'train_ids': ['train'], 'val_ids': ['val']}
    export = {'protocol': 'pair-distance-restraints-v1', 'bundle_sha256': 'bundle',
              'edges': torch.tensor(DISTANCE_EDGES), 'entries': [prediction]}
    assert set(validate_export(export, manifest, ['val'])) == {'val'}
    bad = copy.deepcopy(export)
    bad['entries'][0]['pairs'] = prediction['pairs'].flip(0)
    with pytest.raises(ValueError, match='canonical'):
        validate_export(bad, manifest, ['val'])
    bad = copy.deepcopy(export)
    bad['entries'][0]['probabilities'] *= 2
    with pytest.raises(ValueError, match='distributions'):
        validate_export(bad, manifest, ['val'])
    with pytest.raises(ValueError, match='validation-only'):
        validate_export(export, {**manifest, 'train_ids': ['val']}, ['val'])
    with pytest.raises(ValueError, match='IDs'):
        validate_export(export, manifest, ['other'])


def test_reconstruction_objective_never_reads_native_coordinates():
    entry, prediction, local = sample()
    edges = torch.tensor(DISTANCE_EDGES)
    for p in (None, prediction):
        expected = reconstruction_objective(entry['coords'], entry, local, p, edges)
        without_target = {k: v for k, v in entry.items() if k not in ('target', 'angles')}
        actual = reconstruction_objective(entry['coords'], without_target, local, p, edges)
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)


def test_bounded_reconstruction_retains_same_panel_and_initialization(tmp_path):
    entry, prediction, local = sample()
    panel = {'entries': [{'entry': entry, 'local': local,
                          'predictions': {'sequence': prediction, 'decoder': prediction}}],
             'edges': torch.tensor(DISTANCE_EDGES)}
    torch.save(panel, tmp_path / 'panel.pt')
    protocol = {'sources': sources(), 'panel_sha256': digest(tmp_path / 'panel.pt'),
                'ids': ['val'], 'steps': 3, 'seconds_per_case': 60}
    (tmp_path / 'experiment.json').write_text(json.dumps(protocol))
    run(SimpleNamespace(output=tmp_path, device='cpu'))
    assert json.loads((tmp_path / 'completion.json').read_text())['complete']
    summaries = [json.loads((tmp_path / mode / 'summary.json').read_text()) for mode in MODES]
    assert all(s['initial'] == summaries[0]['initial'] for s in summaries)
    for mode, summary in zip(MODES, summaries):
        assert summary['steps'] == 3 and summary['stop_reason'] == 'steps'
        assert summary['best'][0]['id'] == 'val'
        saved = torch.load(tmp_path / mode / 'best.pt', weights_only=False)
        restraint = None if mode == 'local_only' else prediction
        initial = reconstruction_objective(entry['coords'], entry, local, restraint, panel['edges'])[0]
        assert saved['objective'] <= float(initial)
    assert summaries[1]['mean_best_metrics'] == summaries[2]['mean_best_metrics']
