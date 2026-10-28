import copy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.diagnose_se3_restraints import MODES, objective, prepare_restraints


def sample():
    generator = torch.Generator().manual_seed(51)
    target = torch.randn(32, 3, generator=generator).cumsum(0) * 3.8
    return {'target': target, 'coords': target + torch.randn(32, 3, generator=generator),
            'angles': torch.zeros(32, 3),
            'mask': torch.ones(32, dtype=torch.bool),
            'contact_z': torch.nn.functional.normalize(
                torch.randn(32, 30, generator=generator), dim=-1)}


def test_pairs_masks_budget_and_deterministic_selection():
    entry = sample()
    entry['mask'][4] = False
    a, audit = prepare_restraints(entry, 2.4, -1.3)
    b, _ = prepare_restraints(entry, 2.4, -1.3)
    pairs = a['pairs']
    assert ((pairs[:, 1] - pairs[:, 0]) >= 12).all()
    assert not (pairs == 4).any()
    assert len(a['uniform_indices']) == 62
    assert len(a['decoder_indices']) == 62
    assert len(a['uniform_indices'].unique()) == 62
    assert audit['sparse_distance_pairs'] == 62
    torch.testing.assert_close(a, b)
    ranked = a['probabilities'][a['decoder_indices']]
    assert (ranked[:-1] >= ranked[1:]).all()


@pytest.mark.parametrize('mode', MODES)
def test_objectives_have_finite_gradients_and_reduce(mode):
    entry = sample()
    restraints, _ = prepare_restraints(entry, 2.4, -1.3)
    coords = torch.nn.Parameter(entry['coords'].clone())
    optimizer = torch.optim.Adam([coords], lr=.03)
    initial = float(objective(coords, entry, restraints, mode)[0].detach())
    for _ in range(20):
        optimizer.zero_grad()
        loss, _ = objective(coords, entry, restraints, mode)
        loss.backward()
        assert torch.isfinite(coords.grad).all()
        optimizer.step()
    assert float(objective(coords, entry, restraints, mode)[0].detach()) < initial


@pytest.mark.parametrize('mode', ['local_only', 'predicted_contacts', 'predicted_top_pairs'])
def test_nonoracle_objective_and_pair_selection_do_not_use_native_geometry(mode):
    entry = sample()
    restraints, _ = prepare_restraints(entry, 2.4, -1.3)
    changed = copy.deepcopy(entry)
    changed['target'] = torch.randn_like(entry['target']) * 100
    other, _ = prepare_restraints(changed, 2.4, -1.3)
    for key in ('pairs', 'probabilities', 'uniform_indices', 'decoder_indices'):
        torch.testing.assert_close(restraints[key], other[key], atol=0, rtol=0)
    torch.testing.assert_close(
        objective(entry['coords'], entry, restraints, mode),
        objective(entry['coords'], changed, other, mode), atol=0, rtol=0)


def test_oracle_distance_is_zero_at_native_and_contact_labels_are_explicit():
    entry = sample()
    restraints, _ = prepare_restraints(entry, 2.4, -1.3)
    for mode in ('oracle_contacts', *MODES[4:]):
        _, terms = objective(entry['target'], entry, restraints, mode)
        assert terms['restraint'] == 0
    assert torch.equal(restraints['native_contacts'], (restraints['native_distances'] < 8).float())
    with pytest.raises(ValueError, match='Unknown'):
        objective(entry['coords'], entry, restraints, 'bad')
    entry['contact_z'] *= 2
    with pytest.raises(ValueError, match='normalized'):
        prepare_restraints(entry, 2.4, -1.3)


def test_bounded_runner_completes_every_case_with_identical_initial_coordinates(tmp_path):
    from foldtree2.se3_cache import digest
    from scripts.diagnose_se3_restraints import run, sources
    entry = {**sample(), 'id': 'synthetic'}
    restraints, _ = prepare_restraints(entry, 2.4, -1.3)
    torch.save([{'entry': entry, 'restraints': restraints}], tmp_path / 'panel.pt')
    protocol = {'sources': sources(), 'panel_sha256': digest(tmp_path / 'panel.pt'),
                'ids': ['synthetic'], 'steps': 2, 'seconds_per_case': 60}
    (tmp_path / 'experiment.json').write_text(json.dumps(protocol))
    run(SimpleNamespace(output=tmp_path, device='cpu'))
    assert json.loads((tmp_path / 'completion.json').read_text())['complete']
    initials = []
    for mode in MODES:
        summary = json.loads((tmp_path / mode / 'summary.json').read_text())
        assert summary['steps'] == 2 and summary['stop_reason'] == 'steps'
        assert len(summary['best']) == len(summary['final']) == 1
        checkpoint = torch.load(tmp_path / mode / 'best.pt', weights_only=False)
        initial_objective = float(objective(entry['coords'], entry, restraints, mode)[0])
        assert checkpoint['objective'] <= initial_objective
        initials.append(summary['initial'])
    assert all(x == initials[0] for x in initials)
