import pytest
import torch
import json
from types import SimpleNamespace

from scripts.se3.diagnose_se3_reconstruction import (
    atomic_json, initialize, optimize_step, reconstruction_metrics, select_panel,
    model_forward, perfect_target_audit,
    audit_model,
)
from foldtree2.src.losses.se3_balanced import BalancedGeometryLoss


def example():
    generator = torch.Generator().manual_seed(104)
    bonds = torch.nn.functional.normalize(torch.randn(20, 3, generator=generator), dim=-1) * 3.8
    target = bonds.cumsum(0)
    return {'coords': target + .2, 'target': target,
            'mask': torch.ones(20, dtype=torch.bool), 'angles': torch.zeros(20, 3)}


def test_panel_is_training_only_unique_and_length_spread():
    manifest = {'train_ids': ['a', 'b', 'c', 'd'], 'val_ids': ['v'],
                'files': {i: {'length': n} for i, n in zip('abcdv', [64, 80, 96, 128, 70])}}
    assert select_panel(manifest, 3) == ['a', 'b', 'd']
    assert select_panel(manifest, 1) == ['a']
    with pytest.raises(ValueError, match='Insufficient'):
        select_panel(manifest, 5)
    manifest['val_ids'] = ['a']
    with pytest.raises(ValueError, match='overlaps'):
        select_panel(manifest, 1)


def test_initialization_is_deterministic_masked_and_not_aliased():
    entry = example()
    seed = initialize(entry, 'seed', torch.Generator().manual_seed(42), 1.)
    torch.testing.assert_close(seed, entry['coords'])
    assert seed.data_ptr() != entry['coords'].data_ptr()
    entry['mask'][-1] = False
    entry['target'][-1] = float('nan')
    a = initialize(entry, 'native-noise', torch.Generator().manual_seed(42), 1.)
    b = initialize(entry, 'native-noise', torch.Generator().manual_seed(42), 1.)
    torch.testing.assert_close(a, b)
    assert a[-1].eq(0).all() and torch.isfinite(a).all()
    with pytest.raises(ValueError):
        initialize(entry, 'native-noise', torch.Generator(), 0.)


def test_direct_coordinate_control_reduces_loss():
    entry = example()
    generator = torch.Generator().manual_seed(42)
    pred = torch.nn.Parameter(initialize(entry, 'native-noise', generator, .5))
    loss = BalancedGeometryLoss({'angle': 0., 'local_distance': 1., 'bend': .5, 'torsion': .5})
    optimizer = torch.optim.AdamW([pred], lr=.03, weight_decay=0.)
    predict = lambda _: (pred, torch.zeros(20, 3))
    initial = float(loss(pred, entry['angles'], entry)[0].detach())
    for _ in range(30):
        optimize_step([pred], optimizer, loss, predict, [entry])
    assert float(loss(pred, entry['angles'], entry)[0].detach()) < initial * .5


def test_metrics_perfect_coordinates_and_json_rejects_nan(tmp_path):
    entry = example()
    result = reconstruction_metrics(entry['target'], entry['angles'], entry)
    assert result['fape_saturated_fraction'] == 0
    assert result['frame_error_unclamped_angstrom'] < 1e-5
    if result['long_range_target_contacts']:
        assert result['long_range_contact_recall'] == 1.
    path = tmp_path / 'record.json'
    atomic_json(path, {'finite': 1.})
    assert path.exists() and not path.with_suffix('.json.tmp').exists()
    with pytest.raises(ValueError):
        atomic_json(path, {'invalid': float('nan')})


def test_perfect_target_audit_and_local_contact_control():
    entry = {**example(), 'id': 'synthetic', 'features': torch.randn(20, 8),
             'tokens': torch.arange(20)}
    rows = perfect_target_audit([entry], BalancedGeometryLoss(
        {'local_distance': 1., 'bend': .5, 'torsion': .5}))
    assert rows[0]['coordinate_objective'] < 1e-5

    class Capture:
        def __call__(self, graph, **kwargs):
            edges = kwargs['edge_attr_dict']
            assert edges['use_distance_contacts'] is False
            assert int(edges['dot_prod'].sum()) == 38
            assert not edges['dot_prod'][0, 0, 12]
            return {'coors_out_flat': kwargs['coords_pred'], 'angles': entry['angles']}

    pred, _ = model_forward(Capture(), entry, 'local-only')
    torch.testing.assert_close(pred, entry['coords'])
    with pytest.raises(ValueError):
        model_forward(Capture(), entry, 'invalid')


@pytest.mark.parametrize('gpu,parents,expected_cases', [
    (0, ['coordinates_seed', 'coordinates_native_noise'], 2),
    (1, ['model_one', 'model_panel'], 3),
])
def test_queue_requires_successful_parent_and_preserves_bounds(
        tmp_path, monkeypatch, gpu, parents, expected_cases):
    from scripts.se3 import queue_se3_reconstruction_controls as queue
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', ['queue', '--gpu', str(gpu)])
    root = tmp_path / 'runs/se3_reconstruction_20261020'
    for parent in parents:
        path = root / parent
        path.mkdir(parents=True)
        (path / 'summary.json').write_text(json.dumps({'steps': 2}))
        (path / 'events.jsonl').write_text(json.dumps({'event': 'complete'}) + '\n')
    calls = []

    def run(command, **kwargs):
        assert kwargs['check']
        if command[0] == 'systemctl':
            return SimpleNamespace(stdout='inactive\n' if 'ActiveState' in command[-2] else 'success\n')
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(queue.subprocess, 'run', run)
    queue.main()
    assert len(calls) == expected_cases
    assert all(f'cuda:{gpu}' in command for command in calls)
    (root / parents[0] / 'events.jsonl').write_text(json.dumps({'event': 'failure'}) + '\n')
    with pytest.raises(RuntimeError, match='clean completion'):
        queue.main()


def test_model_audit_restores_buffers_and_does_not_accumulate_gradients(monkeypatch):
    import scripts.se3.diagnose_se3_reconstruction as runner
    from foldtree2.train_se3_lightning import build_model
    from foldtree2.src.models.se3_struct_decoder import enable_tensor_ff_input_norm
    entry = {**example(), 'id': 'synthetic', 'features': torch.randn(20, 8),
             'tokens': torch.arange(20), 'dot': torch.eye(20, dtype=torch.bool)}
    model = enable_tensor_ff_input_norm(build_model(8, {'hidden': 32, 'depth': 2, 'heads': 2}))
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    model.eval()
    # Metric geometry is covered separately; keep this audit focused on state.
    monkeypatch.setattr(runner, 'reconstruction_metrics', lambda *args: {})
    result = audit_model(model, [entry, {**entry, 'id': 'copy'}],
                         BalancedGeometryLoss(), 'normal')
    assert not model.training and result['optimizer_steps'] == 0
    assert len(result['gradient_cosines']) == 2
    assert result['rows'][0]['gradient_norm'] > 0
    assert result['gradient_cosines'][0][1] == pytest.approx(1., abs=1e-6)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
    assert all(p.grad is None for p in model.parameters())


def test_information_probe_pairs_and_features_exclude_targets():
    from scripts.se3.probe_se3_information import pair_features, sample_pairs, report
    entry = {**example(), 'id': 'synthetic', 'features': torch.randn(20, 8),
             'codebook': torch.randn(20, 4)}
    pairs = sample_pairs(entry, torch.Generator().manual_seed(42), count=16)
    assert ((pairs[:, 1] - pairs[:, 0]) >= 12).all()
    features = {r: pair_features(entry, pairs, r)
                for r in ['sequence', 'codebook', 'continuous']}
    assert all(v.shape == (16, 25) for v in features.values())
    entry['target'].fill_(float('nan'))
    entry['coords'].fill_(1000.)
    for representation, original in features.items():
        torch.testing.assert_close(pair_features(entry, pairs, representation), original)
    assert features['sequence'][:, :-1].eq(0).all()
    with pytest.raises(ValueError):
        pair_features(entry, pairs, 'invalid')
    result = report(torch.tensor([5., 10., 20.]), torch.tensor([5., 10., 20.]))
    assert result['precision'] == result['recall'] == 1.
    assert result['distance_mae_angstrom'] == 0.


@pytest.mark.parametrize('gpu,count', [(0, 2), (1, 4)])
def test_remaining_commands_use_nested_panels_and_distinct_outputs(gpu, count):
    from scripts.se3.queue_se3_remaining_controls import commands
    cases = commands(gpu)
    assert cases[0][cases[0].index('--count') + 1] == str(count)
    assert cases[0][cases[0].index('--panel-size') + 1] == '8'
    outputs = [command[command.index('--output') + 1] for command in cases]
    assert len(outputs) == len(set(outputs))
    assert all(f'cuda:{gpu}' in command for command in cases)
