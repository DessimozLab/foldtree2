import copy
import json

import pytest
import torch
from torch_geometric.data import HeteroData

from foldtree2.se3_validation import CONTACT_POLICIES, contact_inputs, forward
from foldtree2.train_se3_lightning import (
    Refiner, build_model, check_warm_start_normalization,
)
from foldtree2.src.models.se3_struct_decoder import enable_tensor_ff_input_norm


def entry():
    generator = torch.Generator().manual_seed(71)
    target = torch.randn(20, 3, generator=generator).cumsum(0) * 3.8
    return {
        'id': 'synthetic', 'features': torch.randn(20, 8, generator=generator),
        'coords': target + .2 * torch.randn(20, 3, generator=generator),
        'target': target, 'angles': torch.zeros(20, 3),
        'tokens': torch.arange(20), 'mask': torch.ones(20, dtype=torch.bool),
        'dot': torch.ones(20, 20, dtype=torch.bool),
    }


def test_contact_policies_split_sources_without_target_input():
    e = entry()
    for policy in CONTACT_POLICIES:
        result = contact_inputs(e, policy)
        assert result['use_distance_contacts'] == (policy in ('normal', 'seed-only'))
        if policy in ('normal', 'inferred-only'):
            torch.testing.assert_close(result['dot_prod'][0], e['dot'])
        else:
            assert result['dot_prod'].sum() == 38
        changed = {**e, 'target': e['target'] * float('nan')}
        torch.testing.assert_close(contact_inputs(changed, policy)['dot_prod'], result['dot_prod'])
    with pytest.raises(ValueError, match='Unknown'):
        contact_inputs(e, 'invalid')


@pytest.mark.parametrize('policy', CONTACT_POLICIES)
@pytest.mark.parametrize('normalization', ['per_structure', 'layer'])
def test_mode_independent_normalization_and_rigid_equivariance(policy, normalization):
    e = entry()
    model = enable_tensor_ff_input_norm(build_model(
        8, {'hidden': 32, 'depth': 2, 'heads': 2, 'input_normalization': normalization}))
    with torch.no_grad():
        # Exercise learned nonzero coordinate updates, not just identity init.
        model.gotennet.proj_to_coors[1].weight.normal_(std=.05)
    model.train()
    p, a = forward(model, e, policy)
    model.eval()
    q, b = forward(model, e, policy)
    torch.testing.assert_close(q, p, atol=0, rtol=0)
    torch.testing.assert_close(b, a, atol=0, rtol=0)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotation[:, 0] *= torch.linalg.det(rotation)
    shift = torch.tensor([10., -7., 3.])
    moved = {**e, 'coords': e['coords'] @ rotation + shift}
    rp, ra = forward(model, moved, policy)
    torch.testing.assert_close(rp, p @ rotation + shift, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ra, a, atol=1e-4, rtol=1e-4)
    (q.square().mean() + b.square().mean()).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert getattr(model.bn, 'running_mean', None) is None


def test_per_structure_matches_original_training_features_and_handles_singleton():
    e = entry()
    original = build_model(8, {'hidden': 32, 'depth': 2, 'heads': 2})
    candidate = build_model(8, {
        'hidden': 32, 'depth': 2, 'heads': 2, 'input_normalization': 'per_structure'})
    batch_features, structure_features = [], []
    original.input2gotennet_fallback.register_forward_pre_hook(
        lambda _, values: batch_features.append(values[0].detach().clone()))
    candidate.input2gotennet_fallback.register_forward_pre_hook(
        lambda _, values: structure_features.append(values[0].detach().clone()))
    original.train()
    forward(original, e)
    candidate.eval()
    forward(candidate, e)
    torch.testing.assert_close(batch_features[0], structure_features[0], atol=0, rtol=0)
    e['mask'][1:] = False
    p, a = forward(candidate, e)
    assert torch.isfinite(p).all() and torch.isfinite(a).all()
    assert p[1:].eq(0).all()


def test_per_structure_normalizes_graphs_independently():
    e = entry()
    model = build_model(8, {
        'hidden': 32, 'depth': 2, 'heads': 2, 'input_normalization': 'per_structure'})
    graph = HeteroData()
    graph['res'].x = e['features']
    graph['res'].batch = torch.tensor([0] * 10 + [1] * 10)
    captured = []
    model.input2gotennet_fallback.register_forward_pre_hook(
        lambda _, values: captured.append(values[0].detach().clone()))
    model.eval()
    model(graph, coords_pred=e['coords'], ft2_token_ids=e['tokens'],
          edge_attr_dict={'node_mask': e['mask'], 'use_distance_contacts': False})
    for values in captured[0].chunk(2):
        torch.testing.assert_close(values.mean(0), torch.zeros(8), atol=1e-6, rtol=0)


def test_normalization_and_contact_warm_start_guards():
    base = {'model': {'hidden': 64, 'depth': 4, 'heads': 4}}
    check_warm_start_normalization(base, copy.deepcopy(base))
    candidate = copy.deepcopy(base)
    candidate['model']['input_normalization'] = 'per_structure'
    with pytest.raises(ValueError, match='input normalization'):
        check_warm_start_normalization(base, candidate)
    with pytest.raises(ValueError, match='contact policy'):
        check_warm_start_normalization(base, {**base, 'contact_policy': 'local-only'})
    manifest = {'feature_dim': 8, 'bundle_sha256': 'b', 'input_identity': {}, 'scale': 1.}
    with pytest.raises(ValueError, match='contact policy'):
        Refiner({**base, 'contact_policy': 'invalid'}, manifest)
    with pytest.raises(ValueError, match='input_normalization'):
        build_model(8, {**base['model'], 'input_normalization': 'invalid'})


def test_subset_preparation_is_matched_and_content_verified(tmp_path):
    from foldtree2.se3_cache import digest, identity_hash, open_bundle
    from scripts.se3.run_se3_generalization import prepare, VARIANTS
    source = tmp_path / 'source'
    (source / 'entries').mkdir(parents=True)
    files = {}
    for index in range(18):
        identifier = str(index)
        e = entry()
        e = {k: torch.cat([v, v, v, v[:4]]) if isinstance(v, torch.Tensor) and v.ndim >= 1
             and len(v) == 20 and k != 'dot' else v for k, v in e.items()}
        e['dot'] = torch.ones(64, 64, dtype=torch.bool)
        name = identifier + '.pt'
        torch.save({'id': identifier, 'identity': {}, 'entry': e}, source / 'entries' / name)
        files[identifier] = {'name': name, 'length': 64, 'sha256': digest(source / 'entries' / name)}
    manifest = {'train_ids': [str(i) for i in range(16)], 'val_ids': ['16', '17'],
                'files': files, 'input_identity': {}, 'scale': 1., 'feature_dim': 8,
                'calibration_ids': ['0']}
    manifest['bundle_sha256'] = identity_hash(manifest)
    (source / 'manifest.json').write_text(json.dumps(manifest))
    root = tmp_path / 'trial'
    prepare(source, root, 16, 2, 1)
    subset = open_bundle(root / 'bundle')
    assert set(subset['train_ids']).isdisjoint(subset['val_ids'])
    assert len(subset['train_ids']) == 16
    experiment = json.loads((root / 'experiment.json').read_text())
    assert len(experiment['configs']) == len(VARIANTS) == 8
    import yaml
    configs = [yaml.safe_load((root / (name + '.yaml')).read_text()) for name, _, _ in VARIANTS]
    for c in configs:
        assert c['train_count'] == 16 and c['val_count'] == 2
        assert c['epochs'] == 1 and c['global_batch'] == 16
        assert c['input_baseline_metrics']
    with pytest.raises(ValueError, match='divisible'):
        prepare(source, tmp_path / 'invalid', 17, 2, 1)
