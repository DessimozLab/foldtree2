import copy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from foldtree2.src.distance.distance_restraints import (
    DISTANCE_EDGES, PairwiseDistanceHead, distance_labels, distribution_restraint_loss,
    eligible_pairs, validate_edges,
)


def example():
    generator = torch.Generator().manual_seed(19)
    return {
        'id': 'synthetic', 'features': torch.randn(32, 8, generator=generator),
        'contact_z': F.normalize(torch.randn(32, 30, generator=generator), dim=-1),
        'aa_probs': torch.randn(32, 20, generator=generator).softmax(-1),
        'target': torch.randn(32, 3, generator=generator) * 8.,
        'coords': torch.randn(32, 3, generator=generator),
        'mask': torch.ones(32, dtype=torch.bool),
    }


@pytest.mark.parametrize('context_layers', [0, 2])
def test_symmetric_chunked_logits_masks_and_target_independence(context_layers):
    entry = example()
    entry['mask'][-1] = False
    for key in ('features', 'contact_z', 'aa_probs', 'target'):
        entry[key][-1] = float('nan')
    pairs = eligible_pairs(entry['mask'])
    assert not (pairs == 31).any()
    model = PairwiseDistanceHead(8, hidden=16, context_layers=context_layers).eval()
    full = model(entry, pairs)
    assert full.shape == (len(pairs), len(DISTANCE_EDGES) + 1)
    torch.testing.assert_close(full, model(entry, pairs.flip(-1)), atol=0, rtol=0)
    torch.testing.assert_close(full, model(entry, pairs, chunk_size=7), atol=1e-6, rtol=1e-6)
    changed = {k: v for k, v in entry.items() if k not in ('target', 'coords')}
    torch.testing.assert_close(full, model(changed, pairs), atol=0, rtol=0)
    for key in ('features', 'contact_z', 'aa_probs'):
        changed = copy.deepcopy(entry)
        changed[key] = changed[key].roll(1, -1)
        assert not torch.allclose(full, model(changed, pairs)), key


def test_sequence_control_ignores_feature_values_and_matches_parameter_count():
    entry = example()
    pairs = eligible_pairs(entry['mask'])
    model = PairwiseDistanceHead(8, hidden=16, mode='sequence')
    decoder = PairwiseDistanceHead(8, hidden=16)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in decoder.parameters())
    changed = copy.deepcopy(entry)
    changed['features'] *= 30
    changed['contact_z'] = changed['contact_z'].roll(1, -1)
    changed['aa_probs'] = changed['aa_probs'].roll(1, -1)
    torch.testing.assert_close(model(entry, pairs), model(changed, pairs), atol=0, rtol=0)


def test_contact_average_precision_handles_tied_scores():
    from foldtree2.train_distance_restraints import contact_metrics
    a = contact_metrics(torch.ones(4) * .2, torch.tensor([1, 0, 0, 0]), 2)
    b = contact_metrics(torch.ones(4) * .2, torch.tensor([0, 0, 0, 1]), 2)
    assert a['contact_average_precision'] == b['contact_average_precision'] == .25


def test_bin_boundaries_contact_probability_and_overflow_semantics():
    edges = torch.tensor(DISTANCE_EDGES)
    distances = torch.tensor([0., 3.99, 4., 7.99, 8., 31.99, 32., 100.])
    target = torch.zeros(len(distances) + 1, 3)
    target[1:, 0] = distances
    pairs = torch.stack([torch.zeros(len(distances), dtype=torch.long),
                         torch.arange(1, len(distances) + 1)], dim=-1)
    labels, measured = distance_labels(target, pairs, edges)
    assert labels.tolist() == [0, 0, 1, 2, 3, 8, 9, 9]
    model = PairwiseDistanceHead(8, hidden=16)
    one_hot = F.one_hot(labels, len(edges) + 1).float()
    assert torch.equal(model.contact_probability(one_hot), (distances < 8.).float())
    assert model.capped_distance(one_hot)[-1] == 32.
    torch.testing.assert_close(measured, distances)


@pytest.mark.parametrize('context_layers', [0, 2])
def test_head_learns_with_frozen_inputs_and_exact_reload(tmp_path, context_layers):
    entry = example()
    for key in ('features', 'contact_z', 'aa_probs'):
        entry[key].requires_grad_()
    pairs = eligible_pairs(entry['mask'])
    model = PairwiseDistanceHead(8, hidden=16, context_layers=context_layers)
    labels, _ = distance_labels(entry['target'], pairs, model.edges)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    initial = float(F.cross_entropy(model(entry, pairs), labels).detach())
    for _ in range(30):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(entry, pairs), labels)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert float(F.cross_entropy(model(entry, pairs), labels).detach()) < initial * .8
    for key in ('features', 'contact_z', 'aa_probs'):
        assert entry[key].grad is None
    torch.save({'model': model.state_dict(), 'model_config': model.model_config()}, tmp_path / 'head.pt')
    saved = torch.load(tmp_path / 'head.pt', weights_only=True)
    clone = PairwiseDistanceHead(**saved['model_config'])
    clone.load_state_dict(saved['model'])
    torch.testing.assert_close(model(entry, pairs), clone(entry, pairs), atol=0, rtol=0)


def test_context_exchanges_information_but_ignores_masked_residues():
    entry = example()
    entry['mask'][-1] = False
    pairs = torch.tensor([[0, 12]])
    changed = copy.deepcopy(entry)
    changed['features'][20] *= -7
    for depth in (0, 2):
        model = PairwiseDistanceHead(8, hidden=16, context_layers=depth).eval()
        original = model(entry, pairs)
        if depth:
            assert not torch.allclose(original, model(changed, pairs), atol=1e-7, rtol=1e-7)
        else:
            torch.testing.assert_close(original, model(changed, pairs), atol=0, rtol=0)
        masked = copy.deepcopy(entry)
        for key in ('features', 'contact_z', 'aa_probs'):
            masked[key][-1] = float('nan')
        torch.testing.assert_close(original, model(masked, pairs), atol=0, rtol=0)
        alone = model(entry, eligible_pairs(entry['mask']))[:1]
        torch.testing.assert_close(original, alone, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('depth,heads', [(-1, 8), (True, 8), (1, 0), (1, 3)])
def test_invalid_context_configuration_rejected(depth, heads):
    with pytest.raises(ValueError, match='residue-context'):
        PairwiseDistanceHead(8, hidden=16, context_layers=depth, context_heads=heads)


@pytest.mark.parametrize('edges', [[8., 4.], [0., 8.], [8., 8.], [float('nan')], []])
def test_invalid_bins_rejected(edges):
    with pytest.raises(ValueError):
        validate_edges(torch.tensor(edges))


def test_invalid_pairs_and_features_rejected():
    entry = example()
    model = PairwiseDistanceHead(8)
    pairs = eligible_pairs(entry['mask'])
    for bad in (torch.tensor([[0, 0]]), torch.tensor([[0, 32]]),
                pairs.float(), torch.empty(0, 2, dtype=torch.long)):
        with pytest.raises(ValueError):
            model(entry, bad)
    changed = copy.deepcopy(entry)
    changed['aa_probs'] *= 2
    with pytest.raises(ValueError, match='probability'):
        model(changed, pairs)
    changed = copy.deepcopy(entry)
    changed['contact_z'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='Invalid frozen'):
        model(changed, pairs)
    with pytest.raises(ValueError, match='8-Angstrom'):
        PairwiseDistanceHead(8, edges=[4., 10., 20.])
    with pytest.raises(ValueError, match='No eligible'):
        eligible_pairs(torch.ones(8, dtype=torch.bool))


def test_distribution_restraint_gradients_overflow_and_rigid_invariance():
    model = PairwiseDistanceHead(8)
    pairs = torch.tensor([[0, 1]])
    contact = torch.zeros(1, 10)
    contact[0, 1] = 1.  # [4,6) Angstrom.
    coords = torch.nn.Parameter(torch.tensor([[0., 0., 0.], [15., 0., 0.]]))
    initial = float(distribution_restraint_loss(coords, pairs, contact, model.edges).detach())
    optimizer = torch.optim.Adam([coords], lr=.1)
    for _ in range(40):
        optimizer.zero_grad()
        loss = distribution_restraint_loss(coords, pairs, contact, model.edges)
        loss.backward()
        assert torch.isfinite(coords.grad).all()
        optimizer.step()
    assert float(loss.detach()) < initial * .2
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    moved = coords @ rotation + torch.tensor([2., 5., -3.])
    torch.testing.assert_close(
        distribution_restraint_loss(coords, pairs, contact, model.edges),
        distribution_restraint_loss(moved, pairs, contact, model.edges))
    overflow = torch.zeros_like(contact)
    overflow[0, -1] = 1.
    near = torch.tensor([[0., 0., 0.], [40., 0., 0.]])
    far = torch.tensor([[0., 0., 0.], [100., 0., 0.]])
    assert distribution_restraint_loss(far, pairs, overflow, model.edges) < distribution_restraint_loss(
        near, pairs, overflow, model.edges)
    with pytest.raises(ValueError, match='normalized'):
        distribution_restraint_loss(coords, pairs, contact * 2, model.edges)


@pytest.mark.parametrize('context_layers', [0, 2])
def test_bounded_trainer_outputs_both_controls_and_native_free_exports(tmp_path, context_layers):
    from foldtree2.se3_cache import digest, identity_hash
    from foldtree2.train_distance_restraints import train

    cache = tmp_path / 'cache'
    (cache / 'entries').mkdir(parents=True)
    identity = {'synthetic': True}
    manifest = {'train_ids': ['train1', 'train2'], 'val_ids': ['val'],
                'input_identity': identity, 'files': {}, 'feature_dim': 8,
                'decoder_enrichment': {'contact_dim': 30}}
    for index, identifier in enumerate(['train1', 'train2', 'val']):
        entry = example()
        entry['target'] = entry['target'] * (1 + index * .1)
        entry.pop('id')
        name = identifier + '.pt'
        torch.save({'id': identifier, 'identity': identity, 'entry': entry}, cache / 'entries' / name)
        manifest['files'][identifier] = {'name': name, 'length': 32,
                                         'sha256': digest(cache / 'entries' / name)}
    manifest['bundle_sha256'] = identity_hash(manifest)
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    output = tmp_path / 'trial'
    args = SimpleNamespace(cache=str(cache), output=str(output), device='cpu', epochs=2,
                           seconds=60., pairs=64, hidden=16, learning_rate=.001,
                           train_count=None, val_count=None, modes=['sequence', 'decoder'])
    args.context_layers = context_layers
    args.context_heads = 8
    args.cosine_lr = False
    args.weight_decay = .01
    args.early_stop = 0
    train(args)
    completion = json.loads((output / 'completion.json').read_text())
    for mode in args.modes:
        assert completion[mode]['updates'] == 4
        assert completion[mode]['reload_exact']
        assert completion[mode]['stop_reason'] == 'epochs'
        assert completion[mode]['validation']['chains'] == 1
        assert completion[mode]['training']['chains'] == 2
        saved = torch.load(output / mode / 'validation_restraints.pt', weights_only=True)
        exported = saved['entries'][0]
        assert set(exported) == {'id', 'pairs', 'probabilities', 'mask'}
        assert exported['id'] == 'val'
        assert saved['model_sha256'] == digest(output / mode / 'last.pt')
        loss = distribution_restraint_loss(
            example()['coords'], exported['pairs'], exported['probabilities'], saved['edges'])
        assert torch.isfinite(loss)
