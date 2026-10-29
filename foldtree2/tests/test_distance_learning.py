import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from foldtree2.se3_cache import digest, identity_hash
from foldtree2.src.distance.distance_energy import (
    distance_energy, fit_reference, interval_log_weights, sample_contact_strata, separation_groups,
)
from foldtree2.src.distance.distance_restraints import DISTANCE_EDGES, eligible_pairs
from scripts.se3.diagnose_distance_learning import memorization_pass, run, sampling_pilot


def test_interval_membership_has_unit_mass_with_unequal_bins_and_open_tail():
    edges = torch.tensor(DISTANCE_EDGES, dtype=torch.float64)
    distances = torch.linspace(0, 100, 500, dtype=torch.float64)
    weights = interval_log_weights(distances, edges).exp()
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(distances), atol=1e-12, rtol=1e-12)
    assert weights[-1, -1] > .999999
    assert weights[0, 0] > .98


def test_reference_prediction_has_zero_energy_and_zero_coordinate_force():
    edges = torch.tensor(DISTANCE_EDGES, dtype=torch.float64)
    d = torch.linspace(.1, 80, 200, dtype=torch.float64).requires_grad_()
    background = torch.arange(1., 11., dtype=torch.float64).repeat(len(d), 1)
    background /= background.sum(-1, keepdim=True)
    energies = distance_energy(d, background, edges, reference=background)
    torch.testing.assert_close(energies, torch.zeros_like(d), atol=1e-12, rtol=0)
    energies.sum().backward()
    torch.testing.assert_close(d.grad, torch.zeros_like(d), atol=1e-12, rtol=0)


def test_pair_specific_contact_enrichment_pulls_close_and_keeps_open_tail():
    edges = torch.tensor(DISTANCE_EDGES)
    background = torch.ones(1, 10) / 10
    contact = torch.zeros_like(background)
    contact[0, 1] = 1.
    d = torch.tensor([15.], requires_grad=True)
    distance_energy(d, contact, edges, background).sum().backward()
    assert d.grad.item() > 0  # Gradient descent reduces the distance.
    assert distance_energy(torch.tensor([5.]), contact, edges, background) < distance_energy(
        torch.tensor([15.]), contact, edges, background)
    tail = torch.zeros_like(background)
    tail[0, -1] = 1.
    assert distance_energy(torch.tensor([100.]), tail, edges, background) < distance_energy(
        torch.tensor([40.]), tail, edges, background)
    with pytest.raises(ValueError, match='reference'):
        distance_energy(torch.tensor([5.]), contact, edges, torch.zeros_like(contact))
    with pytest.raises(ValueError, match='predictions'):
        distance_energy(torch.tensor([5.]), contact * 2, edges, background)


def test_reference_uses_only_supplied_training_records_and_fixed_separation_bands():
    pairs = torch.tensor([[0, 12], [0, 24], [0, 48]])
    assert separation_groups(pairs).tolist() == [0, 1, 2]
    target = torch.zeros(49, 3)
    target[12, 0], target[24, 0], target[48, 0] = 5., 9., 40.
    reference, counts = fit_reference([({'target': target}, pairs)], torch.tensor(DISTANCE_EDGES))
    assert counts[0, 1] == counts[1, 3] == counts[2, 9] == 2
    torch.testing.assert_close(reference.sum(-1), torch.ones(3))
    assert (reference > 0).all()


def test_corrected_sampling_targets_population_ce_not_balanced_class_ce():
    labels = torch.tensor([1] * 10 + [8] * 90)
    losses = torch.linspace(0., 5., 100)
    indices, weights = sample_contact_strata(labels, 100000, torch.Generator().manual_seed(12))
    assert (labels[indices] < 3).sum() == 50000
    torch.testing.assert_close(weights.mean(), torch.tensor(1.))
    assert abs(float((losses[indices] * weights).mean() - losses.mean())) < .02
    assert abs(float(losses[indices].mean() - losses.mean())) > .5
    for labels in (torch.ones(10, dtype=torch.long), torch.full((10,), 8, dtype=torch.long)):
        indices, weights = sample_contact_strata(labels, 20, torch.Generator().manual_seed(42))
        assert len(indices) == 20 and weights.eq(1).all()


def test_memorization_gate_requires_contact_precision_and_recall_not_only_average_nll():
    report = {'nll': .1, 'bin_accuracy': .99, 'contact_average_precision': .99,
              'contact_precision': None, 'contact_recall': 0.}
    assert not memorization_pass(report)
    assert memorization_pass({**report, 'contact_precision': .9, 'contact_recall': .9})


def test_sampling_pilot_executes_both_matched_paths(tmp_path):
    generator = torch.Generator().manual_seed(41)
    entry = {'id': 'train', 'mask': torch.ones(32, dtype=torch.bool),
             'features': torch.randn(32, 8, generator=generator),
             'contact_z': F.normalize(torch.randn(32, 30, generator=generator), dim=-1),
             'aa_probs': torch.randn(32, 20, generator=generator).softmax(-1),
             'target': torch.randn(32, 3, generator=generator) * 4.}
    args = SimpleNamespace(device='cpu', pilot_steps=2, seconds=60.)
    manifest = {'feature_dim': 8, 'decoder_enrichment': {'contact_dim': 30}}
    sampling_pilot(args, manifest, [(entry, eligible_pairs(entry['mask']))],
                   tmp_path, lambda *a, **k: None)
    report = json.loads((tmp_path / 'sampling_pilot.json').read_text())
    assert set(report) == {'uniform', 'contact_corrected'}
    assert all(r['steps'] == 2 for r in report.values())


def test_bounded_pipeline_publishes_gates_and_blocks_sampling_on_failure(tmp_path):
    cache = tmp_path / 'cache'
    (cache / 'entries').mkdir(parents=True)
    head, reconstruction = tmp_path / 'head', tmp_path / 'reconstruction'
    head.mkdir()
    reconstruction.mkdir()
    train_ids = ['train' + str(i) for i in range(8)]
    val_ids = ['val']
    manifest = {'train_ids': train_ids, 'val_ids': val_ids, 'files': {},
                'feature_dim': 8, 'scale': 1., 'input_identity': {'synthetic': True},
                'decoder_enrichment': {'contact_dim': 30}}
    generator = torch.Generator().manual_seed(10)
    entries = {}
    for identifier in train_ids + val_ids:
        entry = {'id': identifier, 'mask': torch.ones(64, dtype=torch.bool),
                 'features': torch.randn(64, 8, generator=generator),
                 'contact_z': F.normalize(torch.randn(64, 30, generator=generator), dim=-1),
                 'aa_probs': torch.randn(64, 20, generator=generator).softmax(-1),
                 'target': torch.randn(64, 3, generator=generator) * 4.,
                 'coords': torch.randn(64, 3, generator=generator) * 6.}
        entries[identifier] = entry
        name = identifier + '.pt'
        torch.save({'id': identifier, 'identity': manifest['input_identity'], 'entry': entry},
                   cache / 'entries' / name)
        manifest['files'][identifier] = {'name': name, 'sha256': digest(cache / 'entries' / name),
                                         'length': 64}
    manifest['bundle_sha256'] = identity_hash(manifest)
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    (head / 'provenance.json').write_text(json.dumps({
        'bundle_sha256': manifest['bundle_sha256'], 'selection': {'train': train_ids, 'val': val_ids}}))
    pairs = eligible_pairs(entries['val']['mask'])
    for mode in ('sequence', 'decoder'):
        (head / mode).mkdir()
        (head / mode / 'last.pt').write_bytes(b'checkpoint-identity')
        torch.save({'protocol': 'pair-distance-restraints-v1',
                    'bundle_sha256': manifest['bundle_sha256'],
                    'model_sha256': digest(head / mode / 'last.pt'),
                    'edges': torch.tensor(DISTANCE_EDGES),
                    'entries': [{'id': 'val', 'mask': entries['val']['mask'],
                                 'pairs': pairs, 'probabilities': torch.ones(len(pairs), 10) / 10}]},
                   head / mode / 'validation_restraints.pt')
    (reconstruction / 'experiment.json').write_text(json.dumps({
        'bundle_sha256': manifest['bundle_sha256'], 'ids': val_ids}))
    for mode in ('local_only', 'sequence', 'decoder'):
        (reconstruction / mode).mkdir()
        torch.save({'ids': val_ids, 'coords': [entries['val']['coords']]},
                   reconstruction / mode / 'best.pt')
    output = tmp_path / 'diagnostic'
    run(SimpleNamespace(cache=str(cache), head=str(head), reconstruction=str(reconstruction),
                        output=str(output), device='cpu', steps=1, pilot_steps=1,
                        seconds=60., log_every=1))
    completion = json.loads((output / 'completion.json').read_text())
    assert completion['complete'] and not completion['sampling_launched']
    assert all(c['updates'] == 1 for c in completion['memorization'].values())
    assert json.loads((output / 'sampling_pilot.json').read_text())['status'] == 'blocked'
    reference = torch.load(output / 'reference.pt', weights_only=True)
    assert reference['train_ids'] == train_ids and not set(reference['train_ids']) & set(val_ids)
    energy = json.loads((output / 'energy_gate.json').read_text())
    assert len(energy['per_chain']) == 2
