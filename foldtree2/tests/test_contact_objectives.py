import copy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from foldtree2.src.distance_restraints import PairwiseDistanceHead, eligible_pairs
from foldtree2.tests.test_distance_restraints import example
from scripts.contact.experiment_contact_objectives import balanced_contact_loss, contact_logit, experiment_cases


def test_configurable_cases_preserve_dimension_and_weight():
    assert experiment_cases(SimpleNamespace(cases=['32:0.1', '64:0.3'])) == [
        ('sphere32_w0p1', 32, .1), ('sphere64_w0p3', 64, .3)]
    assert experiment_cases(SimpleNamespace(cases=['baseline', '64:0.3'])) == [
        ('baseline', 0, 0.), ('sphere64_w0p3', 64, .3)]
    for invalid in (['32:-1'], ['0:0.1'], ['32:nan'], ['32:0.1', '32:0.10'], ['baseline', 'baseline']):
        with pytest.raises(ValueError):
            experiment_cases(SimpleNamespace(cases=invalid))


def test_matched_cases_save_progress_checkpoints_and_final_metrics(tmp_path):
    from foldtree2.se3_cache import digest, identity_hash
    from scripts.contact.experiment_contact_objectives import run

    cache = tmp_path / 'cache'
    (cache / 'entries').mkdir(parents=True)
    identity = {'fixture': True}
    manifest = {'train_ids': ['a', 'b'], 'val_ids': ['c'], 'input_identity': identity,
                'files': {}, 'feature_dim': 8, 'decoder_enrichment': {'contact_dim': 30}}
    for identifier in manifest['train_ids'] + manifest['val_ids']:
        entry = example()
        entry.pop('id')
        path = cache / 'entries' / (identifier + '.pt')
        torch.save({'id': identifier, 'identity': identity, 'entry': entry}, path)
        manifest['files'][identifier] = {'name': path.name, 'length': 32, 'sha256': digest(path)}
    manifest['bundle_sha256'] = identity_hash(manifest)
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    output = tmp_path / 'screen'
    args = SimpleNamespace(cache=str(cache), output=str(output), device='cpu', train_count=2,
                           val_count=1, epochs=1, pairs=32, hidden=16, contact_weight=1.,
                           seed=42, cases=['baseline', '64:0.3'])
    run(args)
    summary = json.loads((output / 'summary.json').read_text())
    assert set(summary) == {'baseline', 'sphere64_w0p3'}
    assert all(case['reload_exact'] for case in summary.values())
    for name in summary:
        case = output / name
        assert torch.load(case / 'last.pt', map_location='cpu', weights_only=False)['epoch'] == 1
        assert [json.loads(line)['event'] for line in (case / 'events.jsonl').read_text().splitlines()] == ['ready', 'complete']
        assert len(json.loads((case / 'history.json').read_text())) == 1


@pytest.mark.parametrize('dim', [4, 8, 16])
def test_spherical_contact_mass_symmetry_and_gradients(dim):
    entry = example()
    model = PairwiseDistanceHead(8, hidden=16, spherical_dim=dim).eval()
    pairs = eligible_pairs(entry['mask'])
    nodes = model._nodes(entry)
    unit = F.normalize(model.contact_projection(nodes), dim=-1)
    i, j = pairs.T
    expected = 10. * (unit[i] * unit[j]).sum(-1) - 5.
    logits = model(entry, pairs)
    torch.testing.assert_close(contact_logit(logits, 3), expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(model.contact_probability(logits.softmax(-1)), expected.sigmoid())
    torch.testing.assert_close(logits, model(entry, pairs.flip(-1)), atol=0, rtol=0)
    loss = F.cross_entropy(logits, torch.arange(len(pairs)) % 10)
    loss.backward()
    assert model.contact_projection.weight.grad.norm() > 0
    assert model.contact_log_scale.grad.abs() > 0
    assert model.contact_offset.grad.abs() > 0
    restored = PairwiseDistanceHead(**model.model_config()).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(logits, restored(entry, pairs), atol=0, rtol=0)


def test_balanced_loss_resists_negative_replication_and_has_correct_gradient():
    logits = torch.zeros(2, 10, requires_grad=True)
    labels = torch.tensor([0, 9])
    loss = balanced_contact_loss(logits, labels, 3)
    repeat = torch.tensor([0] + [1] * 20)
    torch.testing.assert_close(loss, balanced_contact_loss(logits[repeat], labels[repeat], 3))
    loss.backward()
    assert logits.grad[0, :3].sum() < 0
    assert logits.grad[1, :3].sum() > 0


def test_contact_logit_recovers_group_probability():
    logits = torch.randn(10, 10) * 20
    expected = logits.softmax(-1)[:, :3].sum(-1)
    torch.testing.assert_close(contact_logit(logits, 3).sigmoid(), expected)
