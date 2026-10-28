import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from foldtree2.se3_cache import digest, identity_hash
from foldtree2.src.contact_lever_metrics import per_protein_range_metrics, summarize_ranges
from foldtree2.src.contact_levers import InspectablePairHead, XSAPairHead
from foldtree2.src.distance_restraints import eligible_pairs, PairwiseDistanceHead
from foldtree2.tests.test_distance_restraints import example
from scripts.train_contact_levers import ranking_loss, train, batched_forward, make_optimizer


def test_range_metrics_use_per_protein_budget_and_separation():
    pairs = torch.tensor([[0, 24], [0, 25], [0, 26], [0, 27], [0, 28], [0, 29],
                          [1, 25], [1, 26], [1, 27], [1, 28], [0, 12]])
    scores = torch.tensor([.99, .98, .97, .96, .95, .94, .93, .92, .91, .90, .8])
    truth = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).bool()
    ranges = per_protein_range_metrics(scores, truth, pairs, length=30)
    assert ranges['long']['pairs'] == 10
    assert ranges['long']['precision_L5'] == pytest.approx(.5)
    assert ranges['medium']['pairs'] == 1
    assert summarize_ranges([ranges])['long']['chains_with_contacts'] == 1


def test_ranking_loss_penalizes_wrong_order_and_propagates_gradients():
    labels = torch.tensor([0, 0, 8, 8, 8, 8])
    group = [torch.arange(6)]
    wrong = torch.tensor([-2., -1., 1., 2., 3., 4.], requires_grad=True)
    right = -wrong.detach()
    assert ranking_loss(wrong, labels, 3, group) > ranking_loss(right, labels, 3, group)
    ranking_loss(wrong, labels, 3, group).backward()
    assert wrong.grad[:2].sum() < 0 and wrong.grad[2:].sum() > 0


def test_configurable_pair_head_reports_additive_components():
    entry = example(); pairs = eligible_pairs(entry['mask'])
    model = InspectablePairHead(8, hidden=16, spherical_dim=64, correction_hidden=8).eval()
    logits, terms = model(entry, pairs, return_components=True)
    score = torch.logsumexp(logits[:, :3], -1) - torch.logsumexp(logits[:, 3:], -1)
    torch.testing.assert_close(score, terms['dot'] + model.contact_offset + terms['correction'], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(logits, model(entry, pairs.flip(-1)), atol=0, rtol=0)
    restored = InspectablePairHead(**model.model_config()).eval(); restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(logits, restored(entry, pairs), atol=0, rtol=0)


@pytest.mark.parametrize('head_kind', ['baseline', 'pair', 'pair_xsa'])
def test_batched_forward_matches_individual_proteins(head_kind):
    entries = [example(), example()]
    entries[1] = {k: (v[:28] if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == 32 else v) for k, v in entries[1].items()}
    pairs = [eligible_pairs(e['mask'])[:30] for e in entries]
    kwargs = dict(feature_dim=8, hidden=16, context_layers=2, context_heads=8)
    cls = {'baseline': PairwiseDistanceHead, 'pair': InspectablePairHead, 'pair_xsa': XSAPairHead}[head_kind]
    model = (cls(**kwargs) if head_kind == 'baseline' else cls(**kwargs, spherical_dim=8, correction_hidden=8)).eval()
    batched, corrections = batched_forward(model, entries, pairs)
    for idx, (entry, pair) in enumerate(zip(entries, pairs)):
        if head_kind != 'baseline':
            expected, terms = model(entry, pair, return_components=True)
            torch.testing.assert_close(corrections[idx], terms['correction'], atol=1e-6, rtol=1e-5)
        else:
            expected = model(entry, pair)
        torch.testing.assert_close(batched[idx], expected, atol=1e-6, rtol=1e-5)


def test_muon_only_uses_context_weight_matrices():
    model = InspectablePairHead(8, hidden=16, context_layers=2, context_heads=8, spherical_dim=8)
    optimizer = make_optimizer(model, SimpleNamespace(optimizer='muon', learning_rate=.0003, muon_lr=.02))
    muon_ids = {id(p) for p in optimizer.param_groups[0]['params']}
    adam_ids = {id(p) for p in optimizer.param_groups[1]['params']}
    for name, p in model.named_parameters():
        assert (id(p) in muon_ids) == (name.startswith('context.') and p.ndim == 2 and not name.endswith('relative_bias.weight'))
        assert (id(p) in adam_ids) != (id(p) in muon_ids)


@pytest.mark.parametrize('optimizer,head_kind', [('adamw', 'pair'), ('muon', 'pair'), ('adamw', 'pair_xsa')])
def test_epoch_resume_restores_optimizer_and_rng_exactly(tmp_path, optimizer, head_kind):
    cache = tmp_path / 'cache'; (cache / 'entries').mkdir(parents=True)
    identity = {'fixture': True}
    manifest = {'train_ids': ['a', 'b'], 'val_ids': ['c'], 'input_identity': identity,
                'files': {}, 'feature_dim': 8, 'decoder_enrichment': {'contact_dim': 30}}
    for identifier in manifest['train_ids'] + manifest['val_ids']:
        entry = example(); entry.pop('id')
        entry['target'] = torch.stack([(torch.arange(32) % 4).float() * 6, torch.zeros(32), torch.zeros(32)], -1)
        path = cache / 'entries' / (identifier + '.pt')
        torch.save({'id': identifier, 'identity': identity, 'entry': entry}, path)
        manifest['files'][identifier] = {'name': path.name, 'length': 32, 'sha256': digest(path)}
    manifest['bundle_sha256'] = identity_hash(manifest)
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    defaults = dict(cache=str(cache), model=head_kind, device='cpu', train_count=2, train_selection='prefix', val_count=1, batch_size=2,
                    epochs=2, pairs=32, hidden=16, context_layers=2, context_heads=8, optimizer=optimizer, muon_lr=.02, seed=42, learning_rate=.001, contact_weight=.3,
                    ranking_weight=.1, correction_hidden=8, correction_l2=.01, patience=0,
                    min_delta=.0001, resume=False, stop_after_epoch=1)
    resumed = SimpleNamespace(**defaults, output=str(tmp_path/'resumed'))
    train(resumed)
    checkpoint = torch.load(Path(resumed.output)/'resume.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 1 and checkpoint['optimizer']['state']
    resumed.resume = True; resumed.stop_after_epoch = None; train(resumed)
    uninterrupted = SimpleNamespace(**{**defaults, 'stop_after_epoch': None}, output=str(tmp_path/'full'))
    train(uninterrupted)
    a = torch.load(Path(resumed.output)/'last.pt', map_location='cpu', weights_only=False)['state_dict']
    b = torch.load(Path(uninterrupted.output)/'last.pt', map_location='cpu', weights_only=False)['state_dict']
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    assert json.loads((Path(resumed.output)/'completion.json').read_text())['reload_exact']
