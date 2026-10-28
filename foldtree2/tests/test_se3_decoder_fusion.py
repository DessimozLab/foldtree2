import copy
import json
from types import SimpleNamespace

import pytest
import torch

from foldtree2.src.decoder_feature_fusion import (
    DecoderFeatureFusion, DecoderFusionRefiner, MODES, soft_contact_context,
)
from foldtree2.train_se3_lightning import build_model
from foldtree2.src.se3_struct_decoder import enable_tensor_ff_input_norm
from foldtree2.se3_validation import forward
from scripts.se3.enrich_se3_decoder_cache import frozen_aa


def example():
    generator = torch.Generator().manual_seed(91)
    coords = torch.randn(20, 3, generator=generator).cumsum(0) * 3.8
    raw = torch.rand(20, 20, generator=generator)
    return {
        'id': 'synthetic', 'features': torch.randn(20, 8, generator=generator),
        'contact_z': torch.randn(20, 30, generator=generator),
        'aa_probs': torch.randn(20, 20, generator=generator).softmax(-1),
        'contact_probs': (raw + raw.T) / 2,
        'coords': coords, 'target': coords, 'angles': torch.zeros(20, 3),
        'tokens': torch.arange(20), 'mask': torch.ones(20, dtype=torch.bool),
        'dot': torch.eye(20, dtype=torch.bool),
    }


@pytest.mark.parametrize('mode', MODES)
def test_fusion_modes_masks_and_gradients(mode):
    e = example()
    module = DecoderFeatureFusion(8, 30, mode)
    e['mask'][-1] = False
    for key in ('features', 'contact_z', 'aa_probs'):
        e[key][-1] = float('nan')
    result = module(e)
    assert result.shape == (20, 8) and result[-1].eq(0).all()
    assert torch.isfinite(result).all()
    if mode == 'baseline':
        torch.testing.assert_close(result[:-1], e['features'][:-1], atol=0, rtol=0)
    else:
        result.square().mean().backward()
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        changed = copy.deepcopy(e)
        if module.aa is not None:
            changed['aa_probs'][:-1] = changed['aa_probs'][:-1].roll(1, -1)
        if module.contact is not None:
            changed['contact_z'][:-1] = changed['contact_z'][:-1].roll(1, -1)
        assert not torch.allclose(module(changed), result)


def test_soft_pairs_respect_confidence_mask_and_sequence_separation():
    e = example()
    zeros = torch.zeros(20, 20)
    result = soft_contact_context(e['contact_z'], zeros, e['mask'])
    assert result.eq(0).all()
    ones = torch.ones(20, 20)
    full = soft_contact_context(e['contact_z'], ones, e['mask'])
    half = soft_contact_context(e['contact_z'], ones * .5, e['mask'])
    torch.testing.assert_close(half, full * .5)
    near = torch.zeros(20, 20)
    near[0, 1] = near[1, 0] = 1.
    assert soft_contact_context(e['contact_z'], near, e['mask']).eq(0).all()
    with pytest.raises(ValueError, match='symmetric'):
        asymmetric = ones.clone()
        asymmetric[0, 3] = 0
        soft_contact_context(e['contact_z'], asymmetric, e['mask'])


def test_bad_features_are_rejected_and_missing_inputs_never_default():
    e = example()
    module = DecoderFeatureFusion(8, 30, 'both')
    invalid = copy.deepcopy(e)
    invalid['aa_probs'] *= 2
    with pytest.raises(ValueError, match='normalized'):
        module(invalid)
    invalid = copy.deepcopy(e)
    invalid['contact_z'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        module(invalid)
    invalid = copy.deepcopy(e)
    del invalid['aa_probs']
    with pytest.raises(KeyError):
        module(invalid)
    with pytest.raises(ValueError, match='Unknown'):
        DecoderFeatureFusion(8, 30, 'invalid')


def test_shared_adapters_match_across_modes_without_consuming_rng():
    before = torch.get_rng_state()
    both = DecoderFeatureFusion(8, 30, 'both')
    for mode in MODES:
        module = DecoderFeatureFusion(8, 30, mode)
        for name in ('contact', 'aa'):
            adapter = getattr(module, name)
            if adapter is not None:
                torch.testing.assert_close(
                    adapter.state_dict(), getattr(both, name).state_dict(), atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize('mode', MODES)
def test_core_initialization_matches_control_and_equivariance(mode):
    e = example()
    architecture = {'hidden': 32, 'depth': 2, 'heads': 2, 'input_normalization': 'layer'}
    original = enable_tensor_ff_input_norm(build_model(8, architecture))
    model = DecoderFusionRefiner(8, 30, mode, architecture)
    torch.testing.assert_close(original.state_dict(), model.core.state_dict(), atol=0, rtol=0)
    with torch.no_grad():
        model.core.gotennet.proj_to_coors[1].weight.normal_(std=.05)
    model.eval()
    p, a = model(e)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotation[:, 0] *= torch.linalg.det(rotation)
    shift = torch.tensor([1., 3., -5.])
    moved = {**e, 'coords': e['coords'] @ rotation + shift}
    rp, ra = model(moved)
    torch.testing.assert_close(rp, p @ rotation + shift, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ra, a, atol=1e-4, rtol=1e-4)
    model.train()
    tp, ta = model(e)
    torch.testing.assert_close(tp, p, atol=0, rtol=0)
    torch.testing.assert_close(ta, a, atol=0, rtol=0)
    if mode == 'baseline':
        expected_p, expected_a = forward(model.core, e, 'local-only')
        torch.testing.assert_close(expected_p, p, atol=0, rtol=0)
        torch.testing.assert_close(expected_a, a, atol=0, rtol=0)


def test_model_backward_and_checkpoint_reload(tmp_path):
    e = example()
    model = DecoderFusionRefiner(8, 30, 'both_soft_pairs', {
        'hidden': 32, 'depth': 2, 'heads': 2, 'input_normalization': 'layer'})
    with torch.no_grad():
        model.core.gotennet.proj_to_coors[1].weight.normal_(std=.05)
    p, a = model(e)
    (p.square().mean() + a.square().mean()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    for name, parameter in model.fusion.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name
    torch.save(model.state_dict(), tmp_path / 'weights.pt')
    clone = copy.deepcopy(model)
    clone.load_state_dict(torch.load(tmp_path / 'weights.pt', weights_only=True))
    cp, ca = clone(e)
    torch.testing.assert_close(cp, p, atol=0, rtol=0)
    torch.testing.assert_close(ca, a, atol=0, rtol=0)


def test_aa_extractor_uses_only_latent_positions():
    e = example()

    class AA(torch.nn.Module):
        def forward(self, graph):
            assert set(graph.node_types) == {'res', 'positions'}
            assert len(graph.edge_types) == 0
            assert graph['res'].x.shape == (20, 8)
            return {'aa': torch.arange(20.).repeat(20, 1).unsqueeze(1)}

    probabilities = frozen_aa(AA(), e['features'], torch.zeros(20, 256))
    assert probabilities.shape == (20, 20)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(20))


class FrozenAAExample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(8, 20)

    def forward(self, graph):
        assert set(graph.node_types) == {'res', 'positions'}
        assert not graph.edge_types
        return {'aa': self.head(graph['res'].x).unsqueeze(1)}


@pytest.mark.parametrize('fail', [False, True])
def test_enrichment_preserves_cache_and_publishes_only_success(tmp_path, monkeypatch, fail):
    from torch_geometric.data import HeteroData
    from foldtree2.se3_cache import FrozenDataset, digest, identity_hash, open_bundle
    from scripts.se3 import enrich_se3_decoder_cache as enrichment

    parent = tmp_path / 'parent'
    (parent / 'entries').mkdir(parents=True)
    encoder = tmp_path / 'encoder.pt'
    encoder.write_bytes(b'encoder-identity')
    dataset_path = tmp_path / 'graphs.h5'
    dataset_path.write_bytes(b'dataset-identity')
    geometry = torch.nn.Module()
    geometry.normalize = True
    geometry.contact_temp = torch.nn.Parameter(torch.tensor(2.4))
    geometry.contact_bias = torch.nn.Parameter(torch.tensor(-1.3))
    decoder = torch.nn.Module()
    decoder.decoders = torch.nn.ModuleDict({
        'sequence_transformer': FrozenAAExample(), 'geometry_cnn': geometry})
    decoder_path = tmp_path / 'decoder.pt'
    torch.save(decoder, decoder_path)
    pair = {'encoder': encoder, 'decoder': decoder_path}
    identity = {key + '_sha256': digest(path) for key, path in pair.items()}
    identity['dataset_sha256'] = digest(dataset_path)
    manifest = {'train_ids': ['train'], 'val_ids': ['val'], 'input_identity': identity,
                'files': {}, 'scale': 1., 'feature_dim': 8}
    graphs = {}
    for identifier in ('train', 'val'):
        entry = example()
        entry.pop('id')
        entry.pop('aa_probs')
        entry.pop('contact_probs')
        entry['latent'] = entry['features'].clone()
        entry['contact_z'] = torch.nn.functional.normalize(entry['contact_z'], dim=-1)
        entry['mask'][-1] = False
        name = identifier + '.pt'
        torch.save({'id': identifier, 'identity': identity, 'entry': entry}, parent / 'entries' / name)
        manifest['files'][identifier] = {
            'name': name, 'sha256': digest(parent / 'entries' / name), 'length': 20}
        graph = HeteroData()
        graph['positions'].x = torch.zeros(20, 256)
        graph['AA'].x = torch.nn.functional.one_hot(torch.arange(20), 20).float()
        graphs[identifier] = graph
    manifest['bundle_sha256'] = identity_hash(manifest)
    (parent / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(enrichment, 'production_pair', lambda: pair)
    monkeypatch.setattr(enrichment.base.pdbgraphmk2, 'StructureDataset', lambda _: graphs)
    if fail:
        def broken(*args):
            raise ValueError('Deliberate extraction failure')
        monkeypatch.setattr(enrichment, 'frozen_aa', broken)
    output = tmp_path / 'enriched'
    args = SimpleNamespace(cache=parent, output=output, dataset=dataset_path, device='cpu')
    if fail:
        with pytest.raises(ValueError, match='Deliberate'):
            enrichment.enrich(args)
        assert (output / 'failure.json').exists()
        assert not (output / 'manifest.json').exists()
        return
    enrichment.enrich(args)
    enriched = open_bundle(output)
    assert enriched['parent_bundle_sha256'] == manifest['bundle_sha256']
    assert enriched['input_identity'] == identity
    for split in ('train', 'val'):
        assert enriched[split + '_ids'] == manifest[split + '_ids']
        original = FrozenDataset(parent, manifest, split)[0]
        saved = FrozenDataset(output, enriched, split)[0]
        assert set(saved) - set(original) == {'aa_probs', 'contact_probs'}
        for key, value in original.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
            else:
                assert value == saved[key]
        torch.testing.assert_close(saved['aa_probs'].sum(-1), torch.ones(20))
        torch.testing.assert_close(saved['contact_probs'], saved['contact_probs'].T)
    first = FrozenDataset(output, enriched, 'train')[0]
    repeated = frozen_aa(decoder.decoders['sequence_transformer'].eval(),
                         first['latent'], graphs['train']['positions'].x)
    torch.testing.assert_close(first['aa_probs'], repeated, atol=0, rtol=0)
    assert json.loads((output / 'preparation.json').read_text())['frozen_weights_unchanged']