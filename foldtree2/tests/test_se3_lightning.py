import copy
import json
import torch
import pytest
from foldtree2.se3_cache import GlobalBatchSampler, global_order, identity_hash, open_bundle


def test_global_stream_equal_across_topologies():
    lengths = [(i * 17) % 256 for i in range(64)]
    groups = global_order(lengths, 42, 3)
    assert sorted(i for group in groups for i in group) == list(range(64))
    for world in [1, 2, 4, 8, 16]:
        streams = [list(GlobalBatchSampler(lengths, {'epoch':3,'cursor':0}, r, world)) for r in range(world)]
        reconstructed = [streams[r][i] for i in range(len(streams[0])) for r in range(world)]
        assert reconstructed == [i for group in groups for i in group]
        rest = [list(GlobalBatchSampler(lengths, {'epoch':3,'cursor':2}, r, world)) for r in range(world)]
        assert [rest[r][i] for i in range(len(rest[0])) for r in range(world)] == [i for g in groups[2:] for i in g]


def test_prefetch_never_advances_consumed_cursor():
    state = {'epoch':0,'cursor':0}
    sampler = GlobalBatchSampler(list(range(32)), state, 0, 4)
    list(sampler)
    assert state['cursor'] == 0
    state['cursor'] = 1
    assert len(list(sampler)) == 4


def test_invalid_global_batches_rejected():
    with pytest.raises(ValueError):
        global_order(list(range(17)),42,0)
    with pytest.raises(ValueError):
        GlobalBatchSampler(list(range(32)),{},0,3)


def test_manifest_detects_corruption(tmp_path):
    (tmp_path/'entries').mkdir(); (tmp_path/'entries'/'x.pt').write_bytes(b'wrong')
    m={'train_ids':['a'],'val_ids':['b'],'files':{'a':{'name':'x.pt','sha256':'bad'},'b':{'name':'x.pt','sha256':'bad'}}}
    m['bundle_sha256']=identity_hash(m)
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='hash mismatch'):
        open_bundle(tmp_path)


def test_resume_provenance_and_boundary():
    from foldtree2.train_se3_lightning import check_resume
    p={'bundle':'x','training':{'model':'small'}}
    saved={'provenance':p,'topology':{'world_size':4,'nodes':1},'stream':{'cursor':0}}
    check_resume(saved,p,saved['topology'])
    check_resume(saved,p,{'world_size':8,'nodes':2},True)
    with pytest.raises(ValueError,match='Topology'):
        check_resume(saved,p,{'world_size':8,'nodes':2})
    bad=copy.deepcopy(saved);bad['stream']['cursor']=1
    with pytest.raises(ValueError,match='epoch boundary'):
        check_resume(bad,p,{'world_size':8,'nodes':2},True)
    with pytest.raises(ValueError,match='configuration'):
        check_resume(saved,{},saved['topology'])


@pytest.mark.parametrize('width,depth,heads',[(32,2,2),(64,4,4),pytest.param(128,6,8,marks=pytest.mark.xfail(strict=True,reason='GotenNet 0.3.1 tensor stream overflows at depth six; rollout blocked'))])
def test_capacity_equivariance_and_gradients(width,depth,heads):
    from foldtree2.train_se3_lightning import build_model
    from torch_geometric.data import HeteroData
    from foldtree2.se3_validation import forward
    torch.manual_seed(42)
    model=build_model(8,{'hidden':width,'depth':depth,'heads':heads}).eval()
    coords=torch.randn(8,3)
    entry={'features':torch.randn(8,8),'coords':coords,'tokens':torch.arange(8),
           'dot':torch.eye(8),'mask':torch.ones(8,dtype=torch.bool)}
    q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
    shifted=dict(entry,coords=coords@q+torch.tensor([1.,2.,3.]))
    p,a=forward(model,entry); rp,ra=forward(model,shifted)
    torch.testing.assert_close(rp,p@q+torch.tensor([1.,2.,3.]),atol=1e-4,rtol=1e-4)
    torch.testing.assert_close(ra,a,atol=1e-4,rtol=1e-4)
    (p.square().mean()+a.square().mean()).backward()
    grads=[v.grad for v in model.parameters() if v.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)>0


def test_safe_tensor_norm_matches_double_reference_and_gradients():
    from foldtree2.src.se3_struct_decoder import StableHighDegreeNorm
    from gotennet_pytorch.gotennet import HighDegreeNorm
    safe=StableHighDegreeNorm(8)
    reference=HighDegreeNorm(8).double()
    for magnitude in [0.,1.,1e20]:
        x=(torch.randn(2,8,3)*magnitude).requires_grad_()
        y=x.detach().double().requires_grad_()
        a=safe(x);b=reference(y)
        torch.testing.assert_close(a.double(),b,atol=1e-6,rtol=1e-4)
        weights=torch.randn_like(a)
        (a*weights).sum().backward();(b*weights.double()).sum().backward()
        assert torch.isfinite(x.grad).all()
        torch.testing.assert_close(x.grad.double(),y.grad,atol=1e-6,rtol=1e-4)


def test_tensor_ff_normalization_is_local_and_limits_quadratic_growth():
    from gotennet_pytorch.gotennet import EquivariantFeedForward
    from foldtree2.src.se3_struct_decoder import enable_tensor_ff_input_norm
    torch.manual_seed(42)
    original = EquivariantFeedForward(8, 2)
    normalized = copy.deepcopy(original)
    enable_tensor_ff_input_norm(normalized)
    assert isinstance(original.mlps[0][0], torch.nn.Identity)
    assert isinstance(normalized.mlps[0][0], torch.nn.LayerNorm)
    # Parameter-free input normalization preserves the initial parameter draw.
    assert set(original.state_dict()) == set(normalized.state_dict())
    h = torch.randn(1, 5, 8)
    x = [torch.randn(1, 5, 8, m) for m in (3, 5)]
    large_x = [(v * 1e8).requires_grad_() for v in x]
    _, large = normalized(h, large_x)
    _, exploding = original(h, large_x)
    assert max(float(v.detach().abs().max()) for v in large) < 1e10
    assert max(float(v.detach().abs().max()) for v in exploding) > 1e14
    sum(v.square().mean() / 1e16 for v in large).backward()
    assert all(torch.isfinite(v.grad).all() for v in large_x)


def test_normalized_medium_remains_equivariant_with_finite_gradients():
    from foldtree2.train_se3_lightning import build_model
    from foldtree2.src.se3_struct_decoder import enable_tensor_ff_input_norm
    from foldtree2.se3_validation import forward
    torch.manual_seed(42)
    model = enable_tensor_ff_input_norm(build_model(8, {'hidden':64,'depth':4,'heads':4})).eval()
    entry = {'features':torch.randn(8,8), 'coords':torch.randn(8,3),
             'tokens':torch.arange(8), 'dot':torch.eye(8), 'mask':torch.ones(8,dtype=torch.bool)}
    q, _ = torch.linalg.qr(torch.randn(3,3)); q[:,0] *= torch.linalg.det(q)
    shift = torch.tensor([1.,2.,3.])
    p, a = forward(model, entry)
    rp, ra = forward(model, dict(entry, coords=entry['coords'] @ q + shift))
    torch.testing.assert_close(rp, p @ q + shift, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ra, a, atol=1e-4, rtol=1e-4)
    (p.square().mean() + a.square().mean()).backward()
    grads = [v.grad for v in model.parameters() if v.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_normalization_ablation_is_explicit_in_resume_provenance():
    from foldtree2.train_se3_lightning import Refiner, check_resume, check_warm_start_normalization
    from gotennet_pytorch.gotennet import EquivariantFeedForward
    manifest = {'feature_dim':8, 'bundle_sha256':'bundle', 'input_identity':{}, 'scale':1.}
    config = {'model':{'hidden':64,'depth':4,'heads':4}}
    original = Refiner(config, manifest)
    candidate = Refiner(dict(config, tensor_ff_input_norm=True), manifest)
    ff = lambda m: next(x for x in m.model.modules() if isinstance(x, EquivariantFeedForward))
    assert isinstance(ff(original).mlps[0][0], torch.nn.Identity)
    assert isinstance(ff(candidate).mlps[0][0], torch.nn.LayerNorm)
    saved = {'provenance':original.provenance, 'topology':{'world_size':1,'nodes':1},
             'stream':{'cursor':0}}
    with pytest.raises(ValueError, match='configuration'):
        check_resume(saved, candidate.provenance, saved['topology'])
    with pytest.raises(ValueError, match='boolean'):
        Refiner(dict(config, tensor_ff_input_norm='true'), manifest)
    check_warm_start_normalization({}, {})
    check_warm_start_normalization({'tensor_ff_input_norm':True}, {'tensor_ff_input_norm':True})
    with pytest.raises(ValueError, match='normalization mismatch'):
        check_warm_start_normalization({}, {'tensor_ff_input_norm':True})
    with pytest.raises(ValueError, match='normalization mismatch'):
        check_warm_start_normalization({'tensor_ff_input_norm':True}, {})


def test_budget_reservation_is_bounded(tmp_path):
    from scripts.se3_budget import reserve
    schedule=tmp_path/'budget.json';ledger=tmp_path/'ledger.jsonl'
    schedule.write_text(json.dumps({'ceiling_gpuh':2,'stages':[{'name':'one','nodes':1,'hours':.5,'gpuh':2}]}))
    cmd=['sbatch','--nodes=1','--account=a0117','--time=30','job.sbatch']
    reserve(schedule,ledger,'one',2,cmd)
    with pytest.raises(ValueError,match='credit'):
        reserve(schedule,ledger,'one',1,cmd)
    with pytest.raises(ValueError,match='resource'):
        reserve(schedule,ledger,'one',2,cmd+['--nodes=2'])
    ledger.write_text(json.dumps({'stage':'other','reserved_gpuh':2})+'\n')
    with pytest.raises(ValueError,match='ceiling'):
        reserve(schedule,ledger,'one',3,cmd)
