"""Architecture gates: control compatibility, conditioning and rigid geometry."""
import copy
import pytest
import torch
from torch_geometric.data import HeteroData
from foldtree2.tests.test_se3_recycling import ARCH,entry,nonzero_head
from foldtree2.train_se3_lightning import (build_model,calibrated,Refiner,
    load_architecture_weights,check_refinement_warm_start)
from foldtree2.se3_validation import forward
from foldtree2.src.pair_features import decoder_pair_features
from foldtree2.src.learned_geometry_seed import LearnedGeometrySeed
from foldtree2.src.geometry_seeds import lift_seed_with_local_normals

PAIR={'protocol':'decoder_pairs_v1','decoder_sha256':'test-decoder',
      'contact_temperature':2.4314587116241455,'contact_bias':-1.3124585151672363}
SEED={'protocol':'learned_local_seed_v1','max_offset_fraction':.25}


def sample(n=12):
    e=entry(n);e['contact_z']=torch.nn.functional.normalize(torch.randn(n,10),dim=-1)
    return e


def enabled(steps=1):
    return build_model(8,ARCH,steps,PAIR,SEED)


def activate(model):
    nonzero_head(model)
    with torch.no_grad():
        model.gotennet.pair_projection[-1].weight.normal_(0,.04)
        model.seed_initializer.mlp[-1].weight.normal_(0,.04)
        model.out_angles[-1].weight.normal_(0,.02)


def test_pair_formula_masks_and_rigid_invariance():
    e=sample();e['mask'][3]=False;e['contact_z'][3]=float('nan');e['coords'][3]=float('nan')
    p=decoder_pair_features(e['contact_z'],e['coords'],e['mask'],PAIR)
    z=e['contact_z'];cos=(z[0]*z[5]).sum().clamp(-1,1)
    torch.testing.assert_close(p[0,5,2],torch.sigmoid(PAIR['contact_temperature']*cos+PAIR['contact_bias']))
    assert p[0,5,3]==5/32 and p[2,4,3]==2/32
    assert p[3].eq(0).all() and p[:,3].eq(0).all() and p.diagonal(dim1=0,dim2=1).eq(0).all()
    q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
    r=decoder_pair_features(z,e['coords']@q+torch.tensor([8.,-4.,3.]),e['mask'],PAIR)
    torch.testing.assert_close(p,r,atol=1e-4,rtol=1e-4)
    e['contact_z'][1,0]=float('inf')
    with pytest.raises(ValueError,match='Nonfinite'):decoder_pair_features(e['contact_z'],e['coords'],e['mask'],PAIR)


def test_zero_extensions_preserve_common_parameters_and_control_exactly():
    a=build_model(8,ARCH).eval();b=enabled().eval()
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
    nonzero_head(a);b.load_state_dict(a.state_dict(),strict=False);e=sample()
    x=forward(a,calibrated(e,1.2));y=forward(b,calibrated(e,1.2,0.))
    for u,v in zip(x,y):torch.testing.assert_close(u,v,atol=0,rtol=0)


@pytest.mark.parametrize('steps',[1,2])
def test_active_extensions_equivariance_padding_and_gradients(steps):
    model=enabled(steps).eval();activate(model);e=sample();e['mask'][-2:]=False
    for k in ('coords','features','contact_z'):e[k][-2:]=float('nan')
    q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q);shift=torch.tensor([5.,-3.,2.])
    p,a=forward(model,e);r,ra=forward(model,{**e,'coords':e['coords']@q+shift})
    torch.testing.assert_close(r[e['mask']],(p@q+shift)[e['mask']],atol=1e-4,rtol=1e-4)
    torch.testing.assert_close(a,ra,atol=1e-4,rtol=1e-4)
    assert p[~e['mask']].eq(0).all() and a[~e['mask']].eq(0).all()
    (p.square().mean()+a.square().mean()).backward()
    for module in (model.gotennet.pair_projection,model.seed_initializer):
        grads=[v.grad for v in module.parameters() if v.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.norm() for g in grads)>0
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)


def test_continuous_decoder_scores_condition_predictions_with_same_graph():
    model=enabled().eval();activate(model);e=sample();seen=[]
    h=model.gotennet.register_forward_pre_hook(lambda m,args,kw:seen.append(kw['adj_mat'].clone()),with_kwargs=True)
    p,a=forward(model,e);other={**e,'contact_z':torch.randn_like(e['contact_z'])};r,ra=forward(model,other);h.remove()
    assert torch.equal(seen[0],seen[1])
    assert (p-r).abs().max()>1e-5 and (a-ra).abs().max()>1e-6


def test_recycling_seed_once_bn_once_pairs_recompute_geometry_only():
    model=enabled(2).train();activate(model);e=sample();seeds=[];pairs=[]
    h=model.seed_initializer.register_forward_hook(lambda *args:seeds.append(1))
    h2=model.gotennet.register_forward_pre_hook(lambda m,args,kw:pairs.append(kw['pair_features'].detach().clone()),with_kwargs=True)
    forward(model,e);h.remove();h2.remove()
    assert len(seeds)==1 and int(model.bn.num_batches_tracked)==1 and len(pairs)==2
    torch.testing.assert_close(pairs[0][...,:6],pairs[1][...,:6],atol=0,rtol=0)
    assert not torch.equal(pairs[0][...,6:],pairs[1][...,6:])


@pytest.mark.parametrize('n',[1,2,5])
@pytest.mark.parametrize('kind',['coincident','collinear'])
def test_degenerate_seed_has_finite_backward_and_no_arbitrary_offsets(n,kind):
    seed=LearnedGeometrySeed(8);e=sample(n);c=torch.zeros(n,3)
    if kind=='collinear':c[:,0]=torch.arange(n)*3.8
    c.requires_grad_()
    with torch.no_grad():seed.mlp[-1].weight.normal_()
    p=seed(c,e['features'],e['mask'])
    torch.testing.assert_close(p,c,atol=0,rtol=0)
    p.square().sum().backward()
    assert torch.isfinite(c.grad).all()
    assert all(torch.isfinite(v.grad).all() for v in seed.parameters() if v.grad is not None)


def test_mixed_length_independent_chain_transforms_and_padding():
    model=enabled(2).eval();activate(model);e=sample();batch=torch.tensor([0]*5+[1]*7)
    e['mask'][3]=False
    for k in ('features','coords','contact_z'):e[k][3]=float('nan')
    def run(c):
        g=HeteroData();g['res'].x=e['features'];g['res'].batch=batch
        return model(g,coords_pred=c,ft2_token_ids=e['tokens'],edge_attr_dict={'node_mask':e['mask'],'contact_z':e['contact_z']})
    qs=[]
    for _ in range(2):
        q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q);qs.append(q)
    qs=torch.stack(qs);shifts=torch.tensor([[4.,2.,-3.],[-6.,1.,5.]])[batch]
    a=run(e['coords']);b=run(torch.einsum('ni,nij->nj',e['coords'],qs[batch])+shifts)
    expected=torch.einsum('ni,nij->nj',a['coors_out_flat'],qs[batch])+shifts
    torch.testing.assert_close(b['coors_out_flat'][e['mask']],expected[e['mask']],atol=1e-4,rtol=1e-4)
    torch.testing.assert_close(a['angles'],b['angles'],atol=1e-4,rtol=1e-4)


def test_decoder_identity_required_and_migration_only_adds_extension_parameters():
    manifest={'feature_dim':8,'bundle_sha256':'b','input_identity':{'decoder_sha256':'test-decoder'},'scale':1.}
    config={'model':ARCH};a=Refiner(config,manifest)
    new={**config,'pair_information':PAIR,'initial_geometry':SEED};b=Refiner(new,manifest)
    load_architecture_weights(b,a.state_dict(),config,new,True)
    broken=dict(a.state_dict());broken.pop('model.token_embedding.weight')
    with pytest.raises(ValueError,match='base weights'):load_architecture_weights(b,broken,config,new,True)
    with pytest.raises(ValueError,match='base weights'):load_architecture_weights(b,a.state_dict(),new,new,True)
    with pytest.raises(ValueError,match='decoder hash'):Refiner(new,{**manifest,'input_identity':{'decoder_sha256':'other'}})
    check_refinement_warm_start(a.provenance,b.provenance)
    bad=copy.deepcopy(b.provenance);bad['preprocessing']['scale']=2.
    with pytest.raises(ValueError,match='preprocessing'):check_refinement_warm_start(a.provenance,bad)
    e=sample();e.pop('contact_z')
    with pytest.raises(ValueError,match='contact_z'):forward(b.model,e)
