import torch
import pytest
from torch_geometric.data import HeteroData
from foldtree2.train_se3_lightning import build_model,Refiner,check_resume
from foldtree2.se3_validation import forward
from foldtree2.src.recycled_refiner import refinement_weights,refinement_objective
from foldtree2.src.losses.se3_ablation import AblationLoss

ARCH={'hidden':32,'depth':2,'heads':2}


def entry(n=12):
    torch.manual_seed(270)
    target=torch.cat([torch.zeros(1,3),3.8*torch.nn.functional.normalize(torch.randn(n-1,3),dim=-1).cumsum(0)])
    return {'features':torch.randn(n,8),'coords':target+.3*torch.randn(n,3),'tokens':torch.arange(n)%40,
            'dot':torch.eye(n),'mask':torch.ones(n,dtype=torch.bool),'target':target,'angles':torch.zeros(n,3)}


def nonzero_head(model):
    with torch.no_grad():model.gotennet.proj_to_coors[1].weight.normal_(0,.002)


def test_shared_parameters_initialization_and_first_pass_match_control_exactly():
    single=build_model(8,ARCH).eval();recycled=build_model(8,ARCH,2).eval()
    assert single.state_dict().keys()==recycled.state_dict().keys()
    for k,v in single.state_dict().items():torch.testing.assert_close(v,recycled.state_dict()[k],atol=0,rtol=0)
    assert sum(p.numel() for p in single.parameters())==sum(p.numel() for p in recycled.parameters())
    nonzero_head(single);recycled.load_state_dict(single.state_dict());e=entry()
    a=forward(single,e);b=forward(recycled,e,return_steps=True)
    assert len(b)==2
    for x,y in zip(a,b[0]):torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert not torch.equal(b[0][0],b[1][0])


def test_features_normalize_once_and_full_backprop_reaches_first_pass():
    model=build_model(8,ARCH,2).train();nonzero_head(model);e=entry()
    before=int(model.bn.num_batches_tracked);calls=[]
    handle=model.bn.register_forward_hook(lambda *args:calls.append(True))
    outputs=forward(model,e,return_steps=True);handle.remove()
    assert len(calls)==1 and int(model.bn.num_batches_tracked)==before+1
    outputs[0][0].retain_grad()
    values,diagnostics=refinement_objective(AblationLoss(True,True),outputs,e,[0,1])
    values[0].backward()
    assert outputs[0][0].grad is not None and torch.isfinite(outputs[0][0].grad).all() and outputs[0][0].grad.norm()>0
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert len(diagnostics)==2 and diagnostics[0]['weight']==0
    torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
    before=model.gotennet.proj_to_coors[1].weight.detach().clone()
    torch.optim.AdamW(model.parameters(),lr=5e-4).step()
    assert not torch.equal(before,model.gotennet.proj_to_coors[1].weight)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())


def test_two_pass_rigid_equivariance_and_masked_nan_padding():
    model=build_model(8,ARCH,2).eval();nonzero_head(model);e=entry()
    e['mask'][-2:]=False;e['features'][-2:]=float('nan');e['coords'][-2:]=float('nan')
    q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q);shift=torch.tensor([9.,-4.,2.])
    moved={**e,'coords':e['coords']@q+shift}
    a=forward(model,e,return_steps=True);b=forward(model,moved,return_steps=True)
    for (p,angles),(rp,ra) in zip(a,b):
        torch.testing.assert_close(rp[e['mask']],(p@q+shift)[e['mask']],atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(ra,angles,atol=1e-4,rtol=1e-4)
        assert p[~e['mask']].eq(0).all() and angles[~e['mask']].eq(0).all()


def test_contacts_are_recomputed_from_updated_coordinates():
    model=build_model(8,ARCH,2).eval();e=entry();seen=[]
    original=model.gotennet.forward
    def capture(*args,**kwargs):
        seen.append(kwargs['adj_mat'].clone())
        invariant,coords=original(*args,**kwargs)
        # Move the first pass into a compact, noncoincident chain. Collapsing
        # all points exactly would correctly exclude zero-distance contacts.
        compact=torch.arange(kwargs['coors'].shape[1],dtype=coords.dtype)[None,:,None]*torch.tensor([[[.001,0.,0.]]])
        return invariant,(compact-kwargs['coors'])/model.coordinate_residual_gain
    model.gotennet.forward=capture
    e['coords']=torch.arange(12.)[:,None]*torch.tensor([[20.,0.,0.]])
    forward(model,e,return_steps=True)
    assert len(seen)==2 and not torch.equal(seen[0],seen[1])


def test_mixed_length_graphs_transform_independently():
    model=build_model(8,ARCH,2).eval();nonzero_head(model)
    e=entry();batch=torch.tensor([0]*5+[1]*7);g=HeteroData();g['res'].x=e['features'];g['res'].batch=batch
    p=model(g,coords_pred=e['coords'],ft2_token_ids=e['tokens'],edge_attr_dict={'node_mask':e['mask']})
    shifts=torch.tensor([[5.,-3.,2.],[-8.,7.,4.]])[batch]
    r=model(g,coords_pred=e['coords']+shifts,ft2_token_ids=e['tokens'],edge_attr_dict={'node_mask':e['mask']})
    torch.testing.assert_close(r['coors_out_flat'],p['coors_out_flat']+shifts,atol=1e-4,rtol=1e-4)
    torch.testing.assert_close(r['angles'],p['angles'],atol=1e-4,rtol=1e-4)


def test_auxiliary_loss_and_resume_provenance_are_explicit():
    e=entry();outputs=[(e['coords'].requires_grad_(),e['angles']),(e['target']+.2,e['angles'])]
    fn=AblationLoss(True,True);a=fn(*outputs[0],e);b=fn(*outputs[1],e)
    values,_=refinement_objective(fn,outputs,e,[.3,.7])
    torch.testing.assert_close(values[0],.3*a[0]+.7*b[0]);torch.testing.assert_close(values[2],b[2])
    manifest={'feature_dim':8,'bundle_sha256':'b','input_identity':{},'scale':1}
    config={'model':ARCH};single=Refiner(config,manifest);two=Refiner({**config,'refinement_steps':2},manifest)
    saved={'provenance':single.provenance,'topology':{'nodes':1,'world_size':1},'stream':{'cursor':0}}
    with pytest.raises(ValueError,match='configuration'):check_resume(saved,two.provenance,saved['topology'])


@pytest.mark.parametrize('steps,weights',[(True,None),(3,None),(2,[1]),(2,[.5,.6]),(2,[-1,2]),(2,[float('nan'),1])])
def test_invalid_recycling_configuration_is_rejected(steps,weights):
    with pytest.raises(ValueError):refinement_weights(steps,weights)



@pytest.mark.parametrize('n',[1,2,5])
def test_short_coincident_chains_remain_finite(n):
    model=build_model(8,ARCH,2).eval();e=entry(n);e['coords'].zero_()
    for p,a in forward(model,e,return_steps=True):
        assert torch.isfinite(p).all() and torch.isfinite(a).all()


def test_dictionary_inputs_are_not_mutated_across_passes():
    model=build_model(8,ARCH,2).eval();e=entry();data={'res':e['features']}
    before=e['features'].clone()
    model(data,coords_pred=e['coords'],ft2_token_ids=e['tokens'],edge_attr_dict={'node_mask':e['mask']})
    assert set(data)=={'res'};torch.testing.assert_close(data['res'],before,atol=0,rtol=0)


def test_weights_only_migration_rejects_loss_data_and_unreviewed_source_changes():
    import copy,json
    from pathlib import Path
    from foldtree2.train_se3_lightning import check_refinement_warm_start,source_identity
    approved=json.loads(Path('foldtree2/src/recycling_parent_sources.json').read_text())['approved_parent_sources'][0]
    prior={'bundle':'b','input_identity':'i','preprocessing':{'scale':1},'feature_dim':8,
           'training':{'model':ARCH,'learning_rate':5e-4,'epochs':16},'sources':approved}
    now=copy.deepcopy(prior);now['training'].update(refinement_steps=2,refinement_loss_weights=[0,1],epochs=4);now['sources']=source_identity()
    check_refinement_warm_start(prior,now)
    for key,value in [('bundle','changed'),('input_identity','changed')]:
        bad=copy.deepcopy(now);bad[key]=value
        with pytest.raises(ValueError):check_refinement_warm_start(prior,bad)
    for key,value in [('loss',{'protocol':'balanced_geometry_v1'}),('learning_rate',1e-4)]:
        bad=copy.deepcopy(now);bad['training'][key]=value
        with pytest.raises(ValueError,match='loss, model or training'):check_refinement_warm_start(prior,bad)
    bad=copy.deepcopy(now);bad['sources']['src/losses/se3_balanced.py']='changed'
    with pytest.raises(ValueError,match='Unreviewed'):check_refinement_warm_start(prior,bad)
    bad=copy.deepcopy(prior);bad['sources']['train_se3_lightning.py']='unrecognized'
    with pytest.raises(ValueError,match='Unrecognized'):check_refinement_warm_start(bad,now)
