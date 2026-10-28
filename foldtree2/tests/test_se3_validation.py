import torch
from foldtree2.se3_validation import production_pair, length_bucket_batches, frame_fape


def test_manifest_pair_matches_epoch_40():
    pair = production_pair()
    assert all('epoch_40.pt' in p for p in pair.values())


def test_length_bucketing_preserves_samples_and_reduces_padding():
    lengths = [4, 120, 3, 100, 5, 110]
    batches = length_bucket_batches(lengths, 2, 42)
    assert sorted(i for batch in batches for i in batch) == list(range(6))
    cost = sum(len(b)*max(lengths[i] for i in b)**2 for b in batches)
    naive = sum(2*max(lengths[i:i+2])**2 for i in range(0, 6, 2))
    assert cost < naive


def test_masked_fape_is_invariant_and_exact_targets_zero():
    torch.manual_seed(12)
    target = torch.randn(12, 3)
    pred = target + torch.randn_like(target)*.05
    mask = torch.ones(12, dtype=torch.bool)
    mask[4] = False
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    q[:, 0] *= torch.linalg.det(q)
    torch.testing.assert_close(frame_fape(pred, target, mask), frame_fape(pred@q.T+50, target@q.T+50, mask), atol=1e-5, rtol=1e-4)
    assert frame_fape(target, target, mask).item() == 0.


def test_endpoint_fape_ignores_masked_third_neighbors():
    torch.manual_seed(73)
    target = torch.randn(10, 3)
    pred = target + .1 * torch.randn_like(target)
    mask = torch.ones(10, dtype=torch.bool)
    mask[[2, -3]] = False
    reference = frame_fape(pred, target, mask)
    pred[~mask] = torch.randn(2, 3) * 100
    torch.testing.assert_close(frame_fape(pred, target, mask), reference)


def test_fape_masked_nan_coordinates_have_finite_valid_gradients():
    torch.manual_seed(82)
    target = torch.randn(10, 3, dtype=torch.float64)
    pred = target + torch.randn_like(target) * .1
    mask = torch.ones(10, dtype=torch.bool); mask[4] = False
    expected = frame_fape(pred, target, mask)
    pred[~mask] = float('nan'); target[~mask] = float('nan')
    pred.requires_grad_()
    actual = frame_fape(pred, target, mask)
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual, pred)[0]
    assert torch.isfinite(grad).all() and grad[~mask].eq(0).all()


def test_fape_reference_independent_rigid_motion_and_gradcheck():
    from foldtree2.src.losses.fape import equivariant_ca_frame_rotmat, coarse_backbone_fape_loss
    torch.manual_seed(83)
    target = torch.randn(7, 3, dtype=torch.float64)
    pred = (target + torch.randn_like(target) * .1).requires_grad_()
    mask = torch.ones(7, dtype=torch.bool)
    pr, _ = equivariant_ca_frame_rotmat(pred)
    tr, _ = equivariant_ca_frame_rotmat(target)
    reference = coarse_backbone_fape_loss(target[:, None], pred[:, None], tr, pr, target, pred) / 10.
    torch.testing.assert_close(frame_fape(pred, target, mask), reference)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64)); q[:, 0] *= torch.linalg.det(q)
    torch.testing.assert_close(frame_fape(pred @ q.T + 17, target, mask), reference)
    assert torch.autograd.gradcheck(lambda p: frame_fape(p, target, mask), (pred,), atol=1e-4, rtol=1e-3)


def test_fape_rejects_nonfinite_valid_coordinates():
    import pytest
    p = torch.randn(6, 3); t = p.clone(); p[2, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite valid FAPE'):
        frame_fape(p, t, torch.ones(6, dtype=torch.bool))


def test_resume_restores_optimizer_scheduler_and_rejects_changed_provenance():
    import copy
    import pytest
    from foldtree2.se3_validation import restore_training_state
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)
    model(torch.ones(1,2)).sum().backward()
    opt.step(); scheduler.step(.3)
    prov = {'dataset': 'hash', 'train_ids': ['a'], 'val_ids': ['b'], 'model': '32-2'}
    ckpt = copy.deepcopy({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                          'provenance': prov, 'epoch': 4, 'global_step': 11, 'best': .3, 'bad': 2,
                          'torch_rng': torch.get_rng_state(), 'cuda_rng': None})
    fresh = torch.nn.Linear(2,1)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=.1)
    fresh_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fresh_opt)
    result = restore_training_state(ckpt, prov, fresh, fresh_opt, fresh_scheduler, torch.device('cpu'))
    assert result == (5,11,.3,2)
    assert fresh_opt.param_groups[0]['lr'] == 5e-4
    assert fresh_scheduler.state_dict() == scheduler.state_dict()
    for p, q in zip(model.parameters(), fresh.parameters()): torch.testing.assert_close(p, q)
    for key in ['dataset','train_ids','val_ids','model']:
        with pytest.raises(ValueError, match='Resume rejected'):
            restore_training_state(ckpt, {**prov,key:'changed'}, fresh, fresh_opt, fresh_scheduler, torch.device('cpu'))


def test_selection_records_ineligible_structures_and_keeps_split_disjoint():
    from types import SimpleNamespace
    from foldtree2.se3_validation import split_structures
    torch.manual_seed(44)
    groups = {}
    for identifier, n, confidence in [('long',400,1.),('masked',12,.1),('a',12,1.),('b',10,1.),('c',8,1.)]:
        groups[identifier] = {'node': {'res': {'x': torch.zeros(n,8).numpy()},
                                      'coords': {'x': torch.randn(n,3).numpy()},
                                      'bondangles': {'x': torch.ones(n,3).numpy()},
                                      'plddt': {'x': torch.full((n,1),confidence).numpy()}}}
    ds = SimpleNamespace(structlist=list(groups), h5dataset={'structs':groups})
    train,val,skipped = split_structures(ds,2,1,256,0)
    assert len(train)==2 and len(val)==1 and not set(train)&set(val)
    assert {s['reason'] for s in skipped} == {'length','fewer_than_three_valid_residues'}


def test_ca_virtual_angle_errors_are_zero_for_identical_coordinates():
    from foldtree2.se3_validation import ca_angle_errors
    c = torch.randn(12, 3)
    result = ca_angle_errors(c,c,torch.ones(12,dtype=torch.bool))
    assert result['ca_bend_mae_radians'] == 0.
    assert result['ca_torsion_mae_radians'] == 0.


def test_cached_batch_collation_preserves_individual_refiner_outputs():
    from scripts.profile_se3_validation import collate_cached
    from foldtree2.src.se3_struct_decoder import se3_denoiser
    from foldtree2.se3_validation import forward
    torch.manual_seed(9)
    entries = []
    for n in (7,3):
        entries.append({'features':torch.randn(n,8),'coords':torch.randn(n,3),
                        'tokens':torch.arange(n),'mask':torch.ones(n,dtype=torch.bool),
                        'dot':torch.zeros(n,n,dtype=torch.bool)})
    entries[0]['mask'][2]=False
    m=se3_denoiser(8,[32],3,40,.25,dropout_p=0.,depth=2,heads=2,dim_head=16,num_atom_types=40).cpu().float().eval()
    graph,kwargs = collate_cached(entries)
    joined=m(graph,**kwargs)
    offset=0
    for entry in entries:
        coords,angles=forward(m,entry)
        n=len(coords)
        torch.testing.assert_close(joined['coors_out_flat'][offset:offset+n],coords,atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(joined['angles'][offset:offset+n],angles,atol=1e-4,rtol=1e-4)
        offset+=n


def test_geometry_loss_remains_invariant_under_large_translation():
    from foldtree2.se3_validation import losses
    torch.manual_seed(19)
    target=torch.randn(50,3)*10
    pred=target+torch.randn_like(target)
    entry={'target':target,'mask':torch.ones(50,dtype=torch.bool),'angles':torch.zeros(50,3)}
    original=losses(pred,torch.zeros(50,3),entry)
    moved=losses(pred+1000,torch.zeros(50,3),{**entry,'target':target+1000})
    for a,b in zip(original,moved): torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-4)


def test_lddt_excludes_nonzero_numerical_self_distances(monkeypatch):
    from foldtree2.se3_validation import metrics
    torch.manual_seed(31)
    target=torch.randn(64,3)*4;pred=torch.randn(64,3)
    real_cdist=torch.cdist
    p=pred-pred.mean(0);t=target-target.mean(0)
    pd,td=real_cdist(p,p),real_cdist(t,t)
    neighbors=(td<15)&(td>0)&~torch.eye(64,dtype=torch.bool)
    expected=torch.stack([((pd-td).abs()[neighbors]<cutoff).float().mean() for cutoff in (.5,1.,2.,4.)]).mean()
    def nonzero_diagonal(x,y,*args,**kwargs):
        result=real_cdist(x,y,*args,**kwargs)
        result.fill_diagonal_(1.)
        return result
    monkeypatch.setattr(torch,'cdist',nonzero_diagonal)
    entry={'target':target,'mask':torch.ones(64,dtype=torch.bool),'angles':torch.zeros(64,3)}
    torch.testing.assert_close(torch.tensor(metrics(pred,None,entry)['lddt']),expected)


def test_train_only_calibration_recovers_uniform_scale_and_is_equivariant():
    from scripts.investigate_se3_scale import fit_scale,calibrated_entry
    torch.manual_seed(15)
    coords=torch.randn(20,3);q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
    mask=torch.ones(20,dtype=torch.bool);mask[4]=False
    entry={'coords':coords,'target':coords@q.T*7+100,'mask':mask}
    factor=fit_scale([entry]);assert abs(factor-7)<1e-5
    calibrated=calibrated_entry(entry,factor)
    moved=calibrated_entry({**entry,'coords':coords@q.T+50},factor)
    torch.testing.assert_close(moved['coords'][mask],calibrated['coords'][mask]@q.T+50,atol=1e-4,rtol=1e-4)
    assert moved['coords'][~mask].eq(0).all()


def test_normal_lift_escapes_plane_and_transforms_under_se3():
    from foldtree2.src.geometry_seeds import lift_seed_with_local_normals
    theta=torch.arange(12)*2*torch.pi/12
    coords=torch.stack([theta.cos(),theta.sin(),torch.zeros(12)],-1)
    features=(3*theta).sin()[:,None]
    mask=torch.ones(12,dtype=torch.bool)
    lifted=lift_seed_with_local_normals(coords,features,mask)
    assert torch.linalg.svdvals(lifted-lifted.mean(0))[-1]>.01
    q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
    moved=lift_seed_with_local_normals(coords@q.T+100,features,mask)
    torch.testing.assert_close(moved,lifted@q.T+100,atol=1e-4,rtol=1e-4)


def test_normal_lift_keeps_degenerate_and_masked_stencils_safe():
    from foldtree2.src.geometry_seeds import lift_seed_with_local_normals
    coords=torch.zeros(7,3,requires_grad=True);features=torch.zeros(7,2,requires_grad=True)
    mask=torch.ones(7,dtype=torch.bool);mask[2]=False
    result=lift_seed_with_local_normals(coords,features,mask)
    assert result.eq(0).all()
    result.sum().backward();assert torch.isfinite(coords.grad).all()
    assert torch.isfinite(features.grad).all()
    coords=torch.randn(7,3);features=torch.randn(7,2);features[2]=float('nan');coords[2]=float('nan')
    result=lift_seed_with_local_normals(coords,features,mask)
    assert torch.isfinite(result).all() and result[2].eq(0).all()
    # Endpoint zero's normal uses residue two and must not move across that gap.
    torch.testing.assert_close(result[0],coords[0])
