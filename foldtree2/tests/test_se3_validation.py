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
