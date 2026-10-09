"""Full residue refiner checks; float32 SE(3) contract and degenerate chains."""
import importlib
import torch
import pytest
from torch_geometric.data import HeteroData, Batch
import gotennet_pytorch.gotennet as backend

original_harmonics = backend.spherical_harmonics
original_dtype = torch.get_default_dtype()
from foldtree2.src.se3_struct_decoder import se3_denoiser


def model():
    return se3_denoiser(8, [32], 3, 40, .25, dropout_p=0., depth=2,
                        heads=2, dim_head=16, num_atom_types=40).cpu().float().eval()


def graph(n):
    g = HeteroData()
    g['res'].x = torch.randn(n, 8)
    return g


def run(m, g, c, mask=None):
    return m(g.clone(), coords_pred=c, ft2_token_ids=torch.arange(c.shape[0]) % 40,
             edge_attr_dict={'node_mask': mask, 'use_distance_contacts': True})


def test_endpoint_frames_mask_third_neighbors():
    m = model()
    c = torch.randn(10, 3)
    mask = torch.ones(10, dtype=torch.bool)
    mask[[2, -3]] = False
    out = run(m, graph(10), c, mask)
    assert out['twist_undefined'][0]
    assert out['twist_undefined'][-1]


def test_import_has_no_global_side_effects():
    import foldtree2.src.se3_struct_decoder as module
    importlib.reload(module)
    assert torch.get_default_dtype() == original_dtype
    assert backend.spherical_harmonics is original_harmonics


@pytest.mark.parametrize('lengths', [[9], [9, 4], [1], [2], [5, 1]])
@pytest.mark.parametrize('kind', ['random', 'coincident', 'collinear'])
def test_full_refiner_equivariance_and_gradients(lengths, kind):
    m = model()
    g = Batch.from_data_list([graph(n) for n in lengths])
    n = sum(lengths)
    c = torch.randn(n, 3) * 100
    if kind == 'coincident':
        c[:] = torch.tensor([5., 8., -3.])
    if kind == 'collinear':
        c = torch.arange(n).float()[:, None] * torch.tensor([[3., 2., -1.]])
    c.requires_grad_()
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    q[:, 0] *= torch.linalg.det(q)
    t = torch.tensor([300., -75., 25.])
    out = run(m, g, c)
    moved = run(m, g, c @ q.T + t)
    torch.testing.assert_close(moved['coors_out_flat'], out['coors_out_flat'] @ q.T + t, atol=1e-4, rtol=1e-4)
    valid_frames = ~out['twist_undefined'] & ~moved['twist_undefined']
    torch.testing.assert_close(moved['rotmat_pred'][valid_frames], q @ out['rotmat_pred'][valid_frames], atol=1e-4, rtol=1e-4)
    for key in ['z', 'angles']:
        torch.testing.assert_close(moved[key], out[key], atol=1e-4, rtol=1e-4)
    loss = out['coors_out_flat'].square().mean() + out['angles'].square().mean()
    loss.backward()
    assert torch.isfinite(c.grad).all()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(v).all() for v in grads)


def test_masked_nonfinite_and_padding():
    m = model()
    g = Batch.from_data_list([graph(7), graph(3)])
    c = torch.randn(10, 3)
    mask = torch.ones(10, dtype=torch.bool)
    mask[2] = False
    c[2] = float('nan')
    g['res'].x[2] = float('nan')
    out = run(m, g, c, mask)
    assert torch.isfinite(out['coors_out']).all()
    assert out['coors_out'][1, 3:].eq(0).all()
    assert out['coors_out_flat'][2].eq(0).all()
    c[1] = float('inf')
    with pytest.raises(ValueError, match='nonfinite valid coordinates'):
        run(m, g, c, mask)


def test_mixed_batch_frames_match_individual_chains():
    m = model()
    gs = [graph(7), graph(4)]
    cs = [torch.randn(7, 3), torch.randn(4, 3)]
    joined = run(m, Batch.from_data_list(gs), torch.cat(cs))
    # Reuse the same token identities in both representations.
    for start, g, c in [(0, gs[0], cs[0]), (7, gs[1], cs[1])]:
        single = m(g.clone(), coords_pred=c, ft2_token_ids=torch.arange(start,start+len(c)) % 40)
        for key in ['coors_out_flat', 'z', 'angles', 'rotmat_pred']:
            torch.testing.assert_close(joined[key][start:start+len(c)], single[key], atol=1e-4, rtol=1e-4)


def test_continuous_features_reach_the_network():
    m = model()
    g = graph(8)
    c = torch.randn(8, 3)
    out = run(m, g, c)
    g['res'].x = g['res'].x + 3.
    changed = run(m, g, c)
    assert not torch.allclose(out['z'], changed['z'])
    out['z'].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.input2gotennet_fallback.parameters())


def test_nonfinite_valid_features_and_outputs_are_rejected(monkeypatch):
    m = model()
    g = graph(6)
    c = torch.randn(6, 3)
    g['res'].x[1, 2] = float('nan')
    with pytest.raises(ValueError, match='nonfinite valid features'):
        run(m, g, c)
    g['res'].x[1, 2] = 0.
    def bad_forward(atoms, coors, **kwargs):
        return torch.full((*atoms.shape[:2], 32), float('nan')), coors
    monkeypatch.setattr(m.gotennet, 'forward', bad_forward)
    with pytest.raises(RuntimeError, match='nonfinite valid invariant'):
        run(m, g, c)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for device-specific equivariance check')
def test_cuda_float32_refiner_equivariance_and_gradients():
    device = torch.device('cuda:0')
    m = model().to(device)
    g = Batch.from_data_list([graph(9),graph(4)]).to(device)
    c = (torch.randn(13,3,device=device)*100).requires_grad_()
    q,_ = torch.linalg.qr(torch.randn(3,3,device=device)); q[:,0] *= torch.linalg.det(q)
    t = torch.tensor([300.,-75.,25.],device=device)
    original = run(m,g,c)
    transformed = run(m,g,c@q.T+t)
    torch.testing.assert_close(transformed['coors_out_flat'], original['coors_out_flat']@q.T+t, atol=1e-4,rtol=1e-4)
    valid = ~original['twist_undefined'] & ~transformed['twist_undefined']
    torch.testing.assert_close(transformed['rotmat_pred'][valid],q@original['rotmat_pred'][valid],atol=1e-4,rtol=1e-4)
    for key in ['z','angles']:
        torch.testing.assert_close(transformed[key],original[key],atol=1e-4,rtol=1e-4)
    (original['coors_out_flat'].square().mean()+original['angles'].square().mean()).backward()
    assert torch.isfinite(c.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
