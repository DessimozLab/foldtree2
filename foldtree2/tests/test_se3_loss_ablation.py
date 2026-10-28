import torch
from foldtree2.src.losses.se3_ablation import AblationLoss, components
from foldtree2.se3_validation import losses


def entry():
    torch.manual_seed(91)
    target = torch.randn(12, 3).cumsum(0)
    return {'target': target, 'mask': torch.ones(12, dtype=torch.bool),
            'angles': torch.randn(12, 3)}


def test_control_exactly_matches_original_and_soft_tail_has_gradient():
    e = entry(); pred = e['target'] * 30; a = torch.zeros(12, 3)
    for x, y in zip(AblationLoss()(pred, a, e), losses(pred, a, e)):
        torch.testing.assert_close(x, y, atol=0., rtol=0.)
    pred = pred.double().requires_grad_(); e['target'] = e['target'].double()
    c = components(pred, a, e, soft_fape=True)
    assert c['training_fape'] > c['evaluation_fape']
    grad = torch.autograd.grad(c['training_fape'], pred)[0]
    assert torch.isfinite(grad).all() and grad.norm() > 0


def test_local_bonds_mask_gaps_and_penalize_stretching():
    t = torch.tensor([[0., 0., 0.], [3.8, 0., 0.], [6., 3., 0.],
                      [25., 4., 1.], [28.8, 4., 1.], [31., 7., 2.]])
    e = {'target': t, 'mask': torch.ones(6, dtype=torch.bool), 'angles': torch.zeros(6, 3)}
    c = components(t * 2, torch.zeros(6, 3), e, bond=True)
    assert c['bond_count'] == 4 and c['bond'] > 0
    exact = components(t, torch.zeros(6, 3), e, bond=True, soft_fape=True)
    assert exact['bond'] == 0 and exact['geometry'] == 0


def test_new_objective_masks_nan_and_is_rigid_motion_invariant():
    torch.manual_seed(92)
    target = torch.randn(14, 3).cumsum(0) * 2
    pred = target + torch.randn_like(target) * .3
    mask = torch.ones(14, dtype=torch.bool); mask[6] = False
    e = {'target': target, 'mask': mask, 'angles': torch.randn(14, 3)}
    a = torch.zeros(14, 3)
    reference = components(pred, a, e, True, True)
    q, _ = torch.linalg.qr(torch.randn(3, 3)); q[:, 0] *= torch.linalg.det(q)
    moved = components(pred @ q.T + 20, a, {**e, 'target': target @ q.T + 20}, True, True)
    for k in ('total', 'training_fape', 'bond'):
        torch.testing.assert_close(reference[k], moved[k], atol=1e-5, rtol=1e-4)
    pred[~mask] = float('nan'); target[~mask] = float('nan'); pred.requires_grad_()
    result = components(pred, a, e, True, True)
    grad = torch.autograd.grad(result['total'], pred)[0]
    assert torch.isfinite(grad).all() and grad[~mask].eq(0).all()
