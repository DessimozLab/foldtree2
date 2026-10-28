"""Measure FAPE saturation and competing coordinate gradients on frozen inputs."""
import argparse
import json
from pathlib import Path
import torch
from foldtree2.se3_validation import frame_fape, losses, forward
from foldtree2.src.losses.fape import equivariant_ca_frame_rotmat
from foldtree2.src.se3_struct_decoder import se3_denoiser
from scripts.investigate_se3_scale import load_entries, calibrated_entry


def diagnostics(pred, entry):
    p = pred.detach().double().requires_grad_()
    target, mask = entry['target'].double(), entry['mask']
    safe_target = target.masked_fill(~mask[:, None], 0.)
    pr, pu = equivariant_ca_frame_rotmat(p)
    tr, tu = equivariant_ca_frame_rotmat(safe_target)
    fm = mask.clone(); fm[1:] &= mask[:-1]; fm[:-1] &= mask[1:]
    if len(mask) > 2: fm[0] &= mask[2]; fm[-1] &= mask[-3]
    fm &= ~pu & ~tu
    pd, td = p[mask][None]-p[fm][:,None], target[mask][None]-target[fm][:,None]
    pl = torch.einsum('fji,fpj->fpi', pr[fm], pd)
    tl = torch.einsum('fji,fpj->fpi', tr[fm], td)
    error = (pl-tl).norm(dim=-1)
    fape = frame_fape(p,target,mask)
    # Independent explicit row-vector projection, using the shared frame builder.
    reference = torch.stack([(pd[i] @ pr[fm][i]-td[i] @ tr[fm][i]).norm(dim=-1)
                             for i in range(int(fm.sum()))]).clamp(max=10.).mean()/10.
    torch.testing.assert_close(fape, reference)
    valid_p, valid_t = p[mask], target[mask]
    distances = torch.nn.functional.smooth_l1_loss(
        torch.cdist(valid_p-valid_p.mean(0),valid_p-valid_p.mean(0)),
        torch.cdist(valid_t-valid_t.mean(0),valid_t-valid_t.mean(0)))/10.
    fg = torch.autograd.grad(fape,p,retain_graph=True)[0]
    dg = torch.autograd.grad(distances,p)[0]
    return {'fape': float(fape), 'distance_loss': float(distances),
            'unclamped_mean_error_angstrom': float(error.mean()),
            'clamped_pair_fraction': float((error>=10.).double().mean()),
            'valid_frames': int(fm.sum()), 'valid_points': int(mask.sum()),
            'fape_coordinate_gradient_norm': float(fg.norm()),
            'distance_coordinate_gradient_norm': float(dg.norm()),
            'gradient_cosine': float(torch.nn.functional.cosine_similarity(fg.flatten(),dg.flatten(),dim=0)),
            'reference_abs_difference': float((fape-reference).abs()),
            'finite_gradients': bool(torch.isfinite(fg).all() and torch.isfinite(dg).all())}


def main(root):
    provenance, entries = load_entries('overfit')
    ckpt = torch.load(root/'overfit_lifted_circular_best.pt',map_location='cpu',weights_only=False)
    spec = ckpt['spec']; ids = provenance['train_ids']
    model = se3_denoiser(entries[ids[0]]['features'].shape[-1],[32],3,40,.25,
        dropout_p=0.,depth=2,heads=2,dim_head=16,num_atom_types=40,
        coordinate_update='residual',zero_init_coordinate_update=True,
        coordinate_normalization='rms_radius',coordinate_residual_gain=32.,
        angle_output='sincos',device='cpu').float().eval()
    model.load_state_dict(ckpt['model']); rows=[]
    torch.set_num_threads(2)
    for identifier in ids:
        e=calibrated_entry(entries[identifier],spec['scale'],spec['lift_fraction'])
        with torch.no_grad(): pred,angles=forward(model,e)
        rows.append({'id':identifier,'initial':diagnostics(e['coords'],e),'trained':diagnostics(pred,e)})
    (root/'loss_audit.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
