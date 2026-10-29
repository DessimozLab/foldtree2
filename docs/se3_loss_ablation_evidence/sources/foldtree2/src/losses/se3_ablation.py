"""Opt-in residue loss controls; historical evaluation FAPE stays unchanged."""
import torch
import torch.nn.functional as F
from foldtree2.se3_validation import frame_point_errors, losses


def components(pred, angles, entry, soft_fape=False, bond=False):
    _, old_geometry, evaluation_fape, angle = losses(pred, angles, entry)
    distance = old_geometry - evaluation_fape
    if soft_fape:
        error = frame_point_errors(pred, entry['target'], entry['mask']) / 10.
        # Pseudo-Huber, delta=10 Å: linear tail with no hard zero-gradient region.
        training_fape = (error.square() / (torch.sqrt(1. + error.square()) + 1.)).mean()
    else:
        training_fape = evaluation_fape
    target_bonds = (entry['target'][1:] - entry['target'][:-1]).norm(dim=-1)
    # Adjacent rows spanning missing residues must not be treated as a CA bond.
    valid_bonds = entry['mask'][1:] & entry['mask'][:-1] & torch.isfinite(target_bonds)
    valid_bonds &= (target_bonds >= 2.5) & (target_bonds <= 4.5)
    if bond and not valid_bonds.any():
        raise ValueError('No contiguous valid CA bonds for local loss')
    # Select endpoints before arithmetic so masked NaNs cannot poison backward.
    pb = (pred[1:][valid_bonds] - pred[:-1][valid_bonds]).norm(dim=-1)
    bond_loss = F.smooth_l1_loss(pb, target_bonds[valid_bonds]) if valid_bonds.any() else pred.new_zeros(())
    geometry = (distance + training_fape + (bond_loss if bond else 0.)) if (soft_fape or bond) else old_geometry
    return {'total': geometry + .1 * angle, 'geometry': geometry,
            'evaluation_fape': evaluation_fape, 'angle': angle,
            'training_fape': training_fape, 'distance': distance,
            'bond': bond_loss, 'bond_count': int(valid_bonds.sum())}


class AblationLoss:
    def __init__(self, soft_fape=False, bond=False):
        self.soft_fape, self.bond = soft_fape, bond

    def __call__(self, pred, angles, entry):
        c = components(pred, angles, entry, self.soft_fape, self.bond)
        return c['total'], c['geometry'], c['evaluation_fape'], c['angle']
