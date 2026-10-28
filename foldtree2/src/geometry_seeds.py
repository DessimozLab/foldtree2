"""Experimental residue seed construction; no target coordinates are consumed."""
import math
import torch
from foldtree2.src.losses.fape import equivariant_ca_frame_rotmat


def lift_seed_with_local_normals(coords, features, mask, fraction=.1):
    """Lift a planar seed using scalar features and oriented chain normals.

    Cross-product normals transform under SO(3), allowing a three-dimensional
    seed without random noise or a fixed global axis. This deliberately adds a
    parity-sensitive component; reflection equivariance is not claimed. The
    prototype uses feature column zero and a fixed fraction of seed RMS radius.
    It is not a validated production preprocessing choice.
    """
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError('fraction must be finite and nonnegative')
    if coords.ndim != 2 or coords.shape[-1] != 3 or features.ndim != 2 or len(features) != len(coords):
        raise ValueError('Expected matching residue coordinates and scalar features')
    mask=mask.to(device=coords.device,dtype=torch.bool).reshape(-1)
    if len(mask) != len(coords) or not mask.any() or features.shape[-1] < 1:
        raise ValueError('Expected a nonempty valid residue mask and scalar features')
    if not torch.isfinite(coords[mask]).all() or not torch.isfinite(features[mask]).all():
        raise ValueError('Nonfinite valid seed inputs')
    center=coords[mask].mean(0)
    centered=(coords-center).masked_fill(~mask[:,None],0.)
    frames,undefined=equivariant_ca_frame_rotmat(centered)
    valid=mask & ~undefined
    valid[1:] &= mask[:-1];valid[:-1] &= mask[1:]
    if len(mask)>2:
        valid[0] &= mask[2];valid[-1] &= mask[-3]
    normal=frames[:,:,1].masked_fill(~valid[:,None],0.)
    scalar=features[:,0].masked_fill(~mask,0.)
    scalar=(scalar-scalar[mask].mean()).masked_fill(~mask,0.)
    scalar=scalar/scalar[mask].square().mean().clamp_min(1e-12).sqrt()
    radius=centered[mask].square().sum(-1).mean().clamp_min(1e-12).sqrt()
    result=(coords+fraction*radius*scalar[:,None]*normal).masked_fill(~mask[:,None],0.)
    if not torch.isfinite(result).all():raise RuntimeError('Nonfinite lifted seed')
    return result
