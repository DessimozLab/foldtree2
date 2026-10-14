"""Target-free invariant pair channels from frozen decoder outputs and geometry."""
import math
import torch
import torch.nn.functional as F

PAIR_DIM=12


def validate_pair_config(config):
    if config is None:return
    keys={'protocol','decoder_sha256','contact_temperature','contact_bias'}
    if not isinstance(config,dict) or set(config)!=keys or config['protocol']!='decoder_pairs_v1':
        raise ValueError('Expected decoder_pairs_v1 with pinned decoder calibration')
    if not isinstance(config['decoder_sha256'],str) or not config['decoder_sha256']:
        raise ValueError('Pair calibration requires decoder identity')
    for k in ('contact_temperature','contact_bias'):
        if isinstance(config[k],bool) or not isinstance(config[k],(int,float)) or not math.isfinite(config[k]):
            raise ValueError('Nonfinite/invalid pair calibration')


def decoder_pair_features(contact_z,coords,mask,config):
    """[N,N,12]: cosine, bounded logit, probability, sequence and distance RBFs.

    Self pairs and invalid nodes are zero. Original indices preserve gaps.
    Geometry is predicted geometry only, in restored physical units.
    """
    validate_pair_config(config)
    if config is None:raise ValueError('Pair configuration required')
    n=len(coords)
    if coords.shape!=(n,3) or contact_z.ndim!=2 or len(contact_z)!=n or mask.shape!=(n,) or mask.dtype!=torch.bool:
        raise ValueError('Pair inputs must be aligned residue tensors')
    if not mask.any():raise ValueError('Empty pair graph')
    if not torch.isfinite(coords[mask]).all() or not torch.isfinite(contact_z[mask]).all():
        raise ValueError('Nonfinite valid pair inputs')
    if (contact_z[mask].norm(dim=-1)<=1e-8).any():
        raise ValueError('Undefined valid decoder contact embedding')
    z=F.normalize(contact_z.masked_fill(~mask[:,None],0.),dim=-1)
    cosine=(z@z.T).clamp(-1,1)
    logits=config['contact_temperature']*cosine+config['contact_bias']
    probability=logits.sigmoid().clamp(1e-6,1-1e-6)
    index=torch.arange(n,device=coords.device)
    sep=(index[:,None]-index[None,:]).abs().to(coords.dtype)
    sequence=torch.stack(((sep/32).clamp(max=1),(sep==1).to(coords.dtype),(sep<=2).to(coords.dtype)),dim=-1)
    safe=coords.masked_fill(~mask[:,None],0.)
    center=safe.sum(0)/mask.sum().clamp_min(1)
    distances=torch.cdist(safe-center,safe-center)
    centers=coords.new_tensor([0.,4.,8.,12.,20.,32.])
    radial=torch.exp(-.5*((distances[...,None]-centers)/4.).square())
    features=torch.cat((cosine[...,None],(logits/8).tanh()[...,None],probability[...,None],sequence,radial),dim=-1)
    valid=mask[:,None]&mask[None,:]&~torch.eye(n,device=mask.device,dtype=torch.bool)
    return features.masked_fill(~valid[...,None],0.)
