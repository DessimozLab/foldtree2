"""Learned equivariant correction to the existing target-free normal seed lift."""
import math
import torch
from torch import nn
from .geometry_seeds import lift_seed_with_local_normals
from .losses.fape import equivariant_ca_frame_rotmat


def validate_seed_config(config):
    if config is None:return
    if not isinstance(config,dict) or set(config)!={'protocol','max_offset_fraction'} or config['protocol']!='learned_local_seed_v1':
        raise ValueError('Unknown initial geometry configuration')
    value=config['max_offset_fraction']
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError('Seed correction bound must be finite positive')


class LearnedGeometrySeed(nn.Module):
    def __init__(self,feature_dim,max_offset_fraction=.25):
        super().__init__();validate_seed_config({'protocol':'learned_local_seed_v1','max_offset_fraction':max_offset_fraction})
        self.max_offset_fraction=max_offset_fraction
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(732)
            self.mlp=nn.Sequential(nn.LayerNorm(feature_dim,elementwise_affine=False),nn.Linear(feature_dim,32),nn.GELU(),nn.Linear(32,3))
            nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)

    def forward(self,coords,features,mask,batch=None):
        if coords.shape!=(len(features),3) or mask.shape!=(len(features),) or mask.dtype!=torch.bool:
            raise ValueError('Invalid learned-seed inputs')
        if not torch.isfinite(coords[mask]).all() or not torch.isfinite(features[mask]).all():
            raise ValueError('Nonfinite valid learned-seed inputs')
        if batch is None:batch=torch.zeros(len(coords),device=coords.device,dtype=torch.long)
        if batch.shape!=mask.shape:raise ValueError('Seed batch mismatch')
        result=torch.zeros_like(coords)
        for group in torch.unique(batch,sorted=True):
            idx=torch.where(batch==group)[0];c=coords[idx];f=features[idx];m=mask[idx]
            if not m.any():raise ValueError('Empty learned-seed chain')
            safe=c.masked_fill(~m[:,None],0.);safe_f=f.masked_fill(~m[:,None],0.)
            base=lift_seed_with_local_normals(safe,safe_f,m,.1)
            centered=(safe-safe[m].mean(0)).masked_fill(~m[:,None],0.)
            frames,undefined=equivariant_ca_frame_rotmat(centered)
            valid=m&~undefined;valid[1:]&=m[:-1];valid[:-1]&=m[1:]
            if len(m)>2:valid[0]&=m[2];valid[-1]&=m[-3]
            amplitudes=self.mlp(safe_f)
            # Smooth vector-norm bound, without selecting a global spatial axis.
            amplitudes=amplitudes/torch.sqrt(1+amplitudes.square().sum(-1,keepdim=True))
            radius=centered[m].square().sum(-1).mean().clamp_min(1e-12).sqrt()
            delta=torch.einsum('nij,nj->ni',frames.masked_fill(~valid[:,None,None],0.),amplitudes)
            out=(base+self.max_offset_fraction*radius*delta).masked_fill(~m[:,None],0.)
            result=result.index_copy(0,idx,out)
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite learned geometry')
        return result
