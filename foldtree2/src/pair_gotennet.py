"""Local dense GotenNet pair-state extension; installed backend is untouched.

The dense forward follows gotennet-pytorch 0.3.1. The only architectural change
is a learned invariant residual at edge-state initialization. No sparse-neighbor
backend is introduced. The default model retains the original safe forward.
"""
import torch
from torch import nn
from gotennet_pytorch import gotennet as backend
from .se3_struct_decoder import StableGotenNet,_safe_gotennet_spherical_harmonics
from .pair_features import PAIR_DIM


class PairGotenNet(StableGotenNet):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.cutoff_radius is not None:
            raise ValueError('Pair extension currently requires the existing dense backend')
        dim=self.node_init.atom_embed.out_features
        # Preserve all original parameter draws when this branch is enabled.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(731)
            self.pair_projection=nn.Sequential(nn.Linear(PAIR_DIM,32),nn.SiLU(),nn.Linear(32,dim,bias=False))
            nn.init.zeros_(self.pair_projection[-1].weight)

    def forward(self,atoms,coors,adj_mat=None,lens=None,mask=None,pair_features=None):
        if pair_features is None:
            raise ValueError('Pair-enabled GotenNet requires continuous pair inputs')
        batch,n=atoms.shape[:2]
        if pair_features.shape!=(batch,n,n,PAIR_DIM):raise ValueError('Pair-state shape mismatch')
        if lens is not None:
            if mask is not None:raise ValueError('Choose mask or lengths')
            mask=torch.arange(n,device=atoms.device)[None,:]<lens[:,None]
        if mask is None:mask=torch.ones((batch,n),device=atoms.device,dtype=torch.bool)
        features,positions=[],[]
        for i in range(batch):
            idx=torch.where(mask[i])[0]
            if not len(idx):raise ValueError('Empty pair graph')
            pairs=pair_features[i][idx][:,idx].unsqueeze(0)
            if not torch.isfinite(pairs).all():raise ValueError('Nonfinite valid pair state')
            adj=None if adj_mat is None else adj_mat[i][idx][:,idx].unsqueeze(0)
            h,c=self._dense_forward(atoms[i,idx].unsqueeze(0),coors[i,idx].unsqueeze(0),adj,pairs)
            features.append(h.new_zeros(n,h.shape[-1]).index_copy(0,idx,h[0]))
            positions.append(c.new_zeros(n,3).index_copy(0,idx,c[0]))
        return torch.stack(features),torch.stack(positions)

    def _dense_forward(self,atoms,coors,adj_mat,pairs):
        # Explicit local adaptation of the backend's dense path. Keep the same
        # operations/order so zero initialization is exactly the control.
        rel_pos=backend.einx.subtract('b i c, b j c -> b i j c',coors,coors)
        rel_dist=rel_pos.norm(dim=-1)
        if adj_mat is not None:
            eye=torch.eye(atoms.shape[1],device=atoms.device,dtype=torch.bool)
            adj_mat=adj_mat&~eye
        h=self.node_init(atoms,rel_dist,adj_mat,mask=None,neighbor_indices=None,neighbor_mask=None)
        t=self.edge_init(h,rel_dist,neighbor_indices=None)
        pair_update=self.pair_projection(pairs.to(t.dtype))
        valid=~torch.eye(atoms.shape[1],device=atoms.device,dtype=torch.bool)
        t=t+pair_update.masked_fill(~valid[None,:,:,None],0.)
        r=[_safe_gotennet_spherical_harmonics(d,rel_pos,normalize=True,normalization='norm') for d in range(1,self.max_degree+1)]
        x=self.high_degree_init(h,t,r,mask=None,neighbor_indices=None,neighbor_mask=None)
        value_residuals=None;h=self.expand_streams(h)
        for (htr,maybe_attn,attn,ff),(maybe_res,attn_res,ff_res) in zip(self.layers,self.residual_fns):
            t=htr(t,x,neighbor_indices=None)+t
            if maybe_attn is not None:
                h,add=maybe_res(h);h=add(maybe_attn(h,mask=None))
            h,add=attn_res(h)
            (hr,xr),next_values=attn(h,t,r,x,mask=None,neighbor_indices=None,neighbor_mask=None,value_residuals=value_residuals,return_value_residual=True)
            h=add(hr);x=[a+b for a,b in zip(x,xr)]
            value_residuals=backend.default(value_residuals,next_values)
            h,add=ff_res(h);hr,xr=ff(h,x);h=add(hr);x=[a+b for a,b in zip(x,xr)]
        h=self.reduce_streams(h)
        if self.final_norm:
            h=self.h_final_norm(h);x=[norm(v) for norm,v in zip(self.x_final_norms,x)]
        if self.proj_invariant is not None:h=self.proj_invariant(h)
        if self.proj_to_coors is None:raise ValueError('Pair refiner requires coordinate output')
        return h,self.proj_to_coors(x[0])
