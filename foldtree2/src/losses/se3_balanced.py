"""Versioned coordinate supervision; legacy objective/evaluation remain unchanged."""
import math
import torch
import torch.nn.functional as F
from .se3_ablation import components as legacy_components, AblationLoss

WEIGHTS = {'distance':1., 'training_fape':1., 'bond':1., 'angle':.1,
           'local_distance':0., 'bend':0., 'torsion':0.}


def coordinate_terms(pred, target, mask, cutoff=15., scale=4.):
    """Target-selected pairs and contiguous CA stencils, in physical Angstroms.

    Targets alone determine eligibility. Degenerate predictions stay penalized;
    undefined target torsions and gaps are excluded. No atan2/acos in training.
    """
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[-1] != 3 or mask.shape != pred.shape[:1] or mask.dtype != torch.bool:
        raise ValueError('Expected [N,3] coordinates and boolean [N] mask')
    if not torch.isfinite(pred[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError('Nonfinite valid coordinate-loss input')
    if mask.sum()<3:
        raise ValueError('Fewer than three valid coordinate-loss residues')
    p=pred.masked_fill(~mask[:,None],0.);t=target.masked_fill(~mask[:,None],0.)
    zero=p.sum()*0.
    # Center cdist inputs to avoid translation-dependent matrix cancellation.
    pc=p[mask];tc=t[mask];idx=torch.arange(len(mask),device=mask.device)[mask]
    pd=torch.cdist(pc-pc.mean(0),pc-pc.mean(0));td=torch.cdist(tc-tc.mean(0),tc-tc.mean(0))
    pairs=(idx[None,:]-idx[:,None]>2) & (td<cutoff) & (td>1e-4)
    local=F.smooth_l1_loss(pd[pairs]/scale,td[pairs]/scale) if pairs.any() else zero
    pb=p[1:]-p[:-1];tb=t[1:]-t[:-1];tl=tb.norm(dim=-1)
    bonds=mask[1:] & mask[:-1] & (tl>=2.5) & (tl<=4.5)
    pu=F.normalize(pb,dim=-1,eps=1e-4);tu=F.normalize(tb,dim=-1,eps=1e-4)
    bm=bonds[:-1]&bonds[1:]
    # Cosine error of consecutive bond directions (equivalent bend convention).
    bend=F.smooth_l1_loss((pu[:-1]*pu[1:]).sum(-1)[bm],(tu[:-1]*tu[1:]).sum(-1)[bm]) if bm.any() else zero
    def circular(u):
        b0,b1,b2=-u[:-2],u[1:-1],u[2:]
        v=b0-(b0*b1).sum(-1,keepdim=True)*b1
        w=b2-(b2*b1).sum(-1,keepdim=True)*b1
        defined=(v.norm(dim=-1)>1e-4)&(w.norm(dim=-1)>1e-4)
        v=F.normalize(v,dim=-1,eps=1e-4);w=F.normalize(w,dim=-1,eps=1e-4)
        return torch.stack(((v*w).sum(-1),(torch.cross(b1,v,dim=-1)*w).sum(-1)),dim=-1),defined
    ps,_=circular(pu);ts,defined=circular(tu)
    tm=bonds[:-2]&bonds[1:-1]&bonds[2:]&defined
    # Squared chord distance is 1-cos(delta) for defined unit torsions, and
    # remains a nonnegative penalty for coincident/collinear predictions.
    torsion=.5*(ps[tm]-ts[tm]).square().sum(-1).mean() if tm.any() else zero
    return {'local_distance':local,'bend':bend,'torsion':torsion,
            'local_distance_count':int(pairs.sum()),'bend_count':int(bm.sum()),'torsion_count':int(tm.sum())}


class BalancedGeometryLoss:
    def __init__(self, weights=None, local_cutoff=15., local_scale=4.):
        if weights is not None and (not isinstance(weights,dict) or set(weights)-set(WEIGHTS)):
            raise ValueError('Unknown coordinate-loss weights')
        self.weights={**WEIGHTS,**(weights or {})}
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in self.weights.values()):
            raise ValueError('Weights must be finite nonnegative numbers')
        if not any(self.weights[k]>0 for k in self.weights if k!='angle'):
            raise ValueError('At least one coordinate term must be enabled')
        if not (math.isfinite(local_cutoff) and math.isfinite(local_scale) and local_cutoff>0 and local_scale>0):
            raise ValueError('Local cutoff/scale must be finite positive')
        self.cutoff,self.scale=local_cutoff,local_scale

    def components(self,pred,angles,entry):
        c=legacy_components(pred,angles,entry,True,True)
        c['control_geometry']=c['geometry']
        if any(self.weights[k] for k in ('local_distance','bend','torsion')):
            c.update(coordinate_terms(pred,entry['target'],entry['mask'],self.cutoff,self.scale))
        else:
            c.update({k:pred.new_zeros(()) for k in ('local_distance','bend','torsion')})
        # Undefined target stencils contribute a differentiable zero, with
        # explicit counts in logs. Other valid terms still supervise the chain.
        if not any(self.weights[k] for k in ('distance','training_fape','bond')) and not any(self.weights[k] and c.get(k+'_count',0) for k in ('local_distance','bend','torsion')):
            raise ValueError('No eligible active coordinate terms')
        # The control follows the legacy operation order bit for bit.
        geometry=c['distance']*self.weights['distance']+c['training_fape']*self.weights['training_fape']+c['bond']*self.weights['bond']
        for k in ('local_distance','bend','torsion'):
            if self.weights[k]:geometry=geometry+self.weights[k]*c[k]
        c['geometry']=geometry;c['total']=geometry+self.weights['angle']*c['angle']
        return c

    def __call__(self,pred,angles,entry):
        c=self.components(pred,angles,entry)
        self.last_components={k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in c.items()}
        return c['total'],c['geometry'],c['evaluation_fape'],c['angle']


def build_loss(config=None):
    if config is None:return AblationLoss(True,True)
    if not isinstance(config,dict) or set(config)-{'protocol','weights','local_cutoff','local_scale'} or config.get('protocol')!='balanced_geometry_v1':
        raise ValueError('Unknown loss configuration/protocol')
    return BalancedGeometryLoss(config.get('weights'),config.get('local_cutoff',15.),config.get('local_scale',4.))
