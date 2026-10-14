"""Opt-in shared-weight coordinate recycling for the residue SE(3) refiner."""
import math
import torch
from .se3_struct_decoder import se3_denoiser


def refinement_weights(steps, weights=None):
    if type(steps) is not int or steps not in (1,2):
        raise ValueError('Only one or two refinement passes are validated candidates')
    if weights is None:
        return [0.]*(steps-1)+[1.]
    if not isinstance(weights,(list,tuple)) or len(weights)!=steps:
        raise ValueError('One loss weight per refinement pass is required')
    if any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<0 for w in weights) or not math.isclose(sum(weights),1.,abs_tol=1e-8,rel_tol=0.):
        raise ValueError('Pass loss weights must be finite, nonnegative and sum to one')
    return list(weights)


class RecycledSE3Denoiser(se3_denoiser):
    """Same parameters/state keys, shared across trained coordinate passes.

    Frozen features are normalized once; geometry contacts are rebuilt by the
    base refiner using current predicted coordinates at every pass. Seed lift
    belongs upstream and is not repeated. No target coordinates are consumed.
    """
    def __init__(self,*args,refinement_steps=2,**kwargs):
        refinement_weights(refinement_steps)
        super().__init__(*args,**kwargs)
        self.refinement_steps=refinement_steps

    def forward(self,data,edge_attr_dict=None,**kwargs):
        if kwargs.get('coords_pred_atoms') is not None:
            raise ValueError('Recycling is residue-only')
        if '_normalized_features' in kwargs:
            raise ValueError('Recycled refiner prepares its own features')
        coords=kwargs.get('coords_pred')
        if coords is None:
            raise ValueError('Recycled refiner requires predicted residue coordinates')
        features=data['res'] if isinstance(data,dict) else data.x_dict['res']
        mask=(edge_attr_dict or {}).get('node_mask')
        if mask is None:
            mask=torch.ones(len(features),device=features.device,dtype=torch.bool)
        prepared=self.normalize_residue_features(features,mask)
        outputs=[]
        for _ in range(self.refinement_steps):
            step_kwargs={**kwargs,'coords_pred':coords,'_normalized_features':prepared}
            # The base implementation adds temporary keys to dictionary inputs.
            step_data=dict(data) if isinstance(data,dict) else data
            out=super().forward(step_data,edge_attr_dict=edge_attr_dict,**step_kwargs)
            coords=out['coors_out_flat']
            if coords is None or not torch.isfinite(coords[mask]).all():
                raise FloatingPointError('Invalid recycled coordinates')
            outputs.append(out)
        result=dict(outputs[-1]);result['refinement_outputs']=outputs
        return result


def refinement_objective(loss_fn,outputs,entry,weights):
    weights=refinement_weights(len(outputs),weights)
    values=[loss_fn(p,a,entry) for p,a in outputs]
    if not all(bool(torch.isfinite(v)) for row in values for v in row):
        raise FloatingPointError('Nonfinite intermediate refinement loss')
    if len(values)==1:
        result=values[0]
    elif weights==[0.,1.]:
        result=values[-1]
    else:
        result=tuple(sum(w*row[k] for w,row in zip(weights,values) if w) if k!=2 else values[-1][2] for k in range(4))
    # Evaluation FAPE is always from the final pass. Component diagnostics on
    # a configurable loss likewise refer to its final invocation.
    diagnostics=[{'pass':i+1,'weight':w,'total':float(row[0].detach()),'geometry':float(row[1].detach()),'evaluation_fape':float(row[2].detach())} for i,(w,row) in enumerate(zip(weights,values))]
    return result,diagnostics
