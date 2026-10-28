"""CPU audit on selected training IDs only; no optimizer or production calls."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from foldtree2.se3_cache import open_bundle,FrozenDataset,digest
from foldtree2.se3_validation import forward
from foldtree2.train_se3_lightning import build_model,calibrated
from foldtree2.src.se3_struct_decoder import enable_tensor_ff_input_norm
from foldtree2.src.losses.se3_balanced import BalancedGeometryLoss,WEIGHTS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=8);a=p.parse_args()
    if a.count<=0:raise ValueError('Positive audit count required')
    out=Path(a.output)
    if out.exists():raise ValueError('Audit output exists')
    out.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    manifest=open_bundle(a.cache);dataset=FrozenDataset(a.cache,manifest,'train')
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    prior=c['se3']['provenance']
    # Local audit can use an exactly retained subset cache; every selected ID is
    # checked against the original hashed growth cohort, never validation IDs.
    cohort=json.loads(Path('docs/se3_clariden_evidence/growth_8192/cohort.json').read_text())
    train=set(cohort['train_ids']);val=set(cohort['val_ids'])
    if prior['input_identity']!=manifest['input_identity']:raise ValueError('Frozen source mismatch')
    model=build_model(manifest['feature_dim'],prior['training']['model'])
    if prior['training'].get('tensor_ff_input_norm'):enable_tensor_ff_input_norm(model)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in c['state_dict'].items()})
    model.eval();rows=[]
    weights={**WEIGHTS,'local_distance':1.,'bend':.5,'torsion':.5}
    loss=BalancedGeometryLoss(weights)
    for i in range(min(a.count,len(dataset))):
        entry=dataset[i]
        if entry['id'] not in train or entry['id'] in val:raise ValueError('Audit ID outside training split')
        e=calibrated(entry,manifest['scale'])
        with torch.no_grad():pred,angles=forward(model,e)
        pred=pred.detach().double().requires_grad_();e={k:v.double() if isinstance(v,torch.Tensor) and v.is_floating_point() else v for k,v in e.items()}
        terms=loss.components(pred,angles.double(),e);grads={}
        for key in WEIGHTS:
            if key=='angle':continue
            grads[key]=torch.autograd.grad(terms[key],pred,retain_graph=True)[0]*weights[key]
        total=torch.autograd.grad(terms['total'],pred)[0]
        if not torch.isfinite(total).all() or not all(torch.isfinite(g).all() for g in grads.values()):raise FloatingPointError('Nonfinite coordinate gradients')
        row={'id':entry['id'],'length':len(pred),'components':{k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in terms.items()},'weighted_coordinate_gradient_norms':{k:float(v.norm()) for k,v in grads.items()},'gradient_cosines':{x+':'+y:float(F.cosine_similarity(grads[x].flatten(),grads[y].flatten(),dim=0)) for x in grads for y in grads if x<y}}
        # One real model backward verifies the new objective reaches parameters.
        if i==0:
            model.zero_grad(set_to_none=True);p2,a2=forward(model,calibrated(entry,manifest['scale']));loss(p2,a2,calibrated(entry,manifest['scale']))[0].backward()
            gs=[p.grad for p in model.parameters() if p.grad is not None]
            if not gs or not all(torch.isfinite(g).all() for g in gs):raise FloatingPointError('Nonfinite model gradients')
            row['model_gradient_norm']=float(torch.stack([g.norm() for g in gs]).norm())
            if row['model_gradient_norm']==0:raise RuntimeError('Zero model gradient')
        rows.append(row)
    record={'protocol':'balanced_geometry_v1-audit','checkpoint_sha256':digest(a.checkpoint),'bundle_sha256':manifest['bundle_sha256'],'weights':weights,'selected_ids':[x['id'] for x in rows],'split':'training only','production_forward_calls':0,'optimizer_steps':0,'torch':torch.__version__,'sources':{n:digest(n) for n in ('scripts/audit_se3_balanced_loss.py','foldtree2/src/losses/se3_balanced.py','foldtree2/train_se3_lightning.py')},'rows':rows}
    (out/'audit.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'structures':len(rows),'finite_gradients':True,'model_gradient_norm':rows[0]['model_gradient_norm'],'output':str(out)}))

if __name__=='__main__':main()
