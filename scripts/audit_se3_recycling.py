"""CPU inference/backward audit of shared recycling; no learning claim or promotion."""
import argparse
import json
from pathlib import Path
import time
import torch
from foldtree2.se3_cache import FrozenDataset,open_bundle,digest
from foldtree2.se3_validation import forward,metrics
from foldtree2.train_se3_lightning import build_model,calibrated,provenance,check_refinement_warm_start,source_identity
from foldtree2.src.se3_struct_decoder import enable_tensor_ff_input_norm
from foldtree2.src.losses.se3_ablation import AblationLoss
from foldtree2.src.recycled_refiner import refinement_objective


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=8);a=p.parse_args()
    root=Path(a.output)
    if root.exists() or a.count<=0:raise ValueError('Use a fresh output and positive count')
    root.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    manifest=open_bundle(a.cache);dataset=FrozenDataset(a.cache,manifest,'train')
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);prior=c['se3']['provenance']
    if prior['input_identity']!=manifest['input_identity'] or prior['preprocessing']['scale']!=manifest['scale']:
        raise ValueError('Frozen source or preprocessing mismatch')
    cfg={**prior['training'],'refinement_steps':2,'refinement_loss_weights':[0.,1.]}
    parent_manifest={'bundle_sha256':prior['bundle'],'feature_dim':prior['feature_dim'],
                     'input_identity':prior['input_identity'],'scale':prior['preprocessing']['scale']}
    check_refinement_warm_start(prior,provenance(cfg,parent_manifest))
    # The local cache is a retained subset; audit only known growth training IDs.
    cohort=json.loads(Path('docs/se3_clariden_evidence/growth_8192/cohort.json').read_text())
    train,val=set(cohort['train_ids']),set(cohort['val_ids'])
    models=[build_model(manifest['feature_dim'],prior['training']['model'],steps) for steps in (1,2)]
    for model in models:
        if prior['training'].get('tensor_ff_input_norm'):enable_tensor_ff_input_norm(model)
        model.load_state_dict({k.removeprefix('model.'):v for k,v in c['state_dict'].items()});model.eval()
    rows=[];timings=[0.,0.]
    for i in range(min(a.count,len(dataset))):
        entry=dataset[i]
        if entry['id'] not in train or entry['id'] in val:raise ValueError('Audit is restricted to training IDs')
        e=calibrated(entry,manifest['scale']);predictions=[]
        with torch.no_grad():
            for j,model in enumerate(models):
                started=time.perf_counter();predictions.append(forward(model,e,return_steps=True));timings[j]+=time.perf_counter()-started
        for x,y in zip(predictions[0][0],predictions[1][0]):torch.testing.assert_close(x,y,atol=0,rtol=0)
        row={'id':entry['id'],'length':len(e['coords']),'single':metrics(*predictions[0][0],e),'two_untrained_recycles':metrics(*predictions[1][-1],e),'first_pass_exact':True}
        if i==0:
            outputs=forward(models[1],e,return_steps=True);outputs[0][0].retain_grad()
            values,steps=refinement_objective(AblationLoss(True,True),outputs,e,[0,1]);values[0].backward()
            grads=[p.grad for p in models[1].parameters() if p.grad is not None]
            g=outputs[0][0].grad
            if g is None or not torch.isfinite(g).all() or g.norm()==0 or not grads or not all(torch.isfinite(v).all() for v in grads):
                raise FloatingPointError('Broken recurrent backward path')
            row.update(first_pass_coordinate_gradient_norm=float(g.norm()),model_gradient_norm=float(torch.stack([v.norm() for v in grads]).norm()),pass_losses=steps)
        rows.append(row)
    record={'protocol':'se3-recycling-audit-v1','branch':'se3-recycled-refiner','checkpoint_sha256':digest(a.checkpoint),'local_bundle_sha256':manifest['bundle_sha256'],'parent_bundle_sha256':prior['bundle'],'split':'training only','production_forward_calls':0,'optimizer_steps':0,'parameters':[sum(p.numel() for p in m.parameters()) for m in models],'model_forward_seconds':timings,'cpu_threads':2,'torch':torch.__version__,'sources':source_identity(),'audit_source_sha256':digest(__file__),'rows':rows,'interpretation':'Weights were trained for one pass. Applying two passes is a stability audit, not trained two-pass performance or a merge gate.'}
    (root/'audit.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'structures':len(rows),'first_pass_exact':True,'parameters':record['parameters'],'forward_seconds':timings,'first_pass_gradient_norm':rows[0]['first_pass_coordinate_gradient_norm'],'model_gradient_norm':rows[0]['model_gradient_norm'],'output':str(root)}))

if __name__=='__main__':main()
