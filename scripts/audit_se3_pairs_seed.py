"""CPU cache audit: zero-init compatibility and finite extension learning paths."""
import argparse
import json
import platform
import resource
import time
from pathlib import Path
import torch
import yaml
from foldtree2.se3_cache import FrozenDataset,open_bundle,digest
from foldtree2.se3_validation import forward,metrics
from foldtree2.train_se3_lightning import Refiner,calibrated,provenance,check_refinement_warm_start,load_architecture_weights,source_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--count',type=int,default=8);a=p.parse_args();root=Path(a.output)
    if root.exists() or a.count<=0:raise ValueError('Fresh output and positive count required')
    root.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);prior=checkpoint['se3']['provenance']
    manifest=open_bundle(a.cache);dataset=FrozenDataset(a.cache,manifest,'train')
    if manifest['input_identity']!=prior['input_identity'] or manifest['scale']!=prior['preprocessing']['scale']:
        raise ValueError('Frozen production or preprocessing mismatch')
    cohort=json.loads(Path('docs/se3_clariden_evidence/growth_8192/cohort.json').read_text())
    train,val=set(cohort['train_ids']),set(cohort['val_ids'])
    template=yaml.safe_load(Path('configs/clariden_se3/pairs_seed_v1_pairs_seed.yaml').read_text())
    configurations={}
    for name,usepair,useseed,steps in [('control',False,False,1),('pairs',True,False,1),('seed',False,True,1),('pairs_seed',True,True,1),('pairs_seed_two',True,True,2)]:
        cfg={**prior['training'],'refinement_steps':steps,'refinement_loss_weights':[0.]*(steps-1)+[1.]}
        if usepair:cfg['pair_information']=template['pair_information']
        if useseed:cfg['initial_geometry']=template['initial_geometry']
        configurations[name]=cfg
    models={};rows=[];gradients={};timings={};began=time.perf_counter()
    parent_manifest={**manifest,'bundle_sha256':prior['bundle']}
    for name,cfg in configurations.items():
        check_refinement_warm_start(prior,provenance(cfg,parent_manifest))
        module=Refiner(cfg,manifest);load_architecture_weights(module,checkpoint['state_dict'],prior['training'],cfg,True)
        models[name]=module.eval();timings[name]=0.
    for i in range(min(a.count,len(dataset))):
        e=dataset[i]
        if e['id'] not in train or e['id'] in val:raise ValueError('Training-only audit required')
        row={'id':e['id'],'length':len(e['coords']),'metrics':{}};reference=None
        for name,module in models.items():
            entry=calibrated(e,manifest['scale'],module.provenance['preprocessing']['lift_fraction'])
            started=time.perf_counter()
            with torch.no_grad():outputs=forward(module.model,entry,return_steps=True)
            timings[name]+=time.perf_counter()-started
            if reference is None:reference=outputs[0]
            for x,y in zip(reference,outputs[0]):torch.testing.assert_close(x,y,atol=0,rtol=0)
            row['metrics'][name]=metrics(*outputs[-1],entry)
            if i==0 and name!='control':
                module.zero_grad();outputs=forward(module.model,entry,return_steps=True)
                total=module.loss_fn(*outputs[-1],entry)[0];total.backward()
                if not torch.isfinite(total) or not all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None):
                    raise FloatingPointError('Nonfinite audit loss or gradient')
                norms={}
                for field in ('gotennet.pair_projection','seed_initializer'):
                    parameters=[p for key,p in module.model.named_parameters() if key.startswith(field) and p.grad is not None]
                    if parameters:
                        value=float(torch.stack([p.grad.norm() for p in parameters]).norm())
                        if value<=0:raise ValueError('Disconnected extension: '+field)
                        norms[field]=value
                gradients[name]={'loss':float(total.detach()),'extension_gradient_norms':norms}
                module.zero_grad()
        row['first_pass_exact']=True;rows.append(row)
    record={'protocol':'se3-pairs-seed-audit-v1','branch':'se3-recycled-refiner',
        'checkpoint_sha256':digest(a.checkpoint),'local_bundle_sha256':manifest['bundle_sha256'],
        'parent_bundle_sha256':prior['bundle'],'input_identity':manifest['input_identity'],
        'split':'training only','production_forward_calls':0,'optimizer_steps':0,
        'configurations':configurations,'parameters':{k:sum(p.numel() for p in v.parameters()) for k,v in models.items()},
        'gradients':gradients,'rows':rows,'model_forward_seconds':timings,'elapsed_seconds':time.perf_counter()-began,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cpu_threads':2,
        'software':{'torch':torch.__version__,'python':platform.python_version()},'sources':source_identity(),
        'audit_source_sha256':digest(__file__),
        'interpretation':'Zero-initialized additions preserve single-pass outputs. Gradient audit only; no trained improvement, held-out claim or merge gate.'}
    (root/'audit.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:record[k] for k in ('parameters','gradients','elapsed_seconds','peak_rss_kib')}))

if __name__=='__main__':main()
