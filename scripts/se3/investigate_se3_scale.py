"""Frozen-cache controls for coordinate units and residual updates.

No production model calls, data regeneration or larger training.
Calibration uses only the original 32 overfit training structures. The 128 pilot
validation structures are reserved for final paired evaluation.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from foldtree2.se3_validation import (production_pair, sha256, frozen_entry,
    forward, losses, metrics, length_bucket_batches)
from foldtree2.src.se3_struct_decoder import se3_denoiser
from foldtree2.src.geometry_seeds import lift_seed_with_local_normals


def calibrated_entry(entry, scale, lift_fraction=0.):
    """Uniform scaling about the valid center preserves rigid-motion behavior."""
    result = dict(entry)
    mask = entry['mask']
    center = entry['coords'][mask].mean(0)
    result['coords'] = ((entry['coords'] - center) * scale + center).masked_fill(~mask[:, None], 0.)
    if lift_fraction:
        result['coords']=lift_seed_with_local_normals(result['coords'],entry['features'],mask,lift_fraction)
    return result


def fit_scale(entries):
    numerator = denominator = 0.
    for e in entries:
        p = torch.pdist(e['coords'][e['mask']].double())
        t = torch.pdist(e['target'][e['mask']].double())
        # Equal structure weights, matching the training sample averaging.
        numerator += float((p*t).mean())
        denominator += float(p.square().mean())
    if denominator <= 1e-12:
        raise ValueError('Cannot calibrate coincident seeds')
    return numerator / denominator


def load_entries(name):
    return load_cohort_entries(f'runs/se3_validation/{name}/provenance.json')


def load_cohort_entries(path):
    provenance = json.loads(Path(path).read_text())
    identity = provenance['identity']
    pair = production_pair()
    if sha256(provenance['config']['dataset']) != identity['dataset_sha256'] or any(
        sha256(pair[key]) != identity[key+'_sha256'] for key in pair):
        raise ValueError('Frozen input provenance changed')
    if identity['preprocessing']['extractor_sha256'] != hashlib.sha256(inspect.getsource(frozen_entry).encode()).hexdigest():
        raise ValueError('Frozen extractor changed')
    directory = Path(provenance['config']['cache']) / hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    entries = {}
    for identifier in dict.fromkeys(provenance['train_ids'] + provenance['val_ids']):
        saved = torch.load(directory/(hashlib.sha256(identifier.encode()).hexdigest()+'.pt'), weights_only=False)
        if saved['identity'] != identity or saved['id'] != identifier:
            raise ValueError('Incompatible frozen cache')
        entries[identifier] = saved['entry']
    return provenance, entries


def evaluate(model, ids, entries, scale, detailed=False, lift_fraction=0., loss_fn=losses):
    model.eval(); rows = []
    with torch.no_grad():
        for identifier in ids:
            e = calibrated_entry(entries[identifier], scale,lift_fraction)
            pred, angle = forward(model, e)
            terms = loss_fn(pred, angle, e)
            row = {'id':identifier, **dict(zip(['loss','geometry','fape','angle'],map(float,terms)))}
            if detailed: row['metrics'] = metrics(pred,angle,e)
            rows.append(row)
    return rows


def train_case(spec, ids, entries, epochs, output, event, loss_fn=losses):
    torch.manual_seed(42)
    device = torch.device(spec.get('device', 'cpu'))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(device)
    model = se3_denoiser(entries[ids[0]]['features'].shape[-1], [32], 3, 40, .25,
        dropout_p=0., depth=2, heads=2, dim_head=16, num_atom_types=40,
        coordinate_update=spec['update'], zero_init_coordinate_update=spec['update']=='residual',
        coordinate_normalization=spec['normalization'],coordinate_residual_gain=spec.get('gain',1.),
        angle_output=spec.get('angles','bounded'),device=device).to(device).float()
    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3)
    initial = evaluate(model,ids,entries,spec['scale'],lift_fraction=spec.get('lift_fraction',0.),loss_fn=loss_fn)
    initial_geometry = statistics.mean(r['geometry'] for r in initial)
    event('initial',case=spec['name'],geometry=initial_geometry,fape=statistics.mean(r['fape'] for r in initial))
    best = float('inf'); step = 0; history=[]
    for epoch in range(epochs):
        started=time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
        rows=[]; norms=[]
        batches=length_bucket_batches([len(entries[i]['coords']) for i in ids],1,42+epoch)
        for i,(index,) in enumerate(batches):
            e=calibrated_entry(entries[ids[index]],spec['scale'],spec.get('lift_fraction',0.))
            pred,angle=forward(model,e); values=loss_fn(pred,angle,e)
            if not all(torch.isfinite(v) for v in values): raise FloatingPointError('Nonfinite loss')
            group_size=min(8,len(batches)-(i//8)*8)
            (values[0]/group_size).backward();rows.append([float(v.detach()) for v in values])
            if (i+1)%8==0 or i+1==len(batches):
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads): raise FloatingPointError('Nonfinite gradients')
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                if norm<=0: raise RuntimeError('Zero gradient')
                norms.append(float(norm));optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
        validation=evaluate(model,ids,entries,spec['scale'],lift_fraction=spec.get('lift_fraction',0.),loss_fn=loss_fn)
        geometry=statistics.mean(r['geometry'] for r in validation)
        fape=statistics.mean(r['fape'] for r in validation)
        scheduler.step(geometry if spec.get('scheduler')=='geometry' else fape)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        record={'epoch':epoch,'case':spec['name'],'samples':len(rows),'step':step,
            'train_geometry':statistics.mean(r[1] for r in rows),'eval_geometry':geometry,'eval_fape':fape,
            'gradient_norm':statistics.mean(norms),'seconds':time.perf_counter()-started}
        record['learning_rate']=optimizer.param_groups[0]['lr']
        record['peak_memory_bytes']=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
        history.append(record);event('epoch',**record)
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'epoch':epoch,'step':step,'spec':spec,'ids':ids,
            'implementation_sha256':sha256('foldtree2/src/se3_struct_decoder.py')}
        if geometry<best:
            best=geometry;torch.save(checkpoint,output/(spec['name']+'_best.pt'))
        torch.save(checkpoint,output/(spec['name']+'_last.pt'))
    selected=torch.load(output/(spec['name']+'_best.pt'),weights_only=False)
    model.load_state_dict(selected['model'])
    final=evaluate(model,ids,entries,spec['scale'],detailed=True,lift_fraction=spec.get('lift_fraction',0.),loss_fn=loss_fn)
    summary={'spec':spec,'initial':initial,'final':final,'history':history,'best_epoch':selected['epoch'],
        'initial_geometry':initial_geometry,'best_geometry':best,'reduction':1-best/initial_geometry}
    return model,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/se3_investigation_20261009')
    parser.add_argument('--screen-count',type=int,default=8)
    parser.add_argument('--screen-epochs',type=int,default=10)
    parser.add_argument('--overfit-epochs',type=int,default=30)
    parser.add_argument('--screen-suite',choices=['basic','parameterization','spatial'],default='basic')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'])
    parser.add_argument('--train-case', default='screen_best',
                        help='Predeclare a screen case for the full learning test, or screen_best')
    args=parser.parse_args()
    if not (1<=args.screen_count<=32 and 1<=args.screen_epochs<=10 and 0<=args.overfit_epochs<=30):
        raise ValueError('Diagnostic bounds exceeded')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'events.jsonl').exists():raise ValueError('Use a fresh investigation output directory')
    def event(kind,**values):
        record={'event':kind,'time':time.time(),**values};text=json.dumps(record)
        with (output/'events.jsonl').open('a') as f:f.write(text+'\n')
        print(text,flush=True)
    try:
        provenance,entries=load_entries('overfit');pilot,heldout=load_entries('pilot')
        device=torch.device(args.device)
        if device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable')
        entries={i:{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in e.items()} for i,e in entries.items()}
        heldout={i:{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in e.items()} for i,e in heldout.items()}
        ids=provenance['train_ids'];val_ids=pilot['val_ids']
        if set(ids)&set(val_ids):raise ValueError('Calibration targets overlap held-out evaluation')
        scale=fit_scale([entries[i] for i in ids]);event('calibration',scale=scale,fit_count=len(ids),heldout_count=len(val_ids))
        angle_values=torch.cat([e['angles'][e['mask']][:,:3] for e in entries.values()])
        valid=torch.isfinite(angle_values);wrapped=torch.atan2(angle_values.sin(),angle_values.cos())
        angle_floor=float((1-torch.cos(wrapped[valid]-wrapped[valid].clamp(-1.,1.))).mean())
        static=[]
        for identifier in ids:
            e=entries[identifier]
            for label,factor in [('raw',1.),('calibrated',scale)]:
                ce=calibrated_entry(e,factor);valid_coords=ce['coords'][ce['mask']]
                d=torch.pdist(valid_coords)
                static.append({'id':identifier,'seed':label,'metrics':metrics(ce['coords'],ce['baseline_angles'],ce),
                    'distance_contact_density':float(((d<8.)&(d>0.)).float().mean())})
        protocol='se3-scale-residual-investigation-v4-full-dimensional-seed' if args.screen_suite=='spatial' else 'se3-scale-residual-investigation-v3-parameterization'
        record={'protocol':protocol,'device':args.device,'seed':42,'seed_applied_per_case':True,
            'train_ids':ids,'heldout_ids':val_ids,'calibration_scale':scale,'fit':'equal-structure pair-distance least squares',
            'source_identity':provenance['identity'],'refiner_sha256':sha256('foldtree2/src/se3_struct_decoder.py'),
            'script_sha256':sha256(__file__),'args':vars(args),'static':static,
            'loss_sha256':hashlib.sha256(inspect.getsource(losses).encode()).hexdigest(),
            'metrics_sha256':hashlib.sha256(inspect.getsource(metrics).encode()).hexdigest(),
            'seed_lift_sha256':hashlib.sha256(inspect.getsource(lift_seed_with_local_normals).encode()).hexdigest(),
            'angles_outside_head_range':float((wrapped.abs()[valid]>1.).float().mean()),'angle_loss_floor':angle_floor}
        (output/'analysis.json').write_text(json.dumps(record,indent=2))
        event('static',raw_fape_mean=statistics.mean(r['metrics']['fape'] for r in static if r['seed']=='raw'),
              angle_outside_fraction=record['angles_outside_head_range'],angle_loss_floor=angle_floor)
        screen_ids=sorted(ids,key=lambda i:len(entries[i]['coords']))[:args.screen_count]
        specs=[{'name':name,'scale':factor,'update':update,'normalization':normalization} for name,factor,update,normalization in [
            ('raw_absolute',1.,'absolute','bounded_radius'),('raw_residual',1.,'residual','bounded_radius'),
            ('calibrated_absolute',scale,'absolute','bounded_radius'),('calibrated_residual',scale,'residual','bounded_radius'),
            ('calibrated_residual_rms',scale,'residual','rms_radius')]]
        if args.screen_suite=='parameterization':
            specs=[{'name':name,'scale':scale,'update':'residual','normalization':'rms_radius',
                    'gain':gain,'angles':angles,'scheduler':'geometry'} for name,gain,angles in [
                ('rms_geometry_scheduler',1.,'bounded'),('rms_gain32',32.,'bounded'),('rms_gain32_circular',32.,'sincos')]]
        elif args.screen_suite=='spatial':
            specs=[{'name':name,'scale':scale,'update':'residual','normalization':'rms_radius',
                    'gain':32.,'angles':angles,'scheduler':'geometry','lift_fraction':fraction}
                for name,fraction,angles in [('planar_control',0.,'bounded'),
                    ('lifted_residual',.1,'bounded'),('lifted_circular',.1,'sincos')]]
        screening=[]
        for spec in specs: spec['device']=args.device
        if args.train_case!='screen_best' and args.train_case not in {s['name'] for s in specs}:
            raise ValueError('Requested training case is not part of this suite')
        for spec in specs:
            _,summary=train_case(spec,screen_ids,entries,args.screen_epochs,output,event)
            screening.append(summary);event('screen_complete',case=spec['name'],reduction=summary['reduction'],geometry=summary['best_geometry'])
            (output/'screening.json').write_text(json.dumps(screening,indent=2))
        winner=(min(screening,key=lambda r:r['best_geometry']) if args.train_case=='screen_best'
                else next(r for r in screening if r['spec']['name']==args.train_case))
        spec={**winner['spec'],'name':'overfit_'+winner['spec']['name']}
        if args.overfit_epochs:
            model,summary=train_case(spec,ids,entries,args.overfit_epochs,output,event)
            (output/'overfit.json').write_text(json.dumps(summary,indent=2))
            evaluated=evaluate(model,val_ids,heldout,spec['scale'],detailed=True,lift_fraction=spec.get('lift_fraction',0.));pairs=[]
            for r in evaluated:
                e=heldout[r['id']];scaled=calibrated_entry(e,spec['scale'])
                seed=calibrated_entry(e,spec['scale'],spec.get('lift_fraction',0.))
                pairs.append({'id':r['id'],'raw':metrics(e['coords'],e['baseline_angles'],e),
                    'calibrated':metrics(scaled['coords'],scaled['baseline_angles'],scaled),
                    'input_seed':metrics(seed['coords'],seed['baseline_angles'],seed),'refined':r['metrics']})
            (output/'heldout_pairs.json').write_text(json.dumps(pairs,indent=2))
        event('complete',winner=spec['name'],frozen_production_calls=0)
    except Exception as exc:
        event('failure',error=repr(exc));raise


if __name__=='__main__':main()
