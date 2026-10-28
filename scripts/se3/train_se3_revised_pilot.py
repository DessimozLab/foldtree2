"""Resumable 512/128 frozen-cache pilot of the validated opt-in residue loss."""
import argparse
import json
from pathlib import Path
import platform
import resource
import time
import statistics
import torch
import yaml
from foldtree2.se3_validation import sha256, forward, metrics, length_bucket_batches, restore_training_state
from foldtree2.src.se3_struct_decoder import se3_denoiser
from foldtree2.src.losses.se3_ablation import AblationLoss
from scripts.se3.investigate_se3_scale import load_entries, load_cohort_entries, calibrated_entry, evaluate


def run(args):
    config=yaml.safe_load(Path(args.config).read_text());root=Path(config['output'])
    root.mkdir(parents=True,exist_ok=True)
    if (root/'events.jsonl').exists() and not args.resume_from:raise ValueError('Use a fresh output or --resume-from')
    def event(kind,**values):
        text=json.dumps({'event':kind,'time':time.time(),**values})
        with (root/'events.jsonl').open('a') as f:f.write(text+'\n')
        print(text,flush=True)
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(config['seed'])
        device=torch.device(config['device'])
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        if not (1<=config['train_count']<=4096 and 1<=config['val_count']<=512 and 1<=config['epochs']<=30):
            raise ValueError('Local experiment bounds exceeded')
        if config['batch_size']!=1:raise ValueError('Only residue batch size one is supported')
        source,entries=(load_cohort_entries(config['cohort_provenance']) if config.get('cohort_provenance') else load_entries('pilot'))
        ids=source['train_ids'];val_ids=source['val_ids']
        assert len(ids)==config['train_count'] and len(val_ids)==config['val_count']
        assert not set(ids)&set(val_ids)
        assert all(len(entries[i]['coords'])<=config['max_residues'] for i in ids+val_ids)
        calibration=json.loads(Path(config['calibration_provenance']).read_text())
        assert not set(calibration['train_ids'])&set(val_ids)
        scale=calibration['scale']
        spec={'scale':scale,'lift_fraction':.1,'gain':32.,'angles':'sincos','update':'residual',
              'normalization':'rms_radius','hidden':32,'depth':2,'heads':2,'dim_head':16}
        files=['scripts/se3/train_se3_revised_pilot.py','scripts/se3/investigate_se3_scale.py',
               'foldtree2/se3_validation.py','foldtree2/src/se3_struct_decoder.py',
               'foldtree2/src/geometry_seeds.py','foldtree2/src/losses/se3_ablation.py','foldtree2/src/losses/fape.py']
        if config.get('cohort_provenance'):files.append('scripts/se3/prepare_se3_larger_cohort.py')
        provenance={'protocol':'se3-larger-local-v1' if config.get('cohort_provenance') else 'se3-revised-pilot-v1','config':config,'input_identity':source['identity'],
                    'train_ids':ids,'val_ids':val_ids,'spec':spec,
                    'calibration_ids':calibration['train_ids'],
                    'calibration_sha256':sha256(config['calibration_provenance']),
                    'sources':{f:sha256(f) for f in files},
                    'validation_scope':source.get('scope','Development validation reused in earlier diagnostics; not untouched test data')}
        if config.get('cohort_provenance'):
            provenance['cohort_sha256']=sha256(config['cohort_provenance'])
        if config.get('warm_start'):
            provenance['warm_start_sha256']=sha256(config['warm_start'])
        if args.resume_from:
            if json.loads((root/'provenance.json').read_text())!=provenance:raise ValueError('Resume provenance mismatch')
        else:
            (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
            for name in files:
                dst=root/'sources'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(name).read_bytes())
        (root/'software.json').write_text(json.dumps({'python':platform.python_version(),'torch':torch.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None},indent=2))
        # Keep frozen cache on host; transfer a single structure at a time.
        def entry(identifier):
            e={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in entries[identifier].items()}
            return calibrated_entry(e,scale,.1)
        model=se3_denoiser(entries[ids[0]]['features'].shape[-1],[32],3,40,.25,dropout_p=0.,depth=2,
            heads=2,dim_head=16,num_atom_types=40,coordinate_update='residual',
            zero_init_coordinate_update=True,coordinate_normalization='rms_radius',
            coordinate_residual_gain=32.,angle_output='sincos',device=device).to(device).float()
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,factor=config.get('scheduler_factor',.1))
        loss_fn=AblationLoss(True,True);start=step=bad=0;best=float('inf')
        if args.resume_from:
            ckpt=torch.load(args.resume_from,map_location=device,weights_only=False)
            start,step,best,bad=restore_training_state(ckpt,provenance,model,optimizer,scheduler,device)
            event('resume',epoch=start,global_step=step)
        elif config.get('warm_start'):
            warm=torch.load(config['warm_start'],map_location=device,weights_only=False)
            if warm['provenance']['spec']!=spec or warm['provenance']['input_identity']!=source['identity']:
                raise ValueError('Warm-start model or input identity differs')
            for name in ['foldtree2/src/se3_struct_decoder.py','foldtree2/src/losses/se3_ablation.py']:
                if warm['provenance']['sources'][name]!=sha256(name):raise ValueError('Warm-start implementation differs')
            model.load_state_dict(warm['model'])
            event('warm_start',checkpoint=config['warm_start'],source_epoch=warm['epoch'],optimizer_reset=True)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        event('ready',train_count=len(ids),val_count=len(val_ids),device=str(device),scale=scale,
              frozen_production_calls=0)
        history=json.loads((root/'history.json').read_text()) if args.resume_from and (root/'history.json').exists() else []
        if config.get('warm_start') and not (root/'initial_losses.json').exists():
            if not args.resume_from:
                initial_checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'epoch':-1,'global_step':0,'best':best,'bad':bad,'provenance':provenance,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
                torch.save(initial_checkpoint,root/'initial.tmp');(root/'initial.tmp').replace(root/'initial.pt')
            model.eval();initial={};initial_validation=[]
            with torch.no_grad():
                for label,selected in [('train',ids),('val',val_ids)]:
                    baseline=[]
                    for index,identifier in enumerate(selected):
                        e=entry(identifier);p,a=forward(model,e);v=loss_fn(p,a,e)
                        if not all(torch.isfinite(x) for x in v):raise FloatingPointError('Nonfinite initial loss')
                        baseline.append([float(x) for x in v])
                        if label=='val':initial_validation.append({'id':identifier,'metrics':metrics(p,a,e)})
                        if (index+1)%256==0:event('initial_progress',split=label,count=index+1,total=len(selected))
                    initial[label]={key:statistics.mean(r[j] for r in baseline) for j,key in enumerate(['loss','geometry','fape','angle'])}
            (root/'initial_validation_metrics.json').write_text(json.dumps(initial_validation,indent=2))
            (root/'initial_losses.json').write_text(json.dumps(initial,indent=2));event('initial',**initial)
        training_began=time.perf_counter()
        prior_training_seconds=sum(r['seconds'] for r in history)
        for epoch in range(start,config['epochs']):
            if bad>=config['patience']:break
            if prior_training_seconds+time.perf_counter()-training_began>config.get('max_training_hours',float('inf'))*3600:
                event('time_limit',hours=config['max_training_hours']);break
            began=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True);rows=[];norms=[]
            batches=length_bucket_batches([len(entries[i]['coords']) for i in ids],1,config['seed']+epoch)
            for i,(index,) in enumerate(batches):
                e=entry(ids[index]);pred,angles=forward(model,e);values=loss_fn(pred,angles,e)
                if not all(torch.isfinite(v) for v in values):raise FloatingPointError('Nonfinite loss')
                size=min(config['effective_batch_size'],len(batches)-(i//config['effective_batch_size'])*config['effective_batch_size'])
                (values[0]/size).backward();rows.append([float(v.detach()) for v in values])
                if (i+1)%config['effective_batch_size']==0 or i+1==len(batches):
                    grads=[p.grad for p in model.parameters() if p.grad is not None]
                    if not grads or not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('Nonfinite gradients')
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
                    if norm<=0:raise RuntimeError('Zero gradient')
                    norms.append(float(norm));optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
                if (i+1)%64==0:event('progress',epoch=epoch,samples=i+1,global_step=step,loss=float(values[0].detach()))
            model.eval();validation=[]
            with torch.no_grad():
                for identifier in val_ids:
                    e=entry(identifier);p,a=forward(model,e);values=loss_fn(p,a,e)
                    if not all(torch.isfinite(v) for v in values):raise FloatingPointError('Nonfinite validation')
                    validation.append([float(v) for v in values])
            score=statistics.mean(r[1 if config.get('monitor')=='geometry' else 2] for r in validation);scheduler.step(score)
            improved=score<best-config['min_delta'];best,bad=(score,0) if improved else (best,bad+1)
            if device.type=='cuda':torch.cuda.synchronize(device)
            row={'epoch':epoch,'global_step':step,'train_geometry':statistics.mean(r[1] for r in rows),
                 'train_fape':statistics.mean(r[2] for r in rows),'val_geometry':statistics.mean(r[1] for r in validation),
                 'val_fape':score,'gradient_norm':statistics.mean(norms),'seconds':time.perf_counter()-began,
                 'train_count':len(rows),'val_count':len(validation),'patience_used':bad,
                 'learning_rate':optimizer.param_groups[0]['lr'],
                 'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                 'peak_host_memory_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
            row['val_fape']=statistics.mean(r[2] for r in validation)
            row['monitor']=config.get('monitor','fape');row['monitor_score']=score
            history.append(row);(root/'history.json').write_text(json.dumps(history,indent=2));event('epoch',**row)
            ckpt={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                  'epoch':epoch,'global_step':step,'best':best,'bad':bad,'provenance':provenance,
                  'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
            torch.save(ckpt,root/'last.tmp');(root/'last.tmp').replace(root/'last.pt')
            if improved:torch.save(ckpt,root/'best.tmp');(root/'best.tmp').replace(root/'best.pt')
        model.load_state_dict(torch.load(root/'best.pt',map_location=device,weights_only=False)['model']);model.eval()
        paired=[]
        warm_metrics={r['id']:r['metrics'] for r in json.loads((root/'initial_validation_metrics.json').read_text())} if config.get('warm_start') else {}
        with torch.no_grad():
            for identifier in val_ids:
                e=entry(identifier);p,a=forward(model,e)
                raw={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in entries[identifier].items()}
                paired.append({'id':identifier,'raw':metrics(raw['coords'],raw['baseline_angles'],raw),
                    'input_seed':metrics(e['coords'],e['baseline_angles'],e),'refined':metrics(p,a,e)})
                if warm_metrics:paired[-1]['warm_start']=warm_metrics[identifier]
        (root/'heldout_pairs.json').write_text(json.dumps(paired,indent=2));event('complete',epochs=len(history),best_score=best,monitor=config.get('monitor','fape'),frozen_production_calls=0)
    except Exception as exc:event('failure',error=repr(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--resume-from')
    run(parser.parse_args())
