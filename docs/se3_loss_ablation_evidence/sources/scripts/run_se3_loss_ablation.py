"""Bounded loss ablation on existing frozen caches, with no model promotion."""
import argparse
import json
from pathlib import Path
import platform
import time
import torch
from foldtree2.se3_validation import sha256, metrics
from foldtree2.src.losses.se3_ablation import AblationLoss, components
from scripts.investigate_se3_scale import load_entries, fit_scale, calibrated_entry, train_case, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0', 'cuda:1'], default='cpu')
    args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    if (root/'events.jsonl').exists(): raise ValueError('Use a fresh output directory')
    def event(kind, **values):
        text = json.dumps({'event': kind, 'time': time.time(), **values})
        with (root/'events.jsonl').open('a') as f: f.write(text+'\n')
        print(text, flush=True)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    try:
        provenance, entries = load_entries('overfit'); pilot, heldout = load_entries('pilot')
        ids, val_ids = provenance['train_ids'], pilot['val_ids']
        assert len(ids) == 32 and len(val_ids) == 128 and not set(ids) & set(val_ids)
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        def move(data):
            return {i:{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in e.items()} for i,e in data.items()}
        entries, heldout = move(entries), move(heldout)
        scale = fit_scale([entries[i] for i in ids])
        sources = ['scripts/run_se3_loss_ablation.py','scripts/investigate_se3_scale.py',
                   'foldtree2/se3_validation.py','foldtree2/src/losses/se3_ablation.py',
                   'foldtree2/src/se3_struct_decoder.py','foldtree2/src/geometry_seeds.py']
        record = {'protocol':'se3-loss-ablation-v1','train_ids':ids,'heldout_ids':val_ids,
                  'source_identity':provenance['identity'],'scale':scale,'args':vars(args),
                  'sources':{name:sha256(name) for name in sources},
                  'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                  'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                  'selection':'predeclared control and soft_fape_bond; train objective selects epoch',
                  'weights':{'distance':.1,'soft_fape_scale_angstrom':10.,'bond':1.,'angle':.1},
                  'bond_eligibility':'adjacent valid rows with target distance in [2.5,4.5] Å',
                  'seed':42,'epochs':30,'effective_batch':8,'batch':1,'lr':5e-4,'clip':1.}
        (root/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
        source_dir=root/'sources';source_dir.mkdir()
        for name in sources:
            p=source_dir/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(name).read_bytes())
        screen_ids=sorted(ids,key=lambda i:len(entries[i]['coords']))[:8]
        controls=[('control',False,False),('bond_only',False,True),('soft_only',True,False),('soft_fape_bond',True,True)]
        screening=[]; full=[]; pairs={}; gradient_diagnostics=[]
        for name,soft,bond in controls:
            spec={'name':'screen_'+name,'scale':scale,'update':'residual','normalization':'rms_radius',
                  'gain':32.,'angles':'sincos','scheduler':'geometry','lift_fraction':.1,
                  'device':args.device,'loss_protocol':name}
            loss_fn=AblationLoss(soft,bond)
            # Physical-coordinate gradients before any optimizer step.
            for identifier in screen_ids:
                e=calibrated_entry(entries[identifier],scale,.1);p=e['coords'].detach().requires_grad_()
                c=components(p,torch.zeros_like(e['angles'][...,:3]),e,soft,bond)
                norms={}
                for key in ('distance','training_fape','bond'):
                    norms[key]=float(torch.autograd.grad(c[key],p,retain_graph=True)[0].norm())
                gradient_diagnostics.append({'case':name,'id':identifier,'norms':norms,
                                             'values':{k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in c.items()}})
            _, summary=train_case(spec,screen_ids,entries,10,root,event,loss_fn=loss_fn)
            screening.append(summary);(root/'screening.json').write_text(json.dumps(screening,indent=2))
        (root/'initial_gradient_diagnostics.json').write_text(json.dumps(gradient_diagnostics,indent=2))
        for name,soft,bond in [controls[0],controls[-1]]:
            spec={**screening[0]['spec'],'name':'overfit_'+name,'loss_protocol':name}
            loss_fn=AblationLoss(soft,bond)
            model,summary=train_case(spec,ids,entries,30,root,event,loss_fn=loss_fn)
            full.append(summary);(root/'overfit.json').write_text(json.dumps(full,indent=2))
            rows=evaluate(model,val_ids,heldout,scale,detailed=True,lift_fraction=.1,loss_fn=loss_fn)
            pairs[name]=[]
            for row in rows:
                e=calibrated_entry(heldout[row['id']],scale,.1)
                pairs[name].append({'id':row['id'],'input_seed':metrics(e['coords'],e['baseline_angles'],e),
                                    'refined':row['metrics']})
            (root/'heldout_pairs.json').write_text(json.dumps(pairs,indent=2))
            event('case_complete',case=name,reduction=summary['reduction'])
        event('complete',frozen_production_calls=0,train_count=32,heldout_count=128)
    except Exception as exc:
        event('failure',error=repr(exc));raise


if __name__=='__main__': main()
