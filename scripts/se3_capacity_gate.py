"""Compare matched capacity runs; reject incomplete, mismatched or regressing runs."""
import argparse
import json
import math
from pathlib import Path
import statistics
import numpy as np
import torch


def assess(control, candidate):
    def read(root):
        root=Path(root)
        ck=torch.load(root/'last.ckpt',map_location='cpu',weights_only=False)['se3']
        config=ck['provenance']['training']
        epochs,count,batch=(config[k] for k in ('epochs','train_count','global_batch'))
        if min(epochs,count,batch)<=0 or count%batch:
            raise ValueError('Invalid capacity schedule')
        if ck['stream'] != {'epoch':epochs,'cursor':0,'updates':epochs*(count//batch)}:
            raise ValueError('Capacity comparison requires the complete declared schedule')
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        trained=[r for r in events if r['event']=='train_epoch']
        if (len(trained)!=epochs or any(r['effective_structures']!=count for r in trained)
                or any(r['event']=='failure' for r in events) or list(root.glob('failure*'))):
            raise ValueError('Incomplete or failed capacity trial')
        best=torch.load(root/'best.ckpt',map_location='cpu',weights_only=False)['se3']
        if best['provenance']!=ck['provenance']:
            raise ValueError('Best checkpoint provenance differs')
        epoch=best['stream']['epoch']-1
        if epoch not in range(epochs):raise ValueError('Best epoch outside schedule')
        rows=json.loads((root/f'validation-{epoch:03d}.json').read_text())
        return ck,rows
    a, ar=read(control);b,br=read(candidate)
    ap,bp=a['provenance'],b['provenance']
    ac,bc=dict(ap['training']),dict(bp['training']);ac.pop('model');bc.pop('model')
    if {k:v for k,v in ap.items() if k!='training'} != {k:v for k,v in bp.items() if k!='training'} or ac != bc:
        raise ValueError('Capacity comparison provenance differs')
    aa={r['id']:r for r in ar};bb={r['id']:r for r in br}
    if (len(aa)!=len(ar) or len(bb)!=len(br) or set(aa)!=set(bb)
            or len(aa)!=ac['val_count']):raise ValueError('Capacity validation IDs differ')
    for i in aa:
        if aa[i]['raw']!=bb[i]['raw'] or aa[i]['refined']['valid_residues']!=bb[i]['refined']['valid_residues']:
            raise ValueError('Capacity baseline or residue masks differ')
    summary={}; regressions=[]
    for key in sorted(ar[0]['refined']):
        pairs=[(aa[i]['refined'][key],bb[i]['refined'][key]) for i in sorted(aa)]
        if any((x is None)!=(y is None) for x,y in pairs):
            raise ValueError('Capacity metric eligibility differs')
        pairs=[(x,y) for x,y in pairs if x is not None and y is not None]
        if not pairs:continue
        if not all(math.isfinite(v) for pair in pairs for v in pair):
            raise ValueError('Nonfinite capacity metric')
        x=statistics.mean(x for x,y in pairs);y=statistics.mean(y for x,y in pairs)
        summary[key]={'control':x,'candidate':y,'paired_count':len(pairs)}
        if key in ('tm_score','lddt'):
            if y<x-.01:regressions.append(key)
        elif key!='valid_residues' and y>x*1.05:
            regressions.append(key)
    if 'fape' not in summary:raise ValueError('No valid capacity FAPE measurements')
    x=statistics.mean(r['geometry'] for r in ar);y=statistics.mean(r['geometry'] for r in br)
    if not math.isfinite(x) or not math.isfinite(y) or x<=0:raise ValueError('Invalid geometry loss')
    improvement=1-y/x
    passed=improvement>=.01 and summary['fape']['candidate']<summary['fape']['control'] and not regressions
    delta=np.array([bb[i]['refined']['fape']-aa[i]['refined']['fape'] for i in sorted(aa)])
    idx=np.random.default_rng(42).integers(0,len(delta),size=(10000,len(delta)))
    return {'passed':passed,'geometry_reduction':improvement,'regressions':regressions,'metrics':summary,
            'paired_fape_delta':float(delta.mean()),'paired_fape_ci95':np.quantile(delta[idx].mean(1),[.025,.975]).tolist(),
            'structures_improved':int((delta<0).sum()),
            'bootstrap':{'seed':42,'draws':10000,'unit':'structure',
                         'caveat':'Conditional on development-set checkpoint selection; not fresh confirmation'},
            'scope':'Structure-level refiner holdouts, not independent of production-model pretraining'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('control');p.add_argument('candidate');p.add_argument('--output',required=True)
    a=p.parse_args(); result=assess(a.control,a.candidate)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
