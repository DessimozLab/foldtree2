"""Compare matched capacity runs; reject incomplete, mismatched or regressing runs."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import json
from pathlib import Path
import statistics
import torch


def assess(control, candidate):
    def read(root):
        root=Path(root)
        ck=torch.load(root/'last.ckpt',map_location='cpu',weights_only=False)['se3']
        if ck['stream'] != {'epoch':16,'cursor':0,'updates':4096}:
            raise ValueError('Capacity comparison requires exactly 16 epochs / 4096 updates')
        best=torch.load(root/'best.ckpt',map_location='cpu',weights_only=False)['se3']
        epoch=best['stream']['epoch']-1
        rows=json.loads((root/f'validation-{epoch:03d}.json').read_text())
        return ck,rows
    a, ar=read(control);b,br=read(candidate)
    ap,bp=a['provenance'],b['provenance']
    ac,bc=dict(ap['training']),dict(bp['training']);ac.pop('model');bc.pop('model')
    if ap['bundle'] != bp['bundle'] or ap['sources'] != bp['sources'] or ac != bc:
        raise ValueError('Capacity comparison provenance differs')
    aa={r['id']:r for r in ar};bb={r['id']:r for r in br}
    if set(aa)!=set(bb):raise ValueError('Capacity validation IDs differ')
    summary={}; regressions=[]
    for key in sorted(ar[0]['refined']):
        pairs=[(aa[i]['refined'][key],bb[i]['refined'][key]) for i in sorted(aa)]
        pairs=[(x,y) for x,y in pairs if x is not None and y is not None]
        if not pairs:continue
        x=statistics.mean(x for x,y in pairs);y=statistics.mean(y for x,y in pairs)
        summary[key]={'control':x,'candidate':y,'paired_count':len(pairs)}
        if key in ('tm_score','lddt'):
            if y<x-.01:regressions.append(key)
        elif key!='valid_residues' and y>x*1.05:
            regressions.append(key)
    x=statistics.mean(r['geometry'] for r in ar);y=statistics.mean(r['geometry'] for r in br)
    improvement=1-y/x
    passed=improvement>=.01 and summary['fape']['candidate']<summary['fape']['control'] and not regressions
    return {'passed':passed,'geometry_reduction':improvement,'regressions':regressions,'metrics':summary,
            'scope':'Structure-level refiner holdouts, not independent of production-model pretraining'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('control');p.add_argument('candidate');p.add_argument('--output',required=True)
    a=p.parse_args(); result=assess(a.control,a.candidate)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
