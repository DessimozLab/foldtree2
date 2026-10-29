"""Check finite two-update loss trials, exact recovery and topology agreement."""
import argparse
import json
from pathlib import Path
import torch
from scripts.se3.verify_se3_normalized_debug import read


def compare(candidate, control, exact=False):
    a,ae,ar=read(candidate,2);b,be,br=read(control,2)
    if a['se3']['provenance']!=b['se3']['provenance']:
        raise ValueError('Loss-check provenance differs')
    if {r['id'] for r in ar}!={r['id'] for r in br}:
        raise ValueError('Loss-check holdouts differ')
    tolerance=0. if exact else 1e-4
    def check(x,y):
        if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,atol=tolerance,rtol=tolerance)
        elif isinstance(x,dict):
            if x.keys()!=y.keys():raise ValueError('State inventory differs')
            for k in x:check(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            if len(x)!=len(y):raise ValueError('State lengths differ')
            for u,v in zip(x,y):check(u,v)
        elif x!=y:raise ValueError('Non-tensor optimizer state differs')
    check(a['state_dict'],b['state_dict']);check(a['optimizer_states'],b['optimizer_states'])
    av=[r for r in ae if r['event']=='validation'][-1]['metrics']
    bv=[r for r in be if r['event']=='validation'][-1]['metrics']
    if av.keys()!=bv.keys():raise ValueError('Metric inventory differs')
    for k in av:
        if av[k] is None or bv[k] is None:
            if av[k]!=bv[k]:raise ValueError('Metric masks differ')
        else:torch.testing.assert_close(torch.tensor(av[k],dtype=torch.float64),torch.tensor(bv[k],dtype=torch.float64),atol=tolerance,rtol=tolerance)
    largest=max(float((v-b['state_dict'][k]).abs().max()) for k,v in a['state_dict'].items() if v.is_floating_point())
    return {'passed':True,'exact_resume':exact,'updates':2,'maximum_parameter_buffer_difference':largest,'validation_structures':len(ar),'validation':av,'topologies':[a['se3']['topology'],b['se3']['topology']]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate');p.add_argument('control');p.add_argument('--exact',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    try:result=compare(a.candidate,a.control,a.exact)
    except Exception as exc:
        Path(a.output).write_text(json.dumps({'passed':False,'error':str(exc)},indent=2)+'\n');raise
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
