"""Create audited continuation branches without resetting optimizer/RNG/stream."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
import yaml

PARENT_SHA = '93ff91d18ff9b4b0df5396889f9cd6c4fc0c91f65f88775caf71d2c90487bb04'


def branch(checkpoint, config, parent_sha, sources):
    saved = checkpoint['se3']; prior = saved['provenance']
    if prior['sources'] != sources:
        raise ValueError('Training source mismatch')
    training = {k:v for k,v in config.items() if k not in ('cache','output','nodes','devices','accelerator')}
    old = prior['training']
    if {k:v for k,v in training.items() if k not in ('epochs','learning_rate')} != {k:v for k,v in old.items() if k not in ('epochs','learning_rate')}:
        raise ValueError('Only epochs and learning rate may change')
    if saved['stream'] != {'epoch':16,'cursor':0,'updates':8192} or config['epochs'] != 32:
        raise ValueError('Expected completed epoch-16 parent and epoch-32 endpoint')
    if config['learning_rate'] not in (5e-4,1e-4) or old['learning_rate'] != 5e-4:
        raise ValueError('Unapproved learning rate')
    if saved['topology'] != {'world_size':config['nodes']*config['devices'],'nodes':config['nodes']}:
        raise ValueError('Topology mismatch')
    result=copy.deepcopy(checkpoint)
    result['se3']['provenance']['training']=training
    result['se3']['provenance']['initialization']={'kind':'continuation_branch','parent_sha256':parent_sha,'parent_initialization':prior['initialization'],'changes':{'epochs':[old['epochs'],32],'learning_rate':[old['learning_rate'],config['learning_rate']]}}
    for optimizer in result['optimizer_states']:
        for group in optimizer['param_groups']: group['lr']=config['learning_rate']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    parent=Path(a.parent);sha=hashlib.sha256(parent.read_bytes()).hexdigest()
    if sha != PARENT_SHA: raise ValueError('Unexpected parent checkpoint hash')
    config=yaml.safe_load(Path(a.config).read_text());c=torch.load(parent,map_location='cpu',weights_only=False)
    root=Path(__file__).resolve().parents[1]/'foldtree2'
    sources={k:hashlib.sha256((root/k).read_bytes()).hexdigest() for k in c['se3']['provenance']['sources']}
    result=branch(c,config,sha,sources)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('Branch checkpoint already exists')
    torch.save(result,out)
    out.with_suffix('.json').write_text(json.dumps({'parent_sha256':sha,'branch_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'stream':result['se3']['stream'],'initialization':result['se3']['provenance']['initialization'],'preserved':'model, optimizer moments, RNG, sampler cursor, validation best; only param-group lr and declared provenance changed'},indent=2)+'\n')

if __name__=='__main__':main()
