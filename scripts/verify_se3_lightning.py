"""Create a small real-cache fixture and compare completed training checkpoints."""
import argparse
import json
import shutil
from pathlib import Path
import torch
import yaml
from foldtree2.se3_cache import identity_hash, open_bundle


def fixture(bundle, output):
    source=Path(bundle); root=Path(output);root.mkdir(parents=True,exist_ok=True)
    m=open_bundle(source)
    m['train_ids']=sorted(m['train_ids'],key=lambda i:m['files'][i]['length'])[:32]
    m['val_ids']=sorted(m['val_ids'],key=lambda i:m['files'][i]['length'])[:3]
    m['files']={i:m['files'][i] for i in m['train_ids']+m['val_ids']}
    del m['bundle_sha256']; m['bundle_sha256']=identity_hash(m)
    cache=root/'cache';(cache/'entries').mkdir(parents=True,exist_ok=True)
    for rec in m['files'].values():
        shutil.copyfile(source/'entries'/rec['name'],cache/'entries'/rec['name'])
    (cache/'manifest.json').write_text(json.dumps(m,indent=2))
    for name,devices in [('serial',1),('ddp',2),('restart',1)]:
        c={'cache':str(cache.resolve()),'output':str((root/name).resolve()),'train_count':32,
           'val_count':3,'seed':42,'global_batch':16,'max_residues':256,'learning_rate':.0005,
           'epochs':1,'devices':devices,'nodes':1,'accelerator':'cpu','checkpoint_every':1,
           'structural_metrics':True,'scheduler':True,'model':{'hidden':32,'depth':2,'heads':2}}
        (root/(name+'.yaml')).write_text(yaml.safe_dump(c))
    print(str(root))


def compare(left,right,atol=1e-4):
    a=torch.load(left,map_location='cpu',weights_only=False)
    b=torch.load(right,map_location='cpu',weights_only=False)
    assert a['se3']['stream']==b['se3']['stream']
    largest=0.
    for key,v in a['state_dict'].items():
        w=b['state_dict'][key]
        if v.is_floating_point():
            largest=max(largest,float((v-w).abs().max()))
        torch.testing.assert_close(v,w,atol=atol,rtol=atol)
    print(json.dumps({'maximum_parameter_difference':largest,'stream':a['se3']['stream']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    f=sub.add_parser('fixture');f.add_argument('--bundle',required=True);f.add_argument('--output',required=True)
    c=sub.add_parser('compare');c.add_argument('left');c.add_argument('right');c.add_argument('--atol',type=float,default=1e-4)
    a=p.parse_args()
    if a.mode=='fixture':fixture(a.bundle,a.output)
    else:compare(a.left,a.right,a.atol)
