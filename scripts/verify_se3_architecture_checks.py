"""Apply existing resume/topology gates and verify new heads actually update."""
import argparse
import json
from pathlib import Path
import torch
from scripts.verify_se3_loss_checks import compare


def verify(candidate,control,exact=False):
    result=compare(candidate,control,exact)
    checkpoint=torch.load(Path(candidate)/'last.ckpt',map_location='cpu',weights_only=False)
    config=checkpoint['se3']['provenance']['training'];state=checkpoint['state_dict'];norms={}
    for option,key in (('pair_information','model.gotennet.pair_projection.2.weight'),
                       ('initial_geometry','model.seed_initializer.mlp.3.weight')):
        if config.get(option) is not None:
            value=state[key]
            if not torch.isfinite(value).all() or not value.abs().sum()>0:
                raise ValueError('New architecture head did not update: '+key)
            norms[key]=float(value.norm())
    return {**result,'extension_weight_norms':norms}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate');p.add_argument('control')
    p.add_argument('--exact',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    try:result=verify(a.candidate,a.control,a.exact)
    except Exception as exc:
        Path(a.output).write_text(json.dumps({'passed':False,'error':str(exc)},indent=2)+'\n');raise
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
