"""Stage explicit weights-only pass migrations; never claim they were trained."""
import argparse
import copy
import json
from pathlib import Path
import torch
from foldtree2.se3_cache import open_bundle,digest
from foldtree2.train_se3_lightning import Refiner,provenance,check_refinement_warm_start,check_loss_warm_start

PARENT_SHA='3006316349da3605336ccd516f79c1eb1d5afebbfd66c386ee23ea855072548b'


def stage(parent,manifest,configs,directory):
    directory=Path(directory)
    if directory.exists():raise ValueError('Initialization output already exists')
    directory.mkdir(parents=True)
    prior=parent['se3']['provenance'];records={}
    for steps,config in configs.items():
        migrated=copy.deepcopy(prior['training'])
        migrated.update(refinement_steps=steps,refinement_loss_weights=[0.]*(steps-1)+[1.],epochs=config['epochs'])
        target=provenance(migrated,manifest)
        check_refinement_warm_start(prior,target)
        check_loss_warm_start(target,provenance(config,manifest))
        module=Refiner(config,manifest);module.load_state_dict(parent['state_dict'])
        for key,value in module.state_dict().items():
            if not torch.equal(value,parent['state_dict'][key]):raise ValueError('Pass migration changed weights')
        target['initialization']={'kind':'weights_only_pass_migration','parent_sha256':PARENT_SHA,
                                  'optimizer_reset':True,'stream_reset':True,'trained_passes':prior['training'].get('refinement_steps',1)}
        path=directory/('init_'+str(steps)+'.ckpt')
        torch.save({'protocol':'se3-weights-only-pass-migration-v1','state_dict':module.state_dict(),
                    'se3':{'provenance':target}},path)
        records[str(steps)]={'sha256':digest(path),'path':str(path),'parent_sha256':PARENT_SHA,
                              'weights_identical':True,'trained_for_new_pass_count':False if steps!=prior['training'].get('refinement_steps',1) else True}
    (directory/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    return records


def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if digest(a.parent)!=PARENT_SHA:raise ValueError('Trained pair parent hash mismatch')
    parent=torch.load(a.parent,map_location='cpu',weights_only=False)
    configs={i:yaml.safe_load(Path('configs/clariden_se3/bend_recycle_v2_'+name+'.yaml').read_text()) for i,name in ((1,'one'),(2,'two'))}
    manifest=open_bundle(configs[1]['cache'])
    print(json.dumps(stage(parent,manifest,configs,a.output)))

if __name__=='__main__':main()
