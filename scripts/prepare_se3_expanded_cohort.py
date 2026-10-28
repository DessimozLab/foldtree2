"""After a capacity win, retain 4096 training IDs and select fresh 4096/1024 IDs."""
import argparse
import json
from pathlib import Path
import sys
import yaml
from scripts.prepare_se3_larger_cohort import fresh_splits, main as populate_cache
from foldtree2.se3_validation import base


def expanded_splits(dataset, retained, prior, seed=42):
    old_train, old_val = retained['train_ids'], retained['val_ids']
    if len(old_train)!=4096 or len(old_val)!=512 or set(old_train)&set(old_val):
        raise ValueError('Expected validated disjoint 4096/512 cohort')
    excluded=set(old_train+old_val)|set(prior)
    new_train,new_val,skipped=fresh_splits(dataset,excluded,4096,1024,256,seed)
    return old_train+new_train,new_val,skipped


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--capacity-gate',required=True);a=p.parse_args()
    if not json.loads(Path(a.capacity_gate).read_text())['passed']:
        raise ValueError('Expanded cache requires a passed matched capacity gate')
    config=yaml.safe_load(Path(a.config).read_text())
    if (config['train_count'],config['val_count'],config['max_residues'])!=(8192,1024,256):
        raise ValueError('Expanded cohort bounds differ from approved plan')
    retained=json.loads(Path(config['retain_cohort']).read_text())
    prior=set(retained['excluded_prior_ids'])
    for path in config.get('prior_provenance',[]):
        record=json.loads(Path(path).read_text());prior.update(record['train_ids']+record['val_ids'])
    output=Path(config['cohort_provenance']);output.parent.mkdir(parents=True,exist_ok=True)
    if not output.exists():
        dataset=base.pdbgraphmk2.StructureDataset(retained['config']['dataset'])
        train,val,skipped=expanded_splits(dataset,retained,prior,config['seed'])
        record={'identity':retained['identity'],'config':retained['config'],'selection_config':config,
                'train_ids':train,'val_ids':val,'skipped':skipped,'excluded_prior_ids':sorted(prior),
                'retained_cohort':config['retain_cohort'],'retired_validation_ids':retained['val_ids'],
                'scope':'New refiner validation structures exclude all earlier used IDs; production pretraining independence not claimed'}
        output.write_text(json.dumps(record,indent=2)+'\n')
    else:
        record=json.loads(output.read_text())
        if record['selection_config']!=config:raise ValueError('Expanded cohort provenance changed')
    sys.argv=[sys.argv[0],'--config',a.config]
    populate_cache()


if __name__=='__main__':main()
