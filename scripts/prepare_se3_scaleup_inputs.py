"""After the passed bend gate, select/export existing mk2 graphs for data growth.

CPU only; no graph regeneration, frozen inference, Slurm submission or merging.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import time
import torch
from foldtree2.se3_cache import digest,identity_hash
from foldtree2.se3_validation import base,frozen_entry
from scripts.prepare_se3_larger_cohort import fresh_splits


def validate_extension(previous,train,val,fresh_val):
    if len(train)!=16384 or len(val)!=1024 or len(fresh_val)!=512:
        raise ValueError('Expected 16384 training, 512 retained + 512 fresh holdouts')
    if train[:8192]!=previous['train_ids'] or val[:512]!=previous['val_ids'] or val[512:]!=fresh_val:
        raise ValueError('Retained split membership changed')
    if len(set(train+val))!=len(train+val):raise ValueError('Training/validation overlap or duplicates')
    if set(fresh_val)&set(previous['files']):raise ValueError('Fresh confirmation IDs were used previously')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous-bundle','cohort','dataset','production-inputs','quality-gate','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    previous=json.loads((Path(a.previous_bundle)/'manifest.json').read_text());cohort=json.loads(Path(a.cohort).read_text())
    gate=json.loads(Path(a.quality_gate).read_text())
    if not gate.get('passed') or not gate.get('scaleup_eligible') or gate.get('winner')!='bend05':raise ValueError('Passed recorded bend05 quality gate required')
    if identity_hash({k:v for k,v in previous.items() if k!='bundle_sha256'})!=previous['bundle_sha256']:
        raise ValueError('Previous cache manifest changed')
    if previous['train_ids']!=cohort['train_ids'] or previous['val_ids']!=cohort['val_ids'] or previous['input_identity']!=cohort['identity']:
        raise ValueError('Cohort/source identity mismatch')
    identity=previous['input_identity'];inputs=Path(a.production_inputs)
    if digest(a.dataset)!=identity['dataset_sha256']:raise ValueError('mk2 dataset changed')
    for name in ('encoder','decoder'):
        if digest(inputs/(name+'.pt'))!=identity[name+'_sha256']:raise ValueError('Frozen production pair changed')
    if hashlib.sha256(inspect.getsource(frozen_entry).encode()).hexdigest()!=identity['preprocessing']['extractor_sha256']:
        raise ValueError('Frozen extraction implementation changed')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);(root/'graphs').mkdir()
    def event(kind,**values):
        record={'event':kind,'time':time.time(),**values};print(json.dumps(record),flush=True)
        with (root/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    dataset=base.pdbgraphmk2.StructureDataset(a.dataset)
    used=set(previous['train_ids']+previous['val_ids'])|set(cohort['excluded_prior_ids'])
    event('selecting',train_count=16384,retained_validation=512,fresh_confirmation=512)
    additional,fresh_val,skipped=fresh_splits(dataset,used,8192,512,256,42)
    train=previous['train_ids']+additional;val=previous['val_ids']+fresh_val
    validate_extension(previous,train,val,fresh_val)
    selection={'protocol':'se3-scaleup-v3-cohort','identity':identity,'train_ids':train,'val_ids':previous['val_ids'],'evaluation_ids':val,
               'fresh_confirmation_ids':fresh_val,'retained_bundle_sha256':previous['bundle_sha256'],
               'excluded_prior_ids':sorted(used),'skipped':skipped,'seed':42,'max_residues':256,
               'quality_gate_winner_sha256':gate['checkpoints']['bend05']['sha256'],
               'scope':'Structure-level confirmation only; production pretraining independence not claimed'}
    (root/'cohort.json').write_text(json.dumps(selection,indent=2)+'\n')
    identifiers=[previous['train_ids'][0]]+additional+fresh_val;graphs={}
    for index,id in enumerate(identifiers):
        name=hashlib.sha256(id.encode()).hexdigest()+'.pt';path=root/'graphs'/name
        torch.save({'id':id,'graph':dataset[id].to_dict()},path)
        graphs[id]={'name':name,'sha256':digest(path)}
        if (index+1)%256==0:event('export_progress',count=index+1,total=len(identifiers))
    for name in ('encoder','decoder'):shutil.copyfile(inputs/(name+'.pt'),root/(name+'.pt'))
    template={k:v for k,v in previous.items() if k not in ('bundle_sha256','files','cohort_sha256','train_ids','val_ids','extraction_producer')}
    template.update(cohort_sha256=digest(root/'cohort.json'),train_ids=train,val_ids=previous['val_ids'],confirmation_ids=fresh_val)
    record={'format':'se3-existing-graph-export-v1','identity':identity,'cohort_sha256':digest(root/'cohort.json'),
            'graphs':graphs,'parity_id':previous['train_ids'][0],'bundle_template':template,
            'retained_bundle_sha256':previous['bundle_sha256'],'new_ids':additional+fresh_val,'source_dataset':a.dataset}
    record['export_sha256']=identity_hash(record)
    (root/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');event('complete',graphs=len(graphs),export_sha256=record['export_sha256'])

if __name__=='__main__':main()
