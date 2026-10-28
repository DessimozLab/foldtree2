"""Select fresh bounded structures and extend the verified frozen-output cache."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch
import yaml
from foldtree2.se3_validation import (base, production_pair, sha256, frozen_entry,
    split_structures, tensor_digest, load_production_decoder)


def fresh_splits(dataset, used_ids, train_count, val_count, max_residues, seed):
    candidates=[i for i in dataset.structlist if i not in used_ids]
    train, validation, skipped=split_structures(SimpleNamespace(
        structlist=candidates,h5dataset=dataset.h5dataset),train_count,val_count,max_residues,seed)
    if set(train+validation)&set(used_ids):raise ValueError('Prior experiment IDs entered fresh cohort')
    return train,validation,skipped


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    def event(kind,**values):
        line=json.dumps({'event':kind,'time':time.time(),**values})
        with (root/'cache_events.jsonl').open('a') as f:f.write(line+'\n')
        print(line,flush=True)
    torch.set_num_threads(2);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    try:
        old=json.loads(Path('runs/se3_validation/pilot/provenance.json').read_text());identity=old['identity']
        pair=production_pair()
        if sha256(old['config']['dataset'])!=identity['dataset_sha256'] or any(sha256(pair[k])!=identity[k+'_sha256'] for k in pair):
            raise ValueError('Dataset or production pair changed')
        if identity['preprocessing']['extractor_sha256']!=hashlib.sha256(inspect.getsource(frozen_entry).encode()).hexdigest():
            raise ValueError('Frozen extractor changed')
        used=set()
        for name in ['runs/se3_validation/smoke/provenance.json','runs/se3_validation/overfit/provenance.json','runs/se3_validation/pilot/provenance.json']:
            p=json.loads(Path(name).read_text());used.update(p['train_ids']+p['val_ids'])
        dataset=base.pdbgraphmk2.StructureDataset(old['config']['dataset'])
        cohort_path=Path(config['cohort_provenance'])
        if cohort_path.exists():
            cohort=json.loads(cohort_path.read_text())
            if cohort['selection_config']!=config or cohort['identity']!=identity:raise ValueError('Cohort provenance changed')
        else:
            event('selecting',excluded_prior_count=len(used),train_count=config['train_count'],val_count=config['val_count'])
            ids,val,skipped=fresh_splits(dataset,used,config['train_count'],config['val_count'],config['max_residues'],config['seed'])
            cohort={'identity':identity,'config':old['config'],'selection_config':config,
                    'train_ids':ids,'val_ids':val,'skipped':skipped,'excluded_prior_ids':sorted(used),
                    'scope':'Fresh structure IDs relative to all local diagnostics; production pretraining independence not claimed'}
            cohort_path.write_text(json.dumps(cohort,indent=2)+'\n')
        ids=cohort['train_ids']+cohort['val_ids'];device=torch.device(config['device'])
        probe=dataset[ids[0]].to(device)
        encoder,_=base.build_encoder(SimpleNamespace(pretrained_encoder_path=pair['encoder'],pretrained_encoder_full_path=pair['encoder']),probe,device)
        if isinstance(encoder,base.FrozenProjectionEncoder):raise RuntimeError('Encoder fallback forbidden')
        decoder=load_production_decoder(pair['decoder'],device).float().eval();encoder=encoder.float().eval()
        for model in [encoder,decoder]:model.requires_grad_(False)
        before=[tensor_digest(m) for m in [encoder,decoder]]
        cache=Path(old['config']['cache'])/hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        cache.mkdir(parents=True,exist_ok=True);new=existing=0
        for index,identifier in enumerate(ids):
            path=cache/(hashlib.sha256(identifier.encode()).hexdigest()+'.pt')
            if path.exists():
                saved=torch.load(path,weights_only=False)
                if saved['identity']!=identity or saved['id']!=identifier:raise ValueError('Cache identity mismatch')
                entry=saved['entry'];existing+=1
            else:
                entry=frozen_entry(dataset[identifier],encoder,decoder,device)
                tmp=path.with_suffix('.tmp');torch.save({'identity':identity,'id':identifier,'entry':entry},tmp);tmp.replace(path);new+=1
            if index==0:
                fresh=frozen_entry(dataset[identifier],encoder,decoder,device)
                for k,v in entry.items():
                    if isinstance(v,torch.Tensor):torch.testing.assert_close(v,fresh[k],atol=1e-6,rtol=1e-6,equal_nan=True)
                event('parity_verified',id=identifier)
            if (index+1)%32==0:event('cache_progress',count=index+1,total=len(ids),new=new,existing=existing)
        if before!=[tensor_digest(m) for m in [encoder,decoder]] or any(p.grad is not None for m in [encoder,decoder] for p in m.parameters()):
            raise RuntimeError('Frozen production model mutated or received gradients')
        event('complete',count=len(ids),new=new,existing=existing,frozen_state_unchanged=True)
    except Exception as exc:event('failure',error=repr(exc));raise


if __name__=='__main__':main()
