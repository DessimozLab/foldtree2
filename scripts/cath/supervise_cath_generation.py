#!/usr/bin/env python3
"""Persistently supervise one frozen CATH generation run and validate its outputs."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import time
from types import SimpleNamespace
import re

from scripts.curnagl.supervise_curnagl_tcs import ACTIVE, RETRYABLE, accounting, save, shutil_disk_free


def verify_source(root):
    for line in (root/'source.sha256').read_text().splitlines():
        expected,relative=line.split(None,1)
        path=root/relative.strip().lstrip('*')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen generation source changed: '+relative)


def validate_family(task):
    root,family=task
    import sys
    sys.path.insert(0,str(Path(root)/'scripts'))
    from scripts.cath.generate_cath_families import validate_generated
    from scripts.curnagl.curnagl_tcs import family_record
    from scripts.rebuild_cath_families import digest
    directory=Path(root)/'data/CATH_data'/family
    proof,frozen=validate_generated(directory)
    with sqlite3.connect(f'file:{frozen["taxonomy_db"]}?mode=ro',uri=True) as db:
        record=family_record(directory,'CATH',Path(family),db,
            SimpleNamespace(SAFE=re.compile(r'^[A-Za-z0-9_.-]+$'),digest=digest))
    return {'family':family,'tips':len(proof['proteins']),
        'input_manifest_sha256':hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()}


def resume_family(task):
    root,family=task
    import sys
    sys.path.insert(0,str(Path(root)/'scripts'))
    from scripts.cath.generate_cath_families import generate_family,validate_generated
    data=Path(root)/'data';report=data/'reports'/(family['family']+'.json')
    if report.exists():
        result=json.loads(report.read_text())
        if result['status']=='passed':validate_generated(data/'CATH_data'/family['family'])
        return result  # Completed scientific exclusions are not retried by infrastructure recovery.
    return generate_family((str(data),family))


def compute(root,action):
    verify_source(root)
    from scripts.cath.generate_cath_families import frozen_plan
    from scripts.rebuild_cath_families import digest,write_json
    plan_path=root/'data/plan.json';plan=frozen_plan(plan_path,digest(plan_path))
    if action=='resume':
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        preserved=root/'recovery-preserved'/stamp;preserved.mkdir(parents=True)
        for family in plan['families']:
            name=family['family'];report=root/'data/reports'/(name+'.json')
            if not report.exists():
                for parent in ('CATH_data','rejected'):
                    path=root/'data'/parent/name
                    if path.exists():path.rename(preserved/(parent+'__'+name))
        # Previous allocation is terminal. Reconcile conservative interrupted-download reservations.
        cache=root/'data/structures'
        if cache.exists():
            actual=sum(p.stat().st_size for p in cache.iterdir() if p.is_file() and p.name!='storage.json')
            write_json(cache/'storage.json',{'reserved_or_used_bytes':actual})
        (root/'data/CATH_data').mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(resume_family,[(str(root),f) for f in plan['families']]))
        write_json(root/'data/summary.json',{'requested_families':len(results),
            'passed_families':sum(r['status']=='passed' for r in results),
            'retained_protein_tips':sum(r['retained'] for r in results if r['status']=='passed'),
            'genus_duplicates_allowed':False,'plan_sha256':digest(plan_path),'results':results,
            'infrastructure_resume':True,'preserved_partial_outputs':str(preserved)})
        if not any(r['status']=='passed' for r in results):raise ValueError('No families passed resumed generation')
        return
    summary=json.loads((root/'data/summary.json').read_text())
    expected={f['family'] for f in plan['families']}
    results=summary['results']
    if len(results)!=len(expected) or {r['family'] for r in results}!=expected or summary['plan_sha256']!=digest(plan_path):
        raise ValueError('Generation summary does not cover the frozen plan')
    passed=[r for r in results if r['status']=='passed']
    if {p.name for p in (root/'data/CATH_data').iterdir() if p.is_dir()}!={r['family'] for r in passed}:
        raise ValueError('Unexpected/missing eligible output family')
    with ProcessPoolExecutor(max_workers=8) as pool:
        verified=list(pool.map(validate_family,[(str(root),r['family']) for r in passed]))
    if len(verified)!=summary['passed_families'] or sum(r['tips'] for r in verified)!=summary['retained_protein_tips']:
        raise ValueError('Validated cohort differs from generation summary')
    write_json(root/'generation-validated.json',{'planned_families':len(expected),
        'validated_families':len(verified),'protein_tips':sum(r['tips'] for r in verified),
        'excluded_families':[r for r in results if r['status']!='passed'],
        'genus_duplicates':0,'plan_sha256':digest(plan_path),
        'summary_sha256':digest(root/'data/summary.json'),'families':verified})


def tick(root):
    import fcntl
    with (root/'.generation-supervision.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=root/'generation-supervision.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {
            'status':'active','phase':'generation',
            'job_id':json.loads((root/'jobs.json').read_text())['full_generation'],'retries':0}
        state['heartbeat_utc']=datetime.now(timezone.utc).isoformat()
        if state['status']!='active':return state
        status,rows=accounting(state['job_id']);state.update(slurm_state=status,accounting=rows)
        reports=list((root/'data/reports').glob('*.json'))
        results=[json.loads(p.read_text()) for p in reports]
        state['progress']={'processed':len(results),'passed':sum(r['status']=='passed' for r in results),
            'tips':sum(r['retained'] for r in results if r['status']=='passed')}
        if status in ACTIVE or status=='ACCOUNTING_PENDING':
            save(state_path,state);return state
        success=status=='COMPLETED' and all(r[2]=='0:0' for r in rows if r[0] in {state['job_id'],state['job_id']+'.batch'})
        if success and state['phase']=='validation':
            marker=json.loads((root/'generation-validated.json').read_text())
            state.update(status='complete',result={k:marker[k] for k in ('planned_families','validated_families','protein_tips','genus_duplicates')},
                excluded_families=len(marker['excluded_families']))
        elif success or status in RETRYABLE and state['retries']<2:
            if shutil_disk_free(root)<20*1024**3:
                state.update(status='needs_attention',reason='Insufficient filesystem/project quota for new job')
            else:
                verify_source(root)
                phase='validation' if success else state['phase']
                batch=root/('generation-validate.sbatch' if phase=='validation' else 'generation-resume.sbatch')
                job=subprocess.check_output(['sbatch','--parsable',str(batch)],text=True,timeout=30).strip().split(';')[0]
                if not job.isdigit():raise ValueError('Unexpected new Slurm job ID')
                state.setdefault('previous_jobs',[]).append({'job_id':state['job_id'],'phase':state['phase'],'state':status})
                state.update(job_id=job,phase=phase)
                state['retries']=0 if success else state['retries']+1
        else:
            state.update(status='needs_attention',reason=f'{state["phase"]}: {status}; inspect log, no scientific/OOM retries')
        save(state_path,state);return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--remote',action='store_true')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--interval',type=int,default=300)
    parser.add_argument('--action',choices=['monitor','resume','validate'],default='monitor')
    args=parser.parse_args();root=args.deployment.resolve()
    if args.interval<60:parser.error('Use at least a one-minute interval')
    if args.action!='monitor':compute(root,args.action);return
    while True:
        if args.remote:
            command=shlex.join(['python3',str(root/'supervise_cath_generation.py'),'--deployment',str(root),'--once'])
            output=subprocess.check_output(['/home/dmoi/.local/bin/unil-access','ssh','--',command],text=True,timeout=90)
            state=json.loads(next(line for line in reversed(output.splitlines()) if line.startswith('{')))
        else:state=tick(root)
        print(json.dumps(state),flush=True)
        if args.once or state['status']!='active':return
        time.sleep(args.interval)


if __name__=='__main__':main()
