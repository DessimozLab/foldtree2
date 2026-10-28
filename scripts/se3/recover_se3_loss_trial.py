"""One bounded recovery of a failed loss trial using its existing reservation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/capstor/store/cscs/swissai/a0117/foldtree2/se3')
REVISION='d7108d96e95e2fa7cf1ce07abdda5fc61abbfb48'
STAGE='loss_v1_local_distances'
RECOVERY=STAGE+'_recovery1'
PARENT_SHA='93ff91d18ff9b4b0df5396889f9cd6c4fc0c91f65f88775caf71d2c90487bb04'


def recovery_cost(elapsed_seconds,remaining_minutes=40):
    if elapsed_seconds<0 or remaining_minutes<=0 or elapsed_seconds+remaining_minutes*60>45*60:
        raise ValueError('Recovery would exceed original 45-minute allocation')
    return (elapsed_seconds+remaining_minutes*60)*8/3600


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--submit',action='store_true');a=p.parse_args()
    code=ROOT/'code'/REVISION;artifact=ROOT/'artifacts';ledger=ROOT/'budget.jsonl'
    result=subprocess.run(['sacct','-j','3663705','--format=JobIDRaw,State,ElapsedRaw,AllocTRES','-nP'],check=True,capture_output=True,text=True)
    row=next(x.split('|') for x in result.stdout.splitlines() if x.split('|')[0]=='3663705')
    if row[1]!='FAILED' or 'gres/gpu=8' not in row[3]:raise ValueError('Unexpected failed job state/allocation')
    elapsed=int(row[2]);ceiling=recovery_cost(elapsed)
    state=subprocess.run(['scontrol','show','node','nid007494'],check=True,capture_output=True,text=True).stdout
    if 'DRAIN' not in state or 'Failed to allocate 90% of the GPU memory after job 3663705' not in state:
        raise ValueError('Expected failed-node health evidence not present')
    if hashlib.sha256((ROOT/'runs/growth_fp64_8192/best.ckpt').read_bytes()).hexdigest()!=PARENT_SHA:raise ValueError('Parent checkpoint changed')
    for name in ('loss_v1_debug_one','loss_v1_debug_two'):
        if not json.loads((artifact/(name+'-gate.json')).read_text())['passed']:raise ValueError('Stability gate missing')
    if (ROOT/'runs'/RECOVERY).exists():raise ValueError('Recovery output exists; do not repeat')
    original=(code/'configs/clariden_se3/loss_v1_local_distances.yaml').read_text()
    old='output: '+str(ROOT/'runs'/STAGE);new='output: '+str(ROOT/'runs'/RECOVERY)
    if original.count(old)!=1:raise ValueError('Unexpected original config')
    config=artifact/(RECOVERY+'.yaml');config.write_text(original.replace(old,new))
    cmd=['sbatch','--account=a0117','--partition=normal','--nodes=2','--time=40','--ntasks-per-node=1','--gpus-per-task=4','--cpus-per-task=8','--exclude=nid007494','--parsable','--job-name=ft2-se3-'+RECOVERY,'--output='+str(code/'logs'/(RECOVERY+'-%j.out')),str(code/'alps/se3/train.sbatch'),'--config',str(config),'--warm-start',str(ROOT/'runs/growth_fp64_8192/best.ckpt'),'--allow-loss-change']
    os.environ.update(TRAIN_WORKDIR=str(code),EDF_PATH=str(code/'alps/se3/extended-pinned.toml'),TRAIN_VENV=str(ROOT/'environments/028cc752968f218de4e5837f75b2e7a8a004162a'))
    preflight=subprocess.run(cmd[:1]+['--test-only']+cmd[1:],capture_output=True,text=True)
    if preflight.returncode:raise RuntimeError(preflight.stderr)
    print(json.dumps({'command':cmd,'failed_seconds':elapsed,'failed_plus_retry_ceiling_gpuh':ceiling,'original_reserved_gpuh':6,'new_reserved_gpuh':0}),flush=True)
    if not a.submit:return
    with ledger.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);rows=[json.loads(x) for x in f if x.strip()]
        parent=[x for x in rows if x['stage']==STAGE]
        if len(parent)!=1 or parent[0]['reserved_gpuh']!=6:raise ValueError('Original reservation not present')
        if any(x.get('parent_reservation')==STAGE or x['stage']==RECOVERY for x in rows):raise ValueError('Recovery allowance already used')
        record={'stage':RECOVERY,'reserved_gpuh':0,'parent_reservation':STAGE,'failed_job':'3663705','failed_seconds':elapsed,'retry_minutes':40,'combined_ceiling_gpuh':ceiling,'time':time.time(),'command':cmd,'state':'submitting'}
        f.seek(0,2);f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
        submit=subprocess.run(cmd,check=True,capture_output=True,text=True)
        job=submit.stdout.strip().split(';')[0]
        if not job.isdigit():raise ValueError('Unexpected receipt')
        record.update(job_id=job,code_revision=REVISION,parent_sha256=PARENT_SHA,node_health=state,controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        receipt=artifact/(RECOVERY+'-submission.json');receipt.write_text(json.dumps(record,indent=2)+'\n')
        # Change only our waiting backbone job; combined already depends on it.
        pending=subprocess.run(['squeue','-h','-j','3663706','-o','%T'],check=True,capture_output=True,text=True).stdout.strip()
        if pending!='PENDING':raise ValueError('Backbone job no longer pending; leave dependencies untouched')
        subprocess.run(['scontrol','update','JobId=3663706','Dependency=afterok:'+job],check=True)
        record['rewired_backbone_job']='3663706';receipt.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'job_id':job,'rewired_backbone_job':'3663706','combined_job':'3663707','total_reserved_gpuh':sum(x['reserved_gpuh'] for x in rows)}),flush=True)

if __name__=='__main__':main()
