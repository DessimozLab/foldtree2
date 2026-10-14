"""Queue five matched architecture trials behind per-variant resume/DDP gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from scripts.se3_budget import reserve
from scripts.submit_se3_loss_trials import ROOT,PARENT,PARENT_SHA,BUNDLE_SHA

ARMS=('control','pairs','seed','pairs_seed','pairs_seed_two')
PREFIX='pairs_seed_v1_'
STAGES=(PREFIX+'debug_one',PREFIX+'debug_two')+tuple(PREFIX+a for a in ARMS)


def check_name(arm,mode):
    if arm not in ARMS or mode not in ('resume','single','two'):raise ValueError('Unknown architecture check')
    return PREFIX+'check_'+arm+'_'+mode


def prepare_checks(code):
    """Derive check configs without altering canonical configs or provenance knobs."""
    code=Path(code);directory=code/'runtime-configs';directory.mkdir(exist_ok=True)
    records={}
    for arm in ARMS:
        source=code/'configs/clariden_se3'/(PREFIX+arm+'.yaml');text=source.read_text()
        for mode in ('resume','single','two'):
            name=check_name(arm,mode)
            old='output: '+str(ROOT/'runs'/(PREFIX+arm))+'\n'
            if text.count(old)!=1 or text.count('nodes: 2\n')!=1:
                raise ValueError('Unexpected canonical configuration: '+arm)
            value=text.replace(old,'output: '+str(ROOT/'runs'/name)+'\n')
            if mode!='two':value=value.replace('nodes: 2\n','nodes: 1\n')
            path=directory/(name+'.yaml')
            if path.exists() and path.read_text()!=value:raise ValueError('Check configuration already differs')
            path.write_text(value);records[str(path.relative_to(code))]=hashlib.sha256(value.encode()).hexdigest()
    return records


def verifier(stage,arm,candidate,control,exact=False):
    check='set -eu\ncd "$TRAIN_WORKDIR"\nsource "$TRAIN_VENV/bin/activate"\n'
    check+='python -m scripts.verify_se3_architecture_checks '+shlex.join([
        str(ROOT/'runs'/candidate),str(ROOT/'runs'/control),
        '--output',str(ROOT/'artifacts'/(stage+'-'+arm+'-gate.json'))]+(['--exact'] if exact else []))+'\n'
    return 'srun --nodes=1 --ntasks=1 --environment="$EDF_PATH" --network=disable_rdzv_get --mpi=none --cpu-bind=none --kill-on-bad-exit=1 bash -c '+shlex.quote(check)+'\n'


def command(code,stage,dependency=None):
    code=Path(code).resolve()
    if stage not in STAGES:raise ValueError('Unknown architecture stage')
    one=stage==PREFIX+'debug_one';check=stage in STAGES[:2]
    nodes=1 if one else 2;minutes=30 if one else 20 if check else 45
    cmd=['sbatch','--account=a0117','--partition=normal',f'--nodes={nodes}',f'--time={minutes}',
         '--ntasks-per-node=1','--gpus-per-task=4','--cpus-per-task=8','--exclude=nid007494',
         '--parsable','--kill-on-invalid-dep=yes',f'--job-name=ft2-se3-{stage}',f'--output={code}/logs/{stage}-%j.out']
    if dependency:cmd.append('--dependency=afterok:'+dependency)
    if not check:
        return cmd+[str(code/'alps/se3/train.sbatch'),'--config','configs/clariden_se3/'+stage+'.yaml',
                    '--warm-start',str(PARENT),'--allow-refinement-change']
    train='bash "$TRAIN_WORKDIR/alps/se3/train.sbatch"'
    warm=' --warm-start '+str(PARENT)+' --allow-refinement-change'
    body='set -eu\n'
    for arm in ARMS:
        single=check_name(arm,'single');candidate=check_name(arm,'resume' if one else 'two')
        config='runtime-configs/'+candidate+'.yaml'
        if one:
            body+=f'{train} --config {config}{warm} --stop-after-updates 1\n'
            body+=f'{train} --config {config} --resume-from {ROOT}/runs/{candidate}/last.ckpt --stop-after-updates 2\n'
            body+=f'{train} --config runtime-configs/{single}.yaml{warm} --stop-after-updates 2\n'
        else:body+=f'{train} --config {config}{warm} --stop-after-updates 2\n'
        body+=verifier(stage,arm,candidate,single,exact=one)
    return cmd+['--wrap',body]


def validate_inputs(code):
    code=Path(code)
    if hashlib.sha256(PARENT.read_bytes()).hexdigest()!=PARENT_SHA:raise ValueError('Parent checkpoint mismatch')
    manifest=json.loads(Path('/iopsstor/scratch/cscs/dmoi/se3/bundle_growth_8192/manifest.json').read_text())
    expected=manifest.pop('bundle_sha256')
    if expected!=BUNDLE_SHA or hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()!=expected:
        raise ValueError('Frozen cache/split mismatch')
    for stage in tuple(PREFIX+a for a in ARMS)+tuple(check_name(a,m) for a in ARMS for m in ('resume','single','two')):
        if (ROOT/'runs'/stage).exists():raise ValueError('Output already exists: '+stage)
    if not (ROOT/'environments/028cc752968f218de4e5837f75b2e7a8a004162a/bin/activate').is_file():
        raise ValueError('Validated training environment missing')
    # The trained parent must be complete before starting a fresh comparison.
    events=[json.loads(x) for x in (ROOT/'runs/growth_fp64_8192/events.jsonl').read_text().splitlines()]
    if len([x for x in events if x['event']=='train_epoch'])!=16 or any(x['event']=='failure' for x in events):
        raise ValueError('Parent training gate failed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--submit',action='store_true')
    p.add_argument('--available-credit',type=float,required=True);a=p.parse_args()
    code=Path(__file__).resolve().parents[1];validate_inputs(code)
    schedule=code/'configs/clariden_se3/pairs_seed_v1_budget.json'
    budget=json.loads(schedule.read_text());cost=sum(x['gpuh'] for x in budget['stages'])
    ledger=ROOT/'budget.jsonl';rows=[json.loads(x) for x in ledger.read_text().splitlines()]
    if sum(x['reserved_gpuh'] for x in rows)+cost>256 or a.available_credit<cost:raise ValueError('Architecture comparison exceeds available budget')
    if any(x['stage'] in STAGES for x in rows):raise ValueError('Architecture comparison already reserved')
    hashes=prepare_checks(code);os.chdir(code);(code/'logs').mkdir(exist_ok=True)
    os.environ.update(TRAIN_WORKDIR=str(code),EDF_PATH=str(code/'alps/se3/extended-pinned.toml'),
        TRAIN_VENV=str(ROOT/'environments/028cc752968f218de4e5837f75b2e7a8a004162a'))
    for stage in STAGES:
        cmd=command(code,stage);result=subprocess.run(cmd[:1]+['--test-only']+cmd[1:],capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Slurm preflight rejected '+stage+': '+result.stderr)
    dependency=None
    for stage in STAGES:
        cmd=command(code,stage,dependency);reserve(schedule,ledger,stage,a.available_credit,cmd,a.submit)
        if a.submit:
            receipt=json.loads(Path(str(ledger)+'.'+stage+'.receipt.json').read_text());job=receipt['stdout'].strip().split(';')[0]
            if not job.isdigit():raise ValueError('Invalid Slurm receipt')
            record={'job_id':job,'stage':stage,'command':cmd,'code_revision':code.name,'parent_sha256':PARENT_SHA,
                'bundle_sha256':BUNDLE_SHA,'dependency':dependency,'check_configs_sha256':hashes,
                'credit_basis':'Existing user confirmation of sufficient credit; within original 256 GPUh rollout ceiling; not an independently retrieved portal balance'}
            (ROOT/'artifacts'/(stage+'-submission.json')).write_text(json.dumps(record,indent=2)+'\n');dependency=job

if __name__=='__main__':main()
