import json
from pathlib import Path
import pytest
import yaml
from scripts.submit_se3_architecture_trials import ARMS,STAGES,PREFIX,ROOT,command,prepare_checks,check_name


def test_checks_only_change_output_and_topology(tmp_path):
    directory=tmp_path/'configs/clariden_se3';directory.mkdir(parents=True)
    for arm in ARMS:
        source=Path('configs/clariden_se3')/(PREFIX+arm+'.yaml')
        (directory/source.name).write_text(source.read_text())
    records=prepare_checks(tmp_path)
    assert len(records)==15
    for arm in ARMS:
        original=yaml.safe_load((directory/(PREFIX+arm+'.yaml')).read_text())
        for mode in ('resume','single','two'):
            name=check_name(arm,mode);config=yaml.safe_load((tmp_path/'runtime-configs'/(name+'.yaml')).read_text())
            assert config.pop('nodes')==(2 if mode=='two' else 1)
            assert config.pop('output')==str(ROOT/'runs'/name)
            assert config=={k:v for k,v in original.items() if k not in ('nodes','output')}
    assert prepare_checks(tmp_path)==records


def test_all_allocations_match_bounded_schedule_and_dependency_chain():
    budget=json.loads(Path('configs/clariden_se3/pairs_seed_v1_budget.json').read_text())
    assert sum(s['gpuh'] for s in budget['stages'])==pytest.approx(34+2/3)
    for stage,row in zip(STAGES,budget['stages']):
        assert stage==row['name'];cmd=command('/tmp/code',stage,'123')
        assert '--nodes='+str(row['nodes']) in cmd
        assert '--time='+str(round(row['hours']*60)) in cmd
        assert '--dependency=afterok:123' in cmd and '--kill-on-invalid-dep=yes' in cmd
        assert '--account=a0117' in cmd and '--partition=normal' in cmd
        assert '--exclude=nid007494' in cmd
    one=command('/tmp/code',STAGES[0])[-1];two=command('/tmp/code',STAGES[1])[-1]
    for arm in ARMS:
        assert check_name(arm,'resume')+'.yaml' in one
        assert str(ROOT/'runs'/check_name(arm,'resume')/'last.ckpt') in one
        assert check_name(arm,'single')+'.yaml' in one
        assert check_name(arm,'two')+'.yaml' in two
    assert one.count('--exact')==5 and one.count('--resume-from')==5
    assert two.count('scripts.verify_se3_architecture_checks')==5
    for stage in STAGES[2:]:
        cmd=command('/tmp/code',stage,'123')
        assert '--allow-refinement-change' in cmd and '--allow-loss-change' not in cmd
