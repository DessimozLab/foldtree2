import copy
import json
from pathlib import Path
import pytest
import torch
import yaml
from scripts import submit_se3_bend_recycle as submit
from scripts.prepare_se3_bend_recycle import stage
from scripts.gate_se3_bend_recycle import assess,ERRORS,SCORES
from foldtree2.train_se3_lightning import Refiner,check_loss_warm_start,provenance
from foldtree2.tests.test_se3_pairs_seed import PAIR
from foldtree2.tests.test_se3_recycling import ARCH,entry,nonzero_head
from foldtree2.src.losses.se3_balanced import build_loss


def test_weights_only_staging_preserves_every_tensor_and_keeps_migrations_separate(tmp_path):
    manifest={'feature_dim':8,'bundle_sha256':'b','input_identity':{'decoder_sha256':PAIR['decoder_sha256']},'scale':1.}
    config={'model':ARCH,'pair_information':PAIR,'epochs':4,'refinement_steps':1,'refinement_loss_weights':[1.]}
    parent=Refiner(config,manifest)
    configs={s:{**config,'epochs':8,'refinement_steps':s,'refinement_loss_weights':[0.]*(s-1)+[1.],'loss':{'protocol':'balanced_geometry_v1','weights':{'bend':0.}}} for s in (1,2)}
    records=stage({'state_dict':parent.state_dict(),'se3':{'provenance':parent.provenance}},manifest,configs,tmp_path/'init')
    for s in (1,2):
        ck=torch.load(records[str(s)]['path'],weights_only=False)
        assert 'optimizer_states' not in ck and ck['protocol']=='se3-weights-only-pass-migration-v1'
        for k,v in ck['state_dict'].items():torch.testing.assert_close(v,parent.state_dict()[k],atol=0,rtol=0)
        for bend in (.5,2.):
            candidate={**configs[s],'loss':{'protocol':'balanced_geometry_v1','weights':{'bend':bend}}}
            check_loss_warm_start(ck['se3']['provenance'],provenance(candidate,manifest))
        bad=copy.deepcopy(configs[s]);bad['model']['hidden']=64
        with pytest.raises(ValueError):check_loss_warm_start(ck['se3']['provenance'],provenance(bad,manifest))


def test_actual_bend_config_changes_coordinate_gradients_without_changing_evaluation():
    e=entry();p=e['coords'].clone().requires_grad_()
    control=build_loss({'protocol':'balanced_geometry_v1','weights':{'bend':0.}})
    bend=build_loss({'protocol':'balanced_geometry_v1','weights':{'bend':2.}})
    a=control(p,e['angles'],e);b=bend(p,e['angles'],e)
    torch.testing.assert_close(a[2],b[2],atol=0,rtol=0)
    ga=torch.autograd.grad(a[0],p,retain_graph=True)[0];gb=torch.autograd.grad(b[0],p)[0]
    assert torch.isfinite(gb).all() and (gb-ga).norm()>0
    assert b[0]>a[0]


def fixture():
    ids=[str(i) for i in range(512)]
    ref={'validation_ids':ids,'original_control':{k:.7 for k in ERRORS},'previous_best':{'fape':.93,'tm_score':.24,'lddt':.28}}
    ref['original_control'].update(tm_score=.2,lddt=.25)
    base={k:.7 for k in ERRORS};base.update(fape=.94,tm_score=.25,lddt=.28,valid_residues=100)
    control=[{'id':i,'evaluation_fape':.94,'refined':dict(base),'raw':{'fape':.96}} for i in ids]
    candidate=copy.deepcopy(control)
    for row in candidate:row['evaluation_fape']=.92;row['refined']['fape']=.92
    return candidate,control,ref


def test_scaleup_requires_bend_recovery_and_paired_fape_and_matching_masks():
    a,b,ref=fixture();assert assess(a,b,ref,draws=100)['passed']
    for row in a:row['refined']['ca_bend_mae_radians']=.9
    result=assess(a,b,ref,draws=100)
    assert not result['passed'] and any('bend' in x for x in result['reasons'])
    a,b,ref=fixture()
    for row in a:row['evaluation_fape']=.95;row['refined']['fape']=.95
    assert not assess(a,b,ref,draws=100)['passed']
    a,b,ref=fixture();a[0]['refined']['valid_residues']=99
    with pytest.raises(ValueError,match='masks'):assess(a,b,ref,draws=100)


def test_schedule_and_derived_checks_preserve_matched_longer_protocol(tmp_path):
    budget=json.loads(Path('configs/clariden_se3/bend_recycle_v2_budget.json').read_text())
    assert sum(x['gpuh'] for x in budget['stages'])==pytest.approx(37.)
    directory=tmp_path/'configs/clariden_se3';directory.mkdir(parents=True)
    for arm in submit.ARMS:
        p=Path('configs/clariden_se3')/(submit.PREFIX+arm+'.yaml');(directory/p.name).write_text(p.read_text())
    assert len(submit.prepare_checks(tmp_path))==12
    for name,row in zip(submit.STAGES,budget['stages']):
        cmd=submit.command(tmp_path,name,'123')
        assert '--nodes='+str(row['nodes']) in cmd and '--time='+str(round(60*row['hours'])) in cmd
        assert '--dependency=afterok:123' in cmd and '--kill-on-invalid-dep=yes' in cmd
    one=submit.command(tmp_path,submit.STAGES[0])[-1]
    assert one.index('scripts.prepare_se3_bend_recycle')<one.index('--stop-after-updates 1')
    assert one.count('--resume-from')==4 and one.count('--exact')==4
    for arm in submit.ARMS:
        cfg=yaml.safe_load((directory/(submit.PREFIX+arm+'.yaml')).read_text())
        assert cfg['epochs']==8 and cfg.get('initial_geometry') is None
        mode='resume';path=tmp_path/'runtime-configs'/(submit.check_name(arm,mode)+'.yaml')
        derived=yaml.safe_load(path.read_text());derived.pop('output');derived.pop('nodes')
        assert derived=={k:v for k,v in cfg.items() if k not in ('nodes','output')}
        cmd=submit.command(tmp_path,submit.PREFIX+arm)
        assert '--allow-loss-change' in cmd and str(submit.initialization(arm)) in cmd
    quality=submit.command(tmp_path,submit.STAGES[-1])[-1]
    assert 'scripts.gate_se3_bend_recycle' in quality and 'sbatch' not in quality
