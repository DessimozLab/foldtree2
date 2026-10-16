import json
from pathlib import Path
import pytest
import torch
import yaml
from scripts import submit_se3_scale_capacity as submit
from foldtree2.train_se3_lightning import build_model,calibrated
from foldtree2.tests.test_se3_pairs_seed import PAIR
from foldtree2.tests.test_se3_recycling import entry
from foldtree2.se3_validation import forward
from foldtree2.src.losses.se3_balanced import build_loss
from foldtree2.src.se3_struct_decoder import enable_tensor_ff_input_norm


@pytest.mark.parametrize('width',[64,80])
def test_fresh_capacity_has_finite_two_pass_learning_and_pair_head_updates(width):
    model=build_model(8,{'hidden':width,'depth':4,'heads':4},2,PAIR)
    enable_tensor_ff_input_norm(model);model.train();e=calibrated(entry(),1.)
    e['contact_z']=torch.randn(12,8)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4)
    loss=build_loss({'protocol':'balanced_geometry_v1','weights':{'bend':.5}})
    for i in range(2):
        optimizer.zero_grad();p,a=forward(model,e)
        if i==0:torch.testing.assert_close(p,e['coords'],atol=1e-6,rtol=1e-6)
        value=loss(p,a,e)[0];value.backward()
        assert torch.isfinite(value)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    assert model.gotennet.pair_projection[-1].weight.abs().sum()>0


def test_capacity_schedule_is_fresh_matched_and_within_remaining_ceiling(tmp_path):
    budget=json.loads(Path('configs/clariden_se3/scale_capacity_v3_budget.json').read_text())
    assert sum(x['gpuh'] for x in budget['stages'])==pytest.approx(14+2/3)
    directory=tmp_path/'configs/clariden_se3';directory.mkdir(parents=True)
    configs=[]
    for arm in submit.ARMS:
        path=Path('configs/clariden_se3')/(submit.PREFIX+arm+'.yaml');(directory/path.name).write_text(path.read_text())
        cfg=yaml.safe_load(path.read_text());cfg.pop('output');cfg['model'].pop('hidden');configs.append(cfg)
    assert configs[0]==configs[1] and configs[0]['refinement_steps']==2 and configs[0]['loss']['weights']['bend']==.5
    assert len(submit.prepare_checks(tmp_path))==6
    for stage,row in zip(submit.STAGES,budget['stages']):
        cmd=submit.command(tmp_path,stage,'123')
        assert '--nodes='+str(row['nodes']) in cmd and '--time='+str(round(60*row['hours'])) in cmd
        assert '--dependency=afterok:123' in cmd
        assert '--warm-start' not in ' '.join(cmd) and '--allow-loss-change' not in ' '.join(cmd)
