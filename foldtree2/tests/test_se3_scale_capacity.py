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


def test_width80_canonical_config_reaches_cache_preflight_from_cli(tmp_path,monkeypatch):
    import sys
    import foldtree2.train_se3_lightning as trainer
    cfg=yaml.safe_load(Path('configs/clariden_se3/scale_capacity_v3_width80.yaml').read_text())
    path=tmp_path/'width80.yaml';path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys,'argv',['train','--config',str(path),'--stop-after-updates','2'])
    monkeypatch.setattr(trainer.pl,'seed_everything',lambda *a,**kw:None)
    monkeypatch.setattr(trainer.torch,'set_num_threads',lambda *a:None)
    monkeypatch.setattr(trainer.torch,'use_deterministic_algorithms',lambda *a:None)
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',torch.backends.cudnn.allow_tf32)
    def cache(_):raise RuntimeError('cache validation reached')
    monkeypatch.setattr(trainer,'open_bundle',cache)
    with pytest.raises(RuntimeError,match='cache validation reached'):trainer.main()


def test_capacity_policy_still_blocks_depth_six_and_unsafe_width80():
    from foldtree2.train_se3_lightning import validate_experiment_architecture
    cfg=yaml.safe_load(Path('configs/clariden_se3/scale_capacity_v3_width80.yaml').read_text())
    validate_experiment_architecture(cfg)
    cfg['tensor_ff_input_norm']=False
    with pytest.raises(ValueError,match='normalization'):validate_experiment_architecture(cfg)
    cfg['tensor_ff_input_norm']=True;cfg['gradient_accumulation_dtype']='float32'
    with pytest.raises(ValueError,match='float64'):validate_experiment_architecture(cfg)
    cfg['model']={'hidden':80,'depth':6,'heads':4}
    with pytest.raises(ValueError,match='depth-six'):validate_experiment_architecture(cfg)


def test_recovery_reuses_only_never_started_reservations_and_cannot_repeat(tmp_path,monkeypatch):
    from scripts import recover_se3_scale_capacity as recover
    cancelled=['3675418','CANCELLED','0','None']
    assert recover.cancelled_without_start([cancelled],'3675418')==cancelled
    for rows in [[['3675418','CANCELLED','1','None']],
                 [['3675418','CANCELLED','0','2026-10-16T10:00:00']],
                 [cancelled,['3675418.batch','CANCELLED','0','None']]]:
        with pytest.raises(ValueError,match='never-started'):recover.cancelled_without_start(rows,'3675418')
    ledger=tmp_path/'budget.jsonl'
    ledger.write_text(json.dumps({'stage':'scale_capacity_v3_debug_two','reserved_gpuh':4/3})+'\n')
    cmd=submit.command(tmp_path,'scale_capacity_v3r1_debug_two','123',prefix=recover.PREFIX)
    record=recover.submit_reusing(ledger,'scale_capacity_v3r1_debug_two','scale_capacity_v3_debug_two',
        '3675418',cmd,cancelled,False)
    assert record['reserved_gpuh']==0 and record['reservation_ceiling_gpuh']==pytest.approx(4/3)
    original=ledger.read_text()
    ledger.write_text(original+json.dumps(record)+'\n')
    with pytest.raises(ValueError,match='already used'):
        recover.submit_reusing(ledger,record['stage'],record['parent_reservation'],'3675418',cmd,cancelled,False)
