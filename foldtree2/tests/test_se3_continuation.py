import copy
import pytest
import torch
from scripts.prepare_se3_continuation import branch


def fixture():
 config={'epochs':32,'learning_rate':1e-4,'model':{'hidden':64},'nodes':2,'devices':4,'cache':'x','output':'y'}
 old={k:v for k,v in config.items() if k not in ('nodes','devices','cache','output')};old.update(epochs=16,learning_rate=5e-4)
 c={'state_dict':{'w':torch.tensor([2.])},'optimizer_states':[{'state':{0:{'exp_avg':torch.tensor([3.])}},'param_groups':[{'lr':5e-4}]}],'se3':{'stream':{'epoch':16,'cursor':0,'updates':8192},'rng':[{'torch':torch.tensor([1])}],'topology':{'world_size':8,'nodes':2},'provenance':{'sources':{'a':'b'},'training':old,'initialization':{'kind':'fresh'}}}}
 return c,config


def test_branch_preserves_training_state():
 c,config=fixture();r=branch(c,config,'hash',{'a':'b'})
 torch.testing.assert_close(r['state_dict']['w'],c['state_dict']['w']);torch.testing.assert_close(r['optimizer_states'][0]['state'][0]['exp_avg'],c['optimizer_states'][0]['state'][0]['exp_avg'])
 assert r['se3']['stream']==c['se3']['stream'] and r['se3']['rng']==c['se3']['rng']
 assert r['optimizer_states'][0]['param_groups'][0]['lr']==1e-4
 assert c['optimizer_states'][0]['param_groups'][0]['lr']==5e-4


@pytest.mark.parametrize('change',[{'model':{'hidden':128}},{'learning_rate':1e-3},{'epochs':64},{'nodes':1}])
def test_branch_rejects_unapproved_changes(change):
 c,config=fixture();config.update(change)
 with pytest.raises(ValueError):branch(c,config,'hash',{'a':'b'})


def test_branch_rejects_source_or_partial_epoch():
 c,config=fixture()
 with pytest.raises(ValueError):branch(c,config,'hash',{'a':'changed'})
 c['se3']['stream']['cursor']=1
 with pytest.raises(ValueError):branch(c,config,'hash',{'a':'b'})
