import pytest
import torch
from foldtree2.src.losses.se3_balanced import BalancedGeometryLoss,build_loss,coordinate_terms
from foldtree2.src.losses.se3_ablation import AblationLoss


def example(dtype=torch.float64):
 torch.manual_seed(104)
 b=torch.nn.functional.normalize(torch.randn(14,3,dtype=dtype),dim=-1)*3.8
 t=torch.cat([torch.zeros(1,3,dtype=dtype),b.cumsum(0)])
 return {'target':t,'mask':torch.ones(15,dtype=torch.bool),'angles':torch.zeros(15,3,dtype=dtype)}


def test_control_is_exact_and_perfect_prediction_zero():
 e=example();p=e['target']+.2*torch.randn_like(e['target']);a=torch.zeros_like(e['angles'])
 for x,y in zip(BalancedGeometryLoss()(p,a,e),AblationLoss(True,True)(p,a,e)):torch.testing.assert_close(x,y,atol=0,rtol=0)
 c=coordinate_terms(e['target'],e['target'],e['mask'])
 for k in ('local_distance','bend','torsion'):assert c[k]==0 and c[k+'_count']>0


def test_independent_rigid_motion_invariance_and_reflection_sensitivity():
 e=example();t=e['target'];p=t+.5*torch.randn_like(t)
 q,_=torch.linalg.qr(torch.randn(3,3,dtype=t.dtype));q[:,0]*=torch.linalg.det(q)
 ref=coordinate_terms(p,t,e['mask']);moved=coordinate_terms(p@q+10,t-21,e['mask'])
 for k in ('local_distance','bend','torsion'):torch.testing.assert_close(ref[k],moved[k],atol=1e-10,rtol=1e-8)
 mirror=t.clone();mirror[:,0]*=-1
 c=coordinate_terms(mirror,t,e['mask']);assert c['torsion']>.01
 torch.testing.assert_close(c['local_distance'],torch.zeros_like(c['local_distance']),atol=1e-15,rtol=0)


def test_masked_nan_gaps_and_degenerate_predictions_have_finite_gradients():
 e=example();m=e['mask'];m[6]=False;t=e['target'];t[~m]=float('nan')
 p=torch.zeros_like(t);p[~m]=float('nan');p.requires_grad_()
 c=coordinate_terms(p,t,m);g=torch.autograd.grad(sum(c[k] for k in ('local_distance','bend','torsion')),p)[0]
 assert torch.isfinite(g).all() and g[~m].eq(0).all()
 assert c['bend_count']==10 and c['torsion_count']==8
 p2=t.detach().clone()
 p2[0]=float('inf')
 with pytest.raises(ValueError,match='Nonfinite'):coordinate_terms(p2,t,m)


def test_coordinate_terms_pass_gradcheck():
 e=example();p=(e['target']+.2*torch.randn_like(e['target'])).requires_grad_()
 def f(p):
  c=coordinate_terms(p,e['target'],e['mask']);return torch.stack([c[k] for k in ('local_distance','bend','torsion')])
 assert torch.autograd.gradcheck(f,(p,),atol=1e-5,rtol=1e-3)


@pytest.mark.parametrize('weights',[{'unknown':1},{'bend':-1},{'torsion':float('nan')},{'bond':True}])
def test_bad_weights_rejected(weights):
 with pytest.raises(ValueError):BalancedGeometryLoss(weights)


def test_enabled_undefined_geometry_fails_and_default_unchanged():
 assert isinstance(build_loss(),AblationLoss)
 e=example();e['target']=torch.arange(15,dtype=torch.float64)[:,None]*torch.tensor([[3.8,0.,0.]])
 assert coordinate_terms(e['target'],e['target'],e['mask'])['torsion_count']==0
 with pytest.raises(ValueError,match='No valid defined frames'):BalancedGeometryLoss({'torsion':1}).components(e['target'],e['angles'],e)


def test_short_chain_masks_undefined_torsion_without_skipping_other_terms():
 e=example();e={k:v[:3] for k,v in e.items()};p=e['target']+.1*torch.randn_like(e['target'])
 c=BalancedGeometryLoss({'torsion':.5}).components(p,e['angles'],e)
 assert c['torsion_count']==0 and c['torsion']==0 and c['total']>0
 with pytest.raises(ValueError,match='No eligible active'):
  BalancedGeometryLoss({'distance':0,'training_fape':0,'bond':0,'torsion':1}).components(p,e['angles'],e)


def test_loss_warm_start_checks_data_model_and_source_boundary():
 import copy
 from foldtree2.train_se3_lightning import check_loss_warm_start
 prior={'bundle':'b','input_identity':'i','preprocessing':{'scale':1},'feature_dim':8,
        'training':{'model':{'hidden':64},'epochs':16,'learning_rate':.0005},
        'sources':{'train_se3_lightning.py':'f9769631cec5b033560dad8403aa92976d4c2c38dc2e52c8851333eb01e7fa68','model.py':'same'}}
 now=copy.deepcopy(prior);now['training']['loss']={'protocol':'balanced_geometry_v1'}
 now['sources'].update({'train_se3_lightning.py':'new','src/losses/se3_balanced.py':'new'})
 check_loss_warm_start(prior,now)
 for key,value in [('bundle','different'),('input_identity','different')]:
  bad=copy.deepcopy(now);bad[key]=value
  with pytest.raises(ValueError):check_loss_warm_start(prior,bad)
 bad=copy.deepcopy(now);bad['sources']['model.py']='changed'
 with pytest.raises(ValueError):check_loss_warm_start(prior,bad)
 bad=copy.deepcopy(prior);bad['sources']['train_se3_lightning.py']='unrecognized'
 with pytest.raises(ValueError):check_loss_warm_start(bad,now)


def test_float32_rigid_invariance_and_lightning_loss_path():
 from foldtree2.train_se3_lightning import Refiner
 e=example(torch.float32);p=e['target']+.3*torch.randn_like(e['target'])
 q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
 a=coordinate_terms(p,e['target'],e['mask']);b=coordinate_terms(p@q+20,e['target']-10,e['mask'])
 for k in ('local_distance','bend','torsion'):torch.testing.assert_close(a[k],b[k],atol=1e-4,rtol=1e-4)
 config={'model':{'hidden':32,'depth':2,'heads':2},'structural_metrics':False,
         'loss':{'protocol':'balanced_geometry_v1','weights':{'local_distance':1,'bend':.5,'torsion':.5}}}
 manifest={'feature_dim':8,'bundle_sha256':'b','input_identity':{},'scale':1}
 module=Refiner(config,manifest).eval()
 batch={**e,'id':'synthetic','coords':p,'features':torch.randn(15,8),'tokens':torch.arange(15),'dot':torch.eye(15)}
 with torch.no_grad():module.validation_step(batch,0)
 row=module.validation_rows[0]
 assert 'torsion' in row['loss_components'] and row['loss_components']['torsion_count']>0
 assert all(torch.isfinite(torch.tensor(v)) for v in row['loss_components'].values())


def test_fape_selection_and_fixed_control_metric_are_logged(tmp_path):
 from foldtree2.train_se3_lightning import Refiner
 from foldtree2.src.losses.se3_balanced import WEIGHTS
 config={'model':{'hidden':32,'depth':2,'heads':2},'structural_metrics':False,
         'loss':{'protocol':'balanced_geometry_v1'},'checkpoint_metric':'evaluation_fape',
         'global_batch':16,'output':str(tmp_path)}
 manifest={'feature_dim':8,'bundle_sha256':'b','input_identity':{},'scale':1,'train_ids':list(range(16)),'val_ids':['v']}
 module=Refiner(config,manifest);events=[];module.event=lambda kind,**values:events.append((kind,values))
 import pytorch_lightning as pl
 module.trainer=pl.Trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,enable_progress_bar=False)
 module.stream['cursor']=1
 module.validation_rows=[{'id':'v','geometry':3.,'evaluation_fape':.8,'loss_components':{**{k:1. for k in WEIGHTS},'control_geometry':2.}}]
 module.on_validation_epoch_end()
 assert module.best==.8 and module.improved
 assert events[0][1]['metrics']['loss.control_geometry']==2.
 assert events[0][1]['counts']['loss.torsion']==1
