import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import yaml
from foldtree2.src.se3_struct_decoder import se3_denoiser
from scripts import train_se3_revised_pilot as trainer


def test_warm_started_training_records_real_baseline_and_resumes(tmp_path, monkeypatch):
    torch.manual_seed(95)
    phase=torch.arange(12)*2*torch.pi/12
    coords=torch.stack([8*phase.cos(),8*phase.sin(),torch.zeros(12)],-1)
    entries={identifier:{'features':torch.randn(12,8),'coords':coords.clone(),
        'target':coords+.1*torch.randn_like(coords),'tokens':torch.arange(12),
        'dot':torch.zeros(12,12,dtype=torch.bool),'mask':torch.ones(12,dtype=torch.bool),
        'angles':torch.zeros(12,3),'baseline_angles':None} for identifier in ['a','b','c']}
    source={'identity':{'dataset':'fixture'},'train_ids':['a','b'],'val_ids':['c'],'scope':'fixture'}
    monkeypatch.setattr(trainer,'load_cohort_entries',lambda _: (source,entries))
    monkeypatch.setattr(trainer,'sha256',lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest())
    spec={'scale':1.,'lift_fraction':.1,'gain':32.,'angles':'sincos','update':'residual',
          'normalization':'rms_radius','hidden':32,'depth':2,'heads':2,'dim_head':16}
    model=se3_denoiser(8,[32],3,40,.25,dropout_p=0.,depth=2,heads=2,dim_head=16,
        num_atom_types=40,coordinate_update='residual',zero_init_coordinate_update=True,
        coordinate_normalization='rms_radius',coordinate_residual_gain=32.,angle_output='sincos',device='cpu')
    warm=tmp_path/'warm.pt'
    torch.save({'model':model.state_dict(),'epoch':8,'provenance':{'spec':spec,
        'input_identity':source['identity'],'sources':{name:trainer.sha256(name) for name in
        ['foldtree2/src/se3_struct_decoder.py','foldtree2/src/losses/se3_ablation.py']}}},warm)
    calibration=tmp_path/'calibration.json';calibration.write_text(json.dumps({'train_ids':['old'],'scale':1.}))
    cohort=tmp_path/'cohort.json';cohort.write_text('{}')
    output=tmp_path/'run';config={'output':str(output),'seed':42,'device':'cpu','epochs':1,
        'train_count':2,'val_count':1,'max_residues':12,'batch_size':1,'effective_batch_size':8,
        'learning_rate':.0005,'gradient_clip':1.,'patience':8,'min_delta':.001,
        'monitor':'geometry','scheduler_factor':.5,'warm_start':str(warm),
        'calibration_provenance':str(calibration),'cohort_provenance':str(cohort)}
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    trainer.run(SimpleNamespace(config=str(path),resume_from=None))
    history=json.loads((output/'history.json').read_text())
    assert history[0]['global_step']==1
    assert history[0]['monitor_score']==history[0]['val_geometry']
    assert history[0]['val_fape']!=history[0]['monitor_score']
    initial=torch.load(output/'initial.pt',weights_only=False)
    assert initial['epoch']==-1 and initial['global_step']==0
    assert (output/'initial_losses.json').exists()
    pairs=json.loads((output/'heldout_pairs.json').read_text())
    assert 'warm_start' in pairs[0]
    trainer.run(SimpleNamespace(config=str(path),resume_from=str(output/'last.pt')))
    assert json.loads((output/'history.json').read_text())==history
