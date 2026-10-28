import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import scripts.cath.supervise_cath_generation as supervisor


def run_root(tmp_path):
    (tmp_path/'jobs.json').write_text(json.dumps({'full_generation':'123'}))
    return tmp_path


def test_queue_wait_does_not_submit_or_retry(tmp_path,monkeypatch):
    root=run_root(tmp_path)
    monkeypatch.setattr(supervisor,'accounting',lambda job:('PENDING',[]))
    state=supervisor.tick(root)
    assert state['status']=='active' and state['job_id']=='123'
    assert state['retries']==0


def test_scientific_failure_is_not_retried(tmp_path,monkeypatch):
    root=run_root(tmp_path)
    monkeypatch.setattr(supervisor,'accounting',lambda job:('FAILED',[]))
    state=supervisor.tick(root)
    assert state['status']=='needs_attention'
    assert state['job_id']=='123'


def test_infrastructure_retry_uses_resume_and_is_bounded(tmp_path,monkeypatch):
    root=run_root(tmp_path)
    monkeypatch.setattr(supervisor,'accounting',lambda job:('TIMEOUT',[]))
    monkeypatch.setattr(supervisor,'shutil_disk_free',lambda root:50*1024**3)
    monkeypatch.setattr(supervisor,'verify_source',lambda root:None)
    submissions=[]
    def submit(argv,**kwargs):submissions.append(argv);return str(124+len(submissions))
    monkeypatch.setattr(supervisor.subprocess,'check_output',submit)
    assert supervisor.tick(root)['retries']==1
    assert submissions[0][-1].endswith('generation-resume.sbatch')
    assert supervisor.tick(root)['retries']==2
    assert supervisor.tick(root)['status']=='needs_attention'
    assert len(submissions)==2


def test_success_releases_final_validation_not_completion(tmp_path,monkeypatch):
    root=run_root(tmp_path)
    monkeypatch.setattr(supervisor,'accounting',lambda job:('COMPLETED',[['123','COMPLETED','0:0','', '']]))
    monkeypatch.setattr(supervisor,'shutil_disk_free',lambda root:50*1024**3)
    monkeypatch.setattr(supervisor,'verify_source',lambda root:None)
    monkeypatch.setattr(supervisor.subprocess,'check_output',lambda *a,**k:'124')
    state=supervisor.tick(root)
    assert state['phase']=='validation' and state['status']=='active'
