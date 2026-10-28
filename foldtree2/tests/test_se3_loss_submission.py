from pathlib import Path
import pytest
from scripts.se3.submit_se3_loss_trials import command,STAGES


def test_trials_preserve_parent_and_have_explicit_dependencies():
 for stage in STAGES[2:]:
  c=command(Path('/tmp/code'),stage,'42')
  assert '--nodes=2' in c and '--time=45' in c and '--dependency=afterok:42' in c
  assert '--allow-loss-change' in c and '--warm-start' in c and '--resume-from' not in c
  assert c[c.index('--warm-start')+1].endswith('/growth_fp64_8192/best.ckpt')


def test_checks_run_resume_and_verification_inside_container():
 c=command('/tmp/code','loss_v1_debug_one');body=c[-1]
 assert '--nodes=1' in c and '--time=10' in c
 assert '--stop-after-updates 1' in body and '--resume-from' in body and '--exact' in body
 assert 'srun --nodes=1' in body and 'source "$TRAIN_VENV/bin/activate"' in body
 c=command('/tmp/code','loss_v1_debug_two','43')
 assert '--nodes=2' in c and '--dependency=afterok:43' in c and '--exact' not in c[-1]
