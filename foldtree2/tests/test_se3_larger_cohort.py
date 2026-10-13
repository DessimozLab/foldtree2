from types import SimpleNamespace
import numpy as np
import torch
from scripts.prepare_se3_larger_cohort import fresh_splits


def test_fresh_cohort_excludes_prior_structures_and_undefined_geometry():
    torch.manual_seed(94)
    groups={}
    for i in range(12):
        n=30 if i==10 else 10
        coords=torch.randn(n,3)
        if i==11:coords=torch.zeros(n,3)
        groups[str(i)]={'node':{'res':{'x':np.zeros((n,8))},'coords':{'x':coords.numpy()},
                              'bondangles':{'x':np.zeros((n,3))},'plddt':{'x':np.ones((n,1))}}}
    dataset=SimpleNamespace(structlist=list(groups),h5dataset={'structs':groups})
    used={'0','1','2','3','4','5'}
    train,val,skipped=fresh_splits(dataset,used,2,2,20,42)
    assert len(train)==2 and len(val)==2
    assert not set(train)&set(val) and not set(train+val)&used
    assert set(train+val)=={'6','7','8','9'}
