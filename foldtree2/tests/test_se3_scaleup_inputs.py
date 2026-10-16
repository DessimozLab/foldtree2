import pytest
from scripts.prepare_se3_scaleup_inputs import validate_extension


def test_growth_retains_splits_and_reserves_fresh_confirmation():
    previous={'train_ids':['t'+str(i) for i in range(8192)],'val_ids':['v'+str(i) for i in range(512)]}
    previous['files']={i:{} for i in previous['train_ids']+previous['val_ids']}
    fresh=['f'+str(i) for i in range(512)]
    train=previous['train_ids']+['n'+str(i) for i in range(8192)]
    val=previous['val_ids']+fresh
    validate_extension(previous,train,val,fresh)
    changed=list(train);changed[0],changed[1]=changed[1],changed[0]
    with pytest.raises(ValueError,match='membership'):validate_extension(previous,changed,val,fresh)
    bad=list(fresh);bad[0]=previous['train_ids'][0]
    with pytest.raises(ValueError):validate_extension(previous,train,previous['val_ids']+bad,bad)
    bad=list(fresh);bad[0]=bad[1]
    with pytest.raises(ValueError):validate_extension(previous,train,previous['val_ids']+bad,bad)
