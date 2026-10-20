import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.se3_capacity_gate import assess


def trial(path, width, fape=.9, geometry=3.):
    path.mkdir()
    training={'epochs':4,'train_count':8192,'global_batch':16,'val_count':2,
              'model':{'hidden':width,'depth':4,'heads':4}}
    meta={'provenance':{'training':training,'bundle':'pinned','sources':{'trainer':'same'},
                        'preprocessing':{'scale':1.}},
          'stream':{'epoch':4,'cursor':0,'updates':2048}}
    torch.save({'se3':meta},path/'last.ckpt')
    best=copy.deepcopy(meta);best['stream']={'epoch':3,'cursor':0,'updates':1536}
    torch.save({'se3':best},path/'best.ckpt')
    rows=[{'id':str(i),'raw':{'fape':.95},'geometry':geometry,
           'refined':{'fape':fape,'ca_bond_mae':.3,'tm_score':.25,'lddt':.35,
                      'valid_residues':100}} for i in range(2)]
    (path/'validation-002.json').write_text(json.dumps(rows))
    (path/'events.jsonl').write_text(''.join(json.dumps({'event':'train_epoch',
        'effective_structures':8192,'epoch':i})+'\n' for i in range(4)))
    return path


def test_capacity_gate_accepts_complete_declared_four_epoch_trial(tmp_path):
    a=trial(tmp_path/'control',64);b=trial(tmp_path/'candidate',80,.89,2.9)
    result=assess(a,b)
    assert result['passed'] and result['paired_fape_ci95'][1]<0
    assert result['structures_improved']==2


@pytest.mark.parametrize('change',['incomplete','failed','baseline','duplicate','nonfinite','provenance','undefined'])
def test_capacity_gate_rejects_invalid_comparisons(tmp_path,change):
    a=trial(tmp_path/'control',64);b=trial(tmp_path/'candidate',80)
    if change=='failed':
        (b/'failure.json').write_text('{}')
    elif change in ('incomplete','provenance'):
        value=torch.load(b/'last.ckpt',weights_only=False)
        if change=='incomplete':value['se3']['stream']['updates']-=1
        else:value['se3']['provenance']['preprocessing']['scale']=2.
        torch.save(value,b/'last.ckpt')
    else:
        path=b/'validation-002.json';rows=json.loads(path.read_text())
        if change=='baseline':rows[0]['raw']['fape']=.5
        elif change=='duplicate':rows[1]['id']='0'
        elif change=='undefined':rows[0]['refined']['fape']=None
        else:rows[0]['refined']['fape']=float('nan')
        path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):assess(a,b)


def test_small_fape_gain_cannot_hide_bond_regression(tmp_path):
    a=trial(tmp_path/'control',64);b=trial(tmp_path/'candidate',80,.8999,2.99)
    path=b/'validation-002.json';rows=json.loads(path.read_text())
    for row in rows:row['refined']['ca_bond_mae']=.37
    path.write_text(json.dumps(rows))
    result=assess(a,b)
    assert not result['passed'] and 'ca_bond_mae' in result['regressions']
