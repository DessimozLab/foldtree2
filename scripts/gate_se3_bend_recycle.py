"""Predeclared eligibility for more data/capacity; this script never submits jobs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ARMS=('one','two','bend05','bend20')
ERRORS=('ca_rmsd','ca_bond_mae','ca_bend_mae_radians','ca_torsion_mae_radians','angle_mae_radians')
SCORES=('tm_score','lddt')


def assess(rows,control,reference,draws=10000):
    def keyed(values):
        result={r['id']:r for r in values}
        if len(values)!=512 or len(result)!=512 or set(result)!=set(reference['validation_ids']):
            raise ValueError('Gate requires exactly the pinned 512 validation IDs')
        return result
    candidate=keyed(rows);baseline=keyed(control);ids=sorted(candidate)
    for id in ids:
        if candidate[id]['raw']!=baseline[id]['raw'] or candidate[id]['refined']['valid_residues']!=baseline[id]['refined']['valid_residues']:
            raise ValueError('Baseline or residue masks differ')
    means={};base_means={}
    for k in ('fape',)+ERRORS+SCORES:
        vals=[candidate[i]['refined'][k] for i in ids if candidate[i]['refined'][k] is not None]
        base=[baseline[i]['refined'][k] for i in ids if baseline[i]['refined'][k] is not None]
        if not vals or len(vals)!=len(base) or not all(math.isfinite(x) for x in vals+base):raise ValueError('Invalid structural metric')
        means[k]=float(np.mean(vals));base_means[k]=float(np.mean(base))
    delta=np.array([candidate[i]['evaluation_fape']-baseline[i]['evaluation_fape'] for i in ids])
    idx=np.random.default_rng(42).integers(0,512,size=(draws,512))
    ci=np.quantile(delta[idx].mean(1),[.025,.975]).tolist()
    reasons=[]
    if ci[1]>=0:reasons.append('No paired FAPE improvement over matched longer one-pass control')
    if means['fape']>reference['previous_best']['fape']:reasons.append('Previous best FAPE gain not retained')
    for k in ERRORS:
        # Bend must recover against the original pre-pair control; other errors
        # must not materially worsen against original or matched control.
        limit=1.05*min(reference['original_control'][k],base_means[k])
        if means[k]>limit:reasons.append(k+' exceeds 5% regression bound')
    for k in SCORES:
        limit=max(reference['original_control'][k],reference['previous_best'][k],base_means[k])-.01
        if means[k]<limit:reasons.append(k+' loses more than 0.01')
    return {'passed':not reasons,'reasons':reasons,'means':means,'matched_control_means':base_means,
            'paired_fape_mean_difference':float(delta.mean()),'paired_fape_ci95':ci,'structures_improved':int((delta<0).sum()),
            'bootstrap':{'seed':42,'draws':draws,'unit':'structure','selection_caveat':'Development-set checkpoint/arm selection; fresh confirmation needed'}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root);reference=json.loads(Path('configs/clariden_se3/bend_recycle_v2_reference.json').read_text())
    selected={};checkpoints={}
    for arm in ARMS:
        path=root/'runs'/('bend_recycle_v2_'+arm)
        events=[json.loads(s) for s in (path/'events.jsonl').read_text().splitlines()]
        epochs=[r for r in events if r['event']=='train_epoch']
        if len(epochs)!=8 or epochs[-1]['global_step']!=4096 or any(r['effective_structures']!=8192 for r in epochs) or list(path.glob('failure*')):
            raise ValueError('Incomplete or failed longer trial: '+arm)
        best=min([r for r in events if r['event']=='validation' and r['complete_epoch']],key=lambda r:r['metrics']['evaluation_fape'])
        selected[arm]=json.loads((path/('validation-%03d.json'%best['epoch'])).read_text())
        checkpoints[arm]={'path':str(path/'best.ckpt'),'sha256':hashlib.sha256((path/'best.ckpt').read_bytes()).hexdigest(),'epoch':best['epoch']+1}
    gates=[json.loads((root/'artifacts'/('bend_recycle_v2_debug_'+topology+'-'+arm+'-gate.json')).read_text()) for topology in ('one','two') for arm in ARMS]
    if not all(x['passed'] for x in gates):raise ValueError('Stability gate failed')
    results={arm:assess(selected[arm],selected['one'],reference) for arm in ARMS if arm!='one'}
    eligible=[arm for arm,row in results.items() if row['passed']]
    winner=min(eligible,key=lambda arm:results[arm]['means']['fape']) if eligible else None
    result={'protocol':'bend-recycle-v2-quality','passed':winner is not None,'winner':winner,
            'checkpoints':checkpoints,'comparisons':results,'scaleup_authorized_by_user':True,
            'scaleup_eligible':winner is not None,'scaleup_submitted':False,
            'next_action':'Prepare larger-data and capacity checks within remaining budget' if winner else 'Preserve results and diagnose; do not scale',
            'reference':reference}
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
