"""Report a completed frozen-cache loss ablation without rerunning models."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')


def report(root, output):
    load=lambda name:json.loads((root/name).read_text())
    events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    if events[-1]['event']!='complete':raise ValueError('Experiment is incomplete')
    provenance,screen,full,pairs=map(load,['provenance.json','screening.json','overfit.json','heldout_pairs.json'])
    assert len(provenance['train_ids'])==32 and len(provenance['heldout_ids'])==128
    assert not set(provenance['train_ids']) & set(provenance['heldout_ids'])
    keys=['fape','ca_rmsd','tm_score','lddt','ca_bond_mae','ca_bend_mae_radians','ca_torsion_mae_radians']
    summary={};higher={'tm_score','lddt'}
    for case,rows in pairs.items():
        assert len(rows)==128 and {r['id'] for r in rows}==set(provenance['heldout_ids'])
        summary[case]={}
        for key in keys:
            before=[r['input_seed'][key] for r in rows];after=[r['refined'][key] for r in rows]
            assert all(math.isfinite(v) for v in before+after)
            a,b=statistics.mean(before),statistics.mean(after)
            summary[case][key]={'input':a,'refined':b,'delta':b-a,
                               'improved':sum(y>x if key in higher else y<x for x,y in zip(before,after)),
                               'material_regression':b<a-.01 if key in higher else b>a*1.05}
    candidate=next(r for r in full if r['spec']['loss_protocol']=='soft_fape_bond')
    gates={'training_reduction_20_percent':candidate['reduction']>=.2,
           'heldout_fape':summary['soft_fape_bond']['fape']['delta']<0,
           'no_material_regression':not any(v['material_regression'] for v in summary['soft_fape_bond'].values())}
    result={'metrics':summary,'gates':gates,'eligible_for_revised_pilot':all(gates.values()),
            'ready_for_larger_run':False,'failures':sum(e['event']=='failure' for e in events)}
    (root/'gate_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3))
    for r in full:
        for ax,key in zip(axes,['eval_geometry','eval_fape']):
            ax.plot([x['epoch']+1 for x in r['history']],[x[key] for x in r['history']],label=r['spec']['loss_protocol'])
            ax.set(xlabel='Epoch',ylabel=key);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=160);plt.close(fig)
    lines=['# SE(3) loss ablation, 2026-10-09','',
        'Four eight-structure/ten-epoch screens isolate hard versus soft FAPE and the local bond term. Two predeclared cases—original control and combined soft-FAPE/bond objective—were then reinitialized and trained on the same 32 structures for 30 epochs each, with identical seed 42, order, architecture and optimizer. Each was evaluated on the same 128 disjoint structures; holdouts did not select a loss or checkpoint.', '',
        'The original CA-frame evaluation FAPE still clamps at 10 Å and divides by 10. The revised training term is pseudo-Huber: mean(sqrt(1+(frame_error/10 Å)²)−1), with a nonzero linear-tail gradient rather than a hard clamp. The bond term is SmoothL1 in Å (beta 1 Å, weight 1) against contiguous target CA distances. Adjacent masked residues and target distances outside [2.5,4.5] Å are excluded as gaps. Distance SmoothL1/10 and circular angle weight 0.1 are retained.', '',
        'These fixed weights were declared before the experiment. An eligibility window based on target bond length is a training-only continuity heuristic, not a claim of chemically validated atom geometry. Coordinate metrics still use the original identical masks; no evaluation mask was changed to improve a result.', '',
        '| Screen | Initial objective geometry | Best | Reduction | Initial clamped FAPE | Final clamped FAPE |',
        '|---|---:|---:|---:|---:|---:|']
    for r in screen:
        lines.append(f"| {r['spec']['loss_protocol']} | {r['initial_geometry']:.6f} | {r['best_geometry']:.6f} | {r['reduction']*100:.2f}% | {statistics.mean(x['fape'] for x in r['initial']):.6f} | {statistics.mean(x['fape'] for x in r['final']):.6f} |")
    lines+=['','| Full 32-structure case | Initial objective geometry | Best | Reduction | Selected epoch |',
            '|---|---:|---:|---:|---:|']
    for r in full:
        lines.append(f"| {r['spec']['loss_protocol']} | {r['initial_geometry']:.6f} | {r['best_geometry']:.6f} | {r['reduction']*100:.2f}% | {r['best_epoch']+1} |")
    lines+=['','Objective magnitudes and percentage reductions belong to each loss definition and must not be compared as the same geometry metric. Both models start from identical lifted coordinates. Checkpoints use their own training objective, never held-out quality.', '',
            f"![Learning curves]({os.path.relpath(root/'learning_curves.png',output.parent)})", '',
            '| Held-out metric | Lifted input | Original loss | Revised loss | Revised − input | Improved / 128 |',
            '|---|---:|---:|---:|---:|---:|']
    for key in keys:
        a=summary['control'][key];b=summary['soft_fape_bond'][key]
        assert abs(a['input']-b['input'])<1e-8
        lines.append(f"| {key} | {a['input']:.6f} | {a['refined']:.6f} | {b['refined']:.6f} | {b['delta']:+.6f} | {b['improved']} |")
    lines+=['',f"Gates: ≥20% own-objective training reduction **{gates['training_reduction_20_percent']}**; better held-out FAPE **{gates['heldout_fape']}**; no material structural regression **{gates['no_material_regression']}**. Material regression uses the existing >5% error / >0.01 TM-score or lDDT thresholds. Eligible for a revised 512/128 pilot: **{all(gates.values())}**. Larger training remains deferred pending a successful pilot.", '',
            'TM-score is TM-align; lDDT and bend/torsion metrics are CA-only. These are refiner holdouts, not a claim of independence from production pretraining. Per-structure comparisons are retained without significance or uncertainty claims.', '',
            'All cases use the fixed fraction-0.1 oriented-normal lift, circular angles, zero-initialized residual with gain 32 and RMS normalization, width 32/depth 2/two heads/head dimension 16; float32, batch 1/effective 8, AdamW 5e-4/weight decay 0.01, clip 1 and geometry-based scheduler. Only existing frozen epoch-40 caches were read; no production forward calls, checkpoint replacement or graph regeneration occurred.', '',
            f"Failure events: {result['failures']}. Full-run peak allocated memory: {max(x['peak_memory_bytes'] for r in full for x in r['history'])/1024**3:.3f} GiB. Mean epoch times: "+', '.join(f"{r['spec']['loss_protocol']} {statistics.mean(x['seconds'] for x in r['history']):.2f} s" for r in full)+'.', '',
            'Source hashes, executed snapshots, software/device identity, IDs, initial component values/gradient norms, logs and local checkpoint hashes are retained in the evidence archive. The historical objective remains the canonical default; revised training is opt-in.', '']
    output.write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.evidence,args.output)
