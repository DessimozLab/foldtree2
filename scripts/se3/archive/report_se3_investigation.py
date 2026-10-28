"""Render the completed scale/parameterization investigation without model calls."""
import argparse
import json
import os
from pathlib import Path
import statistics
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')


def load(path):return json.loads(path.read_text())
def avg(rows,key):return statistics.mean(row[key] for row in rows)


def report(basic,parameterization,output):
    events=[json.loads(line) for line in (parameterization/'events.jsonl').read_text().splitlines()]
    if not events or events[-1]['event']!='complete':raise RuntimeError('Parameterization experiment is not complete')
    analysis=load(parameterization/'analysis.json');screen=load(parameterization/'screening.json')
    full=load(parameterization/'overfit.json');pairs=load(parameterization/'heldout_pairs.json')
    planar=load(parameterization/'planarity.json');controls=load(parameterization/'planarity_control.json')
    prototype=load(parameterization/'lift_prototype.json')
    lines=['# SE(3) refiner investigation, 2026-10-09','',
        'Status: investigation complete; larger training remains deferred. All experiments used CPU while both GPUs were occupied.',
        'Production checkpoints and the mk2 HDF5 dataset were unchanged. Frozen caches supplied the inputs; no production model forward calls or graph regeneration were needed.',
        'The original validation protocol and its historical metrics remain separate from this investigation.', '',
        '## Coordinate units and seed preservation','',
        f"A single scale factor **{analysis['calibration_scale']:.6f}** was fitted from only the original 32 training structures using equal-structure pair-distance least squares. None of the 128 evaluation structures contributed to calibration or model selection.", '',
        '| Seed on 32 training structures | FAPE | CA RMSD Å | TM-score | CA lDDT | Bond MAE Å |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('raw','calibrated'):
        rows=[r['metrics'] for r in analysis['static'] if r['seed']==name]
        lines.append(f"| {name} | "+' | '.join(f'{avg(rows,k):.6f}' for k in ('fape','ca_rmsd','tm_score','lddt','ca_bond_mae'))+' |')
    lines+=['', 'Calibration improves geometry without optimizing the refiner. Those gains must not count toward the learning gate. A zero-initialized residual preserves this calibrated seed; absolute projected coordinates discard its scale.', '',
        '## Controlled screens','',
        'All screens used the same eight shortest training structures for 10 epochs (10 optimizer updates), width 32, depth 2, two heads, head dimension 16, AdamW 5e-4, effective batch 8 and clipping 1. These are diagnostic screens, not independent validation.', '',
        '| Case | Initial geometry | Best geometry | Reduction | Initial angle loss | Final angle loss |',
        '|---|---:|---:|---:|---:|---:|']
    for r in load(basic/'screening.json')+screen:
        lines.append(f"| {r['spec']['name']} | {r['initial_geometry']:.6f} | {r['best_geometry']:.6f} | {100*r['reduction']:.3f}% | {avg(r['initial'],'angle'):.6f} | {avg(r['final'],'angle'):.6f} |")
    lines+=['', 'The first five cases used a FAPE-driven scheduler. The last three used a geometry-driven scheduler. RMS normalization alone barely helped. Multiplying the normalized vector update by 32 produced a measurable learning signal. Gain 32 compensates for the small initial per-channel degree-1 norms; it is an experimental parameterization, not an established universal constant.', '',
        'Measured degree-1 component RMS was 0.01804 (approximately 1/(32√3)). Geometry projection gradients were nonzero, 0.040–0.143 on four representative structures. The original short-screen scheduler reduced learning rate from 5e-4 to 5e-6 because clamped FAPE barely moved.', '',
        '## Bounded 32-structure learning test','',
        f"The selected `{full['spec']['name']}` case was reinitialized and trained for {len(full['history'])} epochs / {full['history'][-1]['step']} updates on all 32 original structures. Selection used training geometry only. The circular head was not selected by this small screen; this is not evidence that the bounded head is adequate.", '',
        f"Geometry: **{full['initial_geometry']:.6f} → {full['best_geometry']:.6f}**, a **{100*full['reduction']:.2f}%** reduction from the calibrated, identity-initialized model. Selected epoch: {full['best_epoch']+1}.", '',
        f"Training FAPE: {avg(full['initial'],'fape'):.6f} → {avg(full['final'],'fape'):.6f}. The ≥20% geometry learning requirement is **{full['reduction']>=.2}**.", '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.2));history=full['history'];x=[r['epoch']+1 for r in history]
    for ax,key,title in zip(axes,['eval_geometry','eval_fape'],['Training evaluation geometry','Training evaluation FAPE']):
        ax.plot(x,[r[key] for r in history]);ax.set(xlabel='Epoch',ylabel=title);ax.grid(alpha=.2)
    fig.tight_layout();image_path=parameterization/'learning_curves.png';fig.savefig(image_path,dpi=160);plt.close(fig)
    lines += [f"![Learning curves]({Path(os.path.relpath(image_path,output.parent)).as_posix()})", '',
        '## Paired evaluation on 128 disjoint structures','',
        'This evaluates a refiner trained on 32 structures; it is not a repeat of the 512-structure pilot. These structures are refiner holdouts, with no claim of independence from production pretraining. No uncertainty intervals were computed.', '',
        '| Metric | Raw seed | Calibrated seed | Refined | Refined − calibrated |','|---|---:|---:|---:|---:|']
    for key in ('fape','ca_rmsd','tm_score','lddt','ca_bond_mae','ca_bend_mae_radians','ca_torsion_mae_radians'):
        values=[statistics.mean(r[label][key] for r in pairs) for label in ('raw','calibrated','refined')]
        lines.append(f"| {key} | "+' | '.join(f'{v:.6f}' for v in values)+f' | {values[2]-values[1]:+.6f} |')
    baseline_fape=statistics.mean(r['calibrated']['fape'] for r in pairs)
    refined_fape=statistics.mean(r['refined']['fape'] for r in pairs)
    lines+=['',f'Held-out FAPE improves over the calibrated input: **{refined_fape<baseline_fape}**. The learned residual regresses on this metric despite some distance-based improvements.', '']
    lines+=['', '## Fundamental planarity constraint','',
        f"Every training seed is essentially planar: maximum smallest/largest singular-value ratio **{max(r['seed_smallest_largest_ratio'] for r in planar):.3g}**. The production decoder has `residual=False` and a final three-channel LayerNorm. Its affine output satisfies Σᵢ(xᵢ−βᵢ)/γᵢ=0, a plane constraint. These are latent contact features, not a physical three-dimensional coordinate head.", '',
        f"GotenNet builds degree-1 vectors from relative coordinates and scalar-weighted vector combinations. Such updates preserve an exactly planar input. Trained output checks remained within **{max(r['max_distance_from_seed_plane'] for r in controls['predictions']):.6f} Å** of the input plane. Small departures reflect numerical noise. Even an optimal planar approximation of these targets has a mean aligned RMSD lower bound of **{avg(planar,'best_planar_target_rmsd_bound'):.3f} Å**.", '',
        'A prototype uses an invariant latent scalar and the oriented cross-product normal of neighboring seed bonds to lift the coordinates. It uses no target coordinates, random noise or fixed global axis. It is SO(3)-equivariant, deliberately not reflection-equivariant, and masks undefined or incomplete normal stencils.', '',
        f"At fixed lift fraction 0.1, all 32 prototype seeds became full rank (minimum singular-value ratio **{min(r['smallest_largest_ratio'] for r in prototype):.3f}**). Prototype seed FAPE/TM-score/bond MAE were **{avg([r['metrics'] for r in prototype],'fape'):.6f} / {avg([r['metrics'] for r in prototype],'tm_score'):.6f} / {avg([r['metrics'] for r in prototype],'ca_bond_mae'):.3f} Å**. These are untrained seed measurements; the lift has not passed a refiner training gate.", '',
        '## Angle range and metric corrections','',
        f"The legacy tanh angle head is limited to ±1 radian. **{100*analysis['angles_outside_head_range']:.1f}%** of valid training targets are outside that interval; the theoretical mean circular loss floor is **{analysis['angle_loss_floor']:.6f}**. An opt-in sin/cos head covers the full circle and rejects undefined zero pairs. Its initialization, gradients, representable range and invariance were tested.", '',
        'Float32 matrix-product distances can suffer translation cancellation and nonzero self-distances. Pair-distance losses and metrics now center coordinates; lDDT explicitly excludes the diagonal. Historical v2 lDDT figures are retained as originally measured, and should not be compared directly with the corrected figures above.', '',
        '## Remaining gate and reproducibility','',
        'A larger run is still premature. A future bounded test must train from a full-dimensional seed or add a tested SO(3) normal-direction update, use a physically valid angle head, and demonstrate the 20% overfit learning gate plus paired held-out quality. Increasing duration or width cannot remove an exactly planar geometric constraint.', '',
        'All new model modes are opt-in; default absolute coordinates and the bounded angle head remain available for historical comparison. The staged transformer and production model files were not changed. The first metric-control attempt was preserved, and the preliminary 32-structure gain-1 run was stopped after six epochs once its gradient/scale limitations were measured.', '',
        'The evidence directories preserve source snapshots, configurations, IDs, calibration, screens, paired metrics, gradient diagnostics, planarity checks, learning curves, logs and checkpoint inventory. CPU correctness checks passed; CUDA-specific checks were deferred while the GPUs were occupied.', '']
    resource_path=parameterization/'resources.json'
    if resource_path.exists():
        resource=load(resource_path)
        peak_kib=float(resource['VmHWM'].split()[0])
        lines += [f"CPU resource measurements: two configured Torch threads, peak observed resident memory **{peak_kib/1024**2:.2f} GiB**, full-training epoch time **{avg(full['history'],'seconds'):.1f} seconds** on average. No GPU memory was allocated by the investigation entrypoint.", '']
    lines += ['See the [evidence index](se3_investigation_evidence/README.md) for retained files and source identities.', '']
    output.write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basic',type=Path,default=Path('runs/se3_investigation_20261009'))
    parser.add_argument('--parameterization',type=Path,default=Path('runs/se3_parameterization_20261009'))
    parser.add_argument('--output',type=Path,default=Path('docs/se3_investigation_20261009.md'))
    args=parser.parse_args();report(args.basic,args.parameterization,args.output)
