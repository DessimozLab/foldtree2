"""Render a completed bounded 3D-seed learning experiment from retained evidence."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')


def report(root, output):
    load = lambda name: json.loads((root / name).read_text())
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    if events[-1]['event'] != 'complete':
        raise ValueError('Experiment must complete before reporting')
    analysis, full, pairs = load('analysis.json'), load('overfit.json'), load('heldout_pairs.json')
    assert len(analysis['train_ids']) == 32 and len(pairs) == 128
    assert not set(analysis['train_ids']) & set(analysis['heldout_ids'])
    assert {r['id'] for r in pairs} == set(analysis['heldout_ids'])
    assert full['spec']['angles'] == 'sincos' and full['spec']['lift_fraction'] > 0
    keys = ['fape', 'ca_rmsd', 'tm_score', 'lddt', 'ca_bond_mae',
            'ca_bend_mae_radians', 'ca_torsion_mae_radians']
    higher = {'tm_score', 'lddt'}
    summary = {}
    for key in keys:
        baseline = [r['input_seed'][key] for r in pairs]
        refined = [r['refined'][key] for r in pairs]
        if not all(math.isfinite(v) for v in baseline + refined):
            raise ValueError('Nonfinite paired metric')
        before, after = statistics.mean(baseline), statistics.mean(refined)
        delta = [b - a for a, b in zip(baseline, refined)]
        summary[key] = {'baseline': before, 'refined': after, 'mean_delta': after - before,
                        'median_delta': statistics.median(delta),
                        'improved': sum(d > 0 if key in higher else d < 0 for d in delta),
                        'material_regression': after < before - .01 if key in higher else after > before * 1.05}
    gates = {'overfit_20_percent': full['reduction'] >= .2,
             'heldout_fape': summary['fape']['refined'] < summary['fape']['baseline'],
             'no_material_regression': not any(r['material_regression'] for r in summary.values())}
    result = {'gates': gates, 'eligible_for_revised_pilot': all(gates.values()),
              'ready_for_larger_run': False, 'metrics': summary,
              'reduction': full['reduction'], 'failed_events': sum(e['event'] == 'failure' for e in events)}
    (root / 'gate_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, key in zip(axes, ['eval_geometry', 'eval_fape']):
        ax.plot([r['epoch'] + 1 for r in full['history']], [r[key] for r in full['history']])
        ax.set(xlabel='Epoch', ylabel=key); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(root / 'learning_curves.png', dpi=160); plt.close(fig)
    lines = ['# Bounded 3D-seed learning test, 2026-10-09', '',
             f"Completed 32 structures × {len(full['history'])} epochs ({full['history'][-1]['step']} optimizer updates), then 128 disjoint evaluation structures. Device: `{analysis['device']}`. GPU 1 remained occupied by an unrelated workload.", '',
             'The candidate was predeclared as lifted circular: fraction-0.1 oriented normal lift, full-range sin/cos angles, zero-initialized residual, RMS normalization and residual gain 32. Eight-structure control screens did not select the full-run candidate. Checkpoint selection used training geometry only.', '',
             'Width 32, depth 2, two heads, head dimension 16; float32, seed 42 reset per case, batch 1/effective 8, AdamW 5e-4, weight decay 0.01, clipping 1, geometry-driven scheduler. Calibration used only the 32 training chains. Frozen epoch-40 production caches were reused without model calls or graph regeneration.', '',
             f"Training geometry: **{full['initial_geometry']:.6f} → {full['best_geometry']:.6f}**, a **{full['reduction']*100:.2f}%** reduction from the lifted model's own initialized baseline. Selected epoch {full['best_epoch']+1}.", '',
             f"![Learning curves]({os.path.relpath(root / 'learning_curves.png', output.parent)})", '',
             '| Metric | Lifted input | Refined | Mean paired delta | Improved / 128 |',
             '|---|---:|---:|---:|---:|']
    for key, r in summary.items():
        lines.append(f"| {key} | {r['baseline']:.6f} | {r['refined']:.6f} | {r['mean_delta']:+.6f} | {r['improved']} |")
    lines += ['', 'Comparisons use identical structures and masks. lDDT is CA-only with explicit diagonal exclusion; TM-score uses TM-align. These are refiner holdouts, with no claim of independence from production pretraining. Per-structure raw, calibrated, lifted-input and refined measurements are retained; calibration and seed-lift gains do not count as learning.', '',
              f"Gates: ≥20% training geometry reduction **{gates['overfit_20_percent']}**; improved held-out FAPE **{gates['heldout_fape']}**; no material structural regression **{gates['no_material_regression']}**. Material regression means >5% increase in error or >0.01 decrease in TM-score/lDDT. These are pre-existing screening thresholds, without statistical uncertainty claims.", '',
              f"Ready for a larger run: **{result['ready_for_larger_run']}**. A revised 512/128 pilot is still required before larger-data training, even if this bounded test passes.", '',
              f"Full-training mean epoch time {statistics.mean(r['seconds'] for r in full['history']):.2f} s; peak allocated GPU memory {max(r['peak_memory_bytes'] for r in full['history'])/1024**3:.3f} GiB. Failure events: {result['failed_events']}. Gradients and losses were explicitly checked for finiteness at every training step.", '',
              'See the retained evidence for source snapshots, checkpoint hashes, software versions, split IDs, configuration, controls, paired metrics and correctness logs. Production weights, the staged transformer and historical reports were unchanged.', '']
    lines += [f"Refined angle-head mean circular MAE: {statistics.mean(r['refined']['angle_mae_radians'] for r in pairs):.6f} radians. The frozen input has no angle-head prediction, so its head MAE is undefined; CA-derived bend/torsion errors are compared above.", '']
    if (root / 'loss_audit.json').exists():
        audit = load('loss_audit.json')
        mean = lambda label, key: statistics.mean(r[label][key] for r in audit)
        lines += ['## FAPE and loss-path audit', '',
                  'Frame projection agrees exactly with an explicit row-vector reference on all 32 structures. Tests additionally cover the shared coarse-backbone FAPE reference, independent rigid transforms and double-precision finite-difference gradcheck. Coordinates are evaluated in restored physical units; FAPE clamps at 10 Å and divides by 10. There is no detach between predicted coordinates and this loss.', '',
                  'A masking bug was reproduced: masked NaN predictions could yield finite FAPE but nonfinite valid gradients. Predicted coordinates are now masked before constructing frames, and nonfinite valid coordinates are rejected. The correction passed regression tests. It was applied after this run; executed source snapshots preserve the original implementation. This run had finite masked model outputs and finite gradients, so that bug does not explain its regression.', '',
                  f"Mean clamped frame–point fraction: {100*mean('initial','clamped_pair_fraction'):.2f}% initially, {100*mean('trained','clamped_pair_fraction'):.2f}% trained. Unclamped mean error: {mean('initial','unclamped_mean_error_angstrom'):.3f} → {mean('trained','unclamped_mean_error_angstrom'):.3f} Å. FAPE itself increases {mean('initial','fape'):.6f} → {mean('trained','fape'):.6f}, while normalized all-pairs distance loss drops {mean('initial','distance_loss'):.6f} → {mean('trained','distance_loss'):.6f}.", '',
                  f"Mean coordinate-gradient norms (FAPE / distance): {mean('initial','fape_coordinate_gradient_norm'):.6f} / {mean('initial','distance_coordinate_gradient_norm'):.6f} initially and {mean('trained','fape_coordinate_gradient_norm'):.6f} / {mean('trained','distance_coordinate_gradient_norm'):.6f} trained. Mean gradient cosine is {mean('initial','gradient_cosine'):.3f} initially and {mean('trained','gradient_cosine'):.3f} trained. Saturation does not mean the entire FAPE gradient vanishes; the remaining signal is concentrated in unsaturated pairs and sensitive local frames.", '',
                  'The actual objective is all-pairs SmoothL1 distance / 10 + clamped CA-frame FAPE + 0.1 circular angle loss. It has no separately weighted backbone bond constraint. The angle head is an auxiliary scalar prediction; it does not reconstruct the coordinate output. Improved composite geometry loss therefore does not guarantee improved FAPE or chemically plausible backbone bonds.', '',
                  'Next diagnostic: a separately versioned objective ablation with an explicit local bond term and a less saturated FAPE training term, retaining the existing clamped FAPE evaluation metric and identical splits. Verify gradients and relative weighting before another bounded learning test. No loss-weight change or larger training was silently applied here.', '']
    output.write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.evidence, args.output)
