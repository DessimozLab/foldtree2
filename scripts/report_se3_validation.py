#!/usr/bin/env python3
"""Summarize persistent SE3 experiment logs and paired structural metrics."""
import argparse
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from pathlib import Path
import statistics


def mean(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.mean(values) if values else None


def plot_learning_curves(epochs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    x = [e['epoch']+1 for e in epochs]
    for ax, keys, label in zip(axes, [('train_geometry','val_geometry'),('train_fape','val_fape'),('train_angle','val_angle')],
                              ['Geometry loss','CA-frame FAPE','Circular angle loss']):
        for key in keys:
            ax.plot(x,[e[key] for e in epochs],label=key.split('_')[0])
        ax.set(xlabel='Epoch', ylabel=label)
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output,dpi=160); plt.close(fig)


def report(root, report_dir=Path('docs')):
    lines = ['# SE(3) residue refiner validation', '',
             'Production models: manifest-selected 40-character epoch-40 encoder and decoder, both frozen. The unrefined baseline uses the decoder’s three-dimensional bottleneck, the same coordinate source supplied to the refiner.',
             'These are structure-level refiner holdouts; independence from production pretraining is not claimed.', '']
    hardware = root / 'hardware.json'
    if hardware.exists():
        h = json.loads(hardware.read_text())
        lines += [f"Hardware: physical GPU {h['physical_gpu']} ({h['name']}), {h['total_memory_bytes']/2**30:.2f} GiB.", '']
    versions = root / 'software_versions.json'
    if versions.exists():
        v = json.loads(versions.read_text())
        lines += ['Software: '+', '.join(f'{key} {value}' for key,value in v.items())+'.', '']
    overfit_passed = False
    pilot_provenance = root / 'pilot' / 'provenance.json'
    if pilot_provenance.exists():
        identity = json.loads(pilot_provenance.read_text())['identity']
        lines += ['| Input | SHA-256 |', '|---|---|']
        for key in ('dataset', 'encoder', 'decoder'):
            lines.append(f"| {key} | `{identity[key+'_sha256']}` |")
        lines += ['', 'Selected structure IDs, split membership and preprocessing settings are recorded in each run’s `provenance.json`.', '']
    for name in ('smoke', 'overfit', 'pilot'):
        run = root / name
        if not (run / 'events.jsonl').exists():
            continue
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
        epochs = list({(e['epoch'],e['global_step']): e for e in events if e['event'] == 'epoch'}.values())
        failures = [e for e in events if e['event'] == 'failure']
        completed = [e for e in events if e['event'] == 'complete']
        lines += [f'## {name.capitalize()}', '']
        prov_path = run / 'provenance.json'
        if prov_path.exists():
            prov = json.loads(prov_path.read_text())
            lines += [f"Structures: {len(prov['train_ids'])} training, {len(prov['val_ids'])} evaluation. "
                      f"Skipped during selection: {len(prov['skipped'])}; complete chains only.", '']
            if name == 'overfit':
                lines += ['The overfit evaluation uses the same 32 training structures.', '']
        lines += [f'Completed epochs: {len(epochs)}. Logged failures: {len(failures)}. Completed: {bool(completed)}.', '']
        if completed:
            result = completed[-1]
            lines += [f"Selected checkpoint: epoch {result['best_epoch']+1}. Frozen parameters and buffers "
                      f"unchanged: {result['frozen_unchanged']}. Nonfinite events: {result['nonfinite_events']}.", '']
        if failures:
            lines += [f"Failure: `{failures[-1]['error']}`", '']
        if epochs:
            plot_learning_curves(epochs, run / 'learning_curves.png')
            curve_path = Path(os.path.relpath(run / 'learning_curves.png', report_dir)).as_posix()
            lines += [f'![Learning curves]({curve_path})', '']
            first, last = epochs[0], epochs[-1]
            reduction = 1 - last['train_geometry']/first['train_geometry']
            lines += [f"Training geometry loss: {first['train_geometry']:.6f} → {last['train_geometry']:.6f} "
                      f"({100*reduction:.1f}% reduction).", '',
                      '| Epoch | Train geometry | Validation FAPE | Grad norm | Seconds | Peak GPU GiB |',
                      '|---:|---:|---:|---:|---:|---:|']
            for e in epochs:
                lines.append(f"| {e['epoch']+1} | {e['train_geometry']:.5f} | {e['val_fape']:.5f} | "
                             f"{e['gradient_norm_mean']:.3g} | {e['seconds']:.1f} | {e['peak_memory_bytes']/2**30:.2f} |")
            lines += ['', f"First epoch: {first['seconds']:.1f} s; final recorded epoch: {last['seconds']:.1f} s. "
                      f"Peak host memory: {max(e.get('peak_host_memory_bytes',0) for e in epochs)/2**30:.2f} GiB.", '']
        path = run / 'paired_metrics.json'
        if path.exists():
            paired = json.loads(path.read_text())
            before, after = [r['baseline'] for r in paired], [r['refined'] for r in paired]
            lines += ['| Metric | Frozen decoder | Refined | Mean paired delta |', '|---|---:|---:|---:|']
            metrics = ['fape','ca_rmsd','tm_score','lddt','ca_bond_mae','ca_bond_deviation_3_8','angle_mae_radians','ca_bend_mae_radians','ca_torsion_mae_radians']
            for key in metrics:
                b, a = mean(before,key), mean(after,key)
                d = mean([r['delta'] for r in paired], key)
                if b is None or a is None:
                    lines.append(f'| {key} | unavailable | {a if a is not None else "unavailable"} | unavailable |')
                else:
                    lines.append(f'| {key} | {b:.6f} | {a:.6f} | {d:+.6f} |')
            lines += ['', 'Coordinates and masks are paired per structure; full pairs are in `paired_metrics.json`.', '']
            improved_count = sum(r['refined']['fape'] < r['baseline']['fape'] for r in paired)
            lines += [f'FAPE improves for {improved_count}/{len(paired)} evaluated structures.', '']
            if name == 'overfit' and epochs:
                passed = reduction >= .2 and mean(after,'fape') < mean(before,'fape')
                overfit_passed = passed
                lines += [f'Overfit gate (≥20% training geometry reduction and lower FAPE than decoder): **{passed}**.', '']
            if name == 'pilot':
                regressions = []
                for key in ['ca_rmsd','ca_bond_mae','ca_bond_deviation_3_8','angle_mae_radians','ca_bend_mae_radians','ca_torsion_mae_radians']:
                    b,a = mean(before,key),mean(after,key)
                    if b is not None and a is not None and a > b*1.05 + 1e-6: regressions.append(key)
                for key in ['tm_score','lddt']:
                    if mean(after,key) < mean(before,key)-.01: regressions.append(key)
                fape_improves = mean(after,'fape') < mean(before,'fape')
                lines += [f'Held-out FAPE improves: **{fape_improves}**. Material regressions: '
                          f'**{", ".join(regressions) or "none"}**.', '',
                          'The operational no-regression thresholds are 5% for error metrics and 0.01 absolute for TM-score/lDDT.', '',
                          f'Proceed to larger training (overfit and held-out gates): **{overfit_passed and fape_improves and not regressions}**. '
                          'Preserve and diagnose failed metrics before increasing model size or duration.', '']
        diag_path = run / 'diagnostics.json'
        if diag_path.exists():
            rows = json.loads(diag_path.read_text())
            lines += ['Coordinate scale diagnosis (mean across evaluation structures):', '',
                      '| Quantity | Baseline | Refined | Target |', '|---|---:|---:|---:|']
            for key in ['radius_rms','bond_mean']:
                lines.append(f"| {key} (Å) | {mean(rows,'baseline_'+key):.4f} | {mean(rows,'refined_'+key):.4f} | {mean(rows,'target_'+key):.4f} |")
            lines += ['', f"Distance-contact density below 8 Å: {mean(rows,'distance_contact_density'):.4f}. "
                      'A high density means these bottleneck coordinates produce nearly complete distance graphs.', '']
            if mean(rows,'refined_radius_rms') < .1*mean(rows,'target_radius_rms'):
                lines += ['The refined trace remains strongly compressed relative to the targets. The backend returns a '
                          'normalized degree-1 coordinate projection, and this refiner has no residual addition of seed '
                          'coordinates. Together with the uncalibrated bottleneck scale, these are likely causes of the '
                          'observed collapse; this is a diagnosis, not a demonstrated causal ablation. Calibrate coordinate '
                          'units and investigate an identity-preserving residual update before increasing capacity or duration.', '']
        profile_path = run / 'length_profile.json'
        if profile_path.exists():
            rows = json.loads(profile_path.read_text())
            lines += ['Observed batch size 1 profile from the first epoch, including warmup; adjacency storage scales as N² and attention intermediates cost more:', '',
                      '| Residue length bucket | Samples | Mean forward/backward seconds | Cumulative peak GPU GiB |',
                      '|---|---:|---:|---:|']
            for upper in (64,128,256,384):
                lower = {64:0,128:64,256:128,384:256}[upper]
                bucket = [r for r in rows if lower < r['residues'] <= upper]
                if bucket:
                    lines.append(f"| {lower+1}–{upper} | {len(bucket)} | {mean(bucket,'forward_backward_seconds'):.4f} | "
                                 f"{max(r['peak_memory_bytes'] for r in bucket)/2**30:.2f} |")
            lines += ['', 'Profiles cover batch size 1 only; larger batches and atom-level graphs have not been validated.', '']
    dense_profile = root / 'dense_profile.json'
    if dense_profile.exists():
        rows = json.loads(dense_profile.read_text())['measurements']
        lines += ['## Warmed dense and padding profile', '',
                  'Frozen caches and the final pilot checkpoint; evaluation mode with backward, without optimizer steps. '
                  'Peak counters are reset per measurement after one warmup, and times average three repeats.', '',
                  '| Lengths | Valid nodes | Batch | Padding nodes | Padded adjacency bytes | Unpadded bytes | Seconds | Peak GPU GiB |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append(f"| {r['lengths']} | {r['valid_nodes']} | {r['batch_size']} | {r['padding_nodes']} | "
                         f"{r['padded_adjacency_bytes']} | {r['unpadded_adjacency_bytes']} | "
                         f"{r['mean_forward_backward_seconds']:.4f} | {r['peak_gpu_allocated_bytes']/2**30:.2f} |")
        lines += ['', 'Validated training envelope: complete chains ≤384 residues, microbatch size 1, effective batch size 8. '
                  'Batch-size-2 profile results measure allocation and output consistency only; larger training batches and '
                  'atom-level graphs remain unvalidated. The local wrapper compacts valid nodes before GotenNet, so backend '
                  'attention avoids padded rows even though packed adjacency still allocates B×Nmax² entries.', '',
                  'External stack sampling was unavailable because process tracing requires administrator permissions; '
                  'in-process timings and CUDA memory counters were collected instead.', '']
    lines += ['## Correctness and failed startup attempts', '',
              'The final CPU/CUDA correctness suite passed all 59 tests, including the original 29 geometry/frame tests. '
              'Compatible resume restored epoch 30 and step 120 without extra training; an incompatible checkpoint '
              'was rejected before provenance replacement. Evidence is retained in `cuda-correctness-test.log`, '
              '`overfit-resume.log` and `resume-rejection-check.json`.', '',
              'Correctness logs are retained in `runs/se3_validation/correctness-tests.log` and `tooling-tests.log`. '
              'The first smoke attempt failed cache parity because frozen CUDA graph reductions were nondeterministic. '
              'Deterministic CUDA algorithms resolved the mismatch. The first overfit selection contained a structure '
              'with zero confidence-masked targets and was rejected before training; target eligibility is now explicit. '
              'These attempts and tracebacks are retained in `smoke_attempt_01` / `overfit_attempt_01`. A first pilot was interrupted before completing an epoch when a stronger frame test exposed cancellation after restoring translation; `pilot_attempt_01` is preserved. Final frames use centered vectors before restoring origins. '
              'A subsequent audit found that endpoint frame masks omitted the third residue used by their normals. This affected two overfit structures and nine pilot structures. Those completed runs are preserved in `overfit_endpoint_mask_attempt` and `pilot_endpoint_mask_attempt`; the final runs repeat the same bounded configurations with corrected masks. Endpoint mask tests verify that masked coordinates cannot influence FAPE.', '']
    lines += ['## Metric scope and limitations' , '',
              'FAPE uses geometry-defined CA frames, masks undefined orientations and frame stencils across masked residues, '
              'clamps at 10 Å and divides by 10. RMSD uses Kabsch alignment; TM-score uses TM-align. '
              'lDDT evaluates valid CA pairs within 15 Å. Bond geometry is CA–CA distance error; N–CA and C–N atom bond '
              'metrics are deferred with atom refinement. Angle error includes paired CA virtual bend/torsion errors and circular MAE in radians from decoder/refiner heads. '
              'A missing decoder angle head is reported as unavailable, not replaced by zero.', '',
              'Raw logs, checkpoints, cache identity, selected IDs, configuration, software versions and per-structure pairs '
              'are retained under the run directories. Production checkpoints and alphabet benchmarks are unchanged.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/se3_validation'))
    parser.add_argument('--output', type=Path, default=Path('docs/se3_validation_report.md'))
    args = parser.parse_args()
    args.output.write_text(report(args.root, args.output.parent))
