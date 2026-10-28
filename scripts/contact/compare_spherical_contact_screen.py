"""Compare saved contact heads at fixed recall on their shared development split."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from foldtree2.se3_cache import open_bundle
from foldtree2.src.distance_restraints import PairwiseDistanceHead, distance_labels
from foldtree2.train_distance_restraints import contact_metrics, load_records
from scripts.se3.diagnose_se3_reconstruction import atomic_json


def operating_point(probabilities, truth, target_recall):
    order = probabilities.argsort(descending=True, stable=True)
    cumulative = truth[order].long().cumsum(0)
    required = max(1, int(torch.ceil(torch.tensor(target_recall * int(truth.sum())))))
    index = int((cumulative >= required).nonzero()[0])
    threshold = probabilities[order[index]]
    predicted = probabilities >= threshold
    true_positives = int((predicted & truth).sum())
    return {'threshold': float(threshold), 'precision': true_positives / int(predicted.sum()),
            'recall': true_positives / int(truth.sum()), 'predicted': int(predicted.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='runs/contact_objective_screen_20261026')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Use only the baseline case from its reference screen')
    parser.add_argument('--screens', nargs='+', required=True)
    parser.add_argument('--case-name', help='Compare this case from each screen')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--plot', action='store_true', help='Save validation curves and a precision–recall plot')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    base = Path(args.baseline)
    reference = json.loads((base / 'provenance.json').read_text())
    roots = [Path(p) for p in args.screens]
    for root in roots:
        meta = json.loads((root / 'provenance.json').read_text())
        for field in ('cache', 'train_count', 'val_count', 'epochs', 'pairs', 'hidden', 'seed'):
            if meta['args'][field] != reference['args'][field]:
                raise ValueError('Unmatched experiment configuration: ' + field)
        if meta['bundle_sha256'] != reference['bundle_sha256'] or meta['selection'] != reference['selection']:
            raise ValueError('Experiments must use the same training and validation examples')
    manifest = open_bundle(reference['args']['cache'])
    records = load_records(reference['args']['cache'], manifest, 'val', reference['args']['val_count'])
    cases = [(base, 'baseline')]
    if not args.baseline_only:
        cases.append((base, 'sphere16'))
    for root in roots:
        names = list(json.loads((root / 'summary.json').read_text()))
        if args.case_name:
            if args.case_name not in names:
                raise ValueError('Requested case missing from ' + str(root))
            names = [args.case_name]
        cases.extend((root, name) for name in names)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    curves = []
    for root, name in cases:
        metadata = json.loads((root / 'provenance.json').read_text())
        model_class = PairwiseDistanceHead
        variant = metadata.get('variant', {}).get('model_class')
        if variant == 'foldtree2.src.contact_bias_experiment.ResidueBiasDistanceHead':
            from foldtree2.src.contact_bias_experiment import ResidueBiasDistanceHead
            model_class = ResidueBiasDistanceHead
        elif variant == 'foldtree2.src.contact_cross_dot_perturbed.PerturbedCrossDotHead':
            from foldtree2.src.contact_cross_dot_perturbed import PerturbedCrossDotHead
            model_class = PerturbedCrossDotHead
        elif variant in ('foldtree2.src.contact_dot_experiments.SymmetricCrossDotHead',
                         'foldtree2.src.contact_dot_experiments.PairAdjustedDotHead'):
            from foldtree2.src.contact_dot_experiments import PairAdjustedDotHead, SymmetricCrossDotHead
            model_class = {'foldtree2.src.contact_dot_experiments.SymmetricCrossDotHead': SymmetricCrossDotHead,
                           'foldtree2.src.contact_dot_experiments.PairAdjustedDotHead': PairAdjustedDotHead}[variant]
        elif variant:
            raise ValueError('Unsupported comparison model class: ' + variant)
        summary = json.loads((root / 'summary.json').read_text())[name]
        checkpoint = torch.load(root / name / 'last.pt', map_location=device, weights_only=False)
        model = model_class(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        all_probabilities, all_labels = [], []
        with torch.no_grad():
            for entry, pairs in records:
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in entry.items()}
                pairs = pairs.to(device)
                labels, _ = distance_labels(inputs['target'], pairs, model.edges)
                probs = model.contact_probability(model(inputs, pairs).softmax(-1))
                all_probabilities.append(probs.cpu())
                all_labels.append((labels < model.contact_bins).cpu())
        probabilities, truth = torch.cat(all_probabilities), torch.cat(all_labels)
        metrics = contact_metrics(probabilities, truth, sum(int(e['mask'].sum()) for e, _ in records))
        expected = summary['validation']['contact_average_precision']
        ap_delta = metrics['contact_average_precision'] - expected
        if abs(ap_delta) > 1e-5:
            raise RuntimeError(f'Recomputed contact AP differs from saved validation: '
                               f'{metrics["contact_average_precision"]} vs {expected}')
        suffix = metadata.get('variant', {}).get('name', 'residue_bias')
        label = name + '_' + suffix if variant else name
        row = {'case': label, 'source': str(root), 'dimension': summary['spherical_dim'],
               'contact_weight': summary['contact_weight'], 'validation': summary['validation'],
               'recomputed_ap_delta': ap_delta, 'max_validation_contact_probability': float(probabilities.max()),
               'at_10pct_recall': operating_point(probabilities, truth, .10),
               'at_20pct_recall': operating_point(probabilities, truth, .20)}
        per_chain = json.loads((root / name / 'final_validation.json').read_text())['per_chain']
        ap_by_chain = [chain['metrics']['contact_average_precision'] for chain in per_chain
                       if chain['metrics']['contact_average_precision'] is not None]
        row['macro_chain_contact_ap'] = statistics.mean(ap_by_chain) if ap_by_chain else None
        row['macro_ap_chains'] = len(ap_by_chain)
        if summary['spherical_dim']:
            scale = float(model.contact_log_scale.detach().clamp(-5., math.log(100.)).exp())
            offset = float(model.contact_offset.detach())
            row['sphere_scale'] = scale
            row['global_contact_bias'] = offset
            if not variant or variant.endswith(('.SymmetricCrossDotHead', '.PerturbedCrossDotHead')):
                row['theoretical_max_contact_probability'] = float(torch.sigmoid(torch.tensor(scale + offset)))
        results.append(row)
        if args.plot:
            order = probabilities.argsort(descending=True, stable=True)
            ranked_truth = truth[order].to(torch.int32)
            ranked_scores = probabilities[order]
            ends = torch.cat(((ranked_scores[:-1] != ranked_scores[1:]).nonzero().flatten(),
                              torch.tensor([len(ranked_scores) - 1])))
            found = ranked_truth.cumsum(0)[ends]
            # Plot only rank positions that add a true contact; preserve ties.
            increments = found - torch.cat((found.new_zeros(1), found[:-1]))
            selected = increments > 0
            recall = (found[selected] / truth.sum()).numpy()
            precision = (found[selected] / (ends[selected] + 1)).numpy()
            curves.append((label, recall, precision,
                           json.loads((root / name / 'history.json').read_text())))
        print(json.dumps(row), flush=True)
    atomic_json(output / 'comparison.json', {'results': results,
        'scope': 'Thresholds selected on this development split; descriptive, optimistic, not test-set performance',
        'val_ids': reference['selection']['val'], 'bundle_sha256': reference['bundle_sha256']})
    configuration = reference['args']
    lines = ['# Spherical contact head comparison', '',
             f"Fixed final epoch {configuration['epochs']}; seed {configuration['seed']}; "
             f"{configuration['train_count']} training and {configuration['val_count']} "
             'development-validation chains in each run.', '',
             '| Model | Weight | Contact AP | Macro-chain AP | Top-L precision | Precision @ 0.5 | Recall @ 0.5 | Precision at 10% recall | Precision at 20% recall | Distance MAE (Å) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in results:
        v = row['validation']
        precision = '—' if v['contact_precision'] is None else f"{v['contact_precision']:.1%}"
        macro_ap = '—' if row['macro_chain_contact_ap'] is None else f"{row['macro_chain_contact_ap']:.4f}"
        lines.append(f"| {row['case']} | {row['contact_weight']:g} | {v['contact_average_precision']:.4f} | {macro_ap} | "
                     f"{v['top_L_precision']:.1%} | {precision} | {v['contact_recall']:.1%} | "
                     f"{row['at_10pct_recall']['precision']:.1%} | {row['at_20pct_recall']['precision']:.1%} | "
                     f"{v['capped_distance_mae_angstrom']:.2f} |")
    train_lengths = sorted(manifest['files'][i]['length'] for i in reference['selection']['train'])
    val_lengths = sorted(manifest['files'][i]['length'] for i in reference['selection']['val'])
    lines += ['', f"Training chain lengths: {train_lengths[0]}–{train_lengths[-1]} residues "
              f"(median {statistics.median(train_lengths):g}); validation: "
              f"{val_lengths[0]}–{val_lengths[-1]} (median {statistics.median(val_lengths):g}).", '',
              'Macro-chain AP averages AP across chains containing at least one contact. '
              'Contact AP and top-L precision are pooled over eligible pairs across chains.', '',
              'Thresholds for the fixed-recall columns are selected on the same validation split; '
              'they describe ranking and are not independently calibrated test results.',
              'The holdout is a reused development set, and production-pretraining independence is unverified.', '']
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.7))
        for name, recall, precision, history in curves:
            epochs = [r['epoch'] for r in history]
            line, = axes[0].plot(epochs, [r['validation']['contact_average_precision'] for r in history], label=name)
            axes[1].plot(epochs, [r['validation']['capped_distance_mae_angstrom'] for r in history], label=name)
            axes[2].plot(epochs, [r['distance_ce'] for r in history], color=line.get_color(), label=name + ' train')
            axes[2].plot(epochs, [r['validation']['macro_chain_nll'] for r in history], color=line.get_color(),
                         linestyle='--', label=name + ' validation')
            axes[3].plot(recall, precision, label=name)
        axes[0].set(xlabel='Epoch', ylabel='Contact AP', title='Contact ranking')
        axes[1].set(xlabel='Epoch', ylabel='Capped MAE (Å)', title='Distance accuracy')
        axes[2].set(xlabel='Epoch', ylabel='Per-chain distance NLL', title='Training and validation')
        axes[3].set(xlabel='Recall', ylabel='Precision', title='Precision–recall', xlim=(0, 1), ylim=(0, 1))
        for ax in axes:
            ax.grid(alpha=.2)
        axes[2].legend(fontsize=7)
        axes[3].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / 'comparison.png', dpi=180)
        plt.close(fig)
        lines += ['![Validation curves](comparison.png)', '']
    (output / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
