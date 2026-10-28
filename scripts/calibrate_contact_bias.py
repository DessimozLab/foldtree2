"""Choose a contact-score bias on one development subset, assess on another."""
import argparse
import json
import math
from pathlib import Path

import torch

from foldtree2.se3_cache import open_bundle
from foldtree2.src.distance_restraints import PairwiseDistanceHead, distance_labels
from foldtree2.train_distance_restraints import contact_metrics, load_records
from scripts.compare_spherical_contact_screen import operating_point
from scripts.se3.diagnose_se3_reconstruction import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', default=['runs/contact_objective_large_baseline_20261026',
                                                     'runs/contact_objective_large_sphere64_20261026'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--target-recall', type=float, default=.2)
    parser.add_argument('--split-seed', type=int, default=31415)
    args = parser.parse_args()
    if not 0 < args.target_recall < 1:
        raise ValueError('Target recall must be in (0, 1)')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    roots = [Path(x) for x in args.roots]
    metadata = [json.loads((p / 'provenance.json').read_text()) for p in roots]
    for candidate in metadata[1:]:
        if metadata[0]['bundle_sha256'] != candidate['bundle_sha256'] or metadata[0]['selection'] != candidate['selection']:
            raise ValueError('Runs do not share the same data and split')
        for field in ('train_count', 'val_count', 'epochs', 'pairs', 'hidden', 'seed'):
            if metadata[0]['args'][field] != candidate['args'][field]:
                raise ValueError('Unmatched experiment configuration: ' + field)
    cache = metadata[0]['args']['cache']
    manifest = open_bundle(cache)
    records = load_records(cache, manifest, 'val', metadata[0]['args']['val_count'])
    order = torch.randperm(len(records), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    calibration_indices = set(order[:len(order) // 2])
    ids = {'calibration': [entry['id'] for i, (entry, _) in enumerate(records) if i in calibration_indices],
           'heldout': [entry['id'] for i, (entry, _) in enumerate(records) if i not in calibration_indices]}
    if set(ids['calibration']) & set(ids['heldout']):
        raise ValueError('Calibration and held-out structures overlap')
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for root, meta in zip(roots, metadata):
        names = list(json.loads((root / 'summary.json').read_text()))
        if len(names) != 1:
            raise ValueError('Each root must contain exactly one model case: ' + str(root))
        name = names[0]
        checkpoint = torch.load(root / name / 'last.pt', map_location=device, weights_only=False)
        variant = meta.get('variant', {}).get('model_class')
        model_class = PairwiseDistanceHead
        if variant == 'foldtree2.src.contact_cross_dot_perturbed.PerturbedCrossDotHead':
            from foldtree2.src.contact_cross_dot_perturbed import PerturbedCrossDotHead
            model_class = PerturbedCrossDotHead
        elif variant in ('foldtree2.src.contact_dot_experiments.SymmetricCrossDotHead',
                       'foldtree2.src.contact_dot_experiments.PairAdjustedDotHead'):
            from foldtree2.src.contact_dot_experiments import PairAdjustedDotHead, SymmetricCrossDotHead
            model_class = {'foldtree2.src.contact_dot_experiments.SymmetricCrossDotHead': SymmetricCrossDotHead,
                           'foldtree2.src.contact_dot_experiments.PairAdjustedDotHead': PairAdjustedDotHead}[variant]
        elif variant:
            raise ValueError('Unsupported calibration model class: ' + variant)
        model = model_class(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        groups = {k: {'prob': [], 'truth': [], 'length': 0} for k in ids}
        with torch.no_grad():
            for i, (entry, pairs) in enumerate(records):
                group = groups['calibration' if i in calibration_indices else 'heldout']
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in entry.items()}
                pairs = pairs.to(device)
                labels, _ = distance_labels(inputs['target'], pairs, model.edges)
                p = model.contact_probability(model(inputs, pairs).softmax(-1))
                group['prob'].append(p.cpu())
                group['truth'].append((labels < model.contact_bins).cpu())
                group['length'] += int(entry['mask'].sum())
        for group in groups.values():
            group['prob'], group['truth'] = torch.cat(group['prob']), torch.cat(group['truth'])
        chosen = operating_point(groups['calibration']['prob'], groups['calibration']['truth'], args.target_recall)
        threshold = chosen['threshold']
        if not 0 < threshold < 1:
            raise ValueError('Calibration returned a nonfinite logit threshold')
        held = groups['heldout']
        predicted = held['prob'] >= threshold
        positives = int(predicted.sum())
        correct = int((predicted & held['truth']).sum())
        original = contact_metrics(held['prob'], held['truth'], held['length'])
        label = name + '_' + meta['variant']['name'] if variant else name
        row = {'case': label, 'checkpoint_case': name, 'source': str(root), 'calibration': chosen,
               'logit_bias_shift': -math.log(threshold / (1 - threshold)),
               'heldout': {'pairs': len(held['truth']), 'contacts_true': int(held['truth'].sum()),
                           'predicted': positives, 'true_positives': correct,
                           'precision': correct / positives if positives else None,
                           'recall': correct / int(held['truth'].sum()) if int(held['truth'].sum()) else None,
                           'contact_average_precision': original['contact_average_precision'],
                           'predicted_at_original_0p5': original['contacts_predicted']}}
        results.append(row)
        print(json.dumps(row), flush=True)
    atomic_json(output / 'calibration.json', {'results': results, 'split_seed': args.split_seed,
        'target_recall': args.target_recall, 'ids': ids, 'bundle_sha256': manifest['bundle_sha256'],
        'scope': 'Chain-disjoint calibration and held-out halves of a reused development validation set; not an independent test'})
    lines = ['# Contact-bias calibration', '',
             f"Random chain split with seed {args.split_seed}: {len(ids['calibration'])} calibration and "
             f"{len(ids['heldout'])} held-out validation chains. Select each threshold for "
             f"{args.target_recall:.0%} calibration recall, then evaluate it on held-out chains.", '',
             '| Model | Calibration threshold | Global logit shift | Held-out precision | Held-out recall | Predicted contacts | Held-out AP |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in results:
        v = row['heldout']
        precision = '—' if v['precision'] is None else f"{v['precision']:.2%}"
        recall = '—' if v['recall'] is None else f"{v['recall']:.2%}"
        lines.append(f"| {row['case']} | {row['calibration']['threshold']:.4f} | {row['logit_bias_shift']:+.3f} | "
                     f"{precision} | {recall} | {v['predicted']} | {v['contact_average_precision']:.4f} |")
    lines += ['', 'The shift is equivalent to subtracting the selected threshold in log-odds space '
              'before applying a 0.5 cutoff. It does not change ranking or AP. The original '
              'development holdout has been reused in prior experiments and is not an independent test.', '']
    (output / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
