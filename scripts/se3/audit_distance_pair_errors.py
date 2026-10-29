"""Read-only error decomposition of exported distance heads and coordinate fits."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from foldtree2.se3_cache import FrozenDataset, digest, open_bundle
from foldtree2.src.distance.distance_restraints import distance_labels
from foldtree2.train_distance_restraints import contact_metrics
from scripts.se3.diagnose_se3_reconstruction import atomic_json
from scripts.se3.reconstruct_distance_restraints import reconstruction_objective, validate_export


def describe(probabilities, distances, labels, edges):
    if not len(distances):
        return {'pairs': 0}
    lower = torch.cat([edges.new_zeros(1), edges])
    upper = torch.cat([edges, edges[-1:]])
    prediction = probabilities @ ((lower + upper) / 2)
    contact = probabilities[:, :int((edges <= 8.).sum())].sum(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    return {
        'pairs': len(distances), 'true_distance_mean': float(distances.mean()),
        'capped_distance_mean': float(prediction.mean()),
        'capped_signed_error': float((prediction - distances.clamp_max(float(edges[-1]))).mean()),
        'capped_mae': float((prediction - distances.clamp_max(float(edges[-1]))).abs().mean()),
        'nll': float(-probabilities.gather(1, labels[:, None]).clamp_min(1e-30).log().mean()),
        'true_bin_probability': float(probabilities.gather(1, labels[:, None]).mean()),
        'contact_probability': float(contact.mean()),
        'contact_frequency': float((distances < 8.).float().mean()),
        'entropy': float(entropy.mean()), 'max_bin_probability': float(probabilities.max(-1).values.mean()),
    }


def restraint_preferences(probabilities, distances, labels, edges):
    grid = torch.arange(2., 65., .5)
    lower = torch.cat([torch.zeros(len(grid), 1),
                       F.logsigmoid(grid[:, None] - edges)], dim=-1)
    upper = torch.cat([F.logsigmoid(edges - grid[:, None]),
                       torch.zeros(len(grid), 1)], dim=-1)
    kernel = (lower + upper).exp()
    scores = probabilities @ kernel.T
    preferred = grid[scores.argmax(-1)]
    native = distances.detach().clone().requires_grad_()
    lo = torch.cat([torch.zeros(len(native), 1),
                    F.logsigmoid(native[:, None] - edges)], dim=-1)
    hi = torch.cat([F.logsigmoid(edges - native[:, None]),
                    torch.zeros(len(native), 1)], dim=-1)
    loss = -torch.logsumexp(probabilities.clamp_min(1e-30).log() + lo + hi, dim=-1)
    derivative = torch.autograd.grad(loss.sum(), native)[0]
    return [{
        'true_bin': index, 'pairs': int((labels == index).sum()),
        'preferred_distance_grid_mean': float(preferred[labels == index].mean()),
        'preferred_at_least_32_fraction': float((preferred[labels == index] >= 32).float().mean()),
        'fraction_force_increases_native_distance': float((derivative[labels == index] < 0).float().mean()),
    } for index in range(len(edges) + 1) if (labels == index).any()]


def audit(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    head = Path(args.head)
    manifest = open_bundle(args.cache)
    provenance = json.loads((head / 'provenance.json').read_text())
    dataset = FrozenDataset(args.cache, manifest, 'val')
    entries = {dataset.ids[i]: dataset[i] for i in range(len(dataset))}
    result = {'bundle_sha256': manifest['bundle_sha256'], 'sources': {__file__: digest(__file__)},
              'scope': 'Existing development predictions; diagnostic evaluation, no tuning', 'modes': {}}
    for mode in ('sequence', 'decoder'):
        path = head / mode / 'validation_restraints.pt'
        export = torch.load(path, map_location='cpu', weights_only=False)
        if export['model_sha256'] != digest(head / mode / 'last.pt'):
            raise ValueError('Prediction checkpoint identity differs')
        rows = validate_export(export, manifest, provenance['selection']['val'])
        edges = export['edges']
        probabilities, distances, labels, separation = [], [], [], []
        per_chain = []
        top_correct = top_count = 0
        for identifier, row in rows.items():
            entry = entries[identifier]
            if not torch.equal(entry['mask'], row['mask']):
                raise ValueError('Prediction mask mismatch')
            p, pairs = row['probabilities'], row['pairs']
            y, d = distance_labels(entry['target'], pairs, edges)
            probabilities.append(p)
            distances.append(d)
            labels.append(y)
            separation.append(pairs[:, 1] - pairs[:, 0])
            c = contact_metrics(p[:, :3].sum(-1), d < 8., int(entry['mask'].sum()))
            top_count += c['top_L_count']
            top_correct += round(c['top_L_precision'] * c['top_L_count'])
            per_chain.append({'id': identifier, **describe(p, d, y, edges), **c})
        p, d, y, sep = map(torch.cat, (probabilities, distances, labels, separation))
        confusion = torch.bincount(y * p.shape[1] + p.argmax(-1),
                                   minlength=p.shape[1] ** 2).reshape(p.shape[1], p.shape[1])
        bins = []
        for index in range(p.shape[1]):
            mask = y == index
            bins.append({'lower': 0. if index == 0 else float(edges[index - 1]),
                         'upper': float(edges[index]) if index < len(edges) else None,
                         **describe(p[mask], d[mask], y[mask], edges)})
        contact = p[:, :3].sum(-1)
        calibration = []
        for lo, hi in ((0., .01), (.01, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .5), (.5, 1.01)):
            mask = (contact >= lo) & (contact < hi)
            calibration.append({'lower': lo, 'upper': hi,
                                **describe(p[mask], d[mask], y[mask], edges)})
        separation_rows = []
        for lo, hi in ((12, 24), (24, 48), (48, 10000)):
            mask = (sep >= lo) & (sep < hi)
            separation_rows.append({'lower': lo, 'upper': hi,
                                    **describe(p[mask], d[mask], y[mask], edges)})
        train = json.loads((head / mode / 'final_training.json').read_text())['aggregate']
        val = json.loads((head / mode / 'final_validation.json').read_text())['aggregate']
        result['modes'][mode] = {
            'aggregate': describe(p, d, y, edges), 'train': train, 'validation': val,
            'per_true_bin': bins, 'per_separation': separation_rows,
            'contact_calibration': calibration, 'confusion_counts_true_rows_predicted_columns': confusion.tolist(),
            'mean_predicted_bin_mass': p.mean(0).tolist(),
            'true_bin_frequency': (torch.bincount(y, minlength=p.shape[1]) / len(y)).tolist(),
            'micro_per_chain_top_L_precision': top_correct / top_count,
            'per_chain_top_L_correct': top_correct, 'per_chain_top_L_budget': top_count,
            'per_chain': per_chain, 'restraint_preferences': restraint_preferences(p, d, y, edges),
        }
    reconstruction = Path(args.reconstruction)
    protocol = json.loads((reconstruction / 'experiment.json').read_text())
    if protocol['bundle_sha256'] != manifest['bundle_sha256']:
        raise ValueError('Reconstruction cache mismatch')
    panel_path = reconstruction / 'panel.pt'
    if digest(panel_path) != protocol['panel_sha256']:
        raise ValueError('Reconstruction panel identity mismatch')
    panel = torch.load(panel_path, weights_only=False, map_location='cpu')
    solver = {}
    for mode in ('local_only', 'sequence', 'decoder'):
        checkpoint = torch.load(reconstruction / mode / 'best.pt', weights_only=False, map_location='cpu')
        if checkpoint['ids'] != protocol['ids']:
            raise ValueError('Coordinate ID/order mismatch')
        changes = []
        for sample, fitted in zip(panel['entries'], checkpoint['coords']):
            entry, local = sample['entry'], sample['local']
            prediction = None if mode == 'local_only' else sample['predictions'][mode]
            records = {}
            for name, coords in (('seed', entry['coords']), ('fitted', fitted), ('native', entry['target'])):
                coords = coords.masked_fill(~entry['mask'][:, None], 0.).clone().requires_grad_()
                _, terms = reconstruction_objective(coords, entry, local, prediction, panel['edges'])
                loss = sum(terms.values())
                gradient = torch.autograd.grad(loss, coords)[0]
                records[name] = {**{k: float(v.detach()) for k, v in terms.items()},
                                 'coordinate_gradient_norm': float(gradient.norm())}
                if mode != 'local_only':
                    pairs = prediction['pairs']
                    dist = (coords[pairs[:, 0]] - coords[pairs[:, 1]]).norm(dim=-1)
                    native = (entry['target'][pairs[:, 0]] - entry['target'][pairs[:, 1]]).norm(dim=-1)
                    records[name].update(nonlocal_mean_distance=float(dist.detach().mean()),
                                         nonlocal_distance_mae=float((dist - native).detach().abs().mean()))
            changes.append({'id': entry['id'], **records})
        solver[mode] = changes
    result['solver_objectives'] = solver
    atomic_json(root / 'audit.json', result)
    print(json.dumps({'output': str(root / 'audit.json'), 'modes': {
        mode: data['aggregate'] for mode, data in result['modes'].items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_decoder_enrichment_v3_20261021')
    parser.add_argument('--head', default='runs/se3_distance_head_20261021')
    parser.add_argument('--reconstruction', default='runs/se3_distance_reconstruction_20261021')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args)


if __name__ == '__main__':
    main()
