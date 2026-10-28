"""Bounded direct-coordinate restraint controls, not a learned folding model."""
import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F

from foldtree2.se3_cache import FrozenDataset, digest, open_bundle
from foldtree2.se3_validation import production_pair, sha256
from foldtree2.train_se3_lightning import calibrated, source_identity
from scripts.se3.diagnose_se3_reconstruction import (
    atomic_checkpoint, atomic_json, reconstruction_metrics,
)


MODES = ('local_only', 'predicted_contacts', 'predicted_top_pairs', 'oracle_contacts',
         'oracle_distances_uniform', 'oracle_distances_decoder_pairs',
         'oracle_distances_all')


def sources():
    result = {str(Path('foldtree2') / k): v for k, v in source_identity().items()}
    for name in ('scripts/se3/diagnose_se3_restraints.py', 'scripts/se3/diagnose_se3_reconstruction.py'):
        result[name] = digest(name)
    return result


def prepare_restraints(entry, temperature, bias, seed=42):
    mask = entry['mask']
    if mask.ndim != 1 or mask.dtype != torch.bool or mask.sum() < 12:
        raise ValueError('Expected at least twelve valid residues')
    n = len(mask)
    idx = torch.arange(n, device=mask.device)
    separation = idx[None, :] - idx[:, None]
    valid = mask[:, None] & mask[None, :]
    pairs = ((separation >= 12) & valid).nonzero()
    if not len(pairs):
        raise ValueError('No eligible long-range pairs')
    z = entry['contact_z']
    if z.shape[0] != n or not torch.isfinite(z[mask]).all():
        raise ValueError('Invalid cached contact embeddings')
    if not torch.allclose(z[mask].norm(dim=-1), torch.ones_like(z[mask, 0]), atol=1e-4):
        raise ValueError('Contact calibration requires normalized embeddings')
    probabilities = (temperature * (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(-1) + bias
                     ).sigmoid().clamp(1e-6, 1 - 1e-6)
    target = entry['target']
    if not torch.isfinite(target[mask]).all():
        raise ValueError('Nonfinite native coordinates')
    distances = (target[pairs[:, 0]] - target[pairs[:, 1]]).norm(dim=-1)
    count = min(2 * int(mask.sum()), len(pairs))
    ranked = probabilities.argsort(descending=True, stable=True)
    random = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed))
    labels = distances < 8.
    positives = int(labels.sum())
    ranked_labels = labels[ranked].float()
    precision = ranked_labels.cumsum(0) / torch.arange(
        1, len(pairs) + 1, device=mask.device)
    audit = {
        'eligible_pairs': len(pairs), 'native_ca_contacts': positives,
        'native_ca_contact_prevalence': float(labels.float().mean()),
        'predicted_probability_mean': float(probabilities.mean()),
        'brier_against_ca_contacts': float((probabilities - labels.float()).square().mean()),
        'average_precision_against_ca_contacts': (
            float((precision * ranked_labels).sum() / positives) if positives else None),
        'precision_top_2L': float(labels[ranked[:count]].float().mean()),
        'sparse_distance_pairs': count,
    }
    return {
        'pairs': pairs, 'probabilities': probabilities.detach(),
        'native_distances': distances.detach(), 'native_contacts': labels.float(),
        'uniform_indices': random[:count].to(mask.device),
        'decoder_indices': ranked[:count],
        'excluded_pairs': ((separation >= 2) & valid).nonzero(),
        'bond_indices': (mask[:-1] & mask[1:]).nonzero().flatten(),
    }, audit


def objective(coords, entry, restraints, mode):
    """Generic local priors; only explicitly oracle modes read native restraints."""
    if mode not in MODES:
        raise ValueError('Unknown restraint mode')
    if not torch.isfinite(coords[entry['mask']]).all():
        raise FloatingPointError('Nonfinite valid coordinates')
    bonds = restraints['bond_indices']
    if not len(bonds):
        raise ValueError('No valid adjacent CA bonds')
    bond_lengths = (coords[bonds + 1] - coords[bonds]).norm(dim=-1)
    bond = F.smooth_l1_loss(bond_lengths, torch.full_like(bond_lengths, 3.8))
    excluded = restraints['excluded_pairs']
    if not len(excluded):
        raise ValueError('No eligible excluded-volume pairs')
    distances = (coords[excluded[:, 0]] - coords[excluded[:, 1]]).norm(dim=-1)
    repulsion = ((3.6 - distances).relu() / 3.6).square().mean()
    restraint = coords.sum() * 0.
    if mode != 'local_only':
        pairs = restraints['pairs']
        d = (coords[pairs[:, 0]] - coords[pairs[:, 1]]).norm(dim=-1)
        if mode in ('predicted_contacts', 'oracle_contacts'):
            probabilities = restraints[
                'probabilities' if mode == 'predicted_contacts' else 'native_contacts']
            # Native binary inequalities can be satisfied without forcing extreme distances.
            attraction = F.smooth_l1_loss(
                (d - 8.).relu() / 4., torch.zeros_like(d), reduction='none')
            separation = F.smooth_l1_loss(
                (8. - d).relu() / 4., torch.zeros_like(d), reduction='none')
            restraint = (10. * probabilities * attraction + (1. - probabilities) * separation).mean()
        elif mode == 'predicted_top_pairs':
            selected = d[restraints['decoder_indices']]
            restraint = F.smooth_l1_loss(
                (selected - 8.).relu() / 4., torch.zeros_like(selected))
        else:
            indices = {
                'oracle_distances_uniform': restraints['uniform_indices'],
                'oracle_distances_decoder_pairs': restraints['decoder_indices'],
                'oracle_distances_all': torch.arange(len(d), device=d.device),
            }[mode]
            restraint = F.smooth_l1_loss(
                d[indices] / 4., restraints['native_distances'][indices] / 4.)
    total = bond + repulsion + restraint
    return total, {'bond': bond, 'repulsion': repulsion, 'restraint': restraint}


def evaluate(coords, entry):
    result = reconstruction_metrics(coords, None, entry)
    p, t = coords[entry['mask']], entry['target'][entry['mask']]
    p, t = p - p.mean(0), t - t.mean(0)
    u, _, vh = torch.linalg.svd(p.T @ t)
    result['reflection_allowed_rmsd'] = float(
        ((p @ u @ vh - t).square().sum(-1).mean()).sqrt())
    return result


def prepare(args):
    if args.steps < 1 or args.seconds <= 0 or not math.isfinite(args.seconds):
        raise ValueError('Positive finite bounds required')
    root = Path(args.output)
    if root.exists():
        raise ValueError('Requires fresh diagnostic output directory')
    manifest = open_bundle(args.cache)
    previous = json.loads(Path(args.panel).read_text())
    ids = previous['ids']
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(manifest['train_ids']):
        raise ValueError('Panel must contain distinct training IDs')
    if previous['bundle_sha256'] != manifest['bundle_sha256']:
        raise ValueError('Original panel cache identity differs')
    pair = production_pair()
    for key, path in pair.items():
        if sha256(path) != manifest['input_identity'][key + '_sha256']:
            raise ValueError('Frozen production identity differs')
    decoder = torch.load(pair['decoder'], map_location='cpu', weights_only=False)
    geometry = decoder.decoders['geometry_cnn']
    if not geometry.normalize:
        raise ValueError('Expected normalized contact decoder')
    temperature, bias = float(geometry.contact_temp.detach()), float(geometry.contact_bias.detach())
    if not all(math.isfinite(v) for v in (temperature, bias)):
        raise ValueError('Nonfinite contact calibration')
    dataset = FrozenDataset(args.cache, manifest, 'train')
    entries, audits = [], []
    for identifier in ids:
        entry = calibrated(dataset[dataset.ids.index(identifier)], manifest['scale'])
        restraints, audit = prepare_restraints(entry, temperature, bias)
        entries.append({'entry': entry, 'restraints': restraints})
        audits.append({'id': identifier, **audit})
    root.mkdir(parents=True)
    atomic_checkpoint(root / 'panel.pt', entries)
    atomic_json(root / 'contact_audit.json', audits)
    atomic_json(root / 'experiment.json', {
        'protocol': 'se3-direct-restraint-v1', 'ids': ids,
        'bundle_sha256': manifest['bundle_sha256'], 'panel_sha256': digest(root / 'panel.pt'),
        'original_panel_sha256': digest(args.panel), 'sources': sources(),
        'steps': args.steps, 'seconds_per_case': args.seconds,
        'modes': list(MODES), 'learning_rate': .05, 'gradient_clip': 1.,
        'contact_threshold': 8., 'distance_loss_scale': 4., 'positive_weight': 10.,
        'local_priors': 'CA bonds 3.8A; nonadjacent exclusion 3.6A; no native local geometry',
        'contact_caveat': 'Decoder trained on CB contacts; apply approximately to CA coordinates. '
                          'Oracle contacts and distance controls use CA geometry.',
        'scope': 'Eight previously used training chains; coordinate fitting, not learned generalization',
        'selection': 'Own restraint objective only, never target reconstruction metrics',
        'chirality': 'Distances cannot distinguish mirror images; report proper and reflection-allowed RMSD',
        'temperature': temperature, 'bias': bias, 'production_forward_calls': 0,
    })
    for name in sources():
        path = root / 'sources' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(name).read_bytes())


def run(args):
    root = Path(args.output)
    protocol = json.loads((root / 'experiment.json').read_text())
    if sources() != protocol['sources'] or digest(root / 'panel.pt') != protocol['panel_sha256']:
        raise ValueError('Staged diagnostic source or panel changed')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    panel = torch.load(root / 'panel.pt', map_location=device, weights_only=False)

    def event(kind, **values):
        text = json.dumps({'event': kind, 'time': time.time(), **values}, allow_nan=False)
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(text + '\n')
        print(text, flush=True)

    try:
        for mode in MODES:
            if sources() != protocol['sources']:
                raise ValueError('Diagnostic source changed')
            output = root / mode
            output.mkdir(exist_ok=False)
            parameters = [torch.nn.Parameter(x['entry']['coords'].clone()) for x in panel]
            optimizer = torch.optim.AdamW(parameters, lr=.05, weight_decay=0.)
            initial = [{'id': x['entry']['id'], 'metrics': evaluate(p.detach(), x['entry'])}
                       for p, x in zip(parameters, panel)]
            atomic_json(output / 'initial.json', initial)
            best = float('inf')
            best_step = 0
            started = time.monotonic()
            event('launch', mode=mode, device=str(device), chains=len(panel))

            def snapshot(step, value):
                atomic_checkpoint(output / 'best.pt', {
                    'coords': [p.detach().clone() for p in parameters], 'step': step,
                    'objective': value, 'ids': protocol['ids'], 'mode': mode})

            with torch.no_grad():
                best = sum(float(objective(p, x['entry'], x['restraints'], mode)[0])
                           for p, x in zip(parameters, panel)) / len(panel)
            snapshot(0, best)
            for step in range(1, protocol['steps'] + 1):
                optimizer.zero_grad(set_to_none=True)
                for p, sample in zip(parameters, panel):
                    loss, _ = objective(p, sample['entry'], sample['restraints'], mode)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite restraint objective')
                    (loss / len(panel)).backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    values = [float(objective(p, x['entry'], x['restraints'], mode)[0])
                              for p, x in zip(parameters, panel)]
                    value = sum(values) / len(values)
                if not math.isfinite(value):
                    raise FloatingPointError('Nonfinite post-update objective')
                if value < best:
                    best, best_step = value, step
                    snapshot(step, value)
                if step == 1 or step % 100 == 0:
                    event('progress', mode=mode, step=step, objective=value,
                          gradient_norm=float(norm), elapsed=time.monotonic() - started)
                if time.monotonic() - started >= protocol['seconds_per_case']:
                    break
            selected = torch.load(output / 'best.pt', map_location=device, weights_only=False)
            with torch.no_grad():
                rows = [{'id': x['entry']['id'], 'metrics': evaluate(p, x['entry'])}
                        for p, x in zip(selected['coords'], panel)]
                final = [{'id': x['entry']['id'], 'metrics': evaluate(p.detach(), x['entry'])}
                         for p, x in zip(parameters, panel)]
            atomic_json(output / 'summary.json', {
                'mode': mode, 'steps': step, 'best_step': best_step, 'best_objective': best,
                'stop_reason': 'steps' if step == protocol['steps'] else 'time_limit',
                'elapsed_seconds': time.monotonic() - started,
                'initial': initial, 'best': rows, 'final': final,
                'passed': sum(r['metrics']['tm_score'] > .8 and
                              r['metrics']['ca_rmsd'] < 3. for r in rows)})
            atomic_checkpoint(output / 'last.pt', {
                'coords': [p.detach() for p in parameters], 'step': step, 'ids': protocol['ids']})
            event('case_complete', mode=mode, steps=step,
                  metrics={k: sum(r['metrics'][k] for r in rows) / len(rows)
                           for k in ('tm_score', 'ca_rmsd', 'lddt', 'reflection_allowed_rmsd')})
        if sources() != protocol['sources']:
            raise ValueError('Diagnostic source changed')
        atomic_json(root / 'completion.json', {'modes': list(MODES), 'complete': True})
        event('complete')
    except Exception as exc:
        atomic_json(root / 'failure.json', {'error': repr(exc)})
        event('failure', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_clariden/bundle_4096')
    parser.add_argument('--panel', default='runs/se3_reconstruction_20261020/coordinates_seed/provenance.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0', 'cuda:1'], default='cuda:1')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seconds', type=float, default=1200.)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.prepare_only:
        prepare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
