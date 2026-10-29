"""Bounded memorization and native/decoy energy gates before further folding runs."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
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
from foldtree2.src.distance.distance_energy import (
    distance_energy, fit_reference, sample_contact_strata, separation_groups,
)
from foldtree2.src.distance.distance_restraints import PairwiseDistanceHead, distance_labels, eligible_pairs
from foldtree2.train_distance_restraints import contact_metrics, load_records
from foldtree2.train_se3_lightning import calibrated, source_identity
from scripts.se3.diagnose_se3_reconstruction import atomic_checkpoint, atomic_json, select_panel
from scripts.se3.reconstruct_distance_restraints import validate_export


def sources():
    files = ('scripts/se3/diagnose_distance_learning.py', 'foldtree2/src/distance/distance_energy.py',
             'foldtree2/src/distance/distance_restraints.py', 'foldtree2/train_distance_restraints.py',
             'foldtree2/se3_cache.py', 'scripts/se3/diagnose_se3_reconstruction.py',
             'scripts/se3/reconstruct_distance_restraints.py')
    return {**{str(Path('foldtree2') / name): value for name, value in source_identity().items()},
            **{name: digest(name) for name in files}}


def describe(model, records, device):
    rows = []
    all_p, all_y, all_d = [], [], []
    model.eval()
    with torch.no_grad():
        for entry, pairs in records:
            entry = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in entry.items()}
            pairs = pairs.to(device)
            labels, distances = distance_labels(entry['target'], pairs, model.edges)
            p = model(entry, pairs).softmax(-1)
            rows.append({'id': entry['id'], 'nll': float(F.nll_loss(p.clamp_min(1e-30).log(), labels)),
                         **contact_metrics(model.contact_probability(p), distances < 8.,
                                           int(entry['mask'].sum()))})
            all_p.append(p)
            all_y.append(labels)
            all_d.append(distances)
        p, y, d = torch.cat(all_p), torch.cat(all_y), torch.cat(all_d)
        per_bin = []
        for i in range(len(model.edges) + 1):
            mask = y == i
            per_bin.append({'bin': i, 'pairs': int(mask.sum()),
                            'nll': float(-p[mask, i].clamp_min(1e-30).log().mean()) if mask.any() else None})
        return {
            'nll': float(F.nll_loss(p.clamp_min(1e-30).log(), y)),
            'bin_accuracy': float((p.argmax(-1) == y).float().mean()),
            **contact_metrics(model.contact_probability(p), d < 8.,
                              sum(int(e['mask'].sum()) for e, _ in records)),
            'per_bin': per_bin, 'per_chain': rows,
        }


def memorization_pass(report):
    return (report['nll'] <= .25 and report['bin_accuracy'] >= .9
            and report['contact_average_precision'] is not None
            and report['contact_average_precision'] >= .8
            and report['contact_precision'] is not None and report['contact_precision'] >= .8
            and report['contact_recall'] is not None and report['contact_recall'] >= .8)


def memorized_energy(model, records, device, reference):
    rows = []
    with torch.no_grad():
        for entry, pairs in records:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in entry.items()}
            pairs = pairs.to(device)
            p = model(inputs, pairs).softmax(-1)
            background = reference.to(device)[separation_groups(pairs)]
            native = inputs['target'].masked_fill(~inputs['mask'][:, None], 0.)
            noise = torch.randn(native.shape, generator=torch.Generator().manual_seed(42)).to(device)
            scores = {}
            for name, coords in (('native', native), ('seed', inputs['coords']),
                                 ('noise_4', native + 4 * noise)):
                distances = (coords[pairs[:, 0]] - coords[pairs[:, 1]]).norm(dim=-1)
                scores[name] = float(distance_energy(distances, p, model.edges, background).mean())
            rows.append({'id': entry['id'], 'scores': scores})
    return rows


def memorize(records, manifest, root, count, args, event):
    output = root / f'memorize_{count}'
    output.mkdir()
    device = torch.device(args.device)
    torch.manual_seed(42)
    model = PairwiseDistanceHead(manifest['feature_dim'],
                                 manifest['decoder_enrichment']['contact_dim'], hidden=64).to(device)
    selected = records[:count]
    data = [({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in e.items()},
             pairs.to(device)) for e, pairs in selected]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
    initial = describe(model, selected, device)
    atomic_json(output / 'initial.json', initial)
    started, history = time.monotonic(), []
    best, best_step = float('inf'), 0
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for entry, pairs in data:
            labels, _ = distance_labels(entry['target'], pairs, model.edges)
            loss = F.cross_entropy(model(entry, pairs), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite memorization loss')
            (loss / count).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps or time.monotonic() - started >= args.seconds:
            report = describe(model, selected, device)
            history.append({'step': step, 'gradient_norm': float(norm), **report})
            atomic_json(output / 'history.json', history)
            if report['nll'] < best:
                best, best_step = report['nll'], step
                atomic_checkpoint(output / 'best.pt', {
                    'model': model.state_dict(), 'model_config': model.model_config(),
                    'step': step, 'report': report, 'ids': [e['id'] for e, _ in selected],
                    'bundle_sha256': manifest['bundle_sha256'], 'resumable': False})
            event('memorization_progress', count=count, step=step, nll=report['nll'],
                  accuracy=report['bin_accuracy'], contact_ap=report['contact_average_precision'],
                  contact_recall=report['contact_recall'])
            if memorization_pass(report):
                break
        if time.monotonic() - started >= args.seconds:
            break
    saved = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    clone = PairwiseDistanceHead(**saved['model_config']).to(device)
    clone.load_state_dict(saved['model'])
    if describe(clone, selected, device) != saved['report']:
        raise ValueError('Memorization checkpoint reload mismatch')
    reference = torch.load(root / 'reference.pt', map_location='cpu', weights_only=False)['probabilities']
    atomic_json(output / 'training_energy.json', memorized_energy(clone, selected, device, reference))
    summary = {'updates': step, 'best_step': best_step, 'passed': memorization_pass(saved['report']),
               'stop_reason': 'gate' if memorization_pass(report) else
               ('steps' if step == args.steps else 'time_limit'),
               'elapsed_seconds': time.monotonic() - started, 'best': saved['report'],
               'final': report, 'reload_exact': True}
    atomic_json(output / 'completion.json', summary)
    return summary


def energy_gate(args, manifest, records, root):
    head = Path(args.head)
    provenance = json.loads((head / 'provenance.json').read_text())
    if (provenance['bundle_sha256'] != manifest['bundle_sha256']
            or provenance['selection']['train'] != manifest['train_ids']):
        raise ValueError('Reference must use exactly the head training split')
    dataset = FrozenDataset(args.cache, manifest, 'val')
    validation = {dataset.ids[i]: calibrated(dataset[i], manifest['scale']) for i in range(len(dataset))}
    protocol = json.loads((Path(args.reconstruction) / 'experiment.json').read_text())
    if protocol['bundle_sha256'] != manifest['bundle_sha256']:
        raise ValueError('Reconstruction bundle differs')
    fitted, decoy_hashes = {}, {}
    for mode in ('local_only', 'sequence', 'decoder'):
        path = Path(args.reconstruction) / mode / 'best.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['ids'] != protocol['ids']:
            raise ValueError('Decoy coordinate ID mismatch')
        if len(payload['coords']) != len(payload['ids']):
            raise ValueError('Decoy coordinate count differs')
        fitted[mode] = dict(zip(payload['ids'], payload['coords']))
        decoy_hashes[mode] = digest(path)
    exports, inputs = {}, {}
    edges = None
    for mode in ('sequence', 'decoder'):
        path = head / mode / 'validation_restraints.pt'
        export = torch.load(path, map_location='cpu', weights_only=False)
        if export['model_sha256'] != digest(head / mode / 'last.pt'):
            raise ValueError('Energy prediction checkpoint mismatch')
        exports[mode] = validate_export(export, manifest, provenance['selection']['val'])
        if edges is not None and not torch.equal(edges, export['edges']):
            raise ValueError('Distance bin mismatch')
        edges = export['edges']
        inputs[mode] = digest(path)
    reference, counts = fit_reference(records, edges)
    atomic_checkpoint(root / 'reference.pt', {
        'probabilities': reference, 'counts_including_laplace': counts,
        'edges': edges, 'train_ids': manifest['train_ids'], 'bundle_sha256': manifest['bundle_sha256']})
    rows = []
    for mode, predictions in exports.items():
        for identifier, prediction in predictions.items():
            entry = validation[identifier]
            if not torch.equal(entry['mask'], prediction['mask']):
                raise ValueError('Energy prediction mask mismatch')
            pairs, p = prediction['pairs'], prediction['probabilities']
            background = reference[separation_groups(pairs)]
            generator = torch.Generator().manual_seed(42)
            native = entry['target'].masked_fill(~entry['mask'][:, None], 0.)
            decoys = {'native': native, 'seed': entry['coords']}
            for noise in (1., 4., 8.):
                decoys[f'noise_{noise:g}'] = native + torch.randn(native.shape, generator=generator) * noise
            valid_indices = entry['mask'].nonzero().flatten()
            shuffled = native.clone()
            shuffled[valid_indices] = native[valid_indices[torch.randperm(len(valid_indices), generator=generator)]]
            decoys['permuted'] = shuffled
            for fitted_mode, mapping in fitted.items():
                if identifier in mapping:
                    decoys['fitted_' + fitted_mode] = mapping[identifier]
            scores = {}
            for name, coords in decoys.items():
                d = (coords[pairs[:, 0]] - coords[pairs[:, 1]]).norm(dim=-1)
                # Historical kernel, retained only for a matched scoring control.
                lower = torch.cat([torch.zeros(len(d), 1), F.logsigmoid(d[:, None] - edges)], dim=-1)
                upper = torch.cat([F.logsigmoid(edges - d[:, None]), torch.zeros(len(d), 1)], dim=-1)
                legacy = -torch.logsumexp(p.log() + lower + upper, dim=-1).mean()
                scores[name] = {
                    'legacy': float(legacy),
                    'partition': float(distance_energy(d, p, edges).mean()),
                    'reference': float(distance_energy(d, p, edges, background).mean())}
            rows.append({'mode': mode, 'id': identifier, 'scores': scores})
    wins = {}
    for mode in exports:
        selected = [r for r in rows if r['mode'] == mode]
        wins[mode] = {}
        for score in ('legacy', 'partition', 'reference'):
            decoy_names = sorted(set(name for row in selected for name in row['scores']) - {'native'})
            wins[mode][score] = {}
            for name in decoy_names:
                subset = [r for r in selected if name in r['scores']]
                margins = [r['scores'][name][score] - r['scores']['native'][score] for r in subset]
                wins[mode][score][name] = {
                    'chains': len(subset), 'native_wins': sum(m > 1e-6 for m in margins),
                    'mean_decoy_minus_native': sum(margins) / len(margins)}
    # Declared gate, not selected from whichever decoys happen to favor the score.
    required = ('seed', 'noise_4', 'noise_8', 'fitted_decoder')
    passed = all(wins['decoder']['reference'][name]['native_wins']
                 / wins['decoder']['reference'][name]['chains'] >= .75 for name in required)
    result = {'passed': passed, 'gate': '>=75% native wins separately against seed,4A/8A noise,fitted_decoder',
              'inputs': inputs, 'decoy_checkpoint_hashes': decoy_hashes,
              'reference_sha256': digest(root / 'reference.pt'),
              'wins': wins, 'per_chain': rows, 'scope': 'Monitored development score diagnostic; no coordinates optimized'}
    atomic_json(root / 'energy_gate.json', result)
    return result


def sampling_pilot(args, manifest, records, root, event):
    """Matched small-panel sampling control, run only after both diagnostic gates."""
    device = torch.device(args.device)
    reports = {}
    for sampling in ('uniform', 'contact_corrected'):
        torch.manual_seed(42)
        model = PairwiseDistanceHead(manifest['feature_dim'],
                                     manifest['decoder_enrichment']['contact_dim'], hidden=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
        generator = torch.Generator().manual_seed(42)
        started = time.monotonic()
        for step in range(1, args.pilot_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            for entry, all_pairs in records:
                labels, _ = distance_labels(entry['target'], all_pairs, model.edges.cpu())
                if sampling == 'uniform':
                    indices = torch.randint(len(labels), (512,), generator=generator)
                    weights = torch.ones(512)
                else:
                    indices, weights = sample_contact_strata(labels, 512, generator)
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in entry.items()}
                logits = model(inputs, all_pairs[indices].to(device))
                loss = (F.cross_entropy(logits, labels[indices].to(device), reduction='none')
                        * weights.to(device)).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite corrected sampling loss')
                (loss / len(records)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if time.monotonic() - started >= args.seconds:
                break
        reports[sampling] = {'steps': step, 'training': describe(model, records, device)}
        atomic_checkpoint(root / (sampling + '.pt'), {
            'model': model.state_dict(), 'model_config': model.model_config(), 'resumable': False})
        event('sampling_case_complete', sampling=sampling, steps=step)
    atomic_json(root / 'sampling_pilot.json', reports)


def run(args):
    if (args.steps < 1 or args.pilot_steps < 1 or args.seconds <= 0
            or not math.isfinite(args.seconds) or args.log_every < 1):
        raise ValueError('Positive finite diagnostic bounds required')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)

    def event(kind, **values):
        text = json.dumps({'event': kind, 'time': time.time(), **values}, allow_nan=False)
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(text + '\n')
        print(text, flush=True)

    try:
        torch.set_num_threads(2)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        manifest = open_bundle(args.cache)
        records = load_records(args.cache, manifest, 'train')
        candidates = {}
        edges = torch.tensor([4., 6., 8., 10., 12., 16., 20., 24., 32.])
        for entry, pairs in records:
            labels, _ = distance_labels(entry['target'], pairs, edges)
            if (labels < 3).any():
                candidates[entry['id']] = (entry, pairs)
        ids = select_panel({**manifest, 'train_ids': list(candidates)}, 8)
        panel = [candidates[i] for i in ids]
        identity = sources()
        atomic_json(root / 'provenance.json', {
            'args': vars(args), 'sources': identity, 'bundle_sha256': manifest['bundle_sha256'],
            'panel_ids': ids, 'excluded_no_native_contacts': len(records) - len(candidates),
            'panel_rule': 'length-spread training panel, requires >=1 native long-range contact',
            'memorization_gate': 'NLL<=.25,accuracy>=.9,contact AP/precision/recall>=.8',
            'scope': 'training memorization, development energy audit; not independent generalization',
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None})
        for name in identity:
            path = root / 'sources' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(name).read_bytes())
        event('ready', panel_ids=ids, device=str(device))
        energy = energy_gate(args, manifest, records, root)
        event('energy_gate_complete', passed=energy['passed'])
        summaries = {}
        for count in (1, 8):
            summaries[str(count)] = memorize(panel, manifest, root, count, args, event)
        allow_sampling = energy['passed'] and all(s['passed'] for s in summaries.values())
        if allow_sampling:
            sampling_pilot(args, manifest, panel, root, event)
        else:
            atomic_json(root / 'sampling_pilot.json', {
                'status': 'blocked', 'reason': 'Both memorization gates and reference-energy gate must pass',
                'memorization': {k: v['passed'] for k, v in summaries.items()},
                'energy': energy['passed']})
        if identity != sources() or open_bundle(args.cache)['bundle_sha256'] != manifest['bundle_sha256']:
            raise ValueError('Diagnostic source or cache changed')
        atomic_json(root / 'completion.json', {
            'complete': True, 'energy_passed': energy['passed'],
            'memorization': {k: {'passed': v['passed'], 'updates': v['updates'],
                                'best_nll': v['best']['nll']} for k, v in summaries.items()},
            'sampling_launched': allow_sampling})
        event('complete', sampling_launched=allow_sampling)
    except Exception as exc:
        atomic_json(root / 'failure.json', {'error': repr(exc)})
        event('failure', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_decoder_enrichment_v3_20261021')
    parser.add_argument('--head', default='runs/se3_distance_head_20261021')
    parser.add_argument('--reconstruction', default='runs/se3_distance_reconstruction_20261021')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0', 'cuda:1'], default='cuda:1')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--pilot-steps', type=int, default=1000)
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--log-every', type=int, default=100)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
