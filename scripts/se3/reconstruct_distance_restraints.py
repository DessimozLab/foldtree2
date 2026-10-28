"""Coordinate reconstruction with frozen predicted distance distributions."""
import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from foldtree2.se3_cache import FrozenDataset, digest, open_bundle
from foldtree2.src.distance_restraints import distribution_restraint_loss, eligible_pairs, validate_edges
from foldtree2.train_se3_lightning import calibrated
from scripts.se3.diagnose_se3_reconstruction import atomic_checkpoint, atomic_json, select_panel
from scripts.se3.diagnose_se3_restraints import evaluate, objective, sources as diagnostic_sources


MODES = ('local_only', 'sequence', 'decoder')


def sources():
    names = ('scripts/se3/reconstruct_distance_restraints.py',
             'foldtree2/src/distance_restraints.py')
    return {**diagnostic_sources(), **{name: digest(name) for name in names}}


def validate_export(export, manifest, expected_ids):
    if (export['protocol'] != 'pair-distance-restraints-v1'
            or export['bundle_sha256'] != manifest['bundle_sha256']):
        raise ValueError('Distance export protocol or bundle mismatch')
    validate_edges(export['edges'])
    rows = export['entries']
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError('Export IDs differ from trained validation selection')
    result = {}
    for row in rows:
        if row['id'] not in manifest['val_ids'] or row['id'] in manifest['train_ids']:
            raise ValueError('Reconstruction requires validation-only restraints')
        mask, pairs, p = row['mask'], row['pairs'], row['probabilities']
        expected = eligible_pairs(mask)
        if not torch.equal(pairs, expected):
            raise ValueError('Export must contain exactly all eligible pairs in canonical order')
        if (p.shape != (len(pairs), len(export['edges']) + 1)
                or not p.is_floating_point() or not torch.isfinite(p).all()
                or (p < 0).any() or (p > 1).any()
                or not torch.allclose(p.sum(-1), torch.ones_like(p[:, 0]), atol=1e-5, rtol=1e-5)):
            raise ValueError('Invalid exported distance distributions')
        result[row['id']] = row
    return result


def reconstruction_objective(coords, entry, local, prediction, edges):
    loss, terms = objective(coords, entry, local, 'local_only')
    if prediction is not None:
        restraint = distribution_restraint_loss(
            coords, prediction['pairs'], prediction['probabilities'], edges)
        loss = loss + restraint
        terms['restraint'] = restraint
    return loss, terms


def prepare(args):
    if (args.count < 1 or args.steps < 1 or args.seconds <= 0
            or not math.isfinite(args.seconds)):
        raise ValueError('Expected positive finite reconstruction bounds')
    root, head = Path(args.output), Path(args.head)
    if root.exists():
        raise ValueError('Requires fresh reconstruction output directory')
    manifest = open_bundle(args.cache)
    provenance = json.loads((head / 'provenance.json').read_text())
    if provenance['bundle_sha256'] != manifest['bundle_sha256']:
        raise ValueError('Head training cache identity differs')
    if set(provenance['selection']['train']) & set(provenance['selection']['val']):
        raise ValueError('Distance head trained on validation identifiers')
    exports, inputs = {}, {}
    edges = None
    for mode in ('sequence', 'decoder'):
        completion = json.loads((head / mode / 'completion.json').read_text())
        if completion['mode'] != mode or completion['stop_reason'] != 'epochs':
            raise ValueError('Requires completed fixed-epoch trained distance heads')
        path = head / mode / 'validation_restraints.pt'
        export = torch.load(path, weights_only=False, map_location='cpu')
        model_hash = digest(head / mode / 'last.pt')
        if export['model_sha256'] != model_hash or completion['checkpoint_sha256'] != model_hash:
            raise ValueError('Export checkpoint identity mismatch')
        exports[mode] = validate_export(export, manifest, provenance['selection']['val'])
        if edges is not None and not torch.equal(edges, export['edges']):
            raise ValueError('Distance heads have different bin definitions')
        edges = export['edges']
        inputs[mode] = {'export_sha256': digest(path), 'checkpoint_sha256': model_hash}
    ids = select_panel({**manifest, 'train_ids': provenance['selection']['val'],
                        'val_ids': manifest['train_ids']}, args.count)
    dataset = FrozenDataset(args.cache, manifest, 'val')
    panel = []
    for identifier in ids:
        entry = calibrated(dataset[dataset.ids.index(identifier)], manifest['scale'])
        mask = entry['mask']
        if any(not torch.equal(exports[m][identifier]['mask'], mask) for m in exports):
            raise ValueError('Prediction and cache masks differ')
        index = torch.arange(len(mask))
        separation = index[None] - index[:, None]
        valid = mask[:, None] & mask[None]
        local = {
            'bond_indices': (mask[:-1] & mask[1:]).nonzero().flatten(),
            'excluded_pairs': ((separation >= 2) & valid).nonzero(),
        }
        panel.append({'entry': entry, 'local': local,
                      'predictions': {mode: exports[mode][identifier] for mode in exports}})
    root.mkdir(parents=True)
    atomic_checkpoint(root / 'panel.pt', {'entries': panel, 'edges': edges})
    atomic_json(root / 'experiment.json', {
        'protocol': 'predicted-distance-reconstruction-v1',
        'ids': ids, 'bundle_sha256': manifest['bundle_sha256'],
        'head_provenance_sha256': digest(head / 'provenance.json'), 'inputs': inputs,
        'panel_sha256': digest(root / 'panel.pt'), 'sources': sources(),
        'modes': list(MODES), 'steps': args.steps, 'seconds_per_case': args.seconds,
        'learning_rate': .05, 'gradient_clip': 1.,
        'objective': 'generic CA bond3.8A + exclusion3.6A + unit-weight soft distance mixture',
        'restraint_softness_angstrom': 1.,
        'selection': 'case own objective only, never native reconstruction metrics',
        'scope': 'reused development chains; direct optimization, not a trained SE3 head',
        'native_geometry': 'evaluation only; no native-derived local geometry or restraints',
    })
    for name in sources():
        path = root / 'sources' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(name).read_bytes())


def run(args):
    root = Path(args.output)
    protocol = json.loads((root / 'experiment.json').read_text())
    if sources() != protocol['sources'] or digest(root / 'panel.pt') != protocol['panel_sha256']:
        raise ValueError('Reconstruction implementation or panel changed')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    data = torch.load(root / 'panel.pt', map_location=device, weights_only=False)
    panel, edges = data['entries'], data['edges']

    def event(kind, **values):
        text = json.dumps({'event': kind, 'time': time.time(), **values}, allow_nan=False)
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(text + '\n')
        print(text, flush=True)

    try:
        for mode in MODES:
            output = root / mode
            output.mkdir(exist_ok=False)
            parameters = [torch.nn.Parameter(x['entry']['coords'].clone()) for x in panel]
            optimizer = torch.optim.AdamW(parameters, lr=.05, weight_decay=0.)
            initial = [{'id': x['entry']['id'], 'metrics': evaluate(p.detach(), x['entry'])}
                       for p, x in zip(parameters, panel)]

            def loss_for(p, sample):
                prediction = None if mode == 'local_only' else sample['predictions'][mode]
                return reconstruction_objective(
                    p, sample['entry'], sample['local'], prediction, edges)[0]

            def score():
                with torch.no_grad():
                    return sum(float(loss_for(p, s)) for p, s in zip(parameters, panel)) / len(panel)

            def save_best(step, value):
                atomic_checkpoint(output / 'best.pt', {
                    'coords': [p.detach().clone() for p in parameters],
                    'step': step, 'objective': value, 'ids': protocol['ids']})

            best, best_step = score(), 0
            save_best(0, best)
            event('launch', mode=mode, chains=len(panel), initial_objective=best)
            started = time.monotonic()
            for step in range(1, protocol['steps'] + 1):
                optimizer.zero_grad(set_to_none=True)
                for p, sample in zip(parameters, panel):
                    loss = loss_for(p, sample)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite predicted-restraint loss')
                    (loss / len(panel)).backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                value = score()
                if not math.isfinite(value):
                    raise FloatingPointError('Nonfinite post-update objective')
                if value < best:
                    best, best_step = value, step
                    save_best(step, value)
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
            summary = {
                'mode': mode, 'steps': step, 'best_step': best_step, 'best_objective': best,
                'elapsed_seconds': time.monotonic() - started,
                'stop_reason': 'steps' if step == protocol['steps'] else 'time_limit',
                'initial': initial, 'best': rows, 'final': final,
                'passed': sum(r['metrics']['tm_score'] > .8 and
                              r['metrics']['ca_rmsd'] < 3. for r in rows),
                'mean_best_metrics': {k: sum(r['metrics'][k] for r in rows) / len(rows)
                                      for k in ('tm_score', 'ca_rmsd', 'lddt', 'ca_bond_mae',
                                                'reflection_allowed_rmsd')},
            }
            atomic_json(output / 'summary.json', summary)
            atomic_checkpoint(output / 'last.pt', {
                'coords': [p.detach() for p in parameters], 'step': step, 'ids': protocol['ids']})
            if sources() != protocol['sources']:
                raise ValueError('Reconstruction source changed')
            event('case_complete', mode=mode, steps=step, metrics=summary['mean_best_metrics'])
        atomic_json(root / 'completion.json', {'complete': True, 'modes': list(MODES)})
        event('complete')
    except Exception as exc:
        atomic_json(root / 'failure.json', {'error': repr(exc)})
        event('failure', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_decoder_enrichment_v3_20261021')
    parser.add_argument('--head', default='runs/se3_distance_head_20261021')
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seconds', type=float, default=1200.)
    parser.add_argument('--device', choices=['cpu', 'cuda:0', 'cuda:1'], default='cuda:1')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
