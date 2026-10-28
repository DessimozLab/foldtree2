"""Bounded training-only reconstruction controls; never calls production models."""
import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from foldtree2.se3_cache import FrozenDataset, digest, open_bundle
from foldtree2.se3_validation import forward, frame_point_errors, metrics
from foldtree2.src.losses.se3_balanced import BalancedGeometryLoss
from foldtree2.src.se3_struct_decoder import enable_tensor_ff_input_norm
from foldtree2.train_se3_lightning import build_model, calibrated, source_identity


def select_panel(manifest, count, min_length=64, max_length=128):
    """Deterministic length-spread panel from training IDs only."""
    candidates = sorted(
        (i for i in manifest['train_ids']
         if min_length <= manifest['files'][i]['length'] <= max_length),
        key=lambda i: (manifest['files'][i]['length'], i),
    )
    if count < 1 or len(candidates) < count:
        raise ValueError('Insufficient training structures for requested panel')
    indices = [0] if count == 1 else [
        j * (len(candidates) - 1) // (count - 1) for j in range(count)]
    selected = [candidates[j] for j in indices]
    if set(selected) & set(manifest['val_ids']):
        raise ValueError('Diagnostic panel overlaps validation')
    return selected


def initialize(entry, kind, generator, noise):
    if kind == 'seed':
        return entry['coords'].detach().clone()
    if kind != 'native-noise' or not math.isfinite(noise) or noise <= 0:
        raise ValueError('Expected seed or positive native-noise initialization')
    perturbation = torch.randn(entry['target'].shape, generator=generator)
    perturbation = perturbation.to(entry['target'])
    return (entry['target'] + noise * perturbation).masked_fill(
        ~entry['mask'][:, None], 0.).detach()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def reconstruction_metrics(pred, angles, entry):
    result = metrics(pred, angles, entry)
    errors = frame_point_errors(pred, entry['target'], entry['mask'])
    result['fape_saturated_fraction'] = float((errors >= 10.).float().mean())
    result['frame_error_unclamped_angstrom'] = float(errors.mean())
    mask = entry['mask']
    p, t = pred[mask], entry['target'][mask]
    idx = torch.arange(len(mask), device=mask.device)[mask]
    eligible = (idx[None] - idx[:, None]) >= 12
    pd = torch.cdist(p - p.mean(0), p - p.mean(0))
    td = torch.cdist(t - t.mean(0), t - t.mean(0))
    predicted, target = (pd < 8.) & eligible, (td < 8.) & eligible
    shared = int((predicted & target).sum())
    npred, ntrue = int(predicted.sum()), int(target.sum())
    result.update(long_range_predicted_contacts=npred, long_range_target_contacts=ntrue,
                  long_range_contact_precision=shared / npred if npred else None,
                  long_range_contact_recall=shared / ntrue if ntrue else None)
    return result


def model_forward(model, entry, contacts='normal'):
    return forward(model, entry, contact_policy=contacts)


def perfect_target_audit(entries, loss_fn):
    rows = []
    for entry in entries:
        angles = entry['angles'][..., :3].nan_to_num()
        terms = loss_fn.components(entry['target'], angles, entry)
        coordinate = sum(float(terms[k]) * loss_fn.weights[k] for k in
                         ('distance', 'training_fape', 'bond', 'local_distance', 'bend', 'torsion'))
        if not math.isfinite(coordinate) or coordinate > 1e-5:
            raise ValueError(f"Perfect-target coordinate loss is nonzero: {entry['id']}")
        rows.append({'id': entry['id'], 'coordinate_objective': coordinate})
    return rows


def audit_model(model, entries, loss_fn, contacts):
    """Compare modes without retaining BatchNorm updates or accumulating grads."""
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    was_training = model.training
    rows, gradients = [], []
    parameters = [p for p in model.parameters() if p.requires_grad]
    try:
        for entry in entries:
            model.eval()
            with torch.no_grad():
                ep, ea = model_forward(model, entry, contacts)
                eval_loss = float(loss_fn(ep, ea, entry)[0])
            model.train()
            pred, angles = model_forward(model, entry, contacts)
            loss = loss_fn(pred, angles, entry)[0]
            grads = torch.autograd.grad(loss, parameters, allow_unused=True)
            vector = torch.cat([
                torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                for p, g in zip(parameters, grads)])
            if not torch.isfinite(vector).all():
                raise FloatingPointError('Nonfinite per-chain audit gradient')
            gradients.append(vector.cpu())
            mask = entry['mask']
            rows.append({
                'id': entry['id'], 'eval_objective': eval_loss,
                'train_objective': float(loss.detach()),
                'gradient_norm': float(vector.norm()),
                'train_eval_coordinate_rms_angstrom': float(
                    (pred[mask] - ep[mask]).detach().square().mean().sqrt()),
                'eval_seed_update_rms_angstrom': float(
                    (ep[mask] - entry['coords'][mask]).square().mean().sqrt()),
                'train_metrics': reconstruction_metrics(pred.detach(), angles.detach(), entry),
                'eval_metrics': reconstruction_metrics(ep, ea, entry),
            })
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
        matrix = torch.stack(gradients).double()
        norms = matrix.norm(dim=1)
        denominators = norms[:, None] * norms[None, :]
        cosines = matrix @ matrix.T / denominators.clamp_min(1e-30)
        return {'rows': rows, 'gradient_cosines': cosines.tolist(),
                'nonzero_gradient_pairs': (denominators > 0).tolist(),
                'buffers_restored': True, 'optimizer_steps': 0}
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        model.train(was_training)


def optimize_step(parameters, optimizer, loss_fn, predict, entries):
    optimizer.zero_grad(set_to_none=True)
    values = []
    for entry in entries:
        pred, angles = predict(entry)
        loss = loss_fn(pred, angles, entry)[0]
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite diagnostic loss')
        (loss / len(entries)).backward()
        values.append(float(loss.detach()))
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite diagnostic gradient')
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in parameters):
        raise FloatingPointError('Nonfinite diagnostic parameters')
    return sum(values) / len(values), float(norm)


def run(args):
    if args.steps < 1 or args.seconds <= 0 or args.log_every < 1:
        raise ValueError('Positive steps, seconds and log frequency required')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Positive finite learning rate required')
    if args.heldout_count < 0 or (args.heldout_count and args.mode != 'model'):
        raise ValueError('Held-out monitoring requires model mode and a nonnegative count')
    if args.evaluate_checkpoint and args.mode != 'model':
        raise ValueError('Checkpoint evaluation requires model mode')
    if args.audit_model and not args.evaluate_checkpoint:
        raise ValueError('Model audit requires an evaluation checkpoint')
    if args.panel_size is not None and args.panel_size < args.count:
        raise ValueError('Panel size must be at least the selected count')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)

    def event(kind, **values):
        record = {'event': kind, 'time': time.time(), **values}
        text = json.dumps(record, allow_nan=False)
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
            torch.cuda.reset_peak_memory_stats(device)
        manifest = open_bundle(args.cache)
        dataset = FrozenDataset(args.cache, manifest, 'train')
        eligible, excluded, raw_entries = [], [], {}
        for index, identifier in enumerate(dataset.ids):
            length = manifest['files'][identifier]['length']
            if not args.min_length <= length <= args.max_length:
                continue
            raw = dataset[index]
            fraction = float(raw['mask'].float().mean())
            if fraction < .9:
                excluded.append({'id': identifier, 'valid_fraction': fraction})
            else:
                eligible.append(identifier)
                raw_entries[identifier] = raw
        panel_ids = select_panel({**manifest, 'train_ids': eligible},
                                 args.panel_size or args.count,
                                 args.min_length, args.max_length)
        ids = panel_ids[:args.count]
        atomic_json(root / 'selection.json', {
            'selected': ids, 'eligible_count': len(eligible),
            'parent_panel': panel_ids,
            'excluded_low_confidence': excluded,
        })
        event('selection', ids=ids, eligible_count=len(eligible), excluded_count=len(excluded))
        generator = torch.Generator().manual_seed(42)
        entries = []
        for identifier in ids:
            raw = raw_entries[identifier]
            entry = calibrated(raw, manifest['scale'])
            entry['coords'] = initialize(entry, args.initialization, generator, args.noise)
            entries.append({k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in entry.items()})
        del raw_entries
        heldout = []
        if args.heldout_count:
            validation = FrozenDataset(args.cache, manifest, 'val')
            candidates = {}
            for index, identifier in enumerate(validation.ids):
                length = manifest['files'][identifier]['length']
                if args.min_length <= length <= args.max_length:
                    raw = validation[index]
                    if float(raw['mask'].float().mean()) >= .9:
                        candidates[identifier] = raw
            heldout_ids = select_panel(
                {**manifest, 'train_ids': list(candidates), 'val_ids': ids},
                args.heldout_count, args.min_length, args.max_length)
            heldout_generator = torch.Generator().manual_seed(43)
            for identifier in heldout_ids:
                entry = calibrated(candidates[identifier], manifest['scale'])
                entry['coords'] = initialize(
                    entry, args.initialization, heldout_generator, args.noise)
                heldout.append({k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in entry.items()})
        weights = {'local_distance': 1., 'bend': .5, 'torsion': .5}
        if args.loss == 'control':
            weights = {}
        if args.mode == 'coordinates':
            weights['angle'] = 0.
        loss_fn = BalancedGeometryLoss(weights)
        atomic_json(root / 'perfect_target_audit.json', perfect_target_audit(entries, loss_fn))
        architecture = {'hidden': 64, 'depth': 4, 'heads': 4}
        model = None
        if args.mode == 'model':
            model = enable_tensor_ff_input_norm(
                build_model(manifest['feature_dim'], architecture)).to(device)
            parameters = list(model.parameters())

            def predict(entry):
                return model_forward(model, entry, args.contacts)
        else:
            parameters = [torch.nn.Parameter(e['coords'].clone()) for e in entries]
            positions = {e['id']: p for e, p in zip(entries, parameters)}

            def predict(entry):
                return positions[entry['id']], torch.zeros_like(entry['angles'][..., :3])

        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate,
                                     weight_decay=.01 if model is not None else 0.)
        provenance = {
            'protocol': 'se3-reconstruction-diagnostic-v1', 'args': vars(args),
            'ids': ids, 'split': 'training only; no generalization claim',
            'heldout_ids': [e['id'] for e in heldout],
            'heldout_scope': 'reused development validation; never optimized or used for selection',
            'bundle_sha256': manifest['bundle_sha256'], 'scale': manifest['scale'],
            'weights': loss_fn.weights, 'architecture': architecture if model else None,
            'sources': {**source_identity(), __file__: digest(__file__)},
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'production_forward_calls': 0, 'seed': 42,
            'noise_definition': 'independent Gaussian standard deviation per Cartesian axis',
            'oracle': args.initialization == 'native-noise' or args.mode == 'coordinates',
        }
        atomic_json(root / 'provenance.json', provenance)
        source_root = root / 'sources'
        for name in source_identity():
            source = Path('foldtree2') / name
            target = source_root / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        target = source_root / 'scripts' / Path(__file__).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(__file__).read_bytes())

        def evaluate(evaluation_entries=None):
            if model is not None:
                model.eval()
            rows = []
            with torch.no_grad():
                for entry in entries if evaluation_entries is None else evaluation_entries:
                    pred, angles = predict(entry)
                    terms = loss_fn.components(pred, angles, entry)
                    rows.append({
                        'id': entry['id'], 'metrics': reconstruction_metrics(pred, angles, entry),
                        'components': {k: float(v) if isinstance(v, torch.Tensor) else v
                                       for k, v in terms.items()},
                    })
            return rows

        initial = evaluate()
        atomic_json(root / 'input_baseline.json', [
            {'id': e['id'], 'metrics': reconstruction_metrics(e['coords'], None, e)}
            for e in entries])
        if heldout:
            atomic_json(root / 'heldout_initial.json', evaluate(heldout))
            atomic_json(root / 'heldout_input_baseline.json', [
                {'id': e['id'], 'metrics': reconstruction_metrics(e['coords'], None, e)}
                for e in heldout])
        atomic_json(root / 'initial.json', initial)
        best = sum(r['components']['total'] for r in initial) / len(initial)
        event('ready', ids=ids, objective=best, device=str(device), mode=args.mode)
        if args.evaluate_checkpoint:
            checkpoint = torch.load(args.evaluate_checkpoint, map_location=device, weights_only=False)
            prior = checkpoint['provenance']
            for key in ('ids', 'bundle_sha256', 'scale', 'weights', 'architecture'):
                if prior[key] != provenance[key]:
                    raise ValueError(f'Evaluation checkpoint mismatch: {key}')
            for key, value in source_identity().items():
                if prior['sources'].get(key) != value:
                    raise ValueError(f'Evaluation checkpoint source mismatch: {key}')
            for key, default in (('initialization', 'seed'), ('noise', 1.),
                                 ('contacts', 'normal')):
                if prior['args'].get(key, default) != getattr(args, key):
                    raise ValueError(f'Evaluation checkpoint input mismatch: {key}')
            model.load_state_dict(checkpoint['model'])
            atomic_json(root / 'checkpoint_training_metrics.json', evaluate())
            if args.audit_model:
                atomic_json(root / 'model_audit.json', audit_model(
                    model, entries, loss_fn, args.contacts))
            if heldout:
                atomic_json(root / 'heldout_best.json', evaluate(heldout))
            atomic_json(root / 'summary.json', {
                'evaluation_only': True, 'checkpoint_sha256': digest(args.evaluate_checkpoint),
                'checkpoint_step': checkpoint['step'], 'heldout_count': len(heldout),
            })
            event('complete', evaluation_only=True, checkpoint_step=checkpoint['step'])
            return
        started = time.monotonic()
        best_step = completed = 0
        rows = initial

        def save(name, step):
            atomic_checkpoint(root / name, {
                'model': model.state_dict() if model is not None else None,
                'coordinates': [p.detach().cpu() for p in parameters] if model is None else None,
                'optimizer': optimizer.state_dict(), 'step': step, 'provenance': provenance,
                'rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                'best_objective': best, 'best_step': best_step,
                'resumable': False,
            })

        save('best.pt', 0)
        atomic_json(root / 'best_metrics.json', initial)
        for step in range(1, args.steps + 1):
            if model is not None:
                model.train()
            objective, norm = optimize_step(parameters, optimizer, loss_fn, predict, entries)
            completed = step
            bounded = time.monotonic() - started >= args.seconds
            if step % args.log_every == 0 or step == args.steps or bounded:
                rows = evaluate()
                current = sum(r['components']['total'] for r in rows) / len(rows)
                if current < best:
                    best, best_step = current, step
                    save('best.pt', step)
                    atomic_json(root / 'best_metrics.json', rows)
                save('last.pt', step)
                atomic_json(root / 'last_metrics.json', rows)
                event('progress', step=step, training_objective=objective,
                      evaluated_objective=current, gradient_norm=norm,
                      seconds=time.monotonic() - started,
                      per_chain=[{'id': r['id'], 'objective': r['components']['total'],
                                  'metrics': r['metrics']} for r in rows],
                      metrics={k: sum(r['metrics'][k] for r in rows) / len(rows)
                               for k in ('fape', 'ca_rmsd', 'tm_score', 'lddt')})
            if bounded:
                break
        atomic_json(root / 'summary.json', {
            'steps': completed, 'best_step': best_step, 'best_objective': best,
            'stop_reason': 'steps' if completed == args.steps else 'time_limit',
            'elapsed_seconds': time.monotonic() - started,
            'initial': initial, 'last': rows,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
        })
        if heldout:
            atomic_json(root / 'heldout_last.json', evaluate(heldout))
            checkpoint = torch.load(root / 'best.pt', map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            atomic_json(root / 'heldout_best.json', evaluate(heldout))
        event('complete', steps=completed, best_step=best_step, production_forward_calls=0)
    except Exception as exc:
        event('failure', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_clariden/bundle_4096')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['coordinates', 'model'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0', 'cuda:1'], required=True)
    parser.add_argument('--initialization', choices=['seed', 'native-noise'], default='seed')
    parser.add_argument('--noise', type=float, default=1.)
    parser.add_argument('--loss', choices=['control', 'combined'], default='combined')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--panel-size', type=int)
    parser.add_argument('--heldout-count', type=int, default=0)
    parser.add_argument('--contacts', choices=['normal', 'local-only'], default='normal')
    parser.add_argument('--evaluate-checkpoint')
    parser.add_argument('--audit-model', action='store_true')
    parser.add_argument('--min-length', type=int, default=64)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--log-every', type=int, default=25)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
