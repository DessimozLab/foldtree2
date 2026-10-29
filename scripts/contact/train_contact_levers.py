"""Resumable distance/contact experiments with ranking and range-aware metrics."""

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
from foldtree2.src.distance.contact_levers import InspectablePairHead, XSAPairHead
from foldtree2.src.distance.contact_lever_metrics import per_protein_range_metrics, summarize_ranges
from foldtree2.src.distance.distance_restraints import PairwiseDistanceHead, distance_labels, eligible_pairs
from foldtree2.train_distance_restraints import evaluate, load_records
from scripts.contact.experiment_contact_objectives import balanced_contact_loss, contact_logit
from scripts.se3.diagnose_se3_reconstruction import atomic_checkpoint, atomic_json


def sample_from(pool, count, generator):
    if not len(pool):
        return pool
    return pool[torch.randint(len(pool), (count,), generator=generator)]


def ranking_loss(scores, labels, contact_bins, groups, hard_fraction=.5):
    """Within each separation range, compare positives with random and hard negatives."""
    terms = []
    for positions in groups:
        if not len(positions):
            continue
        group_scores, positive = scores[positions], labels[positions] < contact_bins
        pos, neg = group_scores[positive], group_scores[~positive]
        if not len(pos) or not len(neg):
            continue
        # Candidates were randomly sampled without using validation labels. The
        # first half provides random negatives; the second supplies hard ones.
        split = len(neg) // 2
        count = max(1, min(len(pos), len(neg)))
        hard_count = min(len(neg) - split, round(count * hard_fraction))
        random_count = min(split, count - hard_count)
        selected = torch.cat([neg[:random_count], neg[split:].topk(hard_count).values])
        if len(selected):
            terms.append(F.softplus(selected[:, None] - pos[None, :]).mean())
    return torch.stack(terms).mean() if terms else scores.sum() * 0.


def batched_forward(model, entries, pairs):
    """Run padded proteins together; split variable numbers of pair logits afterward."""
    pad = torch.nn.utils.rnn.pad_sequence
    mask = pad([e['mask'] for e in entries], batch_first=True, padding_value=False)
    values = [pad([e[k].detach() for e in entries], batch_first=True) for k in ('features', 'contact_z', 'aa_probs')]
    values = [v.masked_fill(~mask[..., None], 0.) for v in values]
    if model.mode == 'sequence':
        values = [torch.zeros_like(v) for v in values]
    nodes = model.node(torch.cat([model.base_norm(values[0]), model.contact_norm(values[1]), model.aa_norm(values[2])], -1))
    nodes = nodes.masked_fill(~mask[..., None], 0.)
    length, batch = mask.shape[1], mask.shape[0]
    positions = torch.arange(length, device=mask.device)
    offsets = (positions[None] - positions[:, None]).clamp(-32, 32) + 32
    for layer in model.context:
        if layer.contact_scale is not None:
            raise ValueError('Batched training does not support contact-biased attention')
        bias = layer.relative_bias(offsets).permute(2, 0, 1)[None].expand(batch, -1, -1, -1)
        bias = bias.masked_fill(~mask[:, None, None, :], -torch.inf).reshape(batch * model.context_heads, length, length)
        normalized = layer.norm(nodes)
        attended, _ = layer.attention(normalized, normalized, normalized, attn_mask=bias, need_weights=False)
        nodes = nodes + layer.drop(attended)
        nodes = (nodes + layer.ff(layer.ff_norm(nodes))).masked_fill(~mask[..., None], 0.)
    nodes = model.context_norm(nodes).masked_fill(~mask[..., None], 0.)
    sizes = [len(p) for p in pairs]
    all_pairs = torch.cat(pairs)
    owner = torch.repeat_interleave(torch.arange(batch, device=mask.device), torch.tensor(sizes, device=mask.device))
    i, j = all_pairs.T
    a, b = nodes[owner, i], nodes[owner, j]
    separation = (j - i).abs().to(nodes.dtype).log1p()[:, None]
    contact = (values[1][owner, i] * values[1][owner, j]).sum(-1, keepdim=True)
    features = torch.cat([a + b, (a - b).abs(), a * b, separation, contact], -1)
    logits = model.pair(features)
    correction = None
    if model.spherical_dim:
        unit = F.normalize(model.contact_projection(nodes), dim=-1, eps=1e-6)
        scale = model.contact_log_scale.clamp(-5., math.log(100.)).exp()
        dot = scale * (unit[owner, i] * unit[owner, j]).sum(-1)
        correction = model.pair_bias(features).squeeze(-1)
        score = dot + model.contact_offset + correction
        logits = torch.cat([F.log_softmax(logits[:, :model.contact_bins], -1) + F.logsigmoid(score)[:, None],
                            F.log_softmax(logits[:, model.contact_bins:], -1) + F.logsigmoid(-score)[:, None]], -1)
    return logits.split(sizes), correction.split(sizes) if correction is not None else None


def range_evaluate(model, records, device):
    model.eval()
    rows = []
    totals = torch.zeros(6, dtype=torch.float64)
    with torch.no_grad():
        for entry, pairs in records:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in entry.items()}
            pairs = pairs.to(device)
            if isinstance(model, InspectablePairHead):
                logits, components = model(inputs, pairs, return_components=True)
                dot, correction = components['dot'].double(), components['correction'].double()
                totals += torch.stack([dot.new_tensor(len(dot)), dot.sum(), correction.sum(), dot.square().sum(),
                                       correction.square().sum(), (dot * correction).sum()]).cpu()
            else:
                logits = model(inputs, pairs)
            labels, _ = distance_labels(inputs['target'], pairs, model.edges)
            probabilities = model.contact_probability(logits.softmax(-1))
            rows.append({'id': entry['id'], 'ranges': per_protein_range_metrics(
                probabilities, labels < model.contact_bins, pairs, int(inputs['mask'].sum()))})
    diagnostics = {}
    if totals[0]:
        n, sx, sy, sx2, sy2, sxy = totals.tolist()
        vx, vy = max(0., sx2 / n - (sx / n)**2), max(0., sy2 / n - (sy / n)**2)
        diagnostics = {'dot_mean': sx / n, 'correction_mean': sy / n,
                       'dot_std': math.sqrt(vx), 'correction_std': math.sqrt(vy),
                       'dot_correction_correlation': ((sxy / n - sx * sy / n**2) / math.sqrt(vx * vy)) if vx * vy > 0 else None}
    return summarize_ranges([r['ranges'] for r in rows]), diagnostics, rows


def source_hashes(optimizer='adamw'):
    paths = [__file__, 'foldtree2/src/distance/contact_levers.py', 'foldtree2/src/distance/contact_lever_metrics.py', 'foldtree2/src/models/xsatransformer.py',
             'foldtree2/src/distance/distance_restraints.py', 'foldtree2/train_distance_restraints.py',
             'foldtree2/se3_cache.py', 'scripts/contact/experiment_contact_objectives.py']
    if optimizer == 'muon':
        import muon
        paths.append(muon.__file__)
    return {str(p): digest(p) for p in paths}


def make_optimizer(model, args):
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.05)
    from muon import SingleDeviceMuonWithAuxAdam
    hidden = [p for name, p in model.named_parameters()
              if name.startswith('context.') and p.ndim == 2 and not name.endswith('relative_bias.weight')]
    selected = {id(p) for p in hidden}
    other = [p for p in model.parameters() if id(p) not in selected]
    if not hidden or not other:
        raise ValueError('Muon requires hidden matrices and auxiliary parameters')
    return SingleDeviceMuonWithAuxAdam([
        dict(params=hidden, use_muon=True, lr=args.muon_lr, weight_decay=.01),
        dict(params=other, use_muon=False, lr=args.learning_rate, betas=(.9, .95), weight_decay=.01),
    ])


def training_records(args, manifest):
    if args.train_selection == 'prefix':
        return load_records(args.cache, manifest, 'train', args.train_count)
    dataset = FrozenDataset(args.cache, manifest, 'train')
    if not 1 <= args.train_count <= len(dataset):
        raise ValueError('Requested subset count outside cache split')
    ordered = sorted(range(len(dataset)), key=lambda i: (manifest['files'][dataset.ids[i]]['length'], dataset.ids[i]))
    generator = torch.Generator().manual_seed(args.seed + 100)
    records = []
    for group in range(args.train_count):
        lo, hi = group * len(ordered) // args.train_count, (group + 1) * len(ordered) // args.train_count
        index = ordered[lo + int(torch.randint(hi - lo, (1,), generator=generator))]
        entry = dataset[index]
        records.append((entry, eligible_pairs(entry['mask'])))
    return records


def train(args):
    if (args.epochs < 1 or args.batch_size < 1 or args.pairs < 2 or args.ranking_weight < 0 or args.correction_l2 < 0 or args.patience < 0
            or args.context_layers < 1 or args.context_heads < 1 or args.hidden % args.context_heads
            or args.learning_rate <= 0 or args.muon_lr <= 0
            or any(not math.isfinite(v) for v in (args.ranking_weight, args.correction_l2, args.contact_weight, args.learning_rate, args.muon_lr, args.min_delta))):
        raise ValueError('Invalid training budget or loss weights')
    root = Path(args.output)
    if args.resume and not (root / 'resume.pt').exists():
        raise ValueError('Resume requires an existing resume.pt')
    root.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    manifest = open_bundle(args.cache)
    records = {'train': training_records(args, manifest), 'val': load_records(args.cache, manifest, 'val', args.val_count)}
    settings = {k: v for k, v in vars(args).items() if k not in ('resume', 'stop_after_epoch')}
    provenance = {'protocol': 'contact-levers-v1', 'args': settings, 'sources': source_hashes(args.optimizer),
                  'bundle_sha256': manifest['bundle_sha256'],
                  'selection': {s: [e['id'] for e, _ in rows] for s, rows in records.items()},
                  'scope': 'Reused development validation; production pretraining overlap unverified',
                  'selection_rule': 'Maximum mean per-protein long-range AP; patience in epochs',
                  'contact_definition': 'CA distance <8A; medium separation 12–23; long separation >=24',
                  'sampling': 'Uniform distance pairs; balanced medium/long positive and negative auxiliary pairs; hard negatives selected within sampled training candidates'}
    if args.resume:
        old = json.loads((root / 'provenance.json').read_text())
        if old != provenance:
            raise ValueError('Resume rejected: source, data, split, device or configuration differs')
    else:
        atomic_json(root / 'provenance.json', provenance)
        for index, path in enumerate(provenance['sources']):
            (root / f'source-{index}-{Path(path).name}').write_bytes(Path(path).read_bytes())
    def event(kind, **values):
        row = {'event': kind, 'time': time.time(), **values}
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)
    edges = torch.tensor((4., 6., 8., 10., 12., 16., 20., 24., 32.))
    histogram = torch.ones(10, dtype=torch.float64)
    pools = []
    for entry, pairs in records['train']:
        labels, _ = distance_labels(entry['target'], pairs, edges)
        histogram += torch.bincount(labels, minlength=10)
        sep = (pairs[:, 1] - pairs[:, 0]).abs()
        pools.append([((labels < 3) & mask).nonzero().flatten() for mask in ((sep < 24), (sep >= 24))] +
                     [((labels >= 3) & mask).nonzero().flatten() for mask in ((sep < 24), (sep >= 24))])
    prior = (histogram / histogram.sum()).float().to(device)
    cls = {'baseline': PairwiseDistanceHead, 'pair': InspectablePairHead, 'pair_xsa': XSAPairHead}[args.model]
    kwargs = dict(feature_dim=manifest['feature_dim'], contact_dim=manifest['decoder_enrichment']['contact_dim'],
                  hidden=args.hidden, context_layers=args.context_layers, context_heads=args.context_heads, dropout=.1,
                  spherical_dim=0 if args.model == 'baseline' else 64)
    if args.model != 'baseline':
        kwargs['correction_hidden'] = args.correction_hidden
    model = cls(**kwargs).to(device)
    optimizer = make_optimizer(model, args)
    torch.manual_seed(args.seed + 2)
    uniform_rng = torch.Generator().manual_seed(args.seed)
    auxiliary_rng = torch.Generator().manual_seed(args.seed + 1)
    hard_rng = torch.Generator().manual_seed(args.seed + 2)
    start, best, bad, history = 0, -1., 0, []
    if args.resume:
        saved = torch.load(root / 'resume.pt', map_location='cpu', weights_only=False)
        if saved['provenance'] != provenance:
            raise ValueError('Checkpoint provenance mismatch')
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng_cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(saved['rng_cuda'], device)
        uniform_rng.set_state(saved['uniform_rng']); auxiliary_rng.set_state(saved['auxiliary_rng']); hard_rng.set_state(saved['hard_rng'])
        start, best, bad, history = saved['epoch'], saved['best'], saved['bad'], saved['history']
        atomic_json(root / 'history.json', history)
        if start >= args.epochs:
            if (root / 'completion.json').exists():
                return
            raise ValueError('Checkpoint already reached its configured epoch budget')
    event('resume' if args.resume else 'ready', epoch=start, updates=start * math.ceil(len(records['train']) / args.batch_size))
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        losses = []
        order = torch.randperm(len(records['train']), generator=uniform_rng)
        indices = order.tolist()
        for batch_start in range(0, len(indices), args.batch_size):
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            inputs, choices, targets, uniforms, all_groups = [], [], [], [], []
            for index in batch_indices:
                entry, all_pairs = records['train'][index]
                uniform = torch.randperm(len(all_pairs), generator=uniform_rng)[:args.pairs]
                selected, groups = [uniform], []
                if args.model != 'baseline':
                    pos_med, pos_long, neg_med, neg_long = pools[index]
                    for pos, neg in ((pos_med, neg_med), (pos_long, neg_long)):
                        start_idx = sum(len(x) for x in selected)
                        p = sample_from(pos, 64, auxiliary_rng)
                        n_random = sample_from(neg, 64, auxiliary_rng)
                        n_candidates = sample_from(neg, 64, hard_rng)
                        selected.extend([p, n_random, n_candidates])
                        groups.append(torch.arange(start_idx, start_idx + len(p) + len(n_random) + len(n_candidates), device=device))
                chosen = all_pairs[torch.cat(selected)]
                labels, _ = distance_labels(entry['target'], chosen, edges)
                inputs.append({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in entry.items()})
                choices.append(chosen.to(device)); targets.append(labels.to(device))
                uniforms.append(len(uniform)); all_groups.append(groups)
            optimizer.zero_grad(set_to_none=True)
            batch_logits, corrections = batched_forward(model, inputs, choices)
            batch_losses = []
            for n, (logits, labels, uniform_count, groups) in enumerate(zip(batch_logits, targets, uniforms, all_groups)):
                ce = F.cross_entropy(logits[:uniform_count], labels[:uniform_count])
                if args.model != 'baseline':
                    aux = balanced_contact_loss(logits[uniform_count:], labels[uniform_count:], model.contact_bins)
                    rank = ranking_loss(contact_logit(logits, model.contact_bins), labels, model.contact_bins, groups)
                    penalty = corrections[n][:uniform_count].square().mean()
                    loss = ce + args.contact_weight * aux + args.ranking_weight * rank + args.correction_l2 * penalty
                else:
                    aux = rank = penalty = ce.new_zeros(())
                    loss = ce
                batch_losses.append(torch.stack((loss, ce, aux, rank, penalty)))
            batch_mean = torch.stack(batch_losses).mean(0)
            if not torch.isfinite(batch_mean[0]):
                raise FloatingPointError('Nonfinite training loss')
            batch_mean[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.extend(x.detach().cpu().tolist() for x in batch_losses)
            consumed = batch_start + len(batch_indices)
            if consumed % 256 == 0 or consumed == len(indices):
                event('update', epoch=epoch, updates=(epoch - 1) * math.ceil(len(indices) / args.batch_size) + math.ceil(consumed / args.batch_size),
                      proteins_seen=(epoch - 1) * len(indices) + consumed,
                      loss=losses[-1][0], distance_ce=losses[-1][1], ranking=losses[-1][3])
        report, _ = evaluate(model, records['val'], device, prior)
        ranges, components, per_chain = range_evaluate(model, records['val'], device)
        score = ranges['long']['ap']
        if score is None:
            raise ValueError('Validation split has no long-range positive contacts')
        improved = score > best + args.min_delta
        best, bad = (score, 0) if improved else (best, bad + 1)
        vals = torch.tensor(losses).mean(0).tolist()
        row = {'epoch': epoch, 'loss': vals[0], 'distance_ce': vals[1], 'contact_loss': vals[2],
               'ranking_loss': vals[3], 'correction_penalty': vals[4],
               'validation': {**report['aggregate'], 'ranges': ranges, 'components': components},
               'best_long_range_macro_ap': best, 'patience_used': bad}
        history.append(row)
        if source_hashes(args.optimizer) != provenance['sources']:
            raise RuntimeError('Source files changed during training')
        saved = {'config': model.model_config(), 'model_class': cls.__name__, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'history': history, 'best': best, 'bad': bad,
                 'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                 'uniform_rng': uniform_rng.get_state(), 'auxiliary_rng': auxiliary_rng.get_state(),
                 'hard_rng': hard_rng.get_state(), 'provenance': provenance}
        atomic_checkpoint(root / 'resume.pt', saved)
        atomic_checkpoint(root / 'last.pt', {'config': model.model_config(), 'model_class': cls.__name__,
                                            'state_dict': model.state_dict(), 'epoch': epoch})
        if improved:
            atomic_checkpoint(root / 'best.pt', {'config': model.model_config(), 'model_class': cls.__name__,
                                                'state_dict': model.state_dict(), 'epoch': epoch})
            atomic_json(root / 'best_validation.json', {'aggregate': row['validation'], 'per_chain_ranges': per_chain})
        atomic_json(root / 'history.json', history)
        event('epoch', **row)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            event('paused', epoch=epoch, updates=epoch * math.ceil(len(records['train']) / args.batch_size))
            return
        if args.patience and bad >= args.patience:
            break
    final_epoch = history[-1]['epoch']
    loaded = torch.load(root / 'last.pt', map_location=device, weights_only=False)
    restored = cls(**loaded['config']).to(device); restored.load_state_dict(loaded['state_dict'])
    checked, _ = evaluate(restored, records['val'], device, prior)
    if checked['aggregate'] != report['aggregate']:
        raise RuntimeError('Checkpoint reload changed validation metrics')
    atomic_json(root / 'completion.json', {'epochs': final_epoch, 'best_long_range_macro_ap': best,
                'validation': history[-1]['validation'], 'reload_exact': True})
    event('complete', epoch=final_epoch, updates=final_epoch * math.ceil(len(records['train']) / args.batch_size))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', default='runs/se3_distance_data_scaling_10240_20261024/enriched')
    p.add_argument('--output', required=True); p.add_argument('--model', choices=['baseline', 'pair', 'pair_xsa'], required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--train-count', type=int, default=9988)
    p.add_argument('--train-selection', choices=['prefix', 'stratified'], default='prefix')
    p.add_argument('--val-count', type=int, default=256); p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--pairs', type=int, default=512); p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--context-layers', type=int, default=2); p.add_argument('--context-heads', type=int, default=8)
    p.add_argument('--optimizer', choices=['adamw', 'muon'], default='adamw'); p.add_argument('--muon-lr', type=float, default=.02)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--contact-weight', type=float, default=.3); p.add_argument('--ranking-weight', type=float, default=0.)
    p.add_argument('--correction-hidden', type=int, default=32); p.add_argument('--correction-l2', type=float, default=0.)
    p.add_argument('--patience', type=int, default=3); p.add_argument('--min-delta', type=float, default=.0001)
    p.add_argument('--resume', action='store_true'); p.add_argument('--stop-after-epoch', type=int)
    train(p.parse_args())


if __name__ == '__main__':
    main()
