"""Bounded frozen-feature distance-head comparison; no SE(3) weights are changed."""
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
from foldtree2.src.distance.distance_restraints import PairwiseDistanceHead, distance_labels, eligible_pairs
from foldtree2.src.losses.losses import angles_reconstruction_loss
from scripts.se3.diagnose_se3_reconstruction import atomic_checkpoint, atomic_json


def sources():
    names = ('foldtree2/train_distance_restraints.py', 'foldtree2/src/distance/distance_restraints.py',
             'foldtree2/se3_cache.py', 'scripts/se3/diagnose_se3_reconstruction.py')
    return {name: digest(name) for name in names}


def load_records(cache, manifest, split, count=None, min_separation=12):
    dataset = FrozenDataset(cache, manifest, split)
    if count is not None and (count < 1 or count > len(dataset)):
        raise ValueError('Requested subset count outside cache split')
    records = []
    for index in range(len(dataset) if count is None else count):
        entry = dataset[index]
        pairs = eligible_pairs(entry['mask'], min_separation=min_separation)
        records.append((entry, pairs))
    return records


def contact_metrics(probabilities, labels, top_count):
    truth = labels.bool()
    order = probabilities.argsort(descending=True, stable=True)
    ranked = truth[order].float()
    true_count = int(truth.sum())
    predicted = probabilities >= .5
    shared = int((predicted & truth).sum())
    npred = int(predicted.sum())
    k = min(top_count, len(truth))
    precision = ranked.cumsum(0) / torch.arange(
        1, len(truth) + 1, device=truth.device)
    scores = probabilities[order]
    ends = torch.cat([(scores[:-1] != scores[1:]).nonzero().flatten(),
                      scores.new_tensor([len(scores) - 1], dtype=torch.long)])
    cumulative = ranked.cumsum(0)[ends]
    increments = cumulative - torch.cat([cumulative.new_zeros(1), cumulative[:-1]])
    return {
        'contacts_true': true_count, 'contacts_predicted': npred,
        'contact_precision': shared / npred if npred else None,
        'contact_recall': shared / true_count if true_count else None,
        'contact_average_precision': (
            float((precision[ends] * increments).sum() / true_count) if true_count else None),
        'contact_brier': float((probabilities - truth.float()).square().mean()),
        'top_L_count': k, 'top_L_precision': float(ranked[:k].mean()),
        'top_L_recall': float(ranked[:k].sum()) / true_count if true_count else None,
    }


def pair_metrics(model, probabilities, labels, distances, top_count, prior):
    contact = model.contact_probability(probabilities)
    capped = model.capped_distance(probabilities)
    nll = F.nll_loss(probabilities.clamp_min(1e-30).log(), labels)
    return {
        'pairs': len(labels), 'nll': float(nll),
        'empirical_prior_nll': float(F.nll_loss(prior.log().expand(len(labels), -1), labels)),
        'bin_accuracy': float((probabilities.argmax(-1) == labels).float().mean()),
        'capped_distance_mae_angstrom': float(
            (capped - distances.clamp_max(float(model.edges[-1]))).abs().mean()),
        **contact_metrics(contact, distances < 8., top_count),
    }


def focal_ce(logits, labels, gamma=0., label_smoothing=0., reduction='mean'):
    """Focal cross-entropy: (1 - p_t)^gamma * CE, gamma=0 reduces to CE."""
    logp = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0.:
        n_bins = logits.shape[-1]
        smooth = (1. - label_smoothing) * F.one_hot(labels, n_bins) + label_smoothing / n_bins
        loss = -(smooth * logp).sum(-1)
    else:
        loss = F.nll_loss(logp, labels, reduction='none')
    if gamma > 0.:
        pt = logp.gather(-1, labels[:, None]).exp().squeeze(-1)
        weights = (1. - pt).pow(gamma).detach().clamp_min(1e-8)
        loss = loss * weights
    return loss.mean() if reduction == 'mean' else loss


def focal_bce(logits, labels, gamma=0., reduction='mean'):
    """Focal binary CE on contact (distance < edges[bins]) vs far logits."""
    labels = labels.long()
    pt = torch.sigmoid(logits)
    pt = torch.where(labels == 1, pt, 1. - pt)
    if gamma > 0.:
        weights = (1. - pt).pow(gamma).detach().clamp_min(1e-8)
    else:
        weights = None
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
    if weights is not None:
        loss = loss * weights
    return loss.mean() if reduction == 'mean' else loss


def sample_pairs(entry, all_pairs, count, generator, balanced_ratio=0.0, contact_cutoff=8.):
    """Subsample pairs uniformly, or with a fixed contact fraction.

    ``balanced_ratio`` specifies the fraction of the batch that should be drawn
    from the contact set (with replacement).  The remaining batch is drawn from
    the non-contact set.  Default 0 means uniform draws (the natural contact
    frequency, typically ~2%).  A value of 0.5 gives equal contact and non-
    contact representation.  Both classes are sampled with replacement so the
    exact ratio is maintained regardless of the relative class sizes.
    """
    if balanced_ratio <= 0. or count < 2:
        indices = torch.randperm(len(all_pairs), generator=generator)[:count]
        return all_pairs[indices]
    target = entry['target']
    distances = (target[all_pairs[:, 0]] - target[all_pairs[:, 1]]).norm(dim=-1)
    contact = distances < contact_cutoff
    contact_idx = contact.nonzero().view(-1)
    noncontact_idx = (~contact).nonzero().view(-1)
    contact_half = min(count, max(0, min(count, int(round(count * balanced_ratio)))))
    noncontact_count = count - contact_half
    if len(contact_idx) == 0 or len(noncontact_idx) == 0 or contact_half < 1 or noncontact_count < 1:
        indices = torch.randperm(len(all_pairs), generator=generator)[:count]
        return all_pairs[indices]
    chosen = torch.cat([
        contact_idx[torch.randint(len(contact_idx), (contact_half,), generator=generator)],
        noncontact_idx[torch.randint(len(noncontact_idx), (noncontact_count,), generator=generator)],
    ])
    return all_pairs[chosen[torch.randperm(count, generator=generator)]]


def make_optimizer(model, args):
    """Build AdamW (default) or Muon with auxiliary AdamW."""
    if getattr(args, 'optimizer', 'adamw') == 'muon':
        from muon import SingleDeviceMuonWithAuxAdam
        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Muon applies orthogonalized momentum to 2D weight matrices
            # (the "hidden layers").  1D biases, norms, embeddings and the
            # final output layer use a small AdamW auxiliary.
            if param.ndim == 2 and 'pair.output' not in name and 'kendall' not in name:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        param_groups = [
            {'params': muon_params, 'use_muon': True,
             'lr': float(args.learning_rate)},
            {'params': adamw_params, 'use_muon': False,
             'lr': 3e-4, 'weight_decay': 0.},
        ]
        return SingleDeviceMuonWithAuxAdam(param_groups)
    return torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def evaluate(model, records, device, prior, export=False):
    model.eval()
    rows, exports = [], []
    all_probs, all_labels, all_distances = [], [], []
    with torch.no_grad():
        for entry, pairs in records:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in entry.items()}
            pairs = pairs.to(device)
            labels, distances = distance_labels(inputs['target'], pairs, model.edges)
            probs = model(inputs, pairs).softmax(-1)
            row = {'id': entry['id'], 'metrics': pair_metrics(
                model, probs, labels, distances, int(inputs['mask'].sum()), prior)}
            if 'contact_probs' in inputs:
                row['frozen_decoder_ca_contact_audit'] = contact_metrics(
                    inputs['contact_probs'][pairs[:, 0], pairs[:, 1]], distances < 8.,
                    int(inputs['mask'].sum()))
            rows.append(row)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_distances.append(distances.cpu())
            if export:
                exports.append({'id': entry['id'], 'pairs': pairs.cpu(),
                                'probabilities': probs.cpu(), 'mask': entry['mask'].cpu()})
    probs, labels, distances = torch.cat(all_probs), torch.cat(all_labels), torch.cat(all_distances)
    # The global ranking budget is the sum of per-chain valid lengths.
    total_length = sum(int(e['mask'].sum()) for e, _ in records)
    aggregate = pair_metrics(model, probs, labels, distances, total_length, prior.cpu())
    aggregate['macro_chain_nll'] = sum(r['metrics']['nll'] for r in rows) / len(rows)
    aggregate['chains'] = len(rows)
    return {'aggregate': aggregate, 'per_chain': rows}, exports


def train(args):
    if (args.epochs < 1 or args.pairs < 1 or args.hidden < 1 or args.seconds <= 0
            or not math.isfinite(args.seconds) or args.learning_rate <= 0
            or not math.isfinite(args.learning_rate) or not args.modes
            or len(set(args.modes)) != len(args.modes)
            or set(args.modes) - {'sequence', 'decoder'}):
        raise ValueError('Invalid distance-head experiment configuration')
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
        if 'decoder_enrichment' not in manifest:
            raise ValueError('Requires versioned AA/contact enriched cache')
        min_separation = max(1, int(getattr(args, 'min_separation', 12)))
        if min_separation < 1 or min_separation > 255:
            raise ValueError('min_separation must be between 1 and 255')
        records = {
            split: load_records(args.cache, manifest, split,
                               getattr(args, split + '_count'),
                               min_separation=min_separation)
            for split in ('train', 'val')}
        identity = sources()
        selection = {s: [e['id'] for e, _ in rows] for s, rows in records.items()}
        if set(selection['train']) & set(selection['val']):
            raise ValueError('Training and validation IDs overlap')
        balanced_ratio = max(0., min(1., float(getattr(args, 'balanced_ratio', 0.))))
        focal_gamma = max(0., float(getattr(args, 'focal_gamma', 0.)))
        label_smoothing = max(0., min(1., float(getattr(args, 'label_smoothing', 0.))))
        kendall_terms = tuple(str(name) for name in getattr(args, 'kendall_terms', []) or [])
        if kendall_terms and len(set(kendall_terms)) != len(kendall_terms):
            raise ValueError('Duplicate Kendall term names')
        kl_penalty = float(getattr(args, 'kl_penalty', 0.))
        if kl_penalty < 0. or not math.isfinite(kl_penalty):
            raise ValueError('KL penalty must be a nonnegative finite float')
        atomic_json(root / 'provenance.json', {
            'protocol': 'frozen-pair-distance-v1', 'args': vars(args),
            'sources': identity, 'bundle_sha256': manifest['bundle_sha256'],
            'selection': selection, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'inputs': 'symmetric continuous/codebook + contact embeddings + AA distributions + '
                      'contact cosine + log row separation; no coordinate inputs',
            'residue_context': {
                'layers': getattr(args, 'context_layers', 0),
                'heads': getattr(args, 'context_heads', 8),
                'position': 'learned signed relative row offsets clipped to [-32,32]',
                'contact_bias': bool(getattr(args, 'contact_bias', False)),
                'contact_bias_source': getattr(args, 'contact_bias_source', 'cosine'),
                'mask': 'invalid residues excluded as attention keys and supervised endpoints',
                'dropout': 0.,
            },
            'pair_representation': {
                'bilinear_dim': getattr(args, 'pair_bilinear_dim', 0),
                'symmetric': 'outer product symmetrized by averaging both orderings',
            },
            'contact_head': {
                'spherical_dim': getattr(args, 'spherical_dim', 0),
                'hierarchical': 'sphere fixes contact mass; pair MLP conditions distance on contact',
            },
            'kl_penalty': {'strength': kl_penalty,
                'kl_batch': getattr(args, 'kl_batch', 0),
                'note': 'KL(marginal || empirical_prior) on detached large-pair sample',
                'source': 'empirical distance prior from training chains'} if kl_penalty > 0. else None,
            'pair_sampling': (f'balanced {balanced_ratio:.2f} contact fraction (with replacement per class)'
                              if balanced_ratio > 0. else 'uniform draws'),
            'chains_per_step': max(1, int(getattr(args, 'chain_batch', 1))),
            'loss_terms': {
                'focal_gamma': focal_gamma,
                'kendall_terms': list(kendall_terms),
                'angle_head': getattr(args, 'predict_angles', False),
                'note': ('focal CE on contact+far binary AND 10-bin distance, Kendall-uncertainty '
                         'weighted' if kendall_terms else
                         'focal CE on 10-bin distance' if focal_gamma > 0. else
                         'plain 10-bin distance CE') + (
                     ', + angle head' if getattr(args, 'predict_angles', False) else ''),
            },
            'lr_schedule': 'cosine-to-zero over the epoch budget' if args.cosine_lr else 'constant',
            'regularization': {'dropout': getattr(args, 'dropout', 0.),
                               'label_smoothing': label_smoothing,
                               'min_separation': min_separation,
                               'weight_decay': args.weight_decay,
                               'early_stop_patience': args.early_stop,
                               'best_checkpoint': 'selection score = val NLL - 2 x contact AP'},
            'scope': 'Reused refiner development holdout; production pretraining independence unverified',
            'cache_scope': manifest.get('scope'),
            'objective': (f'distance-bin cross entropy on balanced {balanced_ratio:.2f} contact-fraction pairs'
              if balanced_ratio > 0.
              else 'unweighted distance-bin cross entropy, uniformly sampled nonlocal pairs'),
            'selection_metric': 'none; final fixed-budget weights, not validation-selected',
            'contact_definition': 'CA distance <8A; frozen decoder trained on CB contacts',
            'capped_mae': 'midpoint expectation with overflow represented by 32A; native distances capped at 32A',
            'snapshot_resumable': False, 'production_forward_calls': 0,
        })
        for name in identity:
            target = root / 'sources' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
        summaries = {}
        for mode in args.modes:
            if sources() != identity:
                raise ValueError('Distance-head implementation changed')
            output = root / mode
            output.mkdir()
            torch.manual_seed(42)
            model = PairwiseDistanceHead(
                manifest['feature_dim'], manifest['decoder_enrichment']['contact_dim'],
                hidden=args.hidden, mode=mode,
                context_layers=getattr(args, 'context_layers', 0),
                context_heads=getattr(args, 'context_heads', 8),
                contact_bias=getattr(args, 'contact_bias', False),
                contact_bias_source=getattr(args, 'contact_bias_source', 'cosine'),
                dropout=getattr(args, 'dropout', 0.),
                spherical_dim=getattr(args, 'spherical_dim', 0),
                pair_bilinear_dim=getattr(args, 'pair_bilinear_dim', 0),
                pair_layers=getattr(args, 'pair_layers', 2),
                kendall_terms=kendall_terms,
                predict_angles=getattr(args, 'predict_angles', False)).to(device)
            histogram = torch.ones(len(model.edges) + 1, dtype=torch.float64)
            for entry, pairs in records['train']:
                labels, _ = distance_labels(entry['target'], pairs, model.edges.cpu())
                histogram += torch.bincount(labels, minlength=len(histogram))
            prior = (histogram / histogram.sum()).float().to(device)
            optimizer = make_optimizer(model, args)
            schedule = None
            if args.cosine_lr:
                total_steps = max(1, args.epochs * max(len(records['train']), 1))
                schedule = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda step: 0.5 * (1. + math.cos(math.pi * min(step / total_steps, 1.))))
            sampling = torch.Generator().manual_seed(42)
            initial, _ = evaluate(model, records['val'], device, prior)
            atomic_json(output / 'initial_validation.json', initial)
            history, updates, visits = [], 0, 0
            started = time.monotonic()
            event('ready', mode=mode, train=len(records['train']), val=len(records['val']),
                  parameters=sum(p.numel() for p in model.parameters()))

            def score(report):
                """Combined selection score: lower distance NLL and higher contact AP is better."""
                aggregate = report['aggregate']
                return aggregate['nll'] - 2. * aggregate['contact_average_precision']

            best = {'score': float('inf'), 'epoch': 0}
            exhausted = False
            chains_per_step = max(1, int(getattr(args, 'chain_batch', 1)))
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses, norms = [], []
                order = torch.randperm(len(records['train']), generator=sampling)
                for position, index in enumerate(order.tolist(), start=1):
                    # Start a new optimizer group — clear old gradients.
                    if (position - 1) % chains_per_step == 0:
                        optimizer.zero_grad(set_to_none=True)
                    entry, all_pairs = records['train'][index]
                    pairs = sample_pairs(entry, all_pairs, args.pairs, sampling,
                             balanced_ratio=balanced_ratio).to(device)
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in entry.items()}
                    labels, _ = distance_labels(inputs['target'], pairs, model.edges)
                    logits = model(inputs, pairs)
                    if kendall_terms and hasattr(model, 'kendall_log_vars'):
                        # Two supervised terms over the same pair sample:
                        #  - contact: marginal probability that the distance is
                        #    below the 8A contact boundary (binary),
                        #  - distance: full 10-bin distance classification.
                        # Both use focal weights when focal_gamma > 0.  Kendall
                        # log-variances learn the trade-off between the two so
                        # the rare contact signal cannot be drowned out while
                        # the natural batch distribution stays intact.
                        contact_edges = int((model.edges <= 8.).sum())
                        near_logit = torch.logsumexp(logits[:, :contact_edges], dim=-1)
                        far_logit = torch.logsumexp(logits[:, contact_edges:], dim=-1)
                        contact_target = (labels < contact_edges).float()
                        term_contact = focal_bce(
                            near_logit - far_logit, contact_target, gamma=focal_gamma)
                        term_distance = focal_ce(logits, labels, gamma=focal_gamma,
                                         label_smoothing=label_smoothing)
                        terms = {'contact': term_contact, 'distance': term_distance}
                        if 'angle' in kendall_terms and getattr(model, 'predict_angles', False):
                            angle_pred = model.predict_angles_from_entry(inputs)
                            term_angle = angles_reconstruction_loss(
                                inputs['angles'], angle_pred,
                                plddt_mask=inputs['mask'].float().unsqueeze(-1),
                                plddt_thresh=0.5, kl_penalty=False)
                            terms['angle'] = term_angle
                        loss = torch.stack([
                            torch.exp(-model.kendall_log_vars[name]) * term + model.kendall_log_vars[name]
                            for name, term in terms.items()]).sum()
                    else:
                        loss = focal_ce(logits, labels, gamma=focal_gamma,
                                label_smoothing=label_smoothing) if focal_gamma > 0. \
                            else (F.cross_entropy(logits, labels) if label_smoothing <= 0.
                                  else focal_ce(logits, labels, label_smoothing=label_smoothing))
                    # KL penalty from a separate large pair sample so the
                    # marginal estimate is low-noise (512 pairs with ~1.5%
                    # contacts is too small for a stable KL).  The marginal
                    # uses logits from the same inputs so it reflects the
                    # current model state; no gradients through the extraction.
                    if kl_penalty > 0. and len(all_pairs) >= args.kl_batch:
                        kl_pairs = all_pairs[torch.randperm(len(all_pairs), generator=sampling)[:args.kl_batch]].to(device)
                        kl_labels, _ = distance_labels(inputs['target'], kl_pairs, model.edges)
                        with torch.no_grad():
                            marginal = model(inputs, kl_pairs).softmax(-1).mean(0)
                        kl = (marginal * (marginal / prior).log()).clamp_min(0.).sum()
                        loss = loss + kl_penalty * kl
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite distance-head objective')
                    # Gradient accumulation: average over the chains in one
                    # optimizer step so the rare contact signal from several
                    # structures is combined before stepping. 512 pairs from a
                    # single chain contain only ~8 contacts at the natural
                    # frequency, which is too weak a per-step gradient.
                    (loss / chains_per_step).backward()
                    losses.append(float(loss.detach()))
                    visits += len(pairs)
                    if position % chains_per_step == 0 or position == len(order):
                        norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1., error_if_nonfinite=True)
                        if norm <= 0:
                            raise RuntimeError('Zero distance-head gradient')
                        optimizer.step()
                        if schedule is not None:
                            schedule.step()
                        norms.append(float(norm))
                        updates += 1
                        if updates % 64 == 0:
                            event('update', mode=mode, epoch=epoch, updates=updates,
                                  loss=float(loss.detach()), gradient_norm=float(norm))
                    if time.monotonic() - started >= args.seconds:
                        exhausted = True
                        break
                if time.monotonic() - started >= args.seconds:
                    exhausted = True
                report, _ = evaluate(model, records['val'], device, prior)
                atomic_json(output / f'validation-{epoch:03d}.json', report)
                row = {'epoch': epoch, 'updates': updates, 'pair_visits': visits,
                       'elapsed_seconds': time.monotonic() - started,
                       'train_nll': sum(losses) / len(losses),
                       'mean_gradient_norm': sum(norms) / len(norms),
                       'epoch_chains': len(losses), 'validation': report['aggregate']}
                history.append(row)
                atomic_json(output / 'history.json', history)
                event('epoch_complete', mode=mode, **row)
                if score(report) < best['score']:
                    best = {'score': score(report), 'epoch': epoch}
                    atomic_checkpoint(output / 'best.pt', {
                        'model': model.state_dict(), 'model_config': model.model_config(),
                        'epochs': epoch, 'updates': updates, 'pair_visits': visits,
                        'validation': report['aggregate'], 'selection_score': best['score'],
                        'empirical_prior': prior.cpu(), 'sources': identity,
                        'bundle_sha256': manifest['bundle_sha256']})
                    atomic_json(output / 'best.json', {
                        'epoch': epoch, 'updates': updates, 'pair_visits': visits,
                        'selection_score': best['score'], 'validation': report['aggregate']})
                    event('best_checkpoint', mode=mode, epoch=epoch,
                          selection_score=best['score'], **report['aggregate'])
                if exhausted:
                    break
                if args.early_stop > 0 and epoch - best['epoch'] >= args.early_stop:
                    exhausted = True
                    event('early_stop', mode=mode, epoch=epoch, best_epoch=best['epoch'])
                    break
            if sources() != identity or open_bundle(args.cache)['bundle_sha256'] != manifest['bundle_sha256']:
                raise ValueError('Distance-head source or input cache changed')
            if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
                raise FloatingPointError('Nonfinite trained distance head')
            if any(not torch.isfinite(v).all() for state in optimizer.state.values()
                   for v in state.values() if isinstance(v, torch.Tensor)):
                raise FloatingPointError('Nonfinite distance-head optimizer')
            final, exports = evaluate(model, records['val'], device, prior, export=True)
            atomic_json(output / 'final_validation.json', final)
            train_report, _ = evaluate(model, records['train'], device, prior)
            atomic_json(output / 'final_training.json', train_report)
            atomic_checkpoint(output / 'last.pt', {
                'model': model.state_dict(), 'model_config': model.model_config(),
                'optimizer': optimizer.state_dict(), 'epochs': epoch, 'updates': updates,
                'pair_visits': visits, 'rng': torch.get_rng_state(),
                'sampling_rng': sampling.get_state(),
                'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                'empirical_prior': prior.cpu(), 'sources': identity,
                'bundle_sha256': manifest['bundle_sha256'], 'resumable': False})
            best_validation = None
            if (output / 'best.pt').exists():
                best_state = torch.load(output / 'best.pt', map_location=device, weights_only=False)
                best_model = PairwiseDistanceHead(**best_state['model_config']).to(device)
                best_model.load_state_dict(best_state['model'])
                best_model.eval()
                best_check, _ = evaluate(best_model, records['val'], device, prior)
                reloaded_score = score(best_check)
                if reloaded_score > best['score'] + 1e-9:
                    raise ValueError('Best distance-head checkpoint reload changed predictions')
                best_validation = best_check['aggregate']
                atomic_json(output / 'best_validation.json', best_validation)
                event('best_validation', mode=mode, epoch=best_state['epochs'],
                      selection_score=best['score'], **{k: v for k, v in best_validation.items()
                                                        if isinstance(v, (int, float))})
            clone = PairwiseDistanceHead(**model.model_config()).to(device)
            clone.load_state_dict(torch.load(
                output / 'last.pt', map_location=device, weights_only=False)['model'])
            checked, _ = evaluate(clone, records['val'][:1], device, prior)
            if checked['per_chain'][0] != final['per_chain'][0]:
                raise ValueError('Distance-head checkpoint reload changed predictions')
            atomic_checkpoint(output / 'validation_restraints.pt', {
                'protocol': 'pair-distance-restraints-v1', 'edges': model.edges.cpu(),
                'entries': exports, 'model_sha256': digest(output / 'last.pt'),
                'best_model_sha256': digest(output / 'best.pt') if (output / 'best.pt').exists() else None,
                'bundle_sha256': manifest['bundle_sha256'],
                'scope': 'predicted development-validation restraints; no native coordinates'})
            summary = {
                'mode': mode, 'epochs': epoch, 'updates': updates, 'pair_visits': visits,
                'stop_reason': 'time_limit' if exhausted else 'epochs',
                'elapsed_seconds': time.monotonic() - started, 'reload_exact': True,
                'best_epoch': best['epoch'], 'best_selection_score': best['score'],
                'best_reload_exact': best_validation is not None,
                'validation': final['aggregate'], 'training': train_report['aggregate'],
                'best_validation': best_validation,
                'checkpoint_sha256': digest(output / 'last.pt')}
            atomic_json(output / 'completion.json', summary)
            summaries[mode] = summary
            event('case_complete', mode=mode, updates=updates)
        if sources() != identity:
            raise ValueError('Distance-head implementation changed before completion')
        atomic_json(root / 'completion.json', summaries)
        event('complete')
    except Exception as exc:
        atomic_json(root / 'failure.json', {'error': repr(exc)})
        event('failure', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_decoder_enrichment_v3_20261021')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0', 'cuda:1'], default='cpu')
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--seconds', type=float, default=900.)
    parser.add_argument('--pairs', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--context-layers', type=int, default=0)
    parser.add_argument('--context-heads', type=int, default=8)
    parser.add_argument('--contact-bias', action='store_true',
                        help='Add learned per-head bias from the pairwise contact signal '
                             'inside residue context attention')
    parser.add_argument('--contact-bias-source', choices=['cosine', 'calibrated'], default='cosine',
                        help='Contact signal for the attention bias: cosine uses the raw '
                             'contact_z dot product; calibrated uses the sigmoid-calibrated '
                             'contact_probs from the enriched cache')
    parser.add_argument('--cosine-lr', action='store_true',
                        help='Cosine-anneal the learning rate to zero over the epoch budget')
    parser.add_argument('--balanced-ratio', type=float, default=0., metavar='RATIO',
                        help='Fraction of contact pairs in each batch (0 means the natural '
                             'frequency; 0.5 gives 50/50).  With replacement per class.')
    parser.add_argument('--spherical-dim', type=int, default=0,
                        help='Enable the hierarchical spherical contact head with this '
                             'sphere dimension (0 disables it)')
    parser.add_argument('--pair-bilinear-dim', type=int, default=0,
                        help='Symmetrized low-rank bilinear pair term width (0 disables it)')
    parser.add_argument('--pair-layers', type=int, default=2,
                        help='Number of hidden layers in the pair MLP (2 = current, 4 = deeper)')
    parser.add_argument('--chain-batch', type=int, default=16,
                        help='Chains per optimizer step (gradient accumulation). Natural '
                             'sampling with 512 pairs/chain gives only ~8 contact pairs per '
                             'chain; batching 16 chains yields ~128 contacts per step.')
    parser.add_argument('--kl-penalty', type=float, default=0.,
                        help='KL(marginal || empirical prior) weight applied via a detached '
                             'large-pair sample (0 disables; ~0.01-0.1 typical)')
    parser.add_argument('--kl-batch', type=int, default=8192,
                        help='Number of pairs for the detached KL marginal estimate '
                             '(default 8192; more = lower noise, larger = slower per step)')
    parser.add_argument('--focal-gamma', type=float, default=0.,
                        help='Focal loss exponent: (1-p_t)^gamma down-weights easy pairs '
                             '(0 = plain CE)')
    parser.add_argument('--kendall-terms', nargs='+', default=[],
                        choices=['contact', 'distance', 'angle'],
                        help='Multi-task Kendall-uncertainty terms to learn a balance for '
                             '(e.g. "contact distance angle"); empty disables Kendall weighting')
    parser.add_argument('--predict-angles', action='store_true',
                        help='Add auxiliary backbone-angle prediction head and loss')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Residual/hidden dropout probability applied inside the '
                             'node MLP and context layers (0 disables)')
    parser.add_argument('--weight-decay', type=float, default=.01,
                        help='AdamW weight decay applied to all head parameters')
    parser.add_argument('--early-stop', type=int, default=0,
                        help='Stop after this many epochs without a selection-score improvement '
                             'when positive; 0 disables early stopping')
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], default='adamw',
                        help='Optimizer: AdamW (default) or Muon (orthogonalized momentum '
                             'for 2D param layers + AdamW aux for biases/norms/embeddings)')
    parser.add_argument('--label-smoothing', type=float, default=0.,
                        help='Apply label smoothing to distance-bin targets (0 = hard CE, '
                             '0.1 = mild smoothing)')
    parser.add_argument('--min-separation', type=int, default=12,
                        help='Minimum sequence separation for eligible pairs (12 = nonlocal only, '
                             '3 = also includes short-range local pairs)')
    parser.add_argument('--train-count', type=int)
    parser.add_argument('--val-count', type=int)
    parser.add_argument('--modes', nargs='+', choices=['sequence', 'decoder'], default=['sequence', 'decoder'])
    train(parser.parse_args())


if __name__ == '__main__':
    main()
