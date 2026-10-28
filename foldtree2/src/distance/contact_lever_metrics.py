"""Per-protein metrics for medium- and long-range C-alpha contacts."""
import math
import torch

from foldtree2.train_distance_restraints import contact_metrics


def per_protein_range_metrics(probabilities, truth, pairs, length):
    if len(probabilities) != len(truth) or len(truth) != len(pairs):
        raise ValueError('Mismatched pair predictions and labels')
    sep = (pairs[:, 1] - pairs[:, 0]).abs()
    result = {}
    for name, mask in [('medium', (sep >= 12) & (sep < 24)), ('long', sep >= 24)]:
        if not mask.any():
            continue
        score, labels = probabilities[mask], truth[mask].bool()
        order = score.argsort(descending=True, stable=True)
        row = {'pairs': len(score), 'contacts': int(labels.sum())}
        row['ap'] = contact_metrics(score, labels, min(length, len(score)))['contact_average_precision']
        for label, count in [('L5', max(1, length // 5)), ('L2', max(1, length // 2)), ('L', length)]:
            k = min(count, len(score))
            row['precision_' + label] = float(labels[order[:k]].float().mean())
        result[name] = row
    return result


def summarize_ranges(rows):
    result = {}
    for name in ('medium', 'long'):
        available = [r[name] for r in rows if name in r]
        result[name] = {'chains_with_pairs': len(available),
                        'chains_with_contacts': sum(r['contacts'] > 0 for r in available)}
        for metric in ('ap', 'precision_L5', 'precision_L2', 'precision_L'):
            vals = [r[metric] for r in available if r[metric] is not None and math.isfinite(r[metric])]
            result[name][metric] = sum(vals) / len(vals) if vals else None
    return result
