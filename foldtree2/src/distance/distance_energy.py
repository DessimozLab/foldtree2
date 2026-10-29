"""Opt-in reference-relative distance energies; historical restraints are unchanged."""
import math

import torch
import torch.nn.functional as F

from foldtree2.src.distance.distance_restraints import validate_edges


def interval_log_weights(distances, edges, softness=1.):
    """Smoothed histogram membership with unit total mass, including open tails."""
    validate_edges(edges)
    if (distances.ndim != 1 or not distances.is_floating_point()
            or not torch.isfinite(distances).all() or (distances < 0).any()
            or not math.isfinite(softness) or softness <= 0):
        raise ValueError('Expected finite nonnegative distances and positive softness')
    edges = edges.to(distances)
    left = F.logsigmoid((edges[0] - distances) / softness)[:, None]
    right = F.logsigmoid((distances - edges[-1]) / softness)[:, None]
    width = (edges[1:] - edges[:-1]) / softness
    interior = (F.logsigmoid((distances[:, None] - edges[:-1]) / softness)
                + F.logsigmoid((edges[1:] - distances[:, None]) / softness)
                + torch.log(-torch.expm1(-width)))
    return torch.cat([left, interior, right], dim=-1)


def validate_probabilities(value, rows, bins, name, positive=False):
    if (value.shape != (rows, bins) or not value.is_floating_point()
            or not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any()
            or (positive and (value <= 0).any())
            or not torch.allclose(value.sum(-1), torch.ones_like(value[:, 0]),
                                  atol=1e-5, rtol=1e-5)):
        raise ValueError('Expected normalized ' + name)


def distance_energy(distances, probabilities, edges, reference=None, softness=1.):
    """Per-pair negative log compatibility, optionally relative to a fitted background."""
    log_weights = interval_log_weights(distances, edges, softness)
    validate_probabilities(probabilities, len(distances), len(edges) + 1, 'predictions')
    log_ratio = probabilities.detach().log()
    if reference is not None:
        validate_probabilities(reference, len(distances), len(edges) + 1,
                               'strictly positive reference', positive=True)
        log_ratio = log_ratio - reference.detach().log()
    energy = -torch.logsumexp(log_ratio + log_weights, dim=-1)
    if not torch.isfinite(energy).all():
        raise FloatingPointError('Nonfinite distance energy')
    return energy


def separation_groups(pairs):
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError('Expected long [P,2] residue pairs')
    return torch.bucketize((pairs[:, 1] - pairs[:, 0]).abs(),
                           pairs.new_tensor([24, 48]), right=True)


def fit_reference(records, edges):
    """Laplace-smoothed pair populations for row separations 12--23,24--47,48+."""
    from foldtree2.src.distance.distance_restraints import distance_labels
    counts = torch.ones(3, len(edges) + 1, dtype=torch.float64)
    for entry, pairs in records:
        labels, _ = distance_labels(entry['target'], pairs, edges.cpu())
        groups = separation_groups(pairs)
        counts += torch.bincount(groups * counts.shape[1] + labels,
                                 minlength=counts.numel()).reshape_as(counts)
    return (counts / counts.sum(-1, keepdim=True)).float(), counts


def sample_contact_strata(labels, count, generator):
    """Unbiased population-CE estimator from equal contact/noncontact draws."""
    if labels.ndim != 1 or labels.dtype != torch.long or not len(labels) or count < 2:
        raise ValueError('Expected nonempty labels and at least two draws')
    contact = labels < 3
    positive, negative = contact.nonzero().flatten(), (~contact).nonzero().flatten()
    if not len(positive) or not len(negative):
        indices = torch.randint(len(labels), (count,), generator=generator)
        return indices, torch.ones(count)
    npos = count // 2
    nneg = count - npos
    indices = torch.cat([
        positive[torch.randint(len(positive), (npos,), generator=generator)],
        negative[torch.randint(len(negative), (nneg,), generator=generator)]])
    prevalence = float(contact.float().mean())
    weights = torch.cat([torch.full((npos,), prevalence * count / npos),
                         torch.full((nneg,), (1 - prevalence) * count / nneg)])
    return indices, weights
