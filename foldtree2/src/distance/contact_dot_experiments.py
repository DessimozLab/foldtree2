"""Experimental symmetric dot-product contact-score alternatives."""
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from foldtree2.src.distance.distance_restraints import PairwiseDistanceHead, validate_pairs


class _FlexibleDotHead(PairwiseDistanceHead):
    """Preserve the distance head while changing the contact-score formulation."""
    def forward(self, entry, pairs, chunk_size=2048):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('Expected positive integer pair chunk size')
        nodes = self._nodes(entry, self._contact_dot(entry, entry['mask']))
        validate_pairs(pairs, entry['mask'])
        logits, corrections = [], []
        for chunk in pairs.split(chunk_size):
            i, j = chunk.T
            a, b = nodes[i], nodes[j]
            separation = (j - i).abs().to(nodes.dtype).log1p()[:, None]
            contact = (entry['contact_z'][i].detach() * entry['contact_z'][j].detach()).sum(-1, keepdim=True)
            if self.mode == 'sequence':
                contact = torch.zeros_like(contact)
            features = torch.cat([a + b, (a - b).abs(), a * b, separation, contact], dim=-1)
            logits.append(self.pair(features))
            corrections.append(self.correction(features))
        logits = torch.cat(logits)
        i, j = pairs.T
        unit = F.normalize(self.contact_projection(nodes), dim=-1, eps=1e-6)
        scale = self.contact_log_scale.clamp(-5., math.log(100.)).exp()
        score = scale * self.similarity(unit, nodes, i, j) + self.contact_offset + torch.cat(corrections)
        near = F.log_softmax(logits[:, :self.contact_bins], dim=-1)
        far = F.log_softmax(logits[:, self.contact_bins:], dim=-1)
        result = torch.cat((near + F.logsigmoid(score)[:, None],
                            far + F.logsigmoid(-score)[:, None]), dim=-1)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite distance logits')
        return result


class SymmetricCrossDotHead(_FlexibleDotHead):
    """Symmetric query/key cosine rather than a shared Gram matrix."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.spherical_dim:
            raise ValueError('Cross-dot head requires spherical embeddings')
        self.key_projection = copy.deepcopy(self.contact_projection)

    def similarity(self, query, nodes, i, j):
        key = F.normalize(self.key_projection(nodes), dim=-1, eps=1e-6)
        return ((query[i] * key[j]).sum(-1) + (query[j] * key[i]).sum(-1)) * .5

    def correction(self, features):
        return features.new_zeros(len(features))


class PairAdjustedDotHead(_FlexibleDotHead):
    """Shared sphere cosine plus a symmetric nonlinear pair-specific offset."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.spherical_dim:
            raise ValueError('Pair-adjusted head requires spherical embeddings')
        self.pair_bias = nn.Sequential(nn.Linear(3 * self.hidden + 2, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.pair_bias[-1].weight)
        nn.init.zeros_(self.pair_bias[-1].bias)

    def similarity(self, unit, nodes, i, j):
        return (unit[i] * unit[j]).sum(-1)

    def correction(self, features):
        return self.pair_bias(features).squeeze(-1)
