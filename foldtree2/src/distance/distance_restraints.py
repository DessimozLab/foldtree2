"""Frozen-feature distance distributions and differentiable pair restraints."""
import math

import torch
from torch import nn
import torch.nn.functional as F


DISTANCE_EDGES = (4., 6., 8., 10., 12., 16., 20., 24., 32.)


def validate_edges(edges):
    if (edges.ndim != 1 or not edges.is_floating_point() or not len(edges)
            or not torch.isfinite(edges).all() or edges[0] <= 0
            or not (edges[1:] > edges[:-1]).all()):
        raise ValueError('Expected positive strictly increasing finite distance edges')


def eligible_pairs(mask, min_separation=12):
    if (mask.ndim != 1 or mask.dtype != torch.bool or not mask.any()
            or isinstance(min_separation, bool) or not isinstance(min_separation, int)
            or min_separation < 1):
        raise ValueError('Expected a nonempty boolean mask and positive integer separation')
    indices = torch.arange(len(mask), device=mask.device)
    pairs = ((indices[None] - indices[:, None] >= min_separation)
             & mask[:, None] & mask[None]).nonzero()
    if not len(pairs):
        raise ValueError('No eligible residue pairs')
    return pairs


def validate_pairs(pairs, mask):
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long
            or not len(pairs) or pairs.device != mask.device):
        raise ValueError('Expected nonempty long [P,2] pairs on the residue device')
    if (pairs < 0).any() or (pairs >= len(mask)).any():
        raise ValueError('Pair indices outside residue range')
    if (pairs[:, 0] == pairs[:, 1]).any() or not mask[pairs].all():
        raise ValueError('Pairs must select distinct valid residues')


def distance_labels(target, pairs, edges):
    validate_edges(edges)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError('Expected [N,3] native coordinates')
    validate_pairs(pairs, torch.ones(len(target), device=target.device, dtype=torch.bool))
    if not torch.isfinite(target[pairs]).all():
        raise ValueError('Nonfinite coordinates in supervised pairs')
    distances = (target[pairs[:, 0]] - target[pairs[:, 1]]).norm(dim=-1)
    return torch.bucketize(distances, edges, right=True), distances


class ResidueContextLayer(nn.Module):
    """Pre-normalized attention with learned, clipped relative row-position bias.

    With ``contact_bias`` the attention logits also receive a learned per-head
    scalar times the pairwise contact-embedding dot product, so heads can
    specialize to contact-similar residue pairs. ``dropout`` applies to the
    attention output and the feed-forward hidden activations (residual
    dropout).
    """
    def __init__(self, hidden, heads, contact_bias=False, dropout=0.):
        super().__init__()
        if not isinstance(contact_bias, bool):
            raise ValueError('Expected boolean contact_bias')
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0. <= dropout < 1.:
            raise ValueError('Expected dropout probability in [0, 1)')
        self.dropout = float(dropout)
        self.drop = nn.Dropout(self.dropout) if self.dropout else nn.Identity()
        self.norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.relative_bias = nn.Embedding(65, heads)
        self.contact_scale = nn.Parameter(torch.zeros(heads)) if contact_bias else None
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(),
                                self.drop,
                                nn.Linear(4 * hidden, hidden))

    def forward(self, nodes, mask, contact_dot=None):
        positions = torch.arange(len(nodes), device=nodes.device)
        offsets = (positions[None] - positions[:, None]).clamp(-32, 32) + 32
        bias = self.relative_bias(offsets).permute(2, 0, 1)
        if self.contact_scale is not None:
            if (contact_dot is None or contact_dot.shape != (len(nodes), len(nodes))
                    or contact_dot.device != nodes.device or not contact_dot.is_floating_point()):
                raise ValueError('Contact-biased context requires the pairwise contact-dot matrix')
            bias = bias + self.contact_scale[:, None, None] * contact_dot[None]
        bias = bias.masked_fill(~mask[None, None, :], -torch.inf)
        normalized = self.norm(nodes)[None]
        attended, _ = self.attention(normalized, normalized, normalized,
                                     attn_mask=bias, need_weights=False)
        nodes = nodes + self.drop(attended[0])
        nodes = nodes + self.ff(self.ff_norm(nodes))
        return nodes.masked_fill(~mask[:, None], 0.)


class PairwiseDistanceHead(nn.Module):
    """Symmetric pair MLP with optional residue context; no geometry inputs."""
    def __init__(self, feature_dim, contact_dim=30, hidden=64, mode='decoder',
                 edges=DISTANCE_EDGES, context_layers=0, context_heads=8,
                 contact_bias=False, dropout=0., spherical_dim=0, pair_bilinear_dim=0):
        super().__init__()
        if (any(isinstance(x, bool) or not isinstance(x, int) or x < 1
                for x in (feature_dim, contact_dim, hidden))
                or mode not in ('decoder', 'sequence')):
            raise ValueError('Invalid distance-head dimensions or mode')
        if (isinstance(context_layers, bool) or not isinstance(context_layers, int)
                or context_layers < 0 or isinstance(context_heads, bool)
                or not isinstance(context_heads, int) or context_heads < 1
                or (context_layers and hidden % context_heads)):
            raise ValueError('Invalid residue-context depth or attention heads')
        if not isinstance(contact_bias, bool):
            raise ValueError('Expected boolean contact_bias')
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0. <= dropout < 1.:
            raise ValueError('Expected dropout probability in [0, 1)')
        if (isinstance(pair_bilinear_dim, bool) or not isinstance(pair_bilinear_dim, int)
                or pair_bilinear_dim < 0):
            raise ValueError('Expected nonnegative integer pair bilinear dimension')
        self.dropout = float(dropout)
        edges = torch.tensor(edges, dtype=torch.float32)
        validate_edges(edges)
        if not (edges == 8.).any():
            raise ValueError('Distance bins must have an exact 8-Angstrom contact boundary')
        self.register_buffer('edges', edges)
        self.contact_bins = int((edges <= 8.).sum())
        self.feature_dim, self.contact_dim, self.hidden, self.mode = feature_dim, contact_dim, hidden, mode
        self.base_norm = nn.LayerNorm(feature_dim)
        self.contact_norm = nn.LayerNorm(contact_dim)
        self.aa_norm = nn.LayerNorm(20)
        input_drop = nn.Dropout(self.dropout) if self.dropout else nn.Identity()
        self.node = nn.Sequential(
            nn.Linear(feature_dim + contact_dim + 20, hidden), nn.GELU(),
            input_drop,
            nn.Linear(hidden, hidden), nn.GELU())
        self.pair_bilinear_dim = int(pair_bilinear_dim)
        pair_input_dim = 3 * hidden + 2
        if self.pair_bilinear_dim:
            # Symmetric low-rank bilinear term: (p_i x p_j + p_j x p_i) / 2,
            # projected back to hidden so pair features grow by exactly hidden.
            self.pair_proj = nn.Linear(hidden, self.pair_bilinear_dim, bias=False)
            self.pair_bilin = nn.Linear(self.pair_bilinear_dim * self.pair_bilinear_dim, hidden, bias=False)
            pair_input_dim += hidden
        self.pair = nn.Sequential(
            nn.Linear(pair_input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, len(edges) + 1))
        self.context_layers, self.context_heads = context_layers, context_heads
        self.contact_bias = contact_bias
        self.context = nn.ModuleList(
            ResidueContextLayer(hidden, context_heads, contact_bias=contact_bias,
                                dropout=dropout)
            for _ in range(context_layers))
        self.context_norm = nn.LayerNorm(hidden) if context_layers else nn.Identity()
        if isinstance(spherical_dim, bool) or not isinstance(spherical_dim, int) or spherical_dim < 0:
            raise ValueError('Expected nonnegative spherical contact dimension')
        self.spherical_dim = spherical_dim
        if spherical_dim:
            self.contact_projection = nn.Linear(hidden, spherical_dim)
            self.contact_log_scale = nn.Parameter(torch.tensor(math.log(10.)))
            self.contact_offset = nn.Parameter(torch.tensor(-5.))

    def model_config(self):
        config = {'feature_dim': self.feature_dim, 'contact_dim': self.contact_dim,
                  'hidden': self.hidden, 'mode': self.mode, 'edges': self.edges.tolist()}
        if self.context_layers:
            config.update(context_layers=self.context_layers, context_heads=self.context_heads)
        if self.contact_bias:
            config.update(contact_bias=self.contact_bias)
        if self.dropout:
            config.update(dropout=self.dropout)
        if self.spherical_dim:
            config.update(spherical_dim=self.spherical_dim)
        if self.pair_bilinear_dim:
            config.update(pair_bilinear_dim=self.pair_bilinear_dim)
        return config

    def _contact_dot(self, entry, mask):
        if not self.contact_bias:
            return None
        if self.contact_bias_source == 'calibrated':
            # Calibrated production contact probabilities from the enriched
            # cache: sigmoid(temp * z@z.T + bias), symmetric in [0,1].  These
            # are better calibrated than the raw cosine for attention logits.
            values = entry['contact_probs']
            if (values.shape != (len(mask), len(mask))
                    or not values.is_floating_point()
                    or values.device != mask.device
                    or not torch.isfinite(values).all()
                    or (values < 0.).any() or (values > 1.).any()
                    or not torch.allclose(values, values.T, atol=1e-6, rtol=1e-6)):
                raise ValueError('Invalid calibrated contact probabilities for attention bias')
            return torch.where(mask[:, None] & mask[None], values, torch.zeros_like(values))
        values = entry['contact_z']
        if (values.shape != (len(mask), self.contact_dim) or not values.is_floating_point()
                or values.device != mask.device or not torch.isfinite(values[mask]).all()):
            raise ValueError('Invalid frozen distance-head input: contact_z')
        dot = values @ values.T
        return torch.where(mask[:, None] & mask[None], dot, torch.zeros_like(dot))

    def _nodes(self, entry, contact_dot=None):
        mask = entry['mask']
        if mask.ndim != 1 or mask.dtype != torch.bool or not mask.any():
            raise ValueError('Expected nonempty boolean residue mask')
        values = []
        for key, width in (('features', self.feature_dim),
                           ('contact_z', self.contact_dim), ('aa_probs', 20)):
            value = entry[key]
            if (value.shape != (len(mask), width) or not value.is_floating_point()
                    or value.device != mask.device or not torch.isfinite(value[mask]).all()):
                raise ValueError('Invalid frozen distance-head input: ' + key)
            values.append(value.detach().masked_fill(~mask[:, None], 0.))
        aa = values[2][mask]
        if ((aa < 0) | (aa > 1)).any() or not torch.allclose(
                aa.sum(-1), torch.ones_like(aa[:, 0]), atol=1e-5, rtol=1e-5):
            raise ValueError('AA inputs must be probability distributions')
        if self.mode == 'sequence':
            values = [torch.zeros_like(v) for v in values]
        nodes = self.node(torch.cat([
            self.base_norm(values[0]), self.contact_norm(values[1]),
            self.aa_norm(values[2])], dim=-1))
        nodes = nodes.masked_fill(~mask[:, None], 0.)
        for layer in self.context:
            nodes = layer(nodes, mask, contact_dot)
        return self.context_norm(nodes).masked_fill(~mask[:, None], 0.)

    def forward(self, entry, pairs, chunk_size=2048):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('Expected positive integer pair chunk size')
        contact_dot = self._contact_dot(entry, entry['mask'])
        nodes = self._nodes(entry, contact_dot)
        validate_pairs(pairs, entry['mask'])
        result = []
        for chunk in pairs.split(chunk_size):
            i, j = chunk.T
            a, b = nodes[i], nodes[j]
            separation = (j - i).abs().to(nodes.dtype).log1p()[:, None]
            contact = (entry['contact_z'][i].detach() * entry['contact_z'][j].detach()).sum(-1, keepdim=True)
            if self.mode == 'sequence':
                contact = torch.zeros_like(contact)
            features = [a + b, (a - b).abs(), a * b, separation, contact]
            if self.pair_bilinear_dim:
                # Symmetric low-rank outer product with a learned projection.
                # (p_i x p_j + p_j x p_i) / 2 is invariant to pair order, so
                # logits stay symmetric without a separate symmetry constraint.
                pa = self.pair_proj(a)
                pb = self.pair_proj(b)
                outer = (pa.unsqueeze(-1) * pb.unsqueeze(-2)
                         + pb.unsqueeze(-1) * pa.unsqueeze(-2)) * .5
                features.append(self.pair_bilin(outer.reshape(len(a), -1)))
            result.append(self.pair(torch.cat(features, dim=-1)))
        logits = torch.cat(result)
        if self.spherical_dim:
            unit = F.normalize(self.contact_projection(nodes), dim=-1, eps=1e-6)
            i, j = pairs.T
            scale = self.contact_log_scale.clamp(-5., math.log(100.)).exp()
            contact_logits = scale * (unit[i] * unit[j]).sum(-1) + self.contact_offset
            # Hierarchical distribution: the sphere determines total contact mass,
            # while the pair MLP predicts distances conditional on either group.
            near = F.log_softmax(logits[:, :self.contact_bins], dim=-1)
            far = F.log_softmax(logits[:, self.contact_bins:], dim=-1)
            logits = torch.cat([
                near + F.logsigmoid(contact_logits)[:, None],
                far + F.logsigmoid(-contact_logits)[:, None]], dim=-1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite distance logits')
        return logits

    def contact_probability(self, probabilities):
        return probabilities[..., :self.contact_bins].sum(-1)

    def capped_distance(self, probabilities):
        edges = self.edges.to(probabilities)
        lower = torch.cat([edges.new_zeros(1), edges])
        upper = torch.cat([edges, edges[-1:]])
        return probabilities @ ((lower + upper) / 2)


def distribution_restraint_loss(coords, pairs, probabilities, edges, softness=1.):
    """Mixture likelihood of soft intervals; the overflow bin has no upper bound."""
    validate_edges(edges)
    if not math.isfinite(softness) or softness <= 0:
        raise ValueError('Expected positive finite restraint softness')
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('Expected [N,3] coordinates')
    validate_pairs(pairs, torch.ones(len(coords), dtype=torch.bool, device=coords.device))
    if not torch.isfinite(coords[pairs]).all():
        raise ValueError('Nonfinite restrained coordinates')
    if (probabilities.shape != (len(pairs), len(edges) + 1)
            or not probabilities.is_floating_point() or not torch.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not torch.allclose(probabilities.sum(-1),
                                  torch.ones_like(probabilities[:, 0]), atol=1e-5, rtol=1e-5)):
        raise ValueError('Expected normalized pair distance distributions')
    d = (coords[pairs[:, 0]] - coords[pairs[:, 1]]).norm(dim=-1)
    lower = torch.cat([torch.zeros_like(d[:, None]),
                       F.logsigmoid((d[:, None] - edges) / softness)], dim=-1)
    upper = torch.cat([F.logsigmoid((edges - d[:, None]) / softness),
                       torch.zeros_like(d[:, None])], dim=-1)
    log_probabilities = probabilities.detach().log()
    loss = -torch.logsumexp(log_probabilities + lower + upper, dim=-1).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite distribution restraint')
    return loss
