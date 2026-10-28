"""Inspectable, configurable pair correction for contact-score experiments."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from foldtree2.src.distance_restraints import PairwiseDistanceHead, validate_pairs
from foldtree2.src.xsatransformer import ExclusiveSelfAttention


class InspectablePairHead(PairwiseDistanceHead):
    def __init__(self, *args, correction_hidden=32, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.spherical_dim or correction_hidden < 1:
            raise ValueError('Expected positive sphere and correction dimensions')
        self.correction_hidden = correction_hidden
        self.pair_bias = nn.Sequential(nn.Linear(3 * self.hidden + 2, correction_hidden),
                                       nn.GELU(), nn.Linear(correction_hidden, 1))
        nn.init.zeros_(self.pair_bias[-1].weight)
        nn.init.zeros_(self.pair_bias[-1].bias)

    def model_config(self):
        return {**super().model_config(), 'correction_hidden': self.correction_hidden}

    def forward(self, entry, pairs, chunk_size=2048, return_components=False):
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
            corrections.append(self.pair_bias(features).squeeze(-1))
        logits = torch.cat(logits)
        correction = torch.cat(corrections)
        unit = F.normalize(self.contact_projection(nodes), dim=-1, eps=1e-6)
        i, j = pairs.T
        scale = self.contact_log_scale.clamp(-5., math.log(100.)).exp()
        dot = scale * (unit[i] * unit[j]).sum(-1)
        score = dot + self.contact_offset + correction
        result = torch.cat((F.log_softmax(logits[:, :self.contact_bins], dim=-1) + F.logsigmoid(score)[:, None],
                            F.log_softmax(logits[:, self.contact_bins:], dim=-1) + F.logsigmoid(-score)[:, None]), dim=-1)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite distance logits')
        return (result, {'dot': dot, 'correction': correction}) if return_components else result


class RelativeBiasXSA(nn.Module):
    """Adapt XSA to the context layer's additive per-head attention mask."""
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.xsa = ExclusiveSelfAttention(hidden, heads, batch_first=True)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        if key is not query or value is not query or need_weights:
            raise ValueError('Expected self-attention without attention weights')
        if attn_mask is not None:
            batch, length, _ = query.shape
            attn_mask = attn_mask.reshape(batch, self.heads, length, length)
        return self.xsa(query, attn_mask=attn_mask), None


class XSAPairHead(InspectablePairHead):
    """Pair-correction head with XSA in place of ordinary context attention."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in self.context:
            layer.attention = RelativeBiasXSA(self.hidden, self.context_heads)
