"""Experimental spherical contact head with symmetric residue propensities."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from foldtree2.src.distance_restraints import PairwiseDistanceHead, validate_pairs


class ResidueBiasDistanceHead(PairwiseDistanceHead):
    """Add one learned, context-dependent bias per residue to the sphere score."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.spherical_dim:
            raise ValueError('Residue-bias head requires a spherical contact projection')
        # The existing global offset already provides the intercept.
        self.residue_bias = nn.Linear(self.hidden, 1, bias=False)
        nn.init.zeros_(self.residue_bias.weight)

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
            result.append(self.pair(torch.cat(
                [a + b, (a - b).abs(), a * b, separation, contact], dim=-1)))
        logits = torch.cat(result)
        unit = F.normalize(self.contact_projection(nodes), dim=-1, eps=1e-6)
        propensity = self.residue_bias(nodes).squeeze(-1)
        i, j = pairs.T
        scale = self.contact_log_scale.clamp(-5., math.log(100.)).exp()
        score = (scale * (unit[i] * unit[j]).sum(-1) + self.contact_offset
                 + (propensity[i] + propensity[j]))
        near = F.log_softmax(logits[:, :self.contact_bins], dim=-1)
        far = F.log_softmax(logits[:, self.contact_bins:], dim=-1)
        output = torch.cat((near + F.logsigmoid(score)[:, None],
                            far + F.logsigmoid(-score)[:, None]), dim=-1)
        if not torch.isfinite(output).all():
            raise FloatingPointError('Nonfinite distance logits')
        return output
