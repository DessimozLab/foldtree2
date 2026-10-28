"""Opt-in frozen decoder feature adapters for the residue refiner."""
import torch
from torch import nn

from foldtree2.se3_validation import forward
from foldtree2.train_se3_lightning import build_model
from foldtree2.src.se3_struct_decoder import enable_tensor_ff_input_norm


MODES = ('baseline', 'contact', 'aa', 'both', 'both_soft_pairs')


def checked_features(value, rows, columns, mask, name):
    if value.shape != (rows, columns) or not value.is_floating_point():
        raise ValueError(f'Expected floating {name} with shape [{rows},{columns}]')
    if not torch.isfinite(value[mask]).all():
        raise ValueError('Nonfinite valid ' + name)
    return value.masked_fill(~mask[:, None], 0.)


def soft_contact_context(features, probabilities, mask):
    """Confidence-weighted nonlocal messages with fixed expected-degree scaling."""
    n = len(mask)
    if probabilities.shape != (n, n) or not torch.isfinite(probabilities).all():
        raise ValueError('Expected finite [N,N] contact probabilities')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Contact probabilities must be in [0,1]')
    if not torch.allclose(probabilities, probabilities.T, atol=1e-6, rtol=1e-6):
        raise ValueError('Contact probabilities must be symmetric')
    indices = torch.arange(n, device=mask.device)
    eligible = (indices[:, None] - indices[None]).abs() >= 3
    eligible &= mask[:, None] & mask[None]
    weights = probabilities.masked_fill(~eligible, 0.)
    # Unlike row-normalized softmax, weak evidence stays weak.
    denominator = eligible.sum(-1, keepdim=True).clamp_min(1)
    context = weights @ features / denominator
    return context.masked_fill(~mask[:, None], 0.)


class DecoderFeatureFusion(nn.Module):
    def __init__(self, feature_dim, contact_dim, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown decoder fusion mode')
        self.mode = mode
        self.feature_dim = feature_dim
        self.contact_dim = contact_dim
        # Independent RNG scope leaves the core model initialization unchanged.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(1729)
            self.contact = self._adapter(contact_dim, feature_dim) if mode in (
                'contact', 'both', 'both_soft_pairs') else None
            torch.default_generator.manual_seed(1730)
            self.aa = self._adapter(20, feature_dim) if mode in (
                'aa', 'both', 'both_soft_pairs') else None
            torch.default_generator.manual_seed(1731)
            self.pair = (nn.Sequential(
                nn.LayerNorm(contact_dim), nn.Linear(contact_dim, feature_dim, bias=False),
                nn.GELU(), nn.Linear(feature_dim, feature_dim, bias=False))
                if mode == 'both_soft_pairs' else None)
        for name in ('contact', 'aa', 'pair'):
            if getattr(self, name) is not None:
                self.register_parameter(name + '_gate', nn.Parameter(torch.tensor(.1)))

    @staticmethod
    def _adapter(input_dim, output_dim):
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim),
                             nn.GELU(), nn.Linear(output_dim, output_dim))

    def forward(self, entry):
        mask = entry['mask']
        if mask.dtype != torch.bool or mask.ndim != 1 or not mask.any():
            raise ValueError('Expected a nonempty boolean residue mask')
        n = len(mask)
        result = checked_features(entry['features'], n, self.feature_dim, mask, 'base features')
        contact = None
        if self.contact is not None:
            contact = checked_features(entry['contact_z'], n, self.contact_dim, mask, 'contact_z')
            result = result + self.contact_gate.tanh() * self.contact(contact)
        if self.aa is not None:
            probabilities = checked_features(entry['aa_probs'], n, 20, mask, 'aa_probs')
            valid = probabilities[mask]
            if ((valid < 0) | (valid > 1)).any() or not torch.allclose(
                    valid.sum(-1), torch.ones_like(valid[:, 0]), atol=1e-5, rtol=1e-5):
                raise ValueError('AA probabilities must be normalized distributions')
            result = result + self.aa_gate.tanh() * self.aa(probabilities)
        if self.pair is not None:
            # Keep confidence magnitude; do not LayerNorm the aggregated context.
            normalized_contact = self.pair[0](contact).masked_fill(~mask[:, None], 0.)
            normalized_context = soft_contact_context(
                normalized_contact, entry['contact_probs'], mask)
            message = self.pair[1:](normalized_context)
            result = result + self.pair_gate.tanh() * message
        if not torch.isfinite(result[mask]).all():
            raise FloatingPointError('Nonfinite decoder feature fusion')
        return result.masked_fill(~mask[:, None], 0.)


class DecoderFusionRefiner(nn.Module):
    def __init__(self, feature_dim, contact_dim, mode, architecture):
        super().__init__()
        self.core = enable_tensor_ff_input_norm(build_model(feature_dim, architecture))
        self.fusion = DecoderFeatureFusion(feature_dim, contact_dim, mode)

    def forward(self, entry):
        # Seeds/lifting are constructed from original features before fusion.
        return forward(self.core, {**entry, 'features': self.fusion(entry)},
                       contact_policy='local-only')