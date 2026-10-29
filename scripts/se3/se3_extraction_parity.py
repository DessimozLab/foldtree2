"""Explicit, bounded cross-runtime audit; same-runtime parity stays strict."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import torch


def compare_entries(reference, current, accept_runtime_drift=False):
    if reference.keys() != current.keys():
        raise ValueError('Frozen entry fields differ')
    report = {}
    exact = {'tokens', 'codebook', 'target', 'mask'}
    for key, value in reference.items():
        other = current[key]
        if not isinstance(value, torch.Tensor):
            if value != other:
                raise ValueError('Frozen optional value differs: ' + key)
            continue
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError('Frozen shape/dtype differs: ' + key)
        if key == 'dot' and accept_runtime_drift:
            changed = int((value != other).sum())
            union = int((value | other).sum())
            fraction = changed / max(union, 1)
            report[key] = {'changed_edges_directed': changed, 'union_edges_directed': union,
                           'changed_fraction': fraction}
            if fraction > .01:
                raise ValueError('More than 1% of contact union edges differ')
            continue
        delta = (value.double() - other.double()).abs()
        finite = delta[torch.isfinite(delta)]
        report[key] = {'max_absolute': float(finite.max()) if finite.numel() else 0.,
                       'rmse': float(finite.square().mean().sqrt()) if finite.numel() else 0.}
        if key == 'angles':
            # Raw graph tensors are hash-verified; degree-to-radian arithmetic
            # may differ by one float32 ULP across CUDA/compiler versions.
            atol = rtol = 1e-6
        elif accept_runtime_drift and key not in exact and value.is_floating_point():
            atol = rtol = 1e-3
        elif key in exact or not value.is_floating_point():
            atol = rtol = 0.
        else:
            atol = rtol = 1e-6
        torch.testing.assert_close(value, other, atol=atol, rtol=rtol, equal_nan=True,
                                   msg=lambda msg: key + ': ' + msg)
    return report
