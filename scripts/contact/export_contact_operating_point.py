"""Export a sphere checkpoint with a development-set decision bias shift."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import copy
import json
import math
from pathlib import Path

import torch

from foldtree2.se3_cache import digest
from scripts.se3.diagnose_se3_reconstruction import atomic_checkpoint


def shifted_checkpoint(checkpoint, calibration):
    """Return a copy; a logit shift changes the 0.5 operating point, not ranking."""
    if not checkpoint.get('config', {}).get('spherical_dim'):
        raise ValueError('Expected a spherical checkpoint')
    threshold = float(calibration['calibration']['threshold'])
    shift = float(calibration['logit_bias_shift'])
    if not 0. < threshold < 1. or not math.isfinite(shift):
        raise ValueError('Invalid threshold or shift')
    if not math.isclose(shift, -math.log(threshold / (1. - threshold)), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('Shift is inconsistent with the calibration threshold')
    result = copy.deepcopy(checkpoint)
    result['state_dict']['contact_offset'] = result['state_dict']['contact_offset'] + shift
    result['operating_point'] = {
        'original_threshold': threshold, 'global_logit_shift': shift,
        'selection_scope': 'reused development validation subset; not an independent test',
        'meaning': 'decision bias for a 0.5 cutoff; resulting values are not calibrated probabilities',
        'effect': 'changes the full distance-bin probability distribution as well as contact decisions',
        'heldout_precision': calibration['heldout']['precision'],
        'heldout_recall': calibration['heldout']['recall'],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--case', default='sphere64_w0p3')
    args = parser.parse_args()
    data = json.loads(Path(args.calibration).read_text())
    matches = [row for row in data['results'] if row['case'] == args.case]
    if len(matches) != 1:
        raise ValueError('Calibration must contain exactly one matching case')
    row = matches[0]
    source = Path(row['source']) / row.get('checkpoint_case', row['case']) / 'last.pt'
    if source.stat().st_mtime > Path(args.calibration).stat().st_mtime:
        raise ValueError('Source checkpoint changed after calibration')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    model = torch.load(source, map_location='cpu', weights_only=False)
    shifted = shifted_checkpoint(model, row)
    provenance = json.loads((Path(row['source']) / 'provenance.json').read_text())
    model_class = provenance.get('variant', {}).get(
        'model_class', 'foldtree2.src.distance.distance_restraints.PairwiseDistanceHead')
    shifted['operating_point'].update({
        'source_checkpoint': str(source), 'source_sha256': digest(source),
        'model_class': model_class,
        'calibration_file': args.calibration, 'calibration_sha256': digest(args.calibration),
        'target_recall': data['target_recall'], 'split_seed': data['split_seed'],
    })
    atomic_checkpoint(output, shifted)
    print(json.dumps(shifted['operating_point'], indent=2))


if __name__ == '__main__':
    main()
