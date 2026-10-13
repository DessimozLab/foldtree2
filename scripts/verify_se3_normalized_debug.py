"""Numerical, resume and cross-topology gates for normalized Clariden checks."""
import argparse
import json
import math
from pathlib import Path
import torch


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from tensors(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from tensors(v)


def read(root, expected_updates):
    root = Path(root)
    ck = torch.load(root / 'last.ckpt', map_location='cpu', weights_only=False)
    if ck['se3']['stream'] != {'epoch':0, 'cursor':expected_updates, 'updates':expected_updates}:
        raise ValueError('Debug checkpoint has unexpected stream')
    if not ck['se3']['provenance']['training'].get('tensor_ff_input_norm'):
        raise ValueError('Normalization is not enabled in provenance')
    if list(root.glob('failure*')):
        raise ValueError('Failure artifacts present')
    states = list(tensors({'model':ck['state_dict'], 'optimizer':ck['optimizer_states']}))
    if not all(bool(torch.isfinite(v).all()) for v in states if v.is_floating_point()):
        raise ValueError('Nonfinite model or optimizer state')
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    updates = [r for r in events if r['event'] == 'update']
    if [r['global_step'] for r in updates] != list(range(1, expected_updates + 1)):
        raise ValueError('Debug stream skipped or repeated optimizer updates')
    if not all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
               and r['gradient_norm'] > 0 for r in updates):
        raise ValueError('Nonfinite loss or missing/nonfinite gradients')
    validations = [r for r in events if r['event'] == 'validation']
    if not validations or validations[-1]['global_step'] != expected_updates:
        raise ValueError('Final validation missing')
    config = ck['se3']['provenance']['training']
    if any(r['counts']['geometry'] != config['val_count'] for r in validations):
        raise ValueError('Validation count differs from configuration')
    for row in validations:
        if not all(v is None or math.isfinite(v) for v in row['metrics'].values()):
            raise ValueError('Nonfinite validation metric')
    pairs = json.loads((root / 'validation-000.json').read_text())
    if len(pairs) != config['val_count'] or len({r['id'] for r in pairs}) != len(pairs):
        raise ValueError('Missing or repeated validation structures')
    return ck, events, pairs


def compare(candidate, control, exact, output):
    a, ae, ar = read(candidate, 32)
    b, be, br = read(control, 32)
    if a['se3']['provenance'] != b['se3']['provenance']:
        raise ValueError('Comparison code, configuration or split provenance differs')
    if {r['id'] for r in ar} != {r['id'] for r in br}:
        raise ValueError('Validation IDs differ between topologies')
    atol = rtol = 0. if exact else 1e-4
    largest = 0.
    for k, v in a['state_dict'].items():
        w = b['state_dict'][k]
        torch.testing.assert_close(v, w, atol=atol, rtol=rtol)
        if v.is_floating_point():
            largest = max(largest, float((v - w).abs().max()))
    if exact:
        if a['optimizer_states'][0]['param_groups'] != b['optimizer_states'][0]['param_groups']:
            raise ValueError('Resumed optimizer parameter groups differ')
        ast, bst = a['optimizer_states'][0]['state'], b['optimizer_states'][0]['state']
        if ast.keys() != bst.keys():
            raise ValueError('Resumed optimizer state inventory differs')
        for key in ast:
            for field in ast[key]:
                torch.testing.assert_close(ast[key][field], bst[key][field], atol=0., rtol=0.)
    av = [r for r in ae if r['event']=='validation'][-1]['metrics']
    bv = [r for r in be if r['event']=='validation'][-1]['metrics']
    for key in av:
        if av[key] is None or bv[key] is None:
            if av[key] != bv[key]:
                raise ValueError('Validation metric masks differ')
        else:
            torch.testing.assert_close(torch.tensor(av[key], dtype=torch.float64),
                                       torch.tensor(bv[key], dtype=torch.float64), atol=atol, rtol=rtol)
    result = {'passed':True, 'exact_resume':exact, 'maximum_parameter_buffer_difference':largest,
              'updates':32, 'validation_structures':len(ar), 'validation':av,
              'topologies':[a['se3']['topology'],b['se3']['topology']],
              'peak_gpu_memory_bytes':max(r['peak_memory_bytes'] for r in ae if r['event']=='update')}
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate'); p.add_argument('control'); p.add_argument('--exact', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    compare(args.candidate, args.control, args.exact, args.output)
