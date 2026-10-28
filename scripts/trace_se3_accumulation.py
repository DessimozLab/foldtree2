"""Local emulation of four/eight-rank float32 accumulation on identical samples.

This isolates summation order; it does not emulate NCCL's actual reduction tree.
"""
import argparse
import copy
import json
from pathlib import Path
import torch
from foldtree2.se3_cache import FrozenDataset, open_bundle, global_order
from foldtree2.train_se3_lightning import build_model, calibrated
from foldtree2.src.se3_struct_decoder import enable_tensor_ff_input_norm
from foldtree2.se3_validation import forward
from foldtree2.src.losses.se3_ablation import components


def reduce_gradients(grads, world, dtype=torch.float32):
    if world <= 0 or not grads or len(grads) % world:
        raise ValueError('The world size must divide the nonempty global batch')
    grads = [g.to(dtype=dtype) for g in grads]
    accumulation = len(grads) // world
    ranks = []
    for rank in range(world):
        value = grads[rank] / accumulation
        for micro in range(1, accumulation):
            value = value + grads[rank + micro * world] / accumulation
        ranks.append(value)
    result = ranks[0] / world
    for value in ranks[1:]:
        result = result + value / world
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True); p.add_argument('--output', required=True)
    p.add_argument('--updates', type=int, default=4)
    p.add_argument('--reduction-dtype', choices=['float32','float64'], default='float32')
    args = p.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    manifest = open_bundle(args.cache); dataset = FrozenDataset(args.cache, manifest, 'train')
    groups = global_order([manifest['files'][i]['length'] for i in dataset.ids], 42, 0)
    initial = enable_tensor_ff_input_norm(build_model(manifest['feature_dim'], {'hidden':64,'depth':4,'heads':4})).cuda().train()
    models = [initial, copy.deepcopy(initial)]
    optimizers = [torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=.01) for m in models]
    rows = []
    dtype = getattr(torch, args.reduction_dtype)
    for update, group in enumerate(groups[:args.updates]):
        entries = [{k:v.cuda() if isinstance(v, torch.Tensor) else v
                    for k,v in calibrated(dataset[i], manifest['scale']).items()} for i in group]
        per_model = []
        coordinate_gradients = []
        coordinate_samples = []
        for branch, model in enumerate(models):
            params = list(model.parameters()); present = [False] * len(params); gradients = []
            for entry in entries:
                model.zero_grad(set_to_none=True)
                pred, angles = forward(model, entry)
                c = components(pred, angles, entry, True, True)
                if branch == 0 and update == 2:
                    detail = {'id':entry['id']}
                    for key in ['training_fape','distance','bond']:
                        g, = torch.autograd.grad(c[key], pred, retain_graph=True)
                        detail[key] = float(g.detach().norm())
                    coordinate_gradients.append(detail)
                    coordinate_samples.append({k:v.detach().cpu() if isinstance(v,torch.Tensor) else v
                                               for k,v in {'id':entry['id'],'pred':pred,
                                                           'target':entry['target'],'mask':entry['mask']}.items()})
                c['total'].backward()
                for i, param in enumerate(params):
                    present[i] |= param.grad is not None
                gradients.append(torch.cat([param.grad.detach().flatten() if param.grad is not None
                                             else torch.zeros_like(param).flatten() for param in params]))
            raw4 = reduce_gradients(gradients, 4, dtype); raw8 = reduce_gradients(gradients, 8, dtype)
            reference = torch.stack([g.double() for g in gradients]).mean(0)
            per_model.append({'common_weights_reduction_max_abs':float((raw4-raw8).abs().max()),
                              'four_rank_error_vs_double':float((raw4.double()-reference).abs().max()),
                              'eight_rank_error_vs_double':float((raw8.double()-reference).abs().max())})
            selected = (raw4 if branch == 0 else raw8).float()
            offset = 0
            for i, param in enumerate(params):
                n = param.numel()
                param.grad = selected[offset:offset+n].view_as(param).clone() if present[i] else None
                offset += n
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            per_model[-1]['gradient_norm'] = float(norm)
            optimizers[branch].step(); optimizers[branch].zero_grad(set_to_none=True)
        difference = max(float((a-b).detach().abs().max()) for a,b in zip(models[0].parameters(),models[1].parameters()))
        row = {'update':update+1, 'parameter_max_abs_difference':difference,
               'reduction_dtype':args.reduction_dtype,
               'reductions':per_model, 'coordinate_component_gradients':coordinate_gradients}
        if coordinate_samples:
            torch.save(coordinate_samples, root/'coordinate-update003.pt')
        rows.append(row); (root/'trace.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='coordinate_component_gradients'}),flush=True)
    torch.save({'four_rank_model':models[0].state_dict(),'eight_rank_model':models[1].state_dict()},root/'models.pt')


if __name__ == '__main__':
    main()
