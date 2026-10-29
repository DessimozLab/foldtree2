"""Replay the canonical medium stream and stop at its first nonfinite module.

Uses the frozen cache only; does not allocate cluster resources or modify runs.
The serial global batch has the same samples and mean gradient as the DDP run.
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import gotennet_pytorch.gotennet as _gotennet_backend
import pytorch_lightning as pl
from foldtree2.se3_cache import FrozenDataset, open_bundle, global_order
from foldtree2.train_se3_lightning import build_model, calibrated
from foldtree2.se3_validation import forward
from foldtree2.src.losses.se3_ablation import AblationLoss


class StableHierarchicalTensorRefinement(_gotennet_backend.HierarchicalTensorRefinement):
    """Rescale tensor products before LayerNorm without changing their ratios.

    A shared scale across residues, channels and degrees preserves all query/key
    products. LayerNorm's epsilon is divided by the square of the product scale.
    Scaling only the final normalization cannot recover overflowing products.
    """
    def forward(self, t_ij, x, neighbor_indices=None):
        if not isinstance(self.norm, nn.LayerNorm):
            return super().forward(t_ij, x, neighbor_indices=neighbor_indices)
        scale = torch.stack([v.detach().abs().amax(dim=(-3, -2, -1), keepdim=True)
                             for v in x]).amax(dim=0).clamp_min(1.)
        queries = [torch.einsum('... d m, d e -> ... e m', v / scale, self.to_queries) for v in x]
        keys = [torch.einsum('... d m, d e -> ... e m', v / scale, k) for v, k in zip(x, self.to_keys)]
        if neighbor_indices is not None:
            keys = [_gotennet_backend.get_at('b [n] d m, b i j -> b i j d m', k, neighbor_indices) for k in keys]
            products = [torch.einsum('... i d m, ... i j d m -> ... i j d', q, k) for q, k in zip(queries, keys)]
        else:
            products = [torch.einsum('... i d m, ... j d m -> ... i j d', q, k) for q, k in zip(queries, keys)]
        w = torch.cat(products, dim=-1)
        centered = w - w.mean(dim=-1, keepdim=True)
        eps = self.norm.eps / scale / scale / scale / scale
        w = centered * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + eps)
        if self.norm.weight is not None:
            w = w * self.norm.weight
        if self.norm.bias is not None:
            w = w + self.norm.bias
        return torch.sigmoid(self.edge_proj(w)) * self.residue_update(t_ij)


def enable_stable_tensor_products(model):
    """Opt-in diagnostic fix; leave unrelated GotenNet instances unchanged."""
    for module in model.modules():
        if type(module) is _gotennet_backend.HierarchicalTensorRefinement:
            module.__class__ = StableHierarchicalTensorRefinement
    return model



def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


def summary(value):
    return [{'shape': list(t.shape), 'finite': bool(torch.isfinite(t).all()),
             'max_abs': float(t.detach().abs().max())} for t in tensors(value)
            if t.numel() and t.is_floating_point()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--updates', type=int, default=16)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--stable-products', action='store_true')
    p.add_argument('--normalize-ff', action='store_true', help='Architectural ablation: normalize equivariant MLP inputs')
    p.add_argument('--replay-failure', help='Replay a saved model/input before any further training')
    p.add_argument('--contraction-audit', action='store_true', help='Compare the fourth-layer tensor product in float32 and float64')
    args = p.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); pl.seed_everything(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = open_bundle(args.cache)
    dataset = FrozenDataset(args.cache, manifest, 'train')
    groups = global_order([manifest['files'][i]['length'] for i in dataset.ids], 42, 0)
    model = build_model(manifest['feature_dim'], {'hidden':64, 'depth':4, 'heads':4}).to(args.device).train()
    if args.stable_products:
        enable_stable_tensor_products(model)
    if args.normalize_ff:
        from foldtree2.src.models.se3_struct_decoder import enable_tensor_ff_input_norm
        enable_tensor_ff_input_norm(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.01)
    loss = AblationLoss(True, True)
    context = {}; recent = []

    def hook(name):
        def check(module, inputs, output):
            record = {'module':name, 'type':type(module).__name__,
                      'input':summary(inputs), 'output':summary(output)}
            recent.append(record)
            del recent[:-24]
            if any(not x['finite'] for x in record['output']):
                (root / 'failure.json').write_text(json.dumps({'context':context, 'recent':recent}, indent=2))
                torch.save({'model':model.state_dict(), 'entry':entry, 'context':context}, root / 'failure.pt')
                raise FloatingPointError(f'First nonfinite module: {name}')
        return check

    for name, module in model.named_modules():
        if name.startswith('gotennet.'):
            module.register_forward_hook(hook(name))
    if args.contraction_audit:
        def audit(module, inputs):
            _, x = inputs
            v = x[0]
            q = torch.einsum('... d m, d e -> ... e m', v, module.to_queries)
            k = torch.einsum('... d m, d e -> ... e m', v, module.to_keys[0])
            product = torch.einsum('b i d m, b j d m -> b i j d', q, k)
            reference = torch.einsum('b i d m, b j d m -> b i j d', q.double(), k.double())
            (root / 'contraction.json').write_text(json.dumps({
                'context':context, 'queries':summary(q), 'keys':summary(k),
                'float32_product_finite':bool(torch.isfinite(product).all()),
                'float64_product_finite':bool(torch.isfinite(reference).all()),
                'float64_product_max_abs':float(reference.abs().max()),
                'float32_max':torch.finfo(torch.float32).max}, indent=2))
        model.gotennet.layers[3][0].register_forward_pre_hook(audit)
    if args.replay_failure:
        saved = torch.load(args.replay_failure, map_location=args.device, weights_only=False)
        model.load_state_dict(saved['model'])
        entry = saved['entry']; context.update(saved['context'])
        pred, angle = forward(model, entry)
        values = loss(pred, angle, entry)
        values[0].backward()
        print(json.dumps({'replay':'finite', 'loss':float(values[0]),
                          'finite_gradients':all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)}), flush=True)
        return
    for update, group in enumerate(groups[:args.updates]):
        losses = []
        for position, index in enumerate(group):
            batch = dataset[index]
            entry = {k:v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k,v in calibrated(batch, manifest['scale']).items()}
            context.update(update=update, position=position, id=batch['id'], length=len(batch['coords']))
            pred, angle = forward(model, entry)
            values = loss(pred, angle, entry)
            (values[0] / 16).backward()
            losses.append(float(values[1].detach()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        record = {'update':update+1, 'geometry':sum(losses)/16, 'gradient_norm':float(norm)}
        with (root / 'events.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
