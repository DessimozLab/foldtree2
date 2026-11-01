#!/usr/bin/env python3
"""Evaluate a trained PairwiseDistanceHead on nonlocal pairs (min_sep=12)."""
import torch, sys
from foldtree2.se3_cache import FrozenDataset, open_bundle
from foldtree2.src.distance.distance_restraints import PairwiseDistanceHead, eligible_pairs, distance_labels

device = torch.device('cuda:0')
ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
model = PairwiseDistanceHead(**ckpt['model_config']).to(device)
model.load_state_dict(ckpt['model']); model.eval()
for p in model.parameters(): p.requires_grad = False

manifest = open_bundle('runs/se3_distance_data_scaling_10240_20261024/enriched')
ds = FrozenDataset('runs/se3_distance_data_scaling_10240_20261024/enriched', manifest, 'val')

all_p, all_l, all_d = [], [], []
for idx in range(len(ds)):
    e = ds[idx]; i = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in e.items()}
    pairs = eligible_pairs(i['mask'], min_separation=12).to(device)
    labels, dists = distance_labels(i['target'], pairs, model.edges)
    with torch.no_grad(): logits = model(i, pairs)
    all_p.append(logits.softmax(-1).cpu()); all_l.append(labels.cpu()); all_d.append(dists.cpu())

p = torch.cat(all_p); l = torch.cat(all_l); d = torch.cat(all_d)
contact = p[:, :3].sum(-1); ctc = l < 3; true = int(ctc.sum())

# AP
order = contact.argsort(descending=True, stable=True)
ranked = ctc[order].float()
precision = ranked.cumsum(0) / torch.arange(1, len(ranked)+1)
scores = contact[order]
ends = torch.cat([(scores[:-1]!=scores[1:]).nonzero().flatten(), scores.new_tensor([len(scores)-1], dtype=torch.long)])
cum = ranked.cumsum(0)[ends]
inc = cum - torch.cat([cum.new_zeros(1), cum[:-1]])
ap = float((precision[ends] * inc).sum() / true) if true else 0.

# Bin acc & MAE
bin_acc = (p.argmax(-1) == l).float().mean().item()
edges_t = torch.tensor([4.,6.,8.,10.,12.,16.,20.,24.,32.]).float()
lower = torch.cat([edges_t.new_zeros(1), edges_t])
upper = torch.cat([edges_t, edges_t[-1:]])
mid = (lower + upper) / 2
capped = (p @ mid).cpu(); d_cpu = d.cpu()
mae = (capped - d_cpu.clamp_max(32.)).abs().mean().item()

pred = contact >= 0.5; npred = int(pred.sum()); shared = int((pred & ctc).sum())
print(f'Nonlocal (min_sep=12) metrics:')
print(f'Contact AP:  {ap:.4f}')
print(f'Bin acc:     {bin_acc:.3f}')
print(f'MAE (A):     {mae:.2f}')
print(f'Precision:   {shared/npred if npred else None}')
print(f'Recall:      {shared/true if true else None}')
print(f'nPred/True:  {npred}/{true}')
print(f'Pairs:       {len(p)}')
print()
print(f'Previous best: AP 0.119 bin 0.422 MAE 4.73')