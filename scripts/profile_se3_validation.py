#!/usr/bin/env python3
"""Measure cached refiner forward/backward and mixed-length padding costs.

No optimizer steps or production model calls. Run after the pilot completes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import yaml
from torch_geometric.data import HeteroData, Batch
from foldtree2.se3_validation import losses, length_bucket_batches, production_pair, sha256
from foldtree2.src.se3_struct_decoder import se3_denoiser


def collate_cached(entries):
    """Pack nearby-length frozen entries without cross-chain contact edges."""
    graphs = []
    max_n = max(len(e['coords']) for e in entries)
    dots = []
    for e in entries:
        g = HeteroData(); g['res'].x = e['features']; graphs.append(g)
        n = len(e['coords'])
        dot = e['dot'].new_zeros(max_n, max_n)
        dot[:n, :n] = e['dot']; dots.append(dot)
    return Batch.from_data_list(graphs), {
        'coords_pred': torch.cat([e['coords'] for e in entries]),
        'ft2_token_ids': torch.cat([e['tokens'] for e in entries]),
        'edge_attr_dict': {'dot_prod': torch.stack(dots), 'node_mask': torch.cat([e['mask'] for e in entries]),
                           'use_distance_contacts': True, 'distance_contact_cutoff': 8.},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/se3_validation_pilot.yaml')
    parser.add_argument('--checkpoint', default='runs/se3_validation/pilot/best.pt')
    parser.add_argument('--output', default='runs/se3_validation/dense_profile.json')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(config['device'])
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    identity = ckpt['provenance']['identity']
    pair = production_pair(config['manifest'])
    if identity['dataset_sha256'] != sha256(config['dataset']) or any(
        identity[key+'_sha256'] != sha256(pair[key]) for key in ('encoder','decoder')):
        raise ValueError('Profile dataset/checkpoints differ from the trained refiner provenance')
    if ckpt['provenance']['implementation_sha256'] != sha256('foldtree2/src/se3_struct_decoder.py'):
        raise ValueError('Refiner implementation differs from the checkpoint')
    cache_dir = Path(config['cache']) / hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    entries = []
    for identifier in ckpt['provenance']['train_ids']:
        saved = torch.load(cache_dir/(hashlib.sha256(identifier.encode()).hexdigest()+'.pt'),weights_only=False)
        if saved['identity'] != identity or saved['id'] != identifier:
            raise ValueError('Profile cache identity mismatch')
        entries.append((identifier,saved['entry']))
    chosen = []
    for upper in (64,128,256,384):
        eligible = [(i,e) for i,e in entries if len(e['coords'])<=upper]
        chosen.append(max(eligible,key=lambda pair:len(pair[1]['coords'])))
    groups = [[e] for e in chosen]
    groups += [[chosen[0],chosen[-1]], [chosen[-2],chosen[-1]]]
    # Also demonstrate the same bucket sampler used by the canonical runner.
    lengths = [len(e['coords']) for _,e in chosen]
    groups += [[chosen[i] for i in b] for b in length_bucket_batches(lengths,2,42)]
    model = se3_denoiser(chosen[0][1]['features'].shape[-1],[32],3,40,.25,dropout_p=0.,
                        depth=2,heads=2,dim_head=16,num_atom_types=40).to(device).float().eval()
    model.load_state_dict(ckpt['model'])
    rows=[]
    for group in groups:
        ids = [identifier for identifier,_ in group]
        local = [{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in e.items()} for _,e in group]
        def step():
            model.zero_grad(set_to_none=True)
            graph, kwargs = collate_cached(local)
            out = model(graph,**kwargs)
            loss_parts=[]; offset=0
            for e in local:
                n=len(e['coords'])
                total,*_ = losses(out['coors_out_flat'][offset:offset+n],out['angles'][offset:offset+n],e)
                loss_parts.append(total); offset+=n
            torch.stack(loss_parts).mean().backward()
        step(); torch.cuda.synchronize(device)  # warm this shape first
        durations=[]; peaks=[]
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            started=time.perf_counter(); step(); torch.cuda.synchronize(device)
            durations.append(time.perf_counter()-started); peaks.append(torch.cuda.max_memory_allocated(device))
        ns=[len(e['coords']) for e in local]; maximum=max(ns)
        rows.append({'ids':ids,'lengths':ns,'valid_nodes':[int(e['mask'].sum()) for e in local],
                     'batch_size':len(local),'padding_nodes':len(ns)*maximum-sum(ns),
                     'padded_adjacency_bytes':len(ns)*maximum**2,
                     'unpadded_adjacency_bytes':sum(n*n for n in ns),
                     'finite_gradients':all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
                     'mean_forward_backward_seconds':sum(durations)/len(durations),
                     'peak_gpu_allocated_bytes':max(peaks),'mode':'eval with backward; no optimizer step'})
        print(json.dumps(rows[-1]),flush=True)
    Path(args.output).write_text(json.dumps({'checkpoint':args.checkpoint,'provenance':ckpt['provenance'],
                                            'measurements':rows},indent=2))


if __name__=='__main__':
    main()
