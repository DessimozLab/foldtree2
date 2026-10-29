"""Versioned AA/contact enrichment; parent frozen entries remain unchanged."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import copy
import json
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from foldtree2.se3_cache import FrozenDataset, digest, identity_hash, open_bundle
from foldtree2.se3_validation import base, production_pair, sha256, tensor_digest
from scripts.se3.diagnose_se3_reconstruction import atomic_json, atomic_checkpoint


def frozen_aa(aa_decoder, latent, positions):
    graph = HeteroData()
    graph['res'].x = latent.clone()
    graph['positions'].x = positions
    with torch.no_grad():
        logits = aa_decoder(graph)['aa']
    if logits.shape == (len(latent), 1, 20):
        logits = logits[:, 0, :]
    if logits.shape != (len(latent), 20) or not torch.isfinite(logits).all():
        raise ValueError('Expected finite [N,20] AA logits')
    return logits.float().softmax(-1)


def enrichment_sources():
    files = ['scripts/se3/enrich_se3_decoder_cache.py', 'foldtree2/src/mono_decoders.py',
             'foldtree2/se3_cache.py']
    return {name: digest(name) for name in files}


def enrich(args):
    root = Path(args.output)
    if root.exists():
        raise ValueError('Enrichment requires a fresh output directory')
    parent = open_bundle(args.cache)
    pair = production_pair()
    if any(sha256(pair[key]) != parent['input_identity'][key + '_sha256'] for key in pair):
        raise ValueError('Frozen checkpoint identity differs from parent cache')
    if sha256(args.dataset) != parent['input_identity']['dataset_sha256']:
        raise ValueError('Original graph dataset identity differs')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    decoder = torch.load(pair['decoder'], weights_only=False, map_location='cpu')
    if not isinstance(decoder, torch.nn.Module):
        raise ValueError('Expected serialized MultiMonoDecoder checkpoint')
    aa = decoder.decoders['sequence_transformer'].to(device).eval()
    for parameter in aa.parameters():
        parameter.requires_grad_(False)
    before = tensor_digest(aa)
    geometry = decoder.decoders['geometry_cnn']
    if not geometry.normalize:
        raise ValueError('Cached contact_z requires the normalized contact decoder')
    temperature = float(geometry.contact_temp.detach())
    bias = float(geometry.contact_bias.detach())
    if not torch.isfinite(torch.tensor([temperature, bias])).all():
        raise ValueError('Nonfinite frozen contact calibration')
    graphs = base.pdbgraphmk2.StructureDataset(args.dataset)
    root.mkdir(parents=True)
    (root / 'entries').mkdir()
    atomic_json(root / 'preparation.json', {'status': 'in_progress', 'parent': parent['bundle_sha256']})
    rows, files = [], {}
    sources = enrichment_sources()
    try:
        for split in ('train', 'val'):
            dataset = FrozenDataset(args.cache, parent, split)
            for index, identifier in enumerate(dataset.ids):
                entry = dataset[index]
                graph = graphs[identifier]
                positions = graph['positions'].x.to(device).float()
                latent = entry['latent'].to(device)
                probabilities = frozen_aa(aa, latent, positions).cpu()
                if index == 0:
                    repeated = frozen_aa(aa, latent, positions).cpu()
                    torch.testing.assert_close(repeated, probabilities, atol=0, rtol=0)
                contact = entry['contact_z'].float()
                contact_probs = torch.sigmoid(temperature * (contact @ contact.T) + bias).clamp(1e-6, 1 - 1e-6)
                if not torch.isfinite(contact_probs).all():
                    raise ValueError('Nonfinite frozen contact probabilities')
                target = graph['AA'].x
                if target.shape != probabilities.shape:
                    raise ValueError('AA labels differ from decoder output shape')
                mask = entry['mask'] & torch.isfinite(target).all(-1) & (target.sum(-1) > 0)
                if not mask.any():
                    raise ValueError('No valid AA evaluation residues')
                labels = target.argmax(-1)
                accuracy = float((probabilities.argmax(-1)[mask] == labels[mask]).float().mean())
                p = probabilities[entry['mask']].clamp_min(1e-12)
                rows.append({'id': identifier, 'split': split, 'aa_accuracy': accuracy,
                             'valid_aa_residues': int(mask.sum()),
                             'aa_correct_residues': int((probabilities.argmax(-1)[mask] == labels[mask]).sum()),
                             'mean_entropy': float(-(p * p.log()).sum(-1).mean())})
                # Preserve every parent field, including original features/seed/contact_z.
                saved = torch.load(Path(args.cache) / 'entries' / parent['files'][identifier]['name'],
                                   map_location='cpu', weights_only=False)
                saved['entry']['aa_probs'] = probabilities
                saved['entry']['contact_probs'] = contact_probs
                name = parent['files'][identifier]['name']
                atomic_checkpoint(root / 'entries' / name, saved)
                files[identifier] = {**parent['files'][identifier],
                                     'sha256': digest(root / 'entries' / name)}
                if (index + 1) % 32 == 0:
                    print(json.dumps({'event': 'enrichment_progress', 'split': split,
                                      'count': index + 1, 'total': len(dataset)}), flush=True)
        if tensor_digest(aa) != before or any(p.grad is not None for p in aa.parameters()):
            raise RuntimeError('Frozen AA decoder weights or gradients changed')
        if sources != enrichment_sources():
            raise RuntimeError('Enrichment implementation changed during extraction')
        manifest = copy.deepcopy(parent)
        manifest.pop('bundle_sha256')
        manifest.update(files=files, parent_bundle_sha256=parent['bundle_sha256'],
                        decoder_enrichment={
                            'protocol': 'frozen-decoder-enrichment-v1', 'sources': sources,
                            'input': 'cached continuous latent; not quantized-only',
                            'aa_branch': 'sequence_transformer', 'aa_input': 'latent + existing positions',
                            'contact_dim': int(contact.shape[-1]), 'aa_dim': 20,
                            'temperature': temperature, 'bias': bias, 'torch': torch.__version__,
                            'device': str(device), 'frozen_aa_sha256': before,
                        })
        manifest['bundle_sha256'] = identity_hash(manifest)
        atomic_json(root / 'aa_quality.json', rows)
        atomic_json(root / 'manifest.json', manifest)
        open_bundle(root)
        for name in sources:
            path = root / 'sources' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(name).read_bytes())
        atomic_json(root / 'preparation.json', {
            'status': 'complete', 'entries': len(files), 'frozen_weights_unchanged': True,
            'parent_bundle_sha256': parent['bundle_sha256'], 'bundle_sha256': manifest['bundle_sha256'],
            'aa_accuracy_by_split': {
                split: sum(r['aa_correct_residues'] for r in rows if r['split'] == split) /
                sum(r['valid_aa_residues'] for r in rows if r['split'] == split)
                for split in ('train', 'val')},
        })
        print(json.dumps({'event': 'complete', 'entries': len(files)}), flush=True)
    except Exception as exc:
        (root / 'manifest.json').unlink(missing_ok=True)
        atomic_json(root / 'failure.json', {'error': repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_generalization_20261021/bundle')
    parser.add_argument('--dataset', default='foldtree2/notebooks/structs_training_mk2.h5')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    enrich(args)


if __name__ == '__main__':
    main()