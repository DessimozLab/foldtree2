"""Generate only added frozen entries on a Clariden GPU, with parity gates."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace
import torch
from torch_geometric.data import HeteroData
from foldtree2.se3_cache import digest, identity_hash, open_bundle
from foldtree2.se3_validation import base, frozen_entry, load_production_decoder, tensor_digest
from scripts.submit_se3_growth_trial import check_growth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--previous-bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report = Path(args.report)
    def event(kind, **values):
        row = {'event': kind, 'time': time.time(), **values}
        with report.with_suffix('.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    inputs = Path(args.inputs)
    source = json.loads((inputs / 'manifest.json').read_text())
    check = dict(source)
    if identity_hash({k: v for k, v in check.items() if k != 'export_sha256'}) != check['export_sha256']:
        raise ValueError('Graph export identity differs')
    identity = source['identity']
    if hashlib.sha256(inspect.getsource(frozen_entry).encode()).hexdigest() != identity['preprocessing']['extractor_sha256']:
        raise ValueError('Frozen extractor differs')
    for key in ('encoder', 'decoder'):
        if digest(inputs / (key + '.pt')) != identity[key + '_sha256']:
            raise ValueError('Production checkpoint hash differs')
    def graph(identifier):
        record = source['graphs'][identifier]
        path = inputs / 'graphs' / record['name']
        if path.parent != inputs / 'graphs' or digest(path) != record['sha256']:
            raise ValueError('Source graph hash differs')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['id'] != identifier:
            raise ValueError('Source graph ID differs')
        return HeteroData.from_dict(payload['graph'])
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('Clariden extraction requires CUDA')
    previous_root = Path(args.previous_bundle)
    previous = open_bundle(previous_root)
    parity_id = source['parity_id']
    probe = graph(parity_id).to(device)
    encoder, _ = base.build_encoder(SimpleNamespace(pretrained_encoder_path=str(inputs / 'encoder.pt'),
        pretrained_encoder_full_path=str(inputs / 'encoder.pt')), probe, device)
    if isinstance(encoder, base.FrozenProjectionEncoder):
        raise RuntimeError('Encoder fallback forbidden')
    decoder = load_production_decoder(str(inputs / 'decoder.pt'), device).float().eval()
    encoder = encoder.float().eval()
    for model in (encoder, decoder):
        model.requires_grad_(False)
    before = [tensor_digest(m) for m in (encoder, decoder)]
    fresh = frozen_entry(graph(parity_id), encoder, decoder, device)
    cached = torch.load(previous_root / 'entries' / previous['files'][parity_id]['name'],
                        map_location='cpu', weights_only=False)['entry']
    differences = {}
    for key, value in cached.items():
        if isinstance(value, torch.Tensor):
            delta = (value.float() - fresh[key].float()).abs()
            finite = delta[torch.isfinite(delta)]
            differences[key] = float(finite.max()) if finite.numel() else 0.0
            torch.testing.assert_close(value, fresh[key], atol=1e-6, rtol=1e-6, equal_nan=True)
    event('parity_verified', id=parity_id, differences=differences)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(previous_root / 'entries', root / 'entries')
    files = dict(previous['files'])
    identifiers = source['bundle_template']['train_ids'][4096:]
    for index, identifier in enumerate(identifiers):
        entry = frozen_entry(graph(identifier), encoder, decoder, device)
        if index == 0:
            again = frozen_entry(graph(identifier), encoder, decoder, device)
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, again[key], atol=1e-6, rtol=1e-6, equal_nan=True)
        name = hashlib.sha256(identifier.encode()).hexdigest() + '.pt'
        path = root / 'entries' / name
        torch.save({'id': identifier, 'identity': identity, 'entry': entry}, path)
        files[identifier] = {'name': name, 'sha256': digest(path), 'length': len(entry['coords'])}
        if (index + 1) % 128 == 0:
            event('cache_progress', count=index + 1, total=len(identifiers))
    if before != [tensor_digest(m) for m in (encoder, decoder)] or any(p.grad is not None for m in (encoder, decoder) for p in m.parameters()):
        raise RuntimeError('Frozen production state changed or received gradients')
    manifest = {**source['bundle_template'], 'files': files}
    manifest['extraction_producer'] = {'export_sha256': source['export_sha256'], 'torch': torch.__version__,
        'python': platform.python_version(), 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device),
        'script_sha256': digest(__file__), 'retained_bundle': previous['bundle_sha256']}
    manifest['bundle_sha256'] = identity_hash(manifest)
    check_growth(previous, manifest)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    open_bundle(root)
    result = {'passed': True, 'bundle_sha256': manifest['bundle_sha256'], 'new_entries': len(identifiers),
              'retained_entries': len(previous['files']), 'parity': differences, 'frozen_state_unchanged': True}
    report.write_text(json.dumps(result, indent=2) + '\n')
    event('complete', **result)


if __name__ == '__main__':
    main()
