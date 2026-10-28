import json
from types import SimpleNamespace

import torch

from foldtree2.src.distance_restraints import eligible_pairs
from scripts.se3.compare_distance_capacity import train_width


def test_capacity_trials_preserve_budgets_panel_and_checkpoint_width(tmp_path):
    generator = torch.Generator().manual_seed(71)
    entry = {'id': 'synthetic', 'mask': torch.ones(32, dtype=torch.bool),
             'features': torch.randn(32, 8, generator=generator),
             'contact_z': torch.nn.functional.normalize(torch.randn(32, 30, generator=generator), dim=-1),
             'aa_probs': torch.randn(32, 20, generator=generator).softmax(-1),
             'target': torch.randn(32, 3, generator=generator) * 5.}
    records = [(entry, eligible_pairs(entry['mask']))]
    manifest = {'feature_dim': 8, 'decoder_enrichment': {'contact_dim': 30},
                'bundle_sha256': 'synthetic'}
    args = SimpleNamespace(device='cpu', steps=2, seconds=60., log_every=1)
    for width in (16, 32):
        summary = train_width(records, manifest, width, tmp_path, args, lambda *a, **k: None)
        assert summary['updates'] == 2 and summary['stop_reason'] == 'steps'
        assert summary['reload_exact']
        checkpoint = torch.load(tmp_path / f'width_{width}' / 'last.pt', weights_only=False)
        assert checkpoint['model_config']['hidden'] == width
        assert checkpoint['step'] == 2 and checkpoint['ids'] == ['synthetic']
        assert 'optimizer' in checkpoint and not checkpoint['resumable']
        history = json.loads((tmp_path / f'width_{width}' / 'history.json').read_text())
        assert [x['step'] for x in history] == [1, 2]
        assert all(x['elapsed_seconds'] > 0 for x in history)
