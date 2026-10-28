import json
from types import SimpleNamespace

from scripts import queue_distance_generalization as queue


def test_queue_finishes_parent_cancels_service_and_preserves_matched_width_budgets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parent = tmp_path / 'runs/se3_distance_capacity_20261022/width_128'
    parent.mkdir(parents=True)
    (parent / 'completion.json').write_text(json.dumps({'stop_reason': 'steps', 'updates': 40000}))
    monkeypatch.setattr(queue, 'open_bundle', lambda _: {
        'train_ids': list(range(256)), 'val_ids': list(range(256, 320)), 'bundle_sha256': 'bundle'})
    monkeypatch.setattr(queue, 'sources', lambda: {})
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert kwargs['check']
        if command[0] == 'systemctl':
            return
        assert command[command.index('--epochs') + 1] == '128'
        assert command[command.index('--pairs') + 1] == '512'
        assert command[command.index('--seconds') + 1] == '3600'
        output = tmp_path / command[command.index('--output') + 1] / 'decoder'
        output.mkdir(parents=True)
        (output / 'completion.json').write_text(json.dumps({
            'stop_reason': 'epochs', 'updates': 32768}))

    monkeypatch.setattr(queue.subprocess, 'run', execute)
    queue.run(SimpleNamespace(cache='cache', output='trial'))
    assert calls[0][:3] == ['systemctl', '--user', 'stop']
    assert [c[c.index('--hidden') + 1] for c in calls[1:]] == ['64', '128', '256']
    assert json.loads((tmp_path / 'trial/completion.json').read_text())['complete']
