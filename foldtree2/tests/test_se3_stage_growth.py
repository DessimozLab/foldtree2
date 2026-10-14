import hashlib
import json
import pytest
from scripts import stage_se3_growth


def setup_bundle(tmp_path, monkeypatch, corrupt=False):
    root = tmp_path / 'durable'
    source = root / 'cache/bundle_growth_8192'
    (source / 'entries').mkdir(parents=True)
    (root / 'artifacts').mkdir()
    payload = b'frozen payload'
    manifest = {'files': {'one': {'name': 'one.pt', 'sha256': hashlib.sha256(payload).hexdigest()}}}
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest['bundle_sha256'] = identity
    (source / 'manifest.json').write_text(json.dumps(manifest))
    (source / 'entries/one.pt').write_bytes(b'corrupted' if corrupt else payload)
    (root / 'artifacts/growth-extraction.json').write_text(json.dumps({'passed': True, 'bundle_sha256': identity}))
    target = tmp_path / 'scratch'
    monkeypatch.setattr(stage_se3_growth, 'Path', lambda p: root if str(p).startswith('/capstor') else target)
    return source, target, root


def test_verified_payloads_publish_complete_manifest(tmp_path, monkeypatch):
    source, target, root = setup_bundle(tmp_path, monkeypatch)
    stage_se3_growth.main()
    assert (target / 'manifest.json').read_bytes() == (source / 'manifest.json').read_bytes()
    assert (target / 'entries/one.pt').read_bytes() == b'frozen payload'
    assert json.loads((root / 'artifacts/growth-stage-in.json').read_text())['passed']


def test_corrupt_payload_does_not_publish_manifest(tmp_path, monkeypatch):
    source, target, root = setup_bundle(tmp_path, monkeypatch, corrupt=True)
    with pytest.raises(ValueError, match='Copied entry differs'):
        stage_se3_growth.main()
    assert not (target / 'manifest.json').exists()
    assert not (root / 'artifacts/growth-stage-in.json').exists()
