"""Offline TCS tests: no cluster connection and no taxonomy downloads."""
import sys
from pathlib import Path

import pytest
from ete3 import Tree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from curnagl_tcs import scoring_functions, tcs


PURE_BASELINE = '''
raise RuntimeError("Global code must never execute")
def frac_score(v): return v + 1
def getTaxOverlap(node, treelen=None, scorefun=frac_score):
    if node.is_leaf():
        node.add_feature('score', 0)
        return node.lineage
    sets = [getTaxOverlap(c) for c in node.children]
    node.add_feature('lineage', set.intersection(*sets))
    node.add_feature('score', len(node.lineage) + sum(c.score for c in node.children))
    return node.lineage
def getTaxOverlap_root(node, clades=None):
    if node.is_root():
        universal = set.intersection(*(l.lineage for l in node))
        clades = {x: int(x not in universal) for l in node for x in l.lineage}
        previous = previous_nr = 0
    else:
        previous, previous_nr = node.up.root_score, node.up.root_score_nr
    node.add_feature('root_score', previous + len(node.lineage))
    node.add_feature('root_score_nr', previous_nr + sum(clades[x] for x in node.lineage))
    for child in node.children: getTaxOverlap_root(child, clades)
def sum_rootscore(node):
    return sum(l.root_score for l in node), sum(l.root_score_nr for l in node)
'''


def test_baseline_scoring_is_offline_and_matches_known_scores(tmp_path):
    source = tmp_path / 'baseline.py'
    source.write_text(PURE_BASELINE)
    lineages = {'a': {'root', 'X'}, 'b': {'root', 'X'}, 'c': {'root', 'Y'}, 'd': {'root', 'Y'}}
    scores = tcs(Tree('((a:1,b:1):1,(c:1,d:1):1);'), lineages, scoring_functions(source))
    assert scores == {'score': 5, 'root_score': 20, 'root_score_nr': 8}
    mixed = tcs(Tree('((a:1,c:1):1,(b:1,d:1):1);'), lineages, scoring_functions(source))
    assert mixed == {'score': 3, 'root_score': 16, 'root_score_nr': 4}


def test_missing_extra_duplicate_or_empty_taxonomy_is_rejected(tmp_path):
    source = tmp_path / 'baseline.py'
    source.write_text(PURE_BASELINE)
    funcs = scoring_functions(source)
    with pytest.raises(ValueError, match='taxa'):
        tcs(Tree('(a:1,b:1);'), {'a': {'root'}}, funcs)
    with pytest.raises(ValueError, match='taxa'):
        tcs(Tree('(a:1,a:1);'), {'a': {'root'}}, funcs)
    with pytest.raises(ValueError, match='Missing taxonomy'):
        tcs(Tree('(a:1,b:1);'), {'a': {'root'}, 'b': set()}, funcs)


def test_incomplete_baseline_fails(tmp_path):
    source = tmp_path / 'bad.py'
    source.write_text('def frac_score(v): return v')
    with pytest.raises(ValueError, match='functions missing'):
        scoring_functions(source)


def test_supervisor_retries_only_infrastructure_failures(tmp_path, monkeypatch):
    import json
    import supervise_curnagl_tcs as supervisor
    state = {'phase': 'pilot', 'jobs': {'pilot': '123'}, 'attempts': {'pilot': 0}, 'status': 'active'}
    (tmp_path / 'supervision.json').write_text(json.dumps(state))
    monkeypatch.setattr(supervisor, 'shutil_disk_free', lambda root: 100 * 1024**3)
    monkeypatch.setattr(supervisor, 'accounting', lambda job: ('TIMEOUT', []))
    monkeypatch.setattr(supervisor, 'submit', lambda root, phase: '456')
    updated = supervisor.tick(tmp_path)
    assert updated['jobs']['pilot'] == '456'
    assert updated['attempts']['pilot'] == 1
    monkeypatch.setattr(supervisor, 'accounting', lambda job: ('OUT_OF_MEMORY', []))
    assert supervisor.tick(tmp_path)['status'] == 'needs_attention'


def test_supervisor_retry_cap_and_storage_gate(tmp_path, monkeypatch):
    import json
    import supervise_curnagl_tcs as supervisor
    monkeypatch.setattr(supervisor, 'shutil_disk_free', lambda root: 100 * 1024**3)
    monkeypatch.setattr(supervisor, 'accounting', lambda job: ('NODE_FAIL', []))
    state = {'phase': 'pilot', 'jobs': {'pilot': '123'}, 'attempts': {'pilot': 2}, 'status': 'active'}
    (tmp_path / 'supervision.json').write_text(json.dumps(state))
    assert supervisor.tick(tmp_path)['status'] == 'needs_attention'
    (tmp_path / 'supervision.json').unlink()
    monkeypatch.setattr(supervisor, 'shutil_disk_free', lambda root: 10 * 1024**3)
    assert supervisor.tick(tmp_path)['status'] == 'needs_attention'


def test_failed_slurm_dependency_stops_instead_of_waiting_forever(tmp_path, monkeypatch):
    import json
    import supervise_curnagl_tcs as supervisor
    monkeypatch.setattr(supervisor, 'execute', lambda argv: '123|PENDING|DependencyNeverSatisfied')
    assert supervisor.accounting('123') == ('DEPENDENCY_FAILED', [])
    state = {'phase': 'inventory', 'jobs': {'inventory': '123'},
             'attempts': {'inventory': 0}, 'status': 'active'}
    (tmp_path / 'supervision.json').write_text(json.dumps(state))
    monkeypatch.setattr(supervisor, 'shutil_disk_free', lambda root: 100 * 1024**3)
    updated = supervisor.tick(tmp_path)
    assert updated['status'] == 'needs_attention'
    assert 'DEPENDENCY_FAILED' in updated['reason']
    assert updated['jobs'] == {'inventory': '123'}


def test_pilot_gate_requires_coverage_and_accounted_memory(tmp_path, monkeypatch):
    import hashlib
    import json
    import supervise_curnagl_tcs as supervisor
    monkeypatch.setattr(supervisor, 'shutil_disk_free', lambda root: 100 * 1024**3)
    monkeypatch.setattr(supervisor, 'accounting', lambda job: ('COMPLETED', [['123', 'COMPLETED', '0:0', '', '1:00'],
                                                            ['123.batch', 'COMPLETED', '0:0', '1024M', '1:00']]))
    state = {'phase': 'pilot', 'jobs': {'pilot': '123'}, 'attempts': {'pilot': 0}, 'status': 'active'}
    (tmp_path / 'supervision.json').write_text(json.dumps(state))
    plan = {'families': {'OMA__1': {}, 'CATH__1': {}}, 'methods': {str(i): {} for i in range(13)},
            'config': {'output_dir': str(tmp_path)}}
    (tmp_path / 'pilot.json').write_text(json.dumps(plan))
    marker = {'scores': 26, 'plan_sha256': hashlib.sha256((tmp_path / 'pilot.json').read_bytes()).hexdigest()}
    (tmp_path / 'validated.json').write_text(json.dumps(marker))
    assert supervisor.tick(tmp_path)['phase'] == 'full'
    state['stop_after_pilot'] = True
    (tmp_path / 'supervision.json').write_text(json.dumps(state))
    updated = supervisor.tick(tmp_path)
    assert updated['status'] == 'pilot_complete'
    assert updated['phase'] == 'pilot'
    assert 'full' not in updated['jobs']


def test_project_quota_is_used_instead_of_whole_filesystem(monkeypatch):
    import shutil
    from types import SimpleNamespace
    import supervise_curnagl_tcs as supervisor
    monkeypatch.setattr(supervisor.os, 'access', lambda *args: False)
    monkeypatch.setattr(shutil, 'disk_usage', lambda root: SimpleNamespace(free=200 * 1000**4))
    monkeypatch.setattr(supervisor, 'execute', lambda args, **kwargs: '\n'.join([
        'Project Quota Used Avail Use%',
        '/work/FAC/FBM/DBC/cdessim2/default 27.00 26.10 0.90 97%',
        '/work/FAC/FBM/DBC/cdessim2/oma 10.25 8.63 1.63 84%']))
    root = Path('/work/FAC/FBM/DBC/cdessim2/default/dmoi/tcs_runs/test')
    assert supervisor.shutil_disk_free(root) == 900 * 1000**3


def test_accounting_falls_back_after_completed_job_leaves_queue(monkeypatch):
    import supervise_curnagl_tcs as supervisor
    def execute(argv):
        if argv[0] == 'squeue':
            assert '--me' in argv and '-j' not in argv
            return '456|RUNNING|None'
        return '123|COMPLETED|0:0||01:00\n123.batch|COMPLETED|0:0|1024M|01:00'
    monkeypatch.setattr(supervisor, 'execute', execute)
    state, rows = supervisor.accounting('123')
    assert state == 'COMPLETED' and len(rows) == 2


def test_quota_uses_longer_bounded_timeout(monkeypatch):
    import shutil
    from types import SimpleNamespace
    import supervise_curnagl_tcs as supervisor
    monkeypatch.setattr(supervisor.os, 'access', lambda *args: False)
    monkeypatch.setattr(shutil, 'disk_usage', lambda root: SimpleNamespace(free=10**12))
    def execute(argv, timeout):
        assert timeout == 120
        return '/work/FAC/project 1.00 0.50 0.50 50%'
    monkeypatch.setattr(supervisor, 'execute', execute)
    assert supervisor.shutil_disk_free(Path('/work/FAC/project/run')) == 500 * 1000**3


def test_callback_waits_on_scheduler_and_exits_on_failed_gate(tmp_path, monkeypatch):
    import json
    import advance_curnagl_tcs as callback
    state = {'phase': 'pilot', 'jobs': {'pilot': '123'}, 'status': 'active', 'slurm_state': 'RUNNING'}
    (tmp_path / 'supervision.json').write_text(json.dumps(state))
    monkeypatch.setattr(callback, 'tick', lambda root: state)
    commands = []
    def execute(argv):
        commands.append(argv)
        return '456'
    monkeypatch.setattr(callback, 'execute', execute)
    assert callback.advance(tmp_path)['callback_job'] == '456'
    assert '--dependency=afterany:123' in commands[0]
    state['status'] = 'needs_attention'
    assert callback.advance(tmp_path)['status'] == 'needs_attention'
    assert len(commands) == 1


def test_gpfs_quota_is_bound_to_fileset_and_reserves_in_doubt_blocks(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    import supervise_curnagl_tcs as supervisor
    monkeypatch.setattr(supervisor.os, 'access', lambda *args: True)
    monkeypatch.setattr(shutil, 'disk_usage', lambda root: SimpleNamespace(free=10**12))
    response = [
        'fileset name: project',
        'mmlsquota:fileset:HEADER:filesystemName:name:blockUsage:blockQuota:blockInDoubt:filesUsage:filesQuota:filesInDoubt:\n'
        'mmlsquota:fileset:0:work:project:100:1000:50:10:100:5:',
    ]
    monkeypatch.setattr(supervisor, 'execute', lambda argv, timeout: response.pop(0))
    assert supervisor.shutil_disk_free(Path('/work/FAC/project/run')) == 850 * 1024
    response[:] = ['fileset name: project',
                  'mmlsquota:fileset:HEADER:filesystemName:name:blockUsage:blockQuota:blockInDoubt:filesUsage:filesQuota:filesInDoubt:\n'
                  'mmlsquota:fileset:0:work:other:100:1000:50:10:100:5:']
    with pytest.raises(RuntimeError, match='Cannot verify GPFS'):
        supervisor.shutil_disk_free(Path('/work/FAC/project/run'))


def test_callback_advances_to_full_only_through_existing_gates(tmp_path, monkeypatch):
    import json
    import advance_curnagl_tcs as callback
    state = {'phase': 'pilot', 'jobs': {'pilot': '123'}, 'status': 'active'}
    (tmp_path / 'supervision.json').write_text(json.dumps(state))
    calls = []
    def tick(root):
        calls.append(1)
        if len(calls) == 1:
            state['phase'] = 'full'
        elif len(calls) == 2:
            state['jobs']['full'] = '456'
        else:
            state['slurm_state'] = 'RUNNING'
        (root / 'supervision.json').write_text(json.dumps(state))
        return dict(state)
    monkeypatch.setattr(callback, 'tick', tick)
    commands = []
    monkeypatch.setattr(callback, 'execute', lambda argv: commands.append(argv) or '789')
    result = callback.advance(tmp_path)
    assert len(calls) == 3 and result['phase'] == 'full'
    assert '--dependency=afterany:456' in commands[0]


def test_callback_accounting_wait_is_delayed_and_bounded(tmp_path, monkeypatch):
    import json
    import advance_curnagl_tcs as callback
    state = {'phase': 'pilot', 'jobs': {'pilot': '123'}, 'status': 'active',
             'slurm_state': 'WAITING_FOR_PEAK_MEMORY_ACCOUNTING'}
    (tmp_path / 'supervision.json').write_text(json.dumps(state))
    monkeypatch.setattr(callback, 'tick', lambda root: state)
    commands = []
    monkeypatch.setattr(callback, 'execute', lambda argv: commands.append(argv) or '456')
    for _ in range(10):
        callback.advance(tmp_path)
    assert all('--begin=now+2minutes' in cmd and not any(a.startswith('--dependency') for a in cmd)
               for cmd in commands)
    with pytest.raises(RuntimeError, match='ten callbacks'):
        callback.advance(tmp_path)
    assert len(commands) == 10
    assert json.loads((tmp_path / 'supervision.json').read_text())['status'] == 'needs_attention'


def test_inventory_checks_domain_sequences_taxonomy_and_complete_cohort(tmp_path):
    import csv
    import hashlib
    import re
    import sqlite3
    from types import SimpleNamespace
    from curnagl_tcs import family_record
    structs = tmp_path / 'structs'
    structs.mkdir()
    rows = []
    for name in 'abcd':
        atom = 'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C  \nEND\n'
        (structs / (name + '.pdb')).write_text(atom)
        rows.append({'query': name, 'Entry': name, 'Sequence': 'MAK',
                     'Taxonomic lineage (Ids)': '1 (root), 2 (species)'})
    metadata = tmp_path / 'sequence_dataset.csv'
    def write_rows(data):
        with metadata.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(data)
    write_rows(rows)
    baseline = SimpleNamespace(SAFE=re.compile(r'^[A-Za-z0-9_.-]+$'),
                               digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE species (taxid INTEGER)')
        db.executemany('INSERT INTO species VALUES (?)', [(1,), (2,)])
        record = family_record(tmp_path, 'CATH', Path('1.2'), db, baseline)
        assert set(record['sequences'].values()) == {'A'}  # Selected domain, not full metadata sequence.
        assert len(record['inputs']) == 5
        write_rows(rows + [{**rows[0], 'query': 'e', 'Entry': 'e'}])
        with pytest.raises(ValueError, match='Metadata/structure taxa differ'):
            family_record(tmp_path, 'CATH', Path('1.2'), db, baseline)
        write_rows(rows)
        db.execute('DELETE FROM species WHERE taxid=2')
        with pytest.raises(ValueError, match='absent from frozen SQLite'):
            family_record(tmp_path, 'CATH', Path('1.2'), db, baseline)


def test_runtime_check_rejects_a_wrapper_with_a_missing_binary(tmp_path, monkeypatch):
    import subprocess
    from check_curnagl_tcs import check_native_tools
    wrapper = tmp_path / 'raxml-ng'
    wrapper.write_text('#!/bin/sh\necho "Error: Executable raxml-ng not found"\nexit 1\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path))
    with pytest.raises(subprocess.CalledProcessError):
        check_native_tools({'three_di_route': 'foldmason'})


def test_cluster_archive_includes_and_hashes_the_native_raxml_binary(tmp_path, monkeypatch):
    import hashlib
    import json
    import tarfile
    import stage_curnagl_tcs as stage
    root = tmp_path / 'source'
    resources = [
        'foldtree2/__init__.py', 'foldtree2/config/aaindex1.csv',
        'foldtree2/config/3diphy_submats/Q.3Di.AF',
        'foldtree2/mafft_tools/hex2maffttext', 'foldtree2/mafft_tools/maffttext2hex',
        'foldtree2/madroot/mad', 'foldtree2/raxml-ng/raxml-ng',
        'foldtree2/raxml-ng/LICENSE.txt', 'workflow/curnagl_tcs.smk',
    ]
    for name in resources:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture\n')
    manifest = root / 'configs/production_models.yaml'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({'models': []}))
    for name in ('benchmark/src/topology_benchmark.py', 'original/src/foldtree/foldseek2tree.py',
                 'tool/bin/foldmason', *['tool/lib/' + lib for lib in (
                     'libatomic.so.1', 'libz.so.1', 'libbz2.so.1.0', 'libgomp.so.1',
                     'libstdc++.so.6', 'libgcc_s.so.1')]):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture\n')
    monkeypatch.setattr(stage, 'ROOT', root)
    monkeypatch.setattr(stage.subprocess, 'check_output', lambda *a, **k: 'fixture-head\n')
    archive = tmp_path / 'deployment.tar.gz'
    stage.build(archive, tmp_path / 'benchmark', tmp_path / 'original', tmp_path / 'tool/bin/foldmason')
    name = 'foldtree2/foldtree2/raxml-ng/raxml-ng'
    with tarfile.open(archive) as packed:
        assert packed.extractfile(name).read() == b'fixture\n'
        assert packed.getmember('foldtree2/foldtree2/raxml-ng/LICENSE.txt')
    provenance = json.loads(archive.with_suffix('.gz.manifest.json').read_text())
    assert provenance['files'][name] == hashlib.sha256(b'fixture\n').hexdigest()


def test_configuration_snapshots_only_the_legacy_subdirectory(tmp_path, monkeypatch):
    import configure_curnagl_tcs as configure
    root = tmp_path / 'run'
    source = root / 'foldtree2/scripts/supervise_curnagl_tcs.py'
    source.parent.mkdir(parents=True)
    source.write_text('# fixture\n')
    template = tmp_path / 'template.yaml'
    template.write_text(f'deployment_dir: {root}\n')
    commands = []
    def output(argv, **kwargs):
        commands.append(argv)
        if argv[1] == 'ls-files':
            return b''
        return 'fixture-head\n' if argv[1] == 'rev-parse' else ''
    def run(argv, **kwargs):
        commands.append(argv)
        kwargs['stdout'].write(b'')
    monkeypatch.setattr(configure.subprocess, 'check_output', output)
    monkeypatch.setattr(configure.subprocess, 'run', run)
    configure.configure(root, template)
    assert ['git', 'status', '--short', '--', '.'] in commands
    assert ['git', 'diff', '--binary', 'HEAD', '--', '.'] in commands
