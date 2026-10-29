"""Track geometry JSON/JSONL/TensorBoard logs without modifying training code."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / '.mlflow'
EXPERIMENTS = {'se3': 'foldtree2/se3', 'transformer': 'foldtree2/transformer_geometry'}
SECRET = ('password', 'token', 'secret', 'credential', 'authorization')


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        key = str(key)
        if any(word in key.lower() for word in SECRET):
            continue
        name = f'{prefix}/{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def numeric(value):
    return {k: float(v) for k, v in flatten(value).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


def read_json(path):
    return json.loads(path.read_text())


def read_yaml(path):
    import yaml
    class LogLoader(yaml.SafeLoader):
        pass
    # Legacy Lightning hparams sometimes embed modules. Keep a descriptive
    # placeholder; never reconstruct or execute Python objects from a log.
    LogLoader.add_constructor(None, lambda loader, node: f'[omitted YAML object: {node.tag}]')
    return yaml.load(path.read_text(), Loader=LogLoader)


def family(path):
    name = str(path.relative_to(ROOT))
    if (path / 'hparams.yaml').exists():
        config = read_yaml(path / 'hparams.yaml')
        if config.get('use_se3') or config.get('train_se3_only'):
            return 'se3'
    return 'transformer' if ('distance_' in name or 'contact_objective' in name
                            or 'lightning_logs' in name) else 'se3'


def discover():
    """Only experiment records, never cache entries or copied source snapshots."""
    found = set()
    for base in [ROOT / 'runs', ROOT / 'docs']:
        for current, dirs, files in os.walk(base):
            path = Path(current)
            dirs[:] = [d for d in dirs if d not in ('sources', 'entries', 'graphs', '__pycache__',
                                                   'checkpoints', 'bundle', 'matrices')
                       and not d.startswith('bundle_')]
            relative = str(path.relative_to(ROOT))
            if not any(x in relative for x in ('se3_', 'contact_objective')):
                continue
            if 'history.json' in files or 'events.jsonl' in files:
                found.add(path)
            elif 'summary.json' in files and base.name == 'docs':
                try:
                    if 'learning_curve' in read_json(path / 'summary.json'):
                        found.add(path)
                except (ValueError, OSError):
                    pass
    for base in [ROOT / 'lightning_logs', ROOT / 'foldtree2/lightning_logs']:
        if base.exists():
            found.update(p for p in base.iterdir() if p.is_dir() and (p / 'hparams.yaml').exists())
    # A distance trainer parent emits lifecycle events, but its per-mode children
    # contain the actual independently trained models and learning curves.
    return sorted(p for p in found if not any((p / d / 'history.json').exists()
                                              for d in ('decoder', 'sequence')))


class Tracker:
    def __init__(self, tracking_uri=None):
        import mlflow
        from mlflow.tracking import MlflowClient
        STORE.mkdir(exist_ok=True)
        self.uri = tracking_uri or os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
        # MLflow's proxied-artifact resolver consults the global URI even when
        # log_dict/log_artifact are invoked on an explicitly configured client.
        mlflow.set_tracking_uri(self.uri)
        self.client = MlflowClient(tracking_uri=self.uri)
        self.experiments = {}
        for key, name in EXPERIMENTS.items():
            experiment = self.client.get_experiment_by_name(name)
            storage = ({'artifact_location': (STORE / 'artifacts' / key).as_uri()}
                       if self.uri.startswith('sqlite:') else {})
            self.experiments[key] = (experiment.experiment_id if experiment else
                self.client.create_experiment(name, **storage))
        suffix = hashlib.sha256(self.uri.encode()).hexdigest()[:12]
        self.state_path = STORE / f'sync-state-{suffix}.json'
        self.state = read_json(self.state_path) if self.state_path.exists() else {}

    def save(self):
        temp = self.state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state))
        temp.replace(self.state_path)

    def metrics(self, run, values, step=0, timestamp=None, collector=None):
        from mlflow.entities import Metric
        items = [Metric(k, v, int(timestamp or time.time() * 1000), max(0, int(step)))
                 for k, v in numeric(values).items()]
        if collector is not None:
            collector.extend(items)
            return
        for offset in range(0, len(items), 1000):
            self.client.log_batch(run, metrics=items[offset:offset + 1000])

    def sync(self, path):
        import yaml
        relative = str(path.relative_to(ROOT))
        record = self.state.get(relative)
        if not record:
            experiment = self.experiments[family(path)]
            existing = self.client.search_runs([experiment],
                filter_string=f"tags.`foldtree2.source` = '{relative}'", max_results=1)
            times = [p.stat().st_mtime for p in path.iterdir() if p.is_file()]
            events = path / 'events.jsonl'
            if events.exists():
                with events.open() as stream:
                    first = stream.readline()
                if first.endswith('\n'):
                    times.append(json.loads(first).get('time', min(times)))
            for p in path.glob('events.out.tfevents.*'):
                try:
                    times.append(float(p.name.split('.')[3]))
                except (IndexError, ValueError):
                    pass
            run = existing[0] if existing else self.client.create_run(experiment,
                start_time=int(min(times or [time.time()]) * 1000), tags={
                    'mlflow.runName': relative, 'foldtree2.source': relative,
                    'foldtree2.family': family(path), 'foldtree2.tracking': 'log-sidecar',
                    'foldtree2.scope': 'development; see provenance for holdout limitations',
                    'foldtree2.stage': 'smoke' if 'smoke' in relative else 'experiment'})
            record = self.state.setdefault(relative, {'run_id': run.info.run_id, 'files': {}})
            self.save()
        run_id, files = record['run_id'], record['files']
        previous_record = json.dumps(record, sort_keys=True)
        if 'contact_objective' in relative and not record.get('case_params'):
            summary = path.parent / 'summary.json'
            case = read_json(summary).get(path.name, {}) if summary.exists() else {}
            if 'spherical_dim' in case and 'contact_weight' in case:
                from mlflow.entities import Param
                self.client.log_batch(run_id, params=[
                    Param('model/spherical_dim', str(case['spherical_dim'])),
                    Param('objective/contact_weight', str(case['contact_weight'])),
                    Param('objective/variant', path.name)])
                record['case_params'] = True
        # Parameters are immutable in MLflow. Log once; original provenance remains
        # available as an artifact when a resumed run changes its configuration.
        if not record.get('params'):
            config = {}
            for candidate in [path / 'provenance.json', path.parent / 'provenance.json', path / 'hparams.yaml']:
                if candidate.exists():
                    data = read_json(candidate) if candidate.suffix == '.json' else read_yaml(candidate)
                    config = data.get('args', data.get('training', data.get('config', data)))
                    break
            from mlflow.entities import Param
            params = [Param(k, (v if isinstance(v, str) else json.dumps(v))[:500])
                      for k, v in flatten(config).items() if k not in ('selection', 'sources')]
            for offset in range(0, len(params), 100):
                self.client.log_batch(run_id, params=params[offset:offset + 100])
            record['params'] = True
        for filename in ['history.json', 'summary.json']:
            p = path / filename
            if not p.exists():
                continue
            data = read_json(p)
            rows = data if isinstance(data, list) else data.get('learning_curve', [])
            start = files.get(filename, 0)
            batch = []
            for i, row in enumerate(rows[start:], start):
                values = {k: v for k, v in row.items() if k not in ('epoch', 'global_step', 'updates')}
                if 'validation' in values:
                    values['val'] = values.pop('validation')
                self.metrics(run_id, values, row.get('epoch', i), collector=batch)
            for offset in range(0, len(batch), 1000):
                self.client.log_batch(run_id, metrics=batch[offset:offset + 1000])
            files[filename] = len(rows)
            if len(rows) != start:
                self.save()
        events = path / 'events.jsonl'
        if events.exists():
            batch = []
            offset = files.get('events.jsonl', 0)
            previous_offset = offset
            if offset > events.stat().st_size:
                offset = 0
            with events.open('rb') as stream:
                stream.seek(offset)
                while True:
                    line = stream.readline()
                    if not line.endswith(b'\n'):
                        break
                    row = json.loads(line)
                    event = row.get('event', '')
                    timestamp = int(row.get('time', time.time()) * 1000)
                    step = row.get('global_step', row.get('updates', row.get('epoch', 0)))
                    if event in ('validation', 'train_epoch', 'epoch', 'update', 'progress'):
                        prefix = 'val' if event == 'validation' else 'train' if event in ('train_epoch', 'update', 'progress') else 'epoch'
                        values = row.get('metrics', {k: v for k, v in row.items()
                            if k not in ('time', 'epoch', 'global_step', 'updates', 'counts')})
                        self.metrics(run_id, {prefix: values}, step, timestamp, collector=batch)
                    if event in ('complete', 'failure', 'ready', 'resume', 'update', 'validation', 'epoch'):
                        record['terminal'] = {'complete': 'FINISHED', 'failure': 'FAILED'}.get(event)
                    record['last_event_time'] = row.get('time', time.time())
                    offset = stream.tell()
            for index in range(0, len(batch), 1000):
                self.client.log_batch(run_id, metrics=batch[index:index + 1000])
            files['events.jsonl'] = offset
            if offset != previous_offset:
                self.save()
        # TensorBoard keeps the original scalar names and optimizer steps.
        event_files = sorted(path.glob('events.out.tfevents.*'))
        signature = [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in event_files]
        if event_files and signature != record.get('tb_signature'):
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            accumulator = EventAccumulator(str(path), size_guidance={'scalars': 0})
            accumulator.Reload()
            cursors = record.setdefault('tb_cursors', {})
            from mlflow.entities import Metric
            batch = []
            for tag in accumulator.Tags()['scalars']:
                points = accumulator.Scalars(tag)
                for point in points[cursors.get(tag, 0):]:
                    if math.isfinite(point.value):
                        batch.append(Metric(tag, point.value, int(point.wall_time * 1000), point.step))
                cursors[tag] = len(points)
            for offset in range(0, len(batch), 1000):
                self.client.log_batch(run_id, metrics=batch[offset:offset + 1000])
            record['tb_signature'] = signature
            self.save()
        # Small diagnostics only: no checkpoints, caches, or model binaries.
        for p in sorted(path.iterdir()):
            if (not p.is_file() or p.suffix not in ('.json', '.yaml', '.yml', '.png', '.md')
                    or p.name in ('manifest.json', 'cohort.json') or p.stat().st_size > 5_000_000):
                continue
            if p.name.startswith(('validation-', 'heldout_pairs', 'paired_metrics')):
                continue
            sig = f'{p.stat().st_size}:{p.stat().st_mtime_ns}'
            if files.get('artifact:' + p.name) != sig:
                if p.suffix in ('.json', '.yaml', '.yml'):
                    data = read_json(p) if p.suffix == '.json' else read_yaml(p)
                    # Full experiment provenance is useful; remove credential fields.
                    def scrub(value):
                        if isinstance(value, dict):
                            return {k: scrub(v) for k, v in value.items()
                                    if not any(s in str(k).lower() for s in SECRET)}
                        return [scrub(v) for v in value] if isinstance(value, list) else value
                    self.client.log_dict(run_id, scrub(data), 'logs/' + p.stem + '.json')
                else:
                    self.client.log_artifact(run_id, str(p), artifact_path='logs')
                files['artifact:' + p.name] = sig
        completed = path / 'completion.json'
        final = path / 'final_validation.json'
        failed = path / 'failure.json'
        cancelled = path / 'cancelled.json'
        status = record.get('terminal')
        summary_path = path / 'summary.json'
        if summary_path.exists():
            summary_status = read_json(summary_path)
            if isinstance(summary_status, dict) and summary_status.get('state') == 'COMPLETED':
                status = 'FINISHED' if summary_status.get('exit_code', '0:0') in ('0:0', '0', 0) else 'FAILED'
        if completed.exists() or final.exists():
            status = 'FINISHED'
        if failed.exists():
            status = 'FAILED'
        if cancelled.exists():
            status = 'KILLED'
        if completed.exists():
            summary = read_json(completed)
            if 'validation' in summary:
                self.metrics_summary(run_id, summary, record, completed)
        last_time = record.get('last_event_time', max(
            [p.stat().st_mtime for p in path.iterdir() if p.is_file()] or [time.time()]))
        inferred = not status and time.time() - last_time > 86400
        status = status or ('KILLED' if inferred else 'RUNNING')
        if record.get('status_inferred') != bool(inferred):
            self.client.set_tag(run_id, 'foldtree2.status_inferred', str(bool(inferred)).lower())
            record['status_inferred'] = bool(inferred)
        if record.get('status') != status:
            if status == 'RUNNING':
                self.client.update_run(run_id, status=status)
            else:
                start = self.client.get_run(run_id).info.start_time
                self.client.set_terminated(run_id, status=status, end_time=max(start, int(last_time * 1000)))
            record['status'] = status
        if json.dumps(record, sort_keys=True) != previous_record:
            self.save()

    def metrics_summary(self, run_id, summary, record, path):
        signature = str(path.stat().st_mtime_ns)
        if record.get('completion_metrics') != signature:
            for key in ('validation', 'best_validation', 'training'):
                if isinstance(summary.get(key), dict):
                    self.metrics(run_id, {'final/' + key: summary[key]}, summary.get('epochs', 0))
            record['completion_metrics'] = signature

    def once(self):
        errors = []
        paths = discover()
        for path in paths:
            try:
                self.sync(path)
            except Exception as exc:
                errors.append({'path': str(path.relative_to(ROOT)), 'error': str(exc)})
        (STORE / 'sync-errors.json').write_text(json.dumps(errors, indent=2))
        print(json.dumps({'tracked': len(self.state), 'discovered': len(paths), 'errors': len(errors)}), flush=True)
        return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['sync', 'watch', 'start', 'stop', 'status'])
    parser.add_argument('--interval', type=float, default=30.)
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    STORE.mkdir(exist_ok=True)
    if args.command in ('sync', 'watch'):
        # Prevent simultaneous backfill/watch writers from creating duplicate runs.
        import fcntl
        with (STORE / 'sync.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if args.command == 'sync' else 0))
            tracker = Tracker()
            while True:
                errors = tracker.once()
                if args.command == 'sync':
                    return int(bool(errors))
                time.sleep(args.interval)
    for name in ('server', 'watcher'):
        pidfile = STORE / (name + '.pid')
        pid = int(pidfile.read_text()) if pidfile.exists() else None
        expected = 'mlflow' if name == 'server' else 'geometry_mlflow.py'
        live = bool(pid and Path(f'/proc/{pid}/cmdline').exists()
                    and expected.encode() in Path(f'/proc/{pid}/cmdline').read_bytes())
        if name == 'server' and not live:
            import urllib.request
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=2) as response:
                    healthy = response.status == 200
                if healthy:
                    print(f'server: running on port {args.port} (existing service; managed separately)')
                    continue
            except OSError:
                pass
        if args.command == 'stop' and live:
            os.killpg(pid, signal.SIGTERM)
            pidfile.unlink(missing_ok=True)
            live = False
        elif args.command == 'start' and not live:
            command = ([sys.executable, '-m', 'mlflow', 'server', '--host', '127.0.0.1',
                        '--port', str(args.port), '--workers', '1', '--backend-store-uri',
                        'sqlite:///' + str(STORE / 'tracking.db'), '--no-serve-artifacts',
                        '--default-artifact-root', str(STORE / 'artifacts')]
                       if name == 'server' else
                       [sys.executable, str(Path(__file__).resolve()), 'watch', '--interval', str(args.interval)])
            with (STORE / (name + '.log')).open('a') as log:
                env = dict(os.environ)
                env.setdefault('MLFLOW_TRACKING_URI', f'http://127.0.0.1:{args.port}')
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                                           start_new_session=True, env=env)
            pidfile.write_text(str(process.pid))
            pid, live = process.pid, True
        print(f'{name}: {"running" if live else "stopped"}' + (f' (pid {pid})' if live else ''))
    if args.command != 'stop':
        print(f'UI: http://127.0.0.1:{args.port}')


if __name__ == '__main__':
    sys.exit(main())
