"""Run with .venv-mlflow/bin/python -m unittest foldtree2.tests.test_geometry_mlflow."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import geometry_mlflow as tracking


class TrackingTests(unittest.TestCase):
    def test_legacy_yaml_objects_are_not_constructed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'hparams.yaml'
            path.write_text('learning_rate: 0.001\nmodel: !!python/object:missing.Class {}\n')
            data = tracking.read_yaml(path)
            self.assertEqual(data['learning_rate'], .001)
            self.assertIn('omitted YAML object', data['model'])

    def test_objective_cases_log_actual_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'runs/contact_objective_screen/baseline'
            path.mkdir(parents=True)
            (path / 'history.json').write_text('[{"epoch":1,"validation":{"nll":2.0}}]')
            (path.parent / 'provenance.json').write_text('{"args":{"contact_weight":1.0}}')
            (path.parent / 'summary.json').write_text(
                '{"baseline":{"spherical_dim":0,"contact_weight":0.0}}')
            with patch.object(tracking, 'ROOT', root), patch.object(tracking, 'STORE', root / '.mlflow'):
                tracker = tracking.Tracker('sqlite:///' + str(root / '.mlflow/test.db'))
                self.assertEqual(tracker.once(), [])
                run_id = tracker.state['runs/contact_objective_screen/baseline']['run_id']
                params = tracker.client.get_run(run_id).data.params
                self.assertEqual(params['objective/contact_weight'], '0.0')
                self.assertEqual(params['model/spherical_dim'], '0')

    def test_tensorboard_import_preserves_steps_without_duplicates(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'lightning_logs/version_0'
            path.mkdir(parents=True)
            (path / 'hparams.yaml').write_text('learning_rate: 0.001\nuse_se3: false\n')
            writer = EventFileWriter(str(path))
            writer.add_event(Event(wall_time=100., step=7,
                summary=Summary(value=[Summary.Value(tag='val/loss', simple_value=2.)])))
            writer.flush()
            with patch.object(tracking, 'ROOT', root), patch.object(tracking, 'STORE', root / '.mlflow'):
                tracker = tracking.Tracker('sqlite:///' + str(root / '.mlflow/test.db'))
                self.assertEqual(tracker.once(), [])
                tracker.once()
                run_id = tracker.state['lightning_logs/version_0']['run_id']
                points = tracker.client.get_metric_history(run_id, 'val/loss')
                self.assertEqual([(p.step, p.value) for p in points], [(7, 2.)])
                writer.add_event(Event(wall_time=101., step=8,
                    summary=Summary(value=[Summary.Value(tag='val/loss', simple_value=1.)])))
                writer.flush()
                self.assertEqual(tracker.once(), [])
                points = tracker.client.get_metric_history(run_id, 'val/loss')
                self.assertEqual([(p.step, p.value) for p in points], [(7, 2.), (8, 1.)])
            writer.close()

    def test_numeric_filters_nonfinite_and_credentials(self):
        self.assertEqual(tracking.numeric({'x': 0., 'bad': float('nan'), 'flag': True,
                                          'nested': {'loss': 2, 'token': 123}}),
                         {'x': 0., 'nested/loss': 2.})

    def test_backfill_incremental_status_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'runs/se3_example'
            path.mkdir(parents=True)
            (path / 'provenance.json').write_text(json.dumps({
                'config': {'learning_rate': .001, 'password': 'do-not-log'}}))
            history = [{'epoch': 1, 'validation': {'fape': 2., 'invalid': None}}]
            (path / 'history.json').write_text(json.dumps(history))
            events = path / 'events.jsonl'
            events.write_text(json.dumps({'event': 'validation', 'epoch': 1,
                                         'metrics': {'geometry': 3.}}) + '\n')
            with patch.object(tracking, 'ROOT', root), patch.object(tracking, 'STORE', root / '.mlflow'):
                tracker = tracking.Tracker('sqlite:///' + str(root / '.mlflow/test.db'))
                self.assertEqual(tracker.once(), [])
                run_id = tracker.state['runs/se3_example']['run_id']
                self.assertEqual(tracker.client.get_run(run_id).data.metrics['val/fape'], 2.)
                # Re-sync and restart must neither create duplicate runs nor duplicate metrics.
                tracker.once()
                tracker = tracking.Tracker('sqlite:///' + str(root / '.mlflow/test.db'))
                tracker.once()
                self.assertEqual(len(tracker.client.get_metric_history(run_id, 'val/fape')), 1)
                history.append({'epoch': 2, 'validation': {'fape': 1.}})
                (path / 'history.json').write_text(json.dumps(history))
                # Incomplete trailing events are left pending, not discarded.
                with events.open('a') as stream:
                    stream.write('{"event":"complete"')
                tracker.once()
                with events.open('a') as stream:
                    stream.write('}\n')
                tracker.once()
                run = tracker.client.get_run(run_id)
                self.assertEqual(run.info.status, 'FINISHED')
                self.assertEqual(run.data.metrics['val/fape'], 1.)
                self.assertEqual(len(tracker.client.get_metric_history(run_id, 'val/fape')), 2)
                self.assertNotIn('password', run.data.params)
                artifact = tracker.client.download_artifacts(run_id, 'logs/provenance.json')
                self.assertNotIn('do-not-log', Path(artifact).read_text())
                runs = tracker.client.search_runs([tracker.experiments['se3']])
                self.assertEqual(len(runs), 1)
                (path / 'cancelled.json').write_text('{"reason":"invalid ablation"}')
                tracker.once()
                self.assertEqual(tracker.client.get_run(run_id).info.status, 'KILLED')


if __name__ == '__main__':
    unittest.main()
