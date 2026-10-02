"""Staging must exclude large inputs before copying and preserve runtime tools."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from stage_conda_source import TOP_FILES, TOOLS, stage_source


class CondaStagingTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'checkout'
        source.mkdir()
        for name in TOP_FILES:
            (source / name).write_text('fixture\n')
        for name in (*TOOLS, '__init__.py', 'src/encoder.py', 'config/aaindex1.csv'):
            path = source / 'foldtree2' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture\n')
        (source / 'foldtree2/mafft_tools/hex2maffttext').chmod(0o755)
        recipe = source / 'conda-recipe'
        recipe.mkdir()
        (recipe / 'meta.yaml').write_text('source:\n  path: ..\n')
        return source, recipe

    def test_large_sparse_inputs_never_enter_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, recipe = self.fixture(root)
            for name in ('foldtree2/huge.h5', 'foldtree2/notebooks/huge.py',
                         'models/production/model.pt', 'foldtree2/config/cache.pkl',
                         'foldtree2/tests/fixture.py'):
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('wb') as stream:
                    stream.truncate(80 * 1024**3)
            destination = root / 'stage'
            (source / 'foldtree2/scaling_experiment.py').write_text('56import invalid\n')
            report = stage_source(source, recipe, destination)
            self.assertLess(report['bytes'], 1024)
            self.assertFalse((destination / 'models').exists())
            self.assertFalse((destination / 'foldtree2/huge.h5').exists())
            self.assertFalse((destination / 'foldtree2/notebooks').exists())
            self.assertFalse((destination / 'foldtree2/tests').exists())
            self.assertFalse((destination / 'foldtree2/scaling_experiment.py').exists())
            self.assertEqual((destination / 'foldtree2/mafft_tools/hex2maffttext').stat().st_mode & 0o777, 0o755)
            self.assertTrue((destination / 'foldtree2/src/encoder.py').exists())

    def test_dry_run_and_oversize_preflight_do_not_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, recipe = self.fixture(root)
            destination = root / 'stage'
            stage_source(source, recipe, destination, dry_run=True)
            self.assertFalse(destination.exists())
            with self.assertRaises(ValueError):
                stage_source(source, recipe, destination, max_file_bytes=1)
            self.assertFalse(destination.exists())

    def test_rejects_nested_or_stale_destinations_and_missing_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, recipe = self.fixture(root)
            with self.assertRaises(ValueError):
                stage_source(source, recipe, source / 'stage')
            destination = root / 'stage'
            destination.mkdir()
            (destination / 'stale.h5').write_text('retain\n')
            with self.assertRaises(ValueError):
                stage_source(source, recipe, destination)
            self.assertEqual((destination / 'stale.h5').read_text(), 'retain\n')
            (source / 'foldtree2/raxml-ng/raxml-ng').unlink()
            with self.assertRaises(ValueError):
                stage_source(source, recipe, root / 'new_stage')

    def test_does_not_follow_external_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, recipe = self.fixture(root)
            external = root / 'external.py'
            external.write_text('not package code\n')
            (source / 'foldtree2/external.py').symlink_to(external)
            stage_source(source, recipe, root / 'stage')
            self.assertFalse((root / 'stage/foldtree2/external.py').exists())


if __name__ == '__main__':
    unittest.main()
