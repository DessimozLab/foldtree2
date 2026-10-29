"""The documentation checker is offline and rejects broken links/CLI flags."""
from contextlib import redirect_stderr, redirect_stdout
import ast
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import runpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import check_documentation


class DocumentationTests(unittest.TestCase):
    def check_example(self, source):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'docs').mkdir()
            (root / 'foldtree2').mkdir()
            (root / 'README.md').write_text(source)
            (root / 'docs' / 'guide.md').write_text('# Guide\n')
            (root / 'foldtree2' / 'ft2treebuilder.py').write_text(
                "parser.add_argument('--device')\n")
            with patch.object(check_documentation, 'ROOT', root), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                return check_documentation.check()

    def test_valid_link_and_flag(self):
        self.assertFalse(self.check_example('[Guide](docs/guide.md)\n```bash\nfoldtree2 --device cpu\n```\n'))

    def test_missing_link_fails(self):
        self.assertTrue(self.check_example('[Missing](docs/missing.md)\n'))

    def test_unknown_flag_fails(self):
        self.assertTrue(self.check_example('```bash\nfoldtree2 --old-device cpu\n```\n'))

    def test_missing_module_fails(self):
        self.assertTrue(self.check_example('```bash\npython -m foldtree2.missing --help\n```\n'))

    def test_nested_readme_links_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('# Project\n')
            readme = root / 'foldtree2/notebooks/README.md'
            readme.parent.mkdir(parents=True)
            readme.write_text('[Missing](absent.ipynb)\n')
            with patch.object(check_documentation, 'ROOT', root), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertTrue(check_documentation.check())

    def test_source_modules_have_purpose_docstrings(self):
        for path in (ROOT / 'foldtree2/src').rglob('*.py'):
            with self.subTest(module=path.relative_to(ROOT)):
                self.assertTrue(ast.get_docstring(ast.parse(path.read_text())))

    def test_notebooks_are_catalogued(self):
        notebook_root = ROOT / 'foldtree2/notebooks'
        for path in notebook_root.rglob('*.ipynb'):
            if '.ipynb_checkpoints' in path.parts:
                continue
            with self.subTest(notebook=path.relative_to(ROOT)):
                self.assertIn(path.name, (path.parent / 'README.md').read_text())

    def test_published_docs_resolve_source_and_evidence_links(self):
        config = runpy.run_path(str(ROOT / 'docs/conf.py'))
        app = SimpleNamespace(
            srcdir=str(ROOT / 'docs'),
            env=SimpleNamespace(doc2path=lambda _: str(ROOT / 'docs/experiments.md')),
        )
        source = [
            '[Source](../foldtree2/src/encoder.py) '
            '[Evidence](se3_validation_evidence/smoke/) '
            '[Guide](training.md) [Section](#native-ancestral-character-reconstruction)'
        ]
        config['resolve_repository_links'](app, 'experiments', source)
        self.assertIn('/blob/main/foldtree2/src/encoder.py', source[0])
        self.assertIn('/tree/main/docs/se3_validation_evidence/smoke', source[0])
        self.assertIn('[Guide](training.md)', source[0])
        self.assertIn('[Section](#native-ancestral-character-reconstruction)', source[0])


if __name__ == '__main__':
    unittest.main()
