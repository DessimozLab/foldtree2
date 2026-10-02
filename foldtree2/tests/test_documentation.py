"""The documentation checker is offline and rejects broken links/CLI flags."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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


if __name__ == '__main__':
    unittest.main()
