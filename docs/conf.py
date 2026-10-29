# Configuration file for the Sphinx documentation builder.

import os
import re
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
try:
    import tomllib
except ImportError:
    import tomli as tomllib
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'FoldTree2'
copyright = '2025, DessimozLab'
author = 'DessimozLab'
with (Path(__file__).resolve().parents[1] / 'pyproject.toml').open('rb') as metadata:
    release = tomllib.load(metadata)['project']['version']

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'myst_parser',
]

source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
myst_heading_anchors = 4
templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    # Historical autodoc pages import CLI/research modules with side effects.
    # The maintained API map links to source without executing those modules.
    'foldtree2*.rst', 'modules.rst',
]

# -- Options for HTML output ------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = []

# -- Napoleon settings -------------------------------------------------------
# For Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Mock imports for dependencies that might not be available during doc build
autodoc_mock_imports = [
    'torch',
    'torch_geometric',
    'sklearn',
    'Bio',
    'alphafold',
]


def resolve_repository_links(app, docname, source):
    """Keep repository-relative README links usable on the published docs site."""
    docpath = Path(app.env.doc2path(docname)).resolve()
    if docpath.suffix != '.md':
        return
    docs_root = Path(app.srcdir).resolve()
    repo_root = docs_root.parent

    def replace(match):
        target = match.group(2).strip('<>')
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)
        path = (docpath.parent / unquote(parsed.path)).resolve()
        if repo_root not in path.parents:
            return match.group(0)
        if docs_root in path.parents and not path.is_dir():
            return match.group(0)
        relative = quote(path.relative_to(repo_root).as_posix())
        anchor = f'#{parsed.fragment}' if parsed.fragment else ''
        kind = 'tree' if path.is_dir() else 'blob'
        return f'{match.group(1)}https://github.com/DessimozLab/foldtree2/{kind}/main/{relative}{anchor})'

    source[0] = re.sub(r'(\[[^\]]*\]\()([^)]+)\)', replace, source[0])


def setup(app):
    app.connect('source-read', resolve_repository_links)