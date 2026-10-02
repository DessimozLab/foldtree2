#!/usr/bin/env python3
"""Stage only build/runtime sources; never copy datasets or production models."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = ('README.md', 'LICENSE.txt', 'pyproject.toml', 'setup.py', 'MANIFEST.in')
TOOLS = ('mafft_tools/hex2maffttext', 'mafft_tools/maffttext2hex',
         'raxml-ng/raxml-ng', 'madroot/mad')
EXCLUDED_DIRS = {'notebooks', 'tests', '__pycache__', 'workflow', '.git',
                 'models', 'runs', 'tmp', 'build', 'dist'}
# Legacy scratch code, not imported by the runtime, and not syntactically valid.
EXCLUDED_FILES = {'scaling_experiment.py'}
FORBIDDEN_SUFFIXES = {'.h5', '.hdf5', '.ipynb', '.pt', '.pth', '.pkl',
                      '.npy', '.npz', '.pyc', '.pyo', '.log'}


def walk_files(directory):
    """Prune excluded/symlink directories before examining their contents."""
    for child in sorted(directory.iterdir()):
        if child.is_symlink() or child.name in EXCLUDED_FILES:
            continue
        if child.is_dir():
            if child.name not in EXCLUDED_DIRS and not child.name.startswith(('results', '.')):
                yield from walk_files(child)
        elif child.is_file() and child.suffix.lower() not in FORBIDDEN_SUFFIXES:
            yield child


def stage_source(source, recipe, destination, dry_run=False, max_file_bytes=50 * 1024**2):
    source, recipe, destination = map(lambda p: Path(p).resolve(), (source, recipe, destination))
    if source == destination or source in destination.parents or recipe == destination or recipe in destination.parents:
        raise ValueError('Destination must be outside the checkout and recipe')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('Destination must be new or empty; stale payloads are not reused')
    if not (recipe / 'meta.yaml').is_file():
        raise FileNotFoundError(recipe / 'meta.yaml')

    selected = [(source / name, Path(name)) for name in TOP_FILES]
    selected += [(source / 'foldtree2' / name, Path('foldtree2') / name) for name in TOOLS]
    selected += [(source / 'foldtree2' / '__init__.py', Path('foldtree2/__init__.py')),
                 (source / 'foldtree2/config/aaindex1.csv', Path('foldtree2/config/aaindex1.csv'))]
    # Python runtime code, not arbitrary files under the package directory.
    selected += [(path, path.relative_to(source)) for path in walk_files(source / 'foldtree2')
                 if path.suffix == '.py']
    selected += [(path, path.relative_to(source)) for path in walk_files(source / 'foldtree2/config')
                 if path.suffix.lower() in {'.csv', '.yaml', '.yml'}]
    selected += [(path, Path(recipe.name) / path.relative_to(recipe)) for path in walk_files(recipe)]
    selected = sorted(set(selected), key=lambda pair: str(pair[1]))
    total = 0
    # Complete preflight before making a directory or opening files for copying.
    for path, relative in selected:
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents) or not path.is_file():
            raise ValueError(f'Required source is missing, nonregular, or symlinked: {path}')
        size = path.stat().st_size
        if size > max_file_bytes:
            raise ValueError(f'Refusing oversized build source ({size} bytes): {relative}')
        total += size
    if total > 100 * 1024**2:
        raise ValueError('Selected build sources exceed the 100 MiB staging budget')
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        for path, relative in selected:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return {'source': str(source), 'recipe': str(destination / recipe.name),
            'destination': str(destination), 'files': len(selected), 'bytes': total,
            'dry_run': dry_run, 'models_included': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--recipe-dir', type=Path)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(stage_source(args.source, args.recipe_dir or args.source / 'conda-recipe',
                                 args.destination, dry_run=args.dry_run), indent=2))


if __name__ == '__main__':
    main()
