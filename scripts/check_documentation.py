#!/usr/bin/env python3
"""Check local documentation links and example flags without importing trainers."""
import ast
from pathlib import Path
import re
import shlex
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {
    'foldtree2': 'foldtree2/ft2treebuilder.py',
    'ft2treebuilder': 'foldtree2/ft2treebuilder.py',
    'makesubmat': 'foldtree2/makesubmat.py',
    'pdbs-to-graphs': 'foldtree2/scripts/pdbs_to_graphs.py',
}


def flags_for(path):
    tree = ast.parse(path.read_text())
    flags = {'--help', '-h'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            flags.update(arg.value for arg in node.args
                         if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith('-'))
    return flags


def check():
    errors = []
    docs = [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md'))]
    checked_commands = 0
    for path in docs:
        source = path.read_text()
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', source):
            target = target.strip().strip('<>')
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = path.parent / unquote(parsed.path)
            if not destination.exists():
                errors.append(f'{path.relative_to(ROOT)}: missing link target {target}')
        for block in re.findall(r'```(?:bash|sh)\n(.*?)```', source, flags=re.S):
            for line in block.replace('\\\n', ' ').splitlines():
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                try:
                    words = shlex.split(line)
                except ValueError as exc:
                    errors.append(f'{path.relative_to(ROOT)}: invalid shell example: {exc}')
                    continue
                if not words:
                    continue
                script = ENTRYPOINTS.get(words[0])
                arguments = words[1:]
                if words[0] in ('python', 'python3') and len(words) > 1:
                    if words[1] == '-m' and len(words) > 2 and words[2].startswith('foldtree2.'):
                        script = words[2].replace('.', '/') + '.py'
                        arguments = words[3:]
                    elif words[1].startswith('scripts/') and words[1].endswith('.py'):
                        script = words[1]
                        arguments = words[2:]
                if not script:
                    continue
                script_path = ROOT / script
                if not script_path.exists():
                    errors.append(f'{path.relative_to(ROOT)}: missing CLI module {script}')
                    continue
                valid_flags = flags_for(script_path)
                checked_commands += 1
                for word in arguments:
                    if word.startswith('--') and word.split('=', 1)[0] not in valid_flags:
                        errors.append(f'{path.relative_to(ROOT)}: unknown {script} flag {word}')
    for error in errors:
        print(error, file=sys.stderr)
    print(f'Checked {len(docs)} documents and {checked_commands} CLI examples; {len(errors)} errors.')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(check())
