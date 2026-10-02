"""Strict artifact/provenance primitives shared by local benchmark commands."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AA_ORDER = 'ARNDCQEGHILKMFPSTWYV'  # PAML/IQ-TREE/RAxML protein matrix order
MULTI_ORDER = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,/:;<=>@[\\]^_{|}~'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def read_fasta(path, aligned=False):
    records, current = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            current = line[1:].split()[0]
            if not current or current in records:
                raise ValueError(f'Duplicate/empty FASTA identifier in {path}: {current}')
            records[current] = ''
        elif line.strip():
            if current is None:
                raise ValueError(f'Sequence before header in {path}')
            records[current] += line.strip()
    if not records or any(not s for s in records.values()):
        raise ValueError(f'Empty FASTA sequence: {path}')
    if aligned and len({len(s) for s in records.values()}) != 1:
        raise ValueError(f'Unequal alignment lengths: {path}')
    return records


def write_fasta(path, records):
    Path(path).write_text(''.join(f'>{key}\n{seq}\n' for key, seq in sorted(records.items())))


def project_tokens(anchor, sequences):
    if set(anchor) != set(sequences):
        raise ValueError('Projection requires exactly identical taxa')
    result = {}
    for ident, aa in anchor.items():
        seq = sequences[ident]
        if len(aa.replace('-', '')) != len(seq):
            raise ValueError(f'Residue correspondence mismatch: {ident}')
        iterator = iter(seq)
        result[ident] = ''.join('-' if char == '-' else next(iterator) for char in aa)
    return result


def run(command, log, stdout=None):
    command = list(map(str, command))
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('a') as stream:
        stream.write('\nCOMMAND ' + json.dumps(command) + '\n')
        stream.flush()
        if stdout is None:
            subprocess.run(command, stdout=stream, stderr=stream, check=True)
        else:
            with Path(stdout).open('w') as output:
                subprocess.run(command, stdout=output, stderr=stream, check=True)


def tool_version(tool):
    executable = shutil.which(tool)
    if executable is None:
        raise FileNotFoundError(f'Required tool not installed: {tool}')
    flag = '--version' if tool == 'raxml-ng' else 'version'
    result = subprocess.run([executable, flag], capture_output=True, text=True)
    return {'executable': executable, 'sha256': digest(executable),
            'version': (result.stdout + result.stderr).strip()[:1000]}


def begin_stage(outdir, signature):
    """Reject mixed provenance; only reuse verified complete outputs."""
    outdir = Path(outdir)
    provenance = outdir / 'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text()) != signature:
        raise ValueError(f'Provenance changed: use a new output directory ({outdir})')
    completed = outdir / 'completed.json'
    if completed.exists():
        data = json.loads(completed.read_text())
        if data['signature'] != signature:
            raise ValueError(f'Completion signature differs: {outdir}')
        for name, checksum in data['outputs'].items():
            if digest(outdir / name) != checksum:
                raise ValueError(f'Completed output changed: {outdir / name}')
        return False
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(provenance, signature)
    return True


def complete_stage(outdir, signature, outputs, **metadata):
    outdir = Path(outdir)
    write_json(outdir / 'completed.json', {'signature': signature,
        'outputs': {str(Path(p).relative_to(outdir)): digest(p) for p in outputs}, **metadata})


def read_paml(path, size):
    values = np.fromstring(Path(path).read_text(), sep=' ')
    if len(values) != size * (size - 1) // 2 + size or not np.isfinite(values).all():
        raise ValueError(f'Invalid PAML matrix dimensions/values: {path}')
    s = np.zeros((size, size))
    offset = 0
    for i in range(1, size):
        s[i, :i] = values[offset:offset + i]
        s[:i, i] = s[i, :i]
        offset += i
    pi = values[offset:]
    if (s < 0).any() or (pi <= 0).any() or not np.isclose(pi.sum(), 1, atol=1e-5):
        raise ValueError(f'Invalid exchangeabilities/stationary frequencies: {path}')
    pi = pi / pi.sum()
    q = s * pi[None, :]
    np.fill_diagonal(q, -q.sum(axis=1))
    rate = -pi @ np.diag(q)
    if rate <= 0 or not np.allclose(pi @ q, 0, atol=1e-10):
        raise ValueError('Matrix is not a valid reversible CTMC')
    return s, pi, q / rate


def stationary_from_log(path, size):
    text = Path(path).read_text()
    matches = re.findall(r'Base frequencies \([^)]*\):\s*([^\n]+)', text)
    if not matches:
        raise ValueError(f'No fitted stationary frequencies in {path}')
    values = np.fromstring(matches[-1], sep=' ')
    if len(values) != size or (values <= 0).any() or not np.isclose(values.sum(), 1, atol=2e-5):
        raise ValueError('Invalid fitted stationary frequencies')
    return values / values.sum()


def infer_ml(alignment, model, prefix, threads=8, seed=42, starts=10):
    """Fixed empirical rates/frequencies; only permitted nuisance parameters fit."""
    prefix = Path(prefix)
    run(['raxml-ng', '--search', '--redo', '--msa', alignment, '--model', model,
         '--tree', f'rand{{{starts}}},pars{{{starts}}}', '--seed', seed,
         '--threads', threads, '--workers', '1', '--prefix', prefix], prefix.with_suffix('.command.log'))
    return Path(str(prefix) + '.raxml.bestTree'), Path(str(prefix) + '.raxml.bestModel')
