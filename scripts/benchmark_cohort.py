"""Frozen, representation-independent OMA family/taxon cohort for benchmarks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from benchmark_common import ROOT, digest, read_fasta, write_json

DEFAULT_ROOT = ROOT / 'families/Information_benchmark/marker_genes'
DEFAULT_COHORT = ROOT / 'configs/oma_benchmark_cohort.json'
NOTEBOOK = ROOT / 'foldtree2/notebooks/benchmarks/treelikelihood_info_theory_benchmark.ipynb'


def stored_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def freeze(root=DEFAULT_ROOT, output=DEFAULT_COHORT):
    from ete3 import Tree
    root = Path(root).resolve()
    markers = sorted((root / 'marker_genes').glob('OMAGroup_*.fa'))
    if not markers:
        raise ValueError('Missing OMAGroup_*.fa protein markers used by the notebook')
    requested, included, excluded = [], [], []
    for marker in markers:
        family = marker.stem.removeprefix('OMAGroup_')
        requested.append(family)
        directory = root / family
        alignment, tree = directory / 'sequences.aligned.fa', directory / 'raxml_lg_tree.raxml.bestTree'
        reason = None
        if not alignment.is_file() or not tree.is_file():
            reason = 'missing_reference_alignment_or_tree'
        else:
            aa = read_fasta(alignment, aligned=True)
            accessions = {label: label.split('|')[0] for label in aa}
            if len(set(accessions.values())) != len(accessions):
                raise ValueError(f'Ambiguous UniProt accessions in OMA family {family}')
            if len(aa) < 4:
                reason = 'fewer_than_four_reference_taxa'
            elif set(Tree(str(tree), format=1).get_leaf_names()) != set(aa):
                reason = 'reference_tree_alignment_taxa_mismatch'
            elif any(not (directory / 'structs' / (acc + '.pdb')).is_file() for acc in accessions.values()):
                reason = 'missing_reference_structures'
        if reason:
            excluded.append({'family': family, 'reason': reason, 'marker': stored_path(marker),
                             'marker_sha256': digest(marker)})
            continue
        inputs = {stored_path(p): digest(p) for p in [marker, alignment, tree]}
        taxa = []
        for label, accession in sorted(accessions.items()):
            pdb = directory / 'structs' / (accession + '.pdb')
            inputs[stored_path(pdb)] = digest(pdb)
            taxa.append({'label': label, 'accession': accession, 'pdb': stored_path(pdb),
                         'aa_sha256': hashlib.sha256(aa[label].replace('-', '').encode()).hexdigest()})
        extras = sorted(p.name for p in (directory / 'structs').glob('*.pdb') if p.stem not in accessions.values())
        included.append({'family': family, 'directory': stored_path(directory), 'marker': stored_path(marker),
                         'aa_alignment': stored_path(alignment), 'reference_tree': stored_path(tree),
                         'taxa': taxa, 'excluded_nonreference_pdbs': extras, 'inputs': inputs})
    payload = {'schema': 'oma_common_cohort_v1', 'source_notebook': stored_path(NOTEBOOK),
               'notebook_sha256': digest(NOTEBOOK), 'families_root': stored_path(root),
               'selection': 'notebook OMA protein markers; AA-reference taxa with structures; minimum four taxa',
               'requested_family_ids': requested, 'families': included, 'exclusions': excluded}
    output = Path(output)
    if output.exists() and json.loads(output.read_text()) != payload:
        raise ValueError('Frozen cohort changed; use a new manifest path, do not overwrite')
    write_json(output, payload)
    return payload


def load(path=DEFAULT_COHORT, verify=True):
    cohort = json.loads(Path(path).read_text())
    if cohort['schema'] != 'oma_common_cohort_v1':
        raise ValueError('Unknown common-cohort schema')
    ids = [f['family'] for f in cohort['families']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate OMA family IDs')
    if verify:
        for family in cohort['families']:
            for source, checksum in family['inputs'].items():
                if digest(resolve_path(source)) != checksum:
                    raise ValueError(f'Frozen OMA input changed: {source}')
    return cohort


def select_families(root, manifest=None, limit=None):
    """Use the frozen cohort automatically for the notebook's standard root."""
    root = Path(root).resolve()
    if manifest is None and root == DEFAULT_ROOT.resolve() and DEFAULT_COHORT.exists():
        manifest = DEFAULT_COHORT
    if manifest is None:
        families = sorted(p for p in root.iterdir() if p.is_dir() and len(list((p / 'structs').glob('*.pdb'))) >= 4)
        return families[:limit] if limit else families, None
    cohort = load(manifest)
    if resolve_path(cohort['families_root']).resolve() != root:
        raise ValueError('Cohort manifest does not match the supplied family root')
    families = sorted(resolve_path(f['directory']) for f in cohort['families'])
    return families[:limit] if limit else families, Path(manifest).resolve()


def native_directory(root, representation, family, _visited=None):
    """Resolve reused artifact locations recorded by the native sweep."""
    root = Path(root).resolve()
    visited = set() if _visited is None else _visited
    if root in visited:
        raise ValueError(f'Cycle in native artifact reuse roots: {root}')
    visited.add(root)
    status = root / 'queue_status.json'
    if status.exists():
        record = json.loads(status.read_text())
        jobs = record['jobs']
        key = f'3Di_{family}' if representation == '3Di' else f'AA_{family}' if representation == 'AA' else f'native_{representation.removeprefix("FT2_")}_{family}'
        artifact = jobs.get(key, {}).get('artifacts')
        if isinstance(artifact, str):
            return Path(artifact)
        inherited = record.get('protocol', {}).get('reuse_native_root')
        if inherited:
            return native_directory(inherited, representation, family, visited)
    return root / '3di' / family if representation == '3Di' else root / 'native' / representation / family


def verify_stage(directory):
    """Verify a native stage before consuming its alignments or fitted models."""
    directory = Path(directory)
    record = json.loads((directory / 'completed.json').read_text())
    if record['signature'] != json.loads((directory / 'provenance.json').read_text()):
        raise ValueError(f'Native completion/provenance mismatch: {directory}')
    for name, checksum in record['outputs'].items():
        if digest(directory / name) != checksum:
            raise ValueError(f'Native output changed: {directory / name}')
    return record['signature']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--out', type=Path, default=DEFAULT_COHORT)
    args = parser.parse_args()
    cohort = freeze(args.families, args.out)
    print(json.dumps({'manifest': str(args.out), 'requested': len(cohort['requested_family_ids']),
                      'included': len(cohort['families']), 'excluded': cohort['exclusions']}))


if __name__ == '__main__':
    main()
