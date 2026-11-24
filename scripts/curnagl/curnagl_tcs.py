#!/usr/bin/env python3
"""Offline, frozen-cohort TCS extension of ft2_benchmark's Snakemake baseline."""
from __future__ import annotations


import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import ast
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys


def load_baseline(cfg):
    source = Path(cfg['benchmark_repo']) / 'src/topology_benchmark.py'
    spec = importlib.util.spec_from_file_location('tcs_topology_baseline', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(Path(cfg['foldtree2_repo']) / 'scripts'))
    sys.path.insert(0, cfg['foldtree2_repo'])
    return module


def scoring_functions(source):
    """Load ONLY four pure baseline functions, never global NCBITaxa/downloads."""
    wanted = {'frac_score', 'getTaxOverlap', 'getTaxOverlap_root', 'sum_rootscore'}
    parsed = ast.parse(Path(source).read_text())
    definitions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in definitions} != wanted:
        raise ValueError('Baseline TCS functions missing')
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def tcs(tree, lineages, functions):
    names = tree.get_leaf_names()
    if len(names) != len(set(names)) or set(names) != set(lineages):
        raise ValueError('Tree taxa differ from frozen taxonomy; pruning is forbidden')
    if any(not value for value in lineages.values()):
        raise ValueError('Missing taxonomy')
    for leaf in tree:
        leaf.add_feature('lineage', set(lineages[leaf.name]))
    functions['getTaxOverlap'](tree)
    functions['getTaxOverlap_root'](tree)
    root_score, root_score_nr = functions['sum_rootscore'](tree)
    scores = {'score': tree.score, 'root_score': root_score, 'root_score_nr': root_score_nr}
    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError('Nonfinite TCS')
    return scores


def pdb_sequence(path, filter_plddt=False):
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    structure = PDBParser(QUIET=True).get_structure('protein', str(path))
    models = list(structure)
    if len(models) != 1:
        raise ValueError('Expected one structural model')
    chains = [c for c in models[0] if any(r.id[0] == ' ' and 'CA' in r for r in c)]
    if len(chains) != 1:
        raise ValueError('Expected one selected protein chain')
    residues = [r for r in chains[0] if r.id[0] == ' ' and 'CA' in r]
    if not residues:
        raise ValueError('No protein residues')
    # Match AFDB_tools.filter_plddt: first atom per residue, not CA specifically.
    confidence = [next(r.get_atoms()).bfactor for r in residues]
    if filter_plddt and (min(confidence) < 10 or statistics.mean(confidence) < 50):
        raise ValueError('OMA baseline pLDDT filter: min 10 / mean 50')
    sequence = ''.join(seq1(r.resname, custom_map={'MSE': 'M'}) for r in residues)
    if 'X' in sequence:
        raise ValueError('Unknown structure residue')
    return sequence


def family_record(directory, dataset, relative, db, baseline):
    # Legacy OMA filtering records its retained cohort in finalset.csv.
    metadata = directory / 'finalset.csv'
    if not metadata.is_file():
        metadata = directory / 'sequence_dataset.csv'
    with metadata.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    rebuild_evidence = None
    generated_evidence = None
    if any(row.get('Sequence_source') == 'observed_pdb_residues' for row in rows):
        schema = json.loads((directory / 'rebuild_provenance.json').read_text())['schema']
        if schema == 'cath_generated_genus_v1':
            from scripts.cath.generate_cath_families import validate_generated
            generated_evidence = validate_generated(directory)
        else:
            from scripts.rebuild_cath_families import validate_rebuild
            rebuild_evidence = validate_rebuild(directory)
    structures = sorted((directory / 'structs').glob('*.pdb'))
    if len(structures) < 4:
        raise ValueError('Fewer than four structures')
    sequences, lineages, paths = {}, {}, {}
    for path in structures:
        name = path.stem
        if not baseline.SAFE.fullmatch(name):
            raise ValueError('Unsafe structure identifier')
        matches = [r for r in rows if r.get('query') == name or r.get('Entry') == name]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous/missing metadata: {name}')
        row = matches[0]
        lineage = row.get('Taxonomic lineage (Ids)', '')
        tokens = lineage.split(',')
        ids = []
        for token in tokens:
            match = re.fullmatch(r'\s*(\d+)(?:\s*\([^)]*\))?\s*', token)
            if not match:
                raise ValueError(f'Invalid/missing taxonomy: {name}')
            ids.append(int(match[1]))
        for taxid in ids:
            if db.execute('SELECT 1 FROM species WHERE taxid=?', (taxid,)).fetchone() is None:
                raise ValueError(f'Taxonomy ID absent from frozen SQLite: {taxid}')
        sequence = pdb_sequence(path, filter_plddt=(dataset == 'OMA'))
        # CATH can provide a domain of the metadata's full UniProt sequence.
        if sequence not in row.get('Sequence', '').upper():
            raise ValueError(f'Structure sequence not represented in metadata: {name}')
        sequences[name], lineages[name], paths[name] = sequence, tokens, str(path.resolve())
    # No accidental loss of a metadata taxon with an absent structure.
    metadata_names = {r.get('query') or r.get('Entry') for r in rows}
    if metadata_names != set(sequences):
        raise ValueError('Metadata/structure taxa differ; no silent filtering')
    inputs = {str(p.resolve()): baseline.digest(p) for p in [metadata, *structures]}
    if generated_evidence:
        proof, frozen = generated_evidence
        inputs[str((directory / 'rebuild_provenance.json').resolve())] = baseline.digest(directory / 'rebuild_provenance.json')
        inputs[proof['plan']] = proof['plan_sha256']
        inputs.update(frozen['inputs'])
        for record in proof['proteins'].values():
            inputs[record['source_cif']] = record['source_cif_sha256']
    if rebuild_evidence is not None:
        proof = directory / 'rebuild_provenance.json'
        inputs[str(proof.resolve())] = baseline.digest(proof)
        for record in rebuild_evidence['proteins'].values():
            inputs[record['reference_source']] = record['reference_sha256']
            inputs.update(record.get('reference_evidence', {}))
    return {'dataset': dataset, 'relative_path': str(relative), 'directory': str(directory.resolve()),
            'labels': {name: name for name in sequences}, 'structures': paths,
            'sequences': sequences, 'lineages': lineages, 'inputs': inputs}


def inventory(config_path, destination):
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    baseline = load_baseline(cfg)
    taxonomy = Path(cfg['taxonomy_db'])
    if not taxonomy.is_file():
        raise FileNotFoundError('Offline taxonomy database is required')
    frozen_db = Path(cfg['deployment_dir']) / 'taxonomy/taxa.sqlite'
    frozen_db.parent.mkdir(parents=True, exist_ok=True)
    if not frozen_db.exists():
        shutil.copy2(taxonomy, frozen_db)
    if baseline.digest(frozen_db) != baseline.digest(taxonomy):
        raise ValueError('Taxonomy changed during snapshot')
    cfg['taxonomy_db'] = str(frozen_db)
    included, excluded = {}, []
    with sqlite3.connect(f'file:{frozen_db}?mode=ro', uri=True) as db:
        for dataset, root_name in cfg['datasets'].items():
            root = Path(root_name).resolve()
            if not root.is_dir():
                raise FileNotFoundError(root)
            for folder, children, files in os.walk(root):
                children.sort()
                children[:] = [d for d in children if d not in {'structs', 'logs', 'tmp', '.snakemake'}]
                if 'sequence_dataset.csv' not in files:
                    if 'identifiers.txt' in files:
                        relative = Path(folder).relative_to(root)
                        excluded.append({'dataset': dataset, 'relative_path': str(relative),
                                         'reason': 'Identifiers only: no existing metadata/structures'})
                        children[:] = []
                    continue
                children[:] = []
                relative = Path(folder).relative_to(root)
                ident = dataset + '__' + '__'.join(relative.parts)
                if not baseline.SAFE.fullmatch(ident) or ident in included:
                    raise ValueError('Unsafe/colliding family namespace')
                try:
                    included[ident] = family_record(Path(folder), dataset, relative, db, baseline)
                except (ValueError, OSError) as error:
                    excluded.append({'dataset': dataset, 'relative_path': str(relative), 'reason': str(error)})
                if (len(included) + len(excluded)) % 50 == 0:
                    print(json.dumps({'dataset': dataset, 'eligible': len(included), 'excluded': len(excluded)}), flush=True)
    if any(not any(f['dataset'] == d for f in included.values()) for d in cfg['datasets']):
        baseline.write_json(Path(cfg['deployment_dir']) / 'inventory_exclusions.json', excluded)
        raise ValueError('At least one eligible family from each dataset is required; see exclusions')
    entries = json.loads(Path(cfg['models_manifest']).read_text())['models']
    methods = baseline.methods(entries, cfg)
    if len(methods) != 13 or {e['size'] for e in entries} != {10, 20, 30, 40, 50}:
        raise ValueError('Expected exactly 13 methods and all five production alphabets')
    assets = [Path(cfg['models_manifest']), frozen_db, Path(__file__).resolve(),
              Path(cfg['benchmark_repo']) / 'src/topology_benchmark.py',
              Path(cfg['baseline_score_source']), Path(cfg['mad']),
              Path(cfg['foldtree2_repo']) / 'foldtree2/raxml-ng/raxml-ng',
              Path(cfg['foldtree2_repo']) / 'workflow/curnagl_tcs.smk',
              Path(cfg['original_foldtree_repo']) / 'src/foldtree/foldseek2tree.py']
    assets += list((Path(cfg['foldtree2_repo']) / 'foldtree2').rglob('*.py'))
    assets += list((Path(cfg['foldtree2_repo']) / 'scripts').glob('*.py'))
    assets += [Path(cfg['foldtree2_repo']) / 'foldtree2/config/aaindex1.csv',
               Path(cfg['foldtree2_repo']) / 'foldtree2/config/3diphy_submats/Q.3Di.AF']
    for entry in entries:
        directory = Path(cfg['foldtree2_repo']) / entry['directory']
        assets += [directory / entry[k] for k in ('encoder', 'decoder', 'mafft', 'raxml')]
        assets += list(directory.glob(Path(entry['encoder']).stem + '*json'))
    if cfg.get('three_di_score_matrix'):
        assets.append(Path(cfg['three_di_score_matrix']))
    payload = {'schema': 'foldtree_tcs_v1', 'config': cfg, 'methods': methods,
               'families': included, 'exclusions': excluded,
               'assets': {str(p): baseline.digest(p) for p in assets}, 'missing_assets': [],
               'config_sha256': baseline.digest(config_path)}
    destination = Path(destination)
    if destination.exists() and json.loads(destination.read_text()) != payload:
        raise ValueError('Frozen inventory changed; use a new deployment')
    baseline.write_json(destination, payload)
    pilot = dict(payload)
    pilot['families'] = {next(k for k in sorted(included) if included[k]['dataset'] == d):
                         included[next(k for k in sorted(included) if included[k]['dataset'] == d)] for d in cfg['datasets']}
    pilot['config'] = {**cfg, 'output_dir': str(Path(cfg['deployment_dir']) / 'results/pilot')}
    baseline.write_json(destination.with_name('pilot.json'), pilot)
    print(json.dumps({'eligible': len(included), 'excluded': len(excluded), 'pilot': list(pilot['families']),
                      'expected_full_scores': len(included) * len(methods)}))


def prepare(plan_path, plan, baseline, family):
    from scripts.benchmark_common import begin_stage, complete_stage, write_fasta
    entry = plan['families'][family]
    baseline.verify(entry['inputs'])
    dest = baseline.prepared(plan, family)
    if not begin_stage(dest, baseline.signature(plan_path, plan)):
        return
    structures = dest / 'structs'
    structures.mkdir(exist_ok=True)
    for name, path in entry['structures'].items():
        target = structures / (name + '.pdb')
        if target.is_symlink():
            target.unlink()
        target.symlink_to(path)
    write_fasta(dest / 'aa.fasta', entry['sequences'])
    complete_stage(dest, baseline.signature(plan_path, plan), [dest / 'aa.fasta'])


def reuse_completed(plan_path, plan, baseline):
    """Import verified inference only; all scores use the new rooting policy."""
    from scripts.benchmark_common import begin_stage, complete_stage
    origin = plan['recovery_from']
    original_path = Path(origin['plan'])
    baseline.verify({str(original_path): origin['sha256']})
    original = json.loads(original_path.read_text())
    if original['families'] != plan['families'] or original['methods'] != plan['methods']:
        raise ValueError('Recovery must retain the exact original cohort and methods')
    ignored = {'deployment_dir', 'output_dir', 'recovery_policy'}
    if ({k: v for k, v in original['config'].items() if k not in ignored}
            != {k: v for k, v in plan['config'].items() if k not in ignored}):
        raise ValueError('Recovery changed scientific configuration')
    baseline.verify(original['assets'])
    old_signature = baseline.signature(original_path, original)
    new_signature = baseline.signature(plan_path, plan)
    imported = {}
    for family in plan['families']:
        for stage in ['cohort', 'allvall', *plan['methods']]:
            source = baseline.output(original, family, stage)
            marker = source / 'completed.json'
            if not marker.is_file():
                continue
            record = json.loads(marker.read_text())
            if any(record['signature'].get(k) != v for k, v in old_signature.items()):
                raise ValueError(f'Original completion belongs to another run: {marker}')
            baseline.verify({str(source / p): h for p, h in record['outputs'].items()})
            dest = baseline.output(plan, family, stage)
            signature = {**record['signature'], **new_signature}
            imported[stage] = imported.get(stage, 0) + 1
            if not begin_stage(dest, signature):
                continue
            for name in record['outputs']:
                relative = Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError(f'Unsafe completion output: {name}')
                target = dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    baseline.verify({str(target): record['outputs'][name]})
                else:
                    shutil.copyfile(source / relative, target)
            if stage == 'cohort':
                structures = dest / 'structs'
                structures.mkdir(exist_ok=True)
                for name, pdb in plan['families'][family]['structures'].items():
                    target = structures / (name + '.pdb')
                    if not target.exists():
                        target.symlink_to(pdb)
                    elif target.resolve() != Path(pdb).resolve():
                        raise ValueError(f'Recovery structure changed: {target}')
            complete_stage(dest, signature, [dest / p for p in record['outputs']],
                           imported_from={'marker': str(marker), 'sha256': baseline.digest(marker)})
        print(json.dumps({'reused_family': family, 'stage_counts': imported}), flush=True)
    baseline.write_json(Path(plan['config']['deployment_dir']) / 'reuse-completed.json', imported)


def infer(plan_path, plan, baseline, family, method_id, threads):
    if not plan['config'].get('recovery_policy') or plan['methods'][method_id]['kind'] not in {'AA', 'FT2'}:
        return baseline.infer(plan_path, plan, plan['config'], family, method_id, threads)
    from scripts.benchmark_common import begin_stage, complete_stage, infer_ml, read_fasta, run, write_fasta
    cfg = plan['config']
    baseline.verify(plan['assets'])
    baseline.verify(plan['families'][family]['inputs'])
    dest = baseline.output(plan, family, method_id)
    signature = {**baseline.signature(plan_path, plan), 'threads': threads}
    if not begin_stage(dest, signature):
        return
    method = plan['methods'][method_id]
    recovery = {'selenocysteine_as_unknown': {}, 'branch_optimizer': 'nr_fast'}
    if method['kind'] == 'AA':
        records = read_fasta(baseline.prepared(plan, family) / 'aa.fasta')
        recovery['selenocysteine_as_unknown'] = {
            name: [i + 1 for i, residue in enumerate(seq) if residue == 'U']
            for name, seq in records.items() if 'U' in seq}
        normalized = dest / 'aa.input.fasta'
        write_fasta(normalized, {name: seq.replace('U', 'X') for name, seq in records.items()})
        run(['mafft', '--amino', '--thread', threads, '--localpair', '--maxiterate', '1000', normalized],
            dest / 'command.log', dest / 'alignment.fasta')
        model = 'LG+G+I'
    else:
        entry = method['entry']
        baseline.learned_alignment(cfg, entry, baseline.prepared(plan, family), dest, threads)
        rate = Path(cfg['foldtree2_repo']) / entry['directory'] / entry['raxml']
        model = f'MULTI{entry["size"]}_GTR{{{rate}}}+I'
    records = read_fasta(dest / 'alignment.fasta', aligned=True)
    if set(records) != set(plan['families'][family]['labels']):
        raise ValueError('Alignment taxa differ from the frozen cohort')
    try:
        tree, fitted = infer_ml(dest / 'alignment.fasta', model, dest / 'ml', threads,
                               cfg['seed'], cfg['starts_each'])
    except subprocess.CalledProcessError as error:
        log = dest / 'ml.command.log'
        if (not error.cmd or Path(str(error.cmd[0])).name != 'raxml-ng' or '--blopt' in error.cmd
                or not log.is_file()
                or 'BL opt converged to a worse likelihood score' not in log.read_text().rsplit('\nCOMMAND ', 1)[-1]):
            raise
        shutil.copy2(log, dest / 'ml.nr_fast.failed.log')
        recovery['branch_optimizer'] = 'nr_safe'
        command = [*error.cmd, '--blopt', 'nr_safe']
        run(command, log)
        tree, fitted = dest / 'ml.raxml.bestTree', dest / 'ml.raxml.bestModel'
    shutil.copy2(tree, dest / 'tree.nwk')
    baseline.splits(baseline.tree(dest / 'tree.nwk'), plan['families'][family]['labels'])
    baseline.write_json(dest / 'execution-policy.json', recovery)
    outputs = [dest / name for name in ('tree.nwk', 'alignment.fasta', 'ml.raxml.log', 'execution-policy.json')]
    outputs.append(fitted)
    if method['kind'] == 'AA':
        outputs.append(normalized)
    if recovery['branch_optimizer'] == 'nr_safe':
        outputs.append(dest / 'ml.nr_fast.failed.log')
    complete_stage(dest, signature, outputs, method=method_id, family=family)


def score(plan_path, plan, baseline, family, method):
    from scripts.benchmark_common import begin_stage, complete_stage, run, read_fasta
    cfg = plan['config']
    entry = plan['families'][family]
    baseline.verify(entry['inputs'])
    expected = baseline.signature(plan_path, plan)
    source = baseline.output(plan, family, method)
    record = json.loads((source / 'completed.json').read_text())
    if any(record['signature'].get(k) != v for k, v in expected.items()):
        raise ValueError('Inference provenance does not match this run')
    baseline.verify({str(source / p): h for p, h in record['outputs'].items()})
    if plan['methods'][method]['kind'] != 'FoldTree':
        if set(read_fasta(source / 'alignment.fasta', aligned=True)) != set(entry['labels']):
            raise ValueError('Alignment taxa changed')
    dest = source / 'tcs'
    signature = {**expected, 'tree_sha256': baseline.digest(source / 'tree.nwk'),
                 'tcs_driver_sha256': baseline.digest(__file__)}
    if not begin_stage(dest, signature):
        return
    unrooted = baseline.tree(source / 'tree.nwk')
    names = unrooted.get_leaf_names()
    if (len(names) != len(set(names)) or set(names) != set(entry['labels'])
            or set(names) != set(entry['lineages'])):
        raise ValueError('Tree taxa differ from frozen taxonomy; pruning is forbidden')
    if any(not lineage for lineage in entry['lineages'].values()):
        raise ValueError('Missing taxonomy')
    clipped = 0
    for node in unrooted.traverse():
        if not math.isfinite(node.dist):
            raise ValueError('Nonfinite branch length')
        if node.dist < 0:
            node.dist = 0
            clipped += 1
    # Identical to baseline postprocess(delta=0), applied to every method.
    unrooted.write(outfile=str(dest / 'tree.nwk'), format=1)
    if (cfg.get('recovery_policy') and
            sum(node.dist > 0 for node in unrooted.traverse() if not node.is_root()) < 3):
        result = {'family': family, 'dataset': entry['dataset'], 'method': method,
                  'n_taxa': len(entry['labels']), 'rooting': 'undefined',
                  'negative_branches_clipped': clipped,
                  'score': None, 'root_score': None, 'root_score_nr': None,
                  'plan_sha256': baseline.digest(plan_path), 'rooted_tree_sha256': None,
                  'score_status': 'undefined', 'undefined_reason': 'mad_insufficient_positive_branches'}
        baseline.write_json(dest / 'score.json', result)
        complete_stage(dest, signature, [dest / 'tree.nwk', dest / 'score.json'])
        return
    run([cfg['mad'], '-n', '-s', dest / 'tree.nwk'], dest / 'rooting.log')
    rooted = dest / 'tree.nwk.rooted'
    tree = baseline.tree(rooted)
    for node in tree.traverse():
        if not math.isfinite(node.dist) or node.dist < 0:
            raise ValueError('Invalid rooted branch length')
    result = {'family': family, 'dataset': entry['dataset'], 'method': method, 'n_taxa': len(entry['labels']),
              'rooting': 'MAD', 'negative_branches_clipped': clipped,
              **tcs(tree, entry['lineages'], scoring_functions(cfg['baseline_score_source'])),
              'plan_sha256': baseline.digest(plan_path), 'rooted_tree_sha256': baseline.digest(rooted)}
    if cfg.get('recovery_policy'):
        result.update(score_status='defined', undefined_reason='')
    baseline.write_json(dest / 'score.json', result)
    complete_stage(dest, signature, [rooted, dest / 'score.json'])


def aggregate(plan_path, plan, baseline):
    rows = []
    for family in plan['families']:
        for method in plan['methods']:
            dest = baseline.output(plan, family, method) / 'tcs'
            record = json.loads((dest / 'completed.json').read_text())
            expected = baseline.signature(plan_path, plan)
            if any(record['signature'].get(k) != v for k, v in expected.items()):
                raise ValueError('TCS completion belongs to another run')
            baseline.verify({str(dest / p): h for p, h in record['outputs'].items()})
            row = json.loads((dest / 'score.json').read_text())
            if row['plan_sha256'] != baseline.digest(plan_path) or row['family'] != family or row['method'] != method:
                raise ValueError('Mismatched score provenance')
            rows.append(row)
    root = baseline.output(plan)
    baseline.write_csv(root / 'tcs_scores.csv', rows)
    summaries = []
    for dataset in plan['config']['datasets']:
        paired = {family for family, entry in plan['families'].items() if entry['dataset'] == dataset}
        paired -= {row['family'] for row in rows if row.get('score_status') == 'undefined'}
        for method in plan['methods']:
            selected = [r for r in rows if r['dataset'] == dataset and r['method'] == method]
            comparable = [r for r in selected if r['family'] in paired]
            summaries.append({'dataset': dataset, 'method': method, 'families': len(selected),
                              **{f'mean_{k}': statistics.mean(r[k] for r in comparable) if comparable else None
                                 for k in ('score', 'root_score', 'root_score_nr')}})
            if plan['config'].get('recovery_policy'):
                summaries[-1].update(paired_families=len(comparable),
                                     undefined_families=sum(r['score_status'] == 'undefined' for r in selected))
    baseline.write_csv(root / 'summary.csv', summaries)
    marker = {'plan_sha256': baseline.digest(plan_path), 'families': len(plan['families']),
              'methods': len(plan['methods']), 'scores': len(rows)}
    if plan['config'].get('recovery_policy'):
        marker['undefined_scores'] = sum(r['score_status'] == 'undefined' for r in rows)
    baseline.write_json(root / 'validated.json', marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory', 'preflight', 'prepare', 'allvall', 'infer', 'score', 'aggregate', 'reuse'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--family')
    parser.add_argument('--method')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.action == 'inventory':
        if not args.config or not args.out:
            parser.error('inventory requires --config and --out')
        inventory(args.config, args.out)
        return
    if not args.plan:
        parser.error('--plan required')
    plan = json.loads(args.plan.read_text())
    if plan['schema'] != 'foldtree_tcs_v1':
        raise ValueError('Unsupported TCS schema')
    cfg = plan['config']
    if cfg.get('recovery_policy') not in (None, 'explicit-undefined-u-as-x-nr-safe-v1'):
        raise ValueError('Unsupported recovery policy')
    baseline = load_baseline(cfg)
    if args.action in {'prepare', 'allvall', 'infer', 'score'} and args.family not in plan['families']:
        parser.error('Unknown frozen family')
    if args.action in {'infer', 'score'} and args.method not in plan['methods']:
        parser.error('Unknown frozen method')
    if args.threads < 1:
        parser.error('Threads must be positive')
    os.environ.update({k: str(args.threads) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    if args.action == 'preflight':
        baseline.preflight(args.plan, plan, cfg)
        scoring_functions(cfg['baseline_score_source'])
        if not Path(cfg['mad']).is_file():
            raise FileNotFoundError(cfg['mad'])
    elif args.action == 'prepare': prepare(args.plan, plan, baseline, args.family)
    elif args.action == 'allvall': baseline.allvall(args.plan, plan, cfg, args.family, args.threads)
    elif args.action == 'infer': infer(args.plan, plan, baseline, args.family, args.method, args.threads)
    elif args.action == 'score': score(args.plan, plan, baseline, args.family, args.method)
    elif args.action == 'reuse': reuse_completed(args.plan, plan, baseline)
    else: aggregate(args.plan, plan, baseline)


if __name__ == '__main__':
    main()
