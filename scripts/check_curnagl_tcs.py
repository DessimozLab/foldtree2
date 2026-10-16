#!/usr/bin/env python3
"""Small Slurm-only runtime/bundle and Snakemake syntax check before pilot release."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def check_native_tools(cfg):
    """A wrapper's presence/version text is not proof that its binary works."""
    commands = [('raxml-ng', '--version'), ('foldseek', 'version')]
    if cfg['three_di_route'] == 'foldmason':
        commands.append(('foldmason', 'version'))
    for tool, flag in commands:
        executable = shutil.which(tool)
        if executable is None:
            raise FileNotFoundError(f'Required tool absent from PATH: {tool}')
        subprocess.run([executable, flag], check=True)


def check(deployment):
    import yaml
    sys.path.insert(0, str(deployment / 'foldtree2/scripts'))
    from curnagl_tcs import load_baseline, scoring_functions, tcs
    from ete3 import Tree
    cfg = yaml.safe_load((deployment / 'config.yaml').read_text())
    baseline = load_baseline(cfg)
    # Check the actual remote baseline, including its unmodified scoring formulas.
    functions = scoring_functions(cfg['baseline_score_source'])
    result = tcs(Tree('((a:1,b:1):1,(c:1,d:1):1);'),
                 {'a': {'root', 'X'}, 'b': {'root', 'X'}, 'c': {'root', 'Y'}, 'd': {'root', 'Y'}}, functions)
    if result != {'score': 5, 'root_score': 20, 'root_score_nr': 8}:
        raise ValueError('Frozen baseline does not match the golden TCS test')
    archive_manifest = next(deployment.glob('*.tar.gz.manifest.json'))
    staged = json.loads(archive_manifest.read_text())
    for name, checksum in staged['files'].items():
        if baseline.digest(deployment / name) != checksum:
            raise ValueError('Staged source/bundle differs from archive: ' + name)
    layer_path = deployment / 'runtime-layer.json'
    if layer_path.is_file():
        layer = json.loads(layer_path.read_text())
        for name, checksum in layer['files'].items():
            if baseline.digest(deployment / name) != checksum:
                raise ValueError('Runtime layer changed: ' + name)
    entries = json.loads(Path(cfg['models_manifest']).read_text())['models']
    cfg['output_dir'] = str(deployment / 'results/runtime-check')
    plan = {'schema': 'foldtree_tcs_v1', 'config': cfg, 'methods': baseline.methods(entries, cfg),
            'families': {}, 'assets': {str(deployment / name): value for name, value in staged['files'].items()},
            'missing_assets': []}
    path = deployment / 'runtime-check-plan.json'
    baseline.write_json(path, plan)
    check_native_tools(cfg)
    baseline.preflight(path, plan, cfg)
    snakefile = deployment / 'foldtree2/workflow/curnagl_tcs.smk'
    controller = '/work/FAC/FBM/DBC/cdessim2/default/dmoi/miniconda3/envs/snakemake/bin/snakemake'
    subprocess.run([controller, '--snakefile', str(snakefile), '--directory', str(deployment),
                    '--config', 'plan=' + str(path), '--cores', '4', '--dry-run',
                    str(Path(cfg['output_dir']) / 'preflight.json')], check=True)
    import importlib.metadata
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'torch-geometric', 'pytorch-lightning', 'gemmi', 'h5py', 'ete3')}
    baseline.write_json(deployment / 'runtime-check-passed.json', {'baseline_scores': result, 'bundles': 5,
                        'methods': len(plan['methods']), 'archive_files_verified': len(staged['files']),
                        'python_packages': versions, 'runtime_layer_sha256': baseline.digest(layer_path) if layer_path.is_file() else None})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    args = parser.parse_args()
    check(args.deployment.resolve())
