#!/usr/bin/env python3
"""Configure an already extracted, isolated deployment; never edit live checkouts."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def configure(root, template):
    root = root.resolve()
    if (root / 'config.yaml').exists():
        raise FileExistsError('Deployment is already configured; choose a new run directory')
    work = Path('/work/FAC/FBM/DBC/cdessim2/default/dmoi')
    legacy = work / 'projects/ft2_benchmark/ft2_benchmark'
    snapshot = root / 'legacy_snapshot'
    snapshot.mkdir()
    # Capture tracked workflow/source files from the WORKTREE (including remote edits).
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=legacy).decode().split('\0')
    for relative in tracked:
        path = Path(relative)
        if not relative or not (path.parts[0] in {'src', 'workflow', 'slurmsimple'} or path.suffix == '.sh'):
            continue
        source = legacy / path
        if not source.is_file() or source.stat().st_size > 100 * 1024**2:
            continue
        target = snapshot / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    status = subprocess.check_output(['git', 'status', '--short', '--', '.'], cwd=legacy, text=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=legacy, text=True).strip()
    with (snapshot / 'worktree.patch').open('wb') as stream:
        subprocess.run(['git', 'diff', '--binary', 'HEAD', '--', '.'], cwd=legacy, stdout=stream, check=True)
    hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in snapshot.rglob('*') if p.is_file()}
    (snapshot / 'provenance.json').write_text(json.dumps({'head': head, 'status': status, 'files': hashes}, indent=2))
    configuration = Path(template).read_text().replace('REPLACE_RUN_ID', root.name)
    if str(root) not in configuration:
        raise ValueError('Template deployment root does not match destination')
    (root / 'config.yaml').write_text(configuration)
    import yaml
    cfg = yaml.safe_load(configuration)
    shutil.copy2(root / 'foldtree2/scripts/supervise_curnagl_tcs.py', root / 'supervisor.py')
    env = work / 'miniforge/envs/foldtree2'
    paths = [root / 'bin', root / 'foldtree2/foldtree2/mafft_tools',
             env / 'bin', work / 'miniconda3/envs/foldseek/bin', work / 'miniforge/envs/foldtree/bin']
    setup = '\n'.join([
        'set -euo pipefail',
        'export PATH=' + shlex.quote(':'.join(map(str, paths))) + ':$PATH',
        'export PYTHONPATH=' + shlex.quote(str(root / 'foldtree2') + ':' + str(root / 'foldtree2/scripts')),
        # Ensure the shipped Foldmason resolves its own conda libraries.
        'export LD_LIBRARY_PATH=' + shlex.quote(str(root / 'lib')) + ':${LD_LIBRARY_PATH:-}',
        'export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8',
        'cd ' + shlex.quote(str(root)),
    ])
    controller = work / 'miniconda3/envs/snakemake/bin/snakemake'
    python = env / 'bin/python'
    driver = root / 'foldtree2/scripts/curnagl_tcs.py'
    snakefile = root / 'foldtree2/workflow/curnagl_tcs.smk'
    stages = [('inventory', 4, '16G', '4:00:00'),
                                         ('runtime-check', 4, '16G', '0:30:00'),
                                         ('pilot', 8, '32G', '4:00:00'),
                                         ('full', 32, '96G', '24:00:00')]
    if cfg.get('cath_rebuild'):
        stages.insert(0, ('rebuild', cfg['cath_rebuild']['workers'], '32G', '8:00:00'))
        (root / 'rebuild-result-location.json').write_text(json.dumps({
            'summary': str(Path(cfg['cath_rebuild']['destination']).parent / 'rebuild_summary.json')}))
    for phase, cores, memory, duration in stages:
        command = shlex.join([str(python), str(driver), 'inventory', '--config', str(root / 'config.yaml'),
                              '--out', str(root / 'plan.json')])
        if phase == 'rebuild':
            rebuild = cfg['cath_rebuild']
            command = shlex.join([str(python), str(root / 'foldtree2/scripts/rebuild_cath_families.py'),
                '--source', rebuild['source'], '--out', rebuild['destination'],
                '--taxonomy-dir', rebuild['taxonomy_dir'], '--workers', str(rebuild['workers']),
                '--min-identity', str(rebuild['min_identity']), '--min-coverage', str(rebuild['min_coverage']),
                '--min-matches', str(rebuild['min_matches'])])
        elif phase == 'runtime-check':
            command = shlex.join([str(python), str(root / 'foldtree2/scripts/check_curnagl_tcs.py'),
                                  '--deployment', str(root)])
        elif phase != 'inventory':
            plan = root / ('pilot.json' if phase == 'pilot' else 'plan.json')
            command = shlex.join([str(controller), '--snakefile', str(snakefile), '--directory', str(root),
                                 '--config', 'plan=' + str(plan), '--cores', str(cores),
                                 '--resources', 'mem_mb=' + str(24000 if phase == 'pilot' else 96000),
                                 '--rerun-incomplete', '--printshellcmds', '--latency-wait', '60'])
        if phase in {'pilot', 'full'}:
            command = 'srun --cpu-bind=cores ' + command
        allocated_cores = 16 if phase == 'pilot' else cores
        header = '\n'.join(['#!/bin/bash', '#SBATCH --account=cdessim2_default', '#SBATCH --partition=cpu',
                            '#SBATCH --ntasks=1', '#SBATCH --hint=nomultithread',
                            '#SBATCH --job-name=ft2-tcs-' + phase, '#SBATCH --cpus-per-task=' + str(allocated_cores),
                            '#SBATCH --mem=' + memory, '#SBATCH --time=' + duration,
                            '#SBATCH --output=' + str(root / ('logs/' + phase + '-%j.log'))])
        (root / (phase + '.sbatch')).write_text(header + '\n' + setup + '\n' + command + '\n')
    (root / 'logs').mkdir(exist_ok=True)
    print(json.dumps({'deployment': str(root), 'baseline_head': head, 'snapshot_files': len(hashes)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('--template', required=True, type=Path)
    args = parser.parse_args()
    configure(args.deployment, args.template)
