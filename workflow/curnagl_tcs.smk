"""TCS extension of workflow/topology_benchmark; inference is shared, scoring is offline."""
import json
from pathlib import Path
import re

PLAN_PATH = str(Path(config['plan']).resolve())
PLAN = json.loads(Path(PLAN_PATH).read_text())
CFG = PLAN['config']
ROOT = CFG['output_dir']
DRIVER = str(Path(CFG['foldtree2_repo']) / 'scripts/curnagl_tcs.py')
PYTHON = CFG['python']
FAMILIES = list(PLAN['families'])
METHODS = list(PLAN['methods'])

wildcard_constraints:
    family='(?:' + '|'.join(map(re.escape, FAMILIES)) + ')',
    method='(?:' + '|'.join(map(re.escape, METHODS)) + ')'

rule all:
    input: ROOT + '/validated.json', ROOT + '/tcs_scores.csv', ROOT + '/summary.csv'

rule preflight:
    input: plan=PLAN_PATH, driver=DRIVER, assets=list(PLAN['assets'])
    output: ROOT + '/preflight.json'
    threads: 1
    resources: mem_mb=24000
    log: ROOT + '/logs/preflight.log'
    shell: '{PYTHON:q} {input.driver:q} preflight --threads {threads} --plan {input.plan:q} > {log:q} 2>&1'

rule cohort:
    input:
        plan=PLAN_PATH, driver=DRIVER, preflight=ROOT + '/preflight.json',
        files=lambda w: list(PLAN['families'][w.family]['inputs'])
    output:
        done=ROOT + '/families/{family}/cohort/completed.json',
        aa=ROOT + '/families/{family}/cohort/aa.fasta',
        structs=directory(ROOT + '/families/{family}/cohort/structs')
    threads: 1
    resources: mem_mb=2000
    log: ROOT + '/families/{family}/logs/cohort.log'
    shell: '{PYTHON:q} {input.driver:q} prepare --plan {input.plan:q} --family {wildcards.family:q} > {log:q} 2>&1'

rule allvall:
    input:
        plan=PLAN_PATH, driver=DRIVER, done=ROOT + '/families/{family}/cohort/completed.json',
        structs=ROOT + '/families/{family}/cohort/structs', preflight=ROOT + '/preflight.json'
    output:
        done=ROOT + '/families/{family}/allvall/completed.json',
        table=ROOT + '/families/{family}/allvall/allvall.tsv'
    threads: CFG['threads']
    resources: mem_mb=24000
    log: ROOT + '/families/{family}/logs/allvall.log'
    shell: '{PYTHON:q} {input.driver:q} allvall --plan {input.plan:q} --family {wildcards.family:q} --threads {threads} > {log:q} 2>&1'

rule infer:
    input:
        plan=PLAN_PATH, driver=DRIVER, done=ROOT + '/families/{family}/cohort/completed.json',
        structs=ROOT + '/families/{family}/cohort/structs', preflight=ROOT + '/preflight.json',
        allvall=lambda w: [ROOT + '/families/' + w.family + '/allvall/completed.json',
                          ROOT + '/families/' + w.family + '/allvall/allvall.tsv'] if PLAN['methods'][w.method]['kind'] == 'FoldTree' else []
    output:
        done=ROOT + '/families/{family}/{method}/completed.json',
        tree=ROOT + '/families/{family}/{method}/tree.nwk'
    threads: CFG['threads']
    resources: mem_mb=24000
    log: ROOT + '/families/{family}/logs/{method}.log'
    shell: '{PYTHON:q} {input.driver:q} infer --plan {input.plan:q} --family {wildcards.family:q} --method {wildcards.method:q} --threads {threads} > {log:q} 2>&1'

rule score:
    input:
        plan=PLAN_PATH, driver=DRIVER, done=ROOT + '/families/{family}/{method}/completed.json',
        tree=ROOT + '/families/{family}/{method}/tree.nwk'
    output: done=ROOT + '/families/{family}/{method}/tcs/completed.json', score=ROOT + '/families/{family}/{method}/tcs/score.json'
    threads: 1
    resources: mem_mb=2000
    log: ROOT + '/families/{family}/logs/{method}.tcs.log'
    shell: '{PYTHON:q} {input.driver:q} score --plan {input.plan:q} --family {wildcards.family:q} --method {wildcards.method:q} > {log:q} 2>&1'

rule aggregate:
    input:
        plan=PLAN_PATH, driver=DRIVER,
        scores=expand(ROOT + '/families/{family}/{method}/tcs/score.json', family=FAMILIES, method=METHODS),
        done=expand(ROOT + '/families/{family}/{method}/tcs/completed.json', family=FAMILIES, method=METHODS)
    output: ROOT + '/validated.json', ROOT + '/tcs_scores.csv', ROOT + '/summary.csv'
    threads: 1
    resources: mem_mb=2000
    log: ROOT + '/logs/aggregate.log'
    shell: '{PYTHON:q} {input.driver:q} aggregate --plan {input.plan:q} > {log:q} 2>&1'
