"""Benchmarking Snakemake workflow for foldtree2 alphabets.

Tests each alphabet (AA, 3Di, FT2_*) across three benchmark dimensions:
  1. Information theoretic descriptors
  2. Ancestral character distributions
  3. Phylogenetic gain (controlled + native)

Input families come from the frozen OMA benchmark cohort
(configs/oma_benchmark_cohort.json). TCS is excluded per design.

Usage::
  conda activate foldtree2

  # Full benchmark
  snakemake --snakefile workflow/benchmark.smk          \\
            --configfile workflow/config/benchmark.yaml \\
            --cores 8

  # Individual stage e.g. ancestral only
  snakemake ... --config stages="['ancestral']"

  # Inspect DAG
  snakemake -n --snakefile workflow/benchmark.smk       \\
            --configfile workflow/config/benchmark.yaml
"""

import json
import re as _re
import sys as _sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(config['root']).expanduser().resolve()
OUTDIR = Path(config['output_dir']).expanduser().resolve()
PYTHON = config.get('python', 'python3')
THREADS = int(config.get('threads', 8))
RAXML_THREADS = int(config.get('raxml_threads', THREADS))
FAMILIES_DIR = (ROOT / config.get('families_dir',
                   'families/Information_benchmark/marker_genes')).resolve()
COHORT_PATH = config.get('cohort_path', 'configs/oma_benchmark_cohort.json')
MAX_FAMILIES = config.get('max_families', None)
TARGET_ALPHABETS = list(config.get('alphabets',
                        ['AA', '3Di', 'FT2_20', 'FT2_30', 'FT2_40']))
STAGES = list(config.get('stages',
               ['information', 'controlled-gain', 'native-gain', 'ancestral']))
ANALYSES = list(config.get('analyses',
                 ['entropy', 'mdl', 'mi', 'position', 'kmer']))
GAP_OCC_MAX = float(config.get('gap_occ_max', 0.3))
SEED = int(config.get('seed', 42))

# Derived
FT2_SIZES = sorted({int(a.removeprefix('FT2_'))
                     for a in TARGET_ALPHABETS if a.startswith('FT2_')},
                    reverse=True)
HAS_FT2 = len(FT2_SIZES) > 0

# ---------------------------------------------------------------------------
# Load cohort (parse-time; Snakemake runs on Python)
# ---------------------------------------------------------------------------
_sys.path.insert(0, str(ROOT))
from scripts.benchmarks.benchmark_cohort import load as _load_cohort  # noqa:E402

_cohort_path = (ROOT / COHORT_PATH).resolve()
_cohort = _load_cohort(str(_cohort_path))
ALL_FAMILIES = sorted(f['family'] for f in _cohort['families'])
if MAX_FAMILIES:
    ALL_FAMILIES = ALL_FAMILIES[:int(MAX_FAMILIES)]

FAMILY_MAP = {f['family']: f for f in _cohort['families']}

# Production model entries
_models_cfg = json.loads((ROOT / 'configs/production_models.yaml').read_text())
ENTRIES = {e['size']: e for e in _models_cfg['models']}

# ---------------------------------------------------------------------------
# Wildcard constraints
# ---------------------------------------------------------------------------
_fam_pat = '|'.join(map(_re.escape, ALL_FAMILIES))
wildcard_constraints:
    family = f'(?:{_fam_pat})'

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _script(rel, *args):
    """Build a shell command for a project script (list form)."""
    parts = [str(PYTHON), '-u', str(ROOT / rel)]
    parts.extend(str(a) for a in args)
    return ' '.join(parts)

# ---------------------------------------------------------------------------
# TARGET FILES per stage
# ---------------------------------------------------------------------------
_INFO_DIR = OUTDIR / 'information'
_INFORMATION_DONE = _INFO_DIR / 'completed.json'

_CONTROLLED_TARGETS = [
    OUTDIR / 'controlled_gain' / a / f / 'site_gain.csv'
    for a in TARGET_ALPHABETS for f in ALL_FAMILIES]
_NATIVE_TARGETS = [
    OUTDIR / 'native_gain' / a / f / 'site_gain.csv'
    for a in TARGET_ALPHABETS for f in ALL_FAMILIES]
_ANCESTRAL_TARGETS = [
    OUTDIR / 'ancestral' / a / f / 'site_uncertainty.csv'
    for a in TARGET_ALPHABETS for f in ALL_FAMILIES]
_JOINT_TARGETS = [OUTDIR / 'joint_comparison' / 'provenance.json']

# ---------------------------------------------------------------------------
# all
# ---------------------------------------------------------------------------
rule all:
    input:
        OUTDIR / '.env_validated',
        *([_INFORMATION_DONE] if 'information' in STAGES and HAS_FT2 else []),
        *(_CONTROLLED_TARGETS if 'controlled-gain' in STAGES else []),
        *(_NATIVE_TARGETS if 'native-gain' in STAGES else []),
        *(_ANCESTRAL_TARGETS if 'ancestral' in STAGES else []),
        *(_JOINT_TARGETS
          if ('controlled-gain' in STAGES or 'native-gain' in STAGES) and HAS_FT2
          else []),
    default_target: True

# ---------------------------------------------------------------------------
# validate_environment
# ---------------------------------------------------------------------------
rule validate_environment:
    """Verify cohort, production models, and record tool versions."""
    output: str(OUTDIR / '.env_validated')
    run:
        from scripts.benchmarks.benchmark_cohort import load as _v
        _v(str(_cohort_path))
        for sz in FT2_SIZES:
            from scripts.alphabets.prepare_production_alphabets import validate_bundle
            validate_bundle(ENTRIES[sz], require_convergence=True)
        from scripts.benchmark_common import tool_version as _tv
        vers = {'raxml-ng': _tv('raxml-ng'), 'foldseek': _tv('foldseek'),
                'mafft': _tv('mafft')}
        Path(str(self.output)).write_text(json.dumps(vers, indent=2) + '\n')
        OUTDIR.mkdir(parents=True, exist_ok=True)
        (OUTDIR / 'logs').mkdir(parents=True, exist_ok=True)

# ======================================================================
# PHASE 1 — FT2 ENCODING + ALIGNMENT (per size, per family)
# ======================================================================
# Uses production_alphabet_experiments.py --family for one (size, family).
# This produces:
#   - encoded.hex            — raw encoder tokens
#   - ft2.aligned.fasta      — MAFFT-aligned FT2 sequences
#   - aa.aligned.fasta       — corresponding AA co-alignment
#   - results.json           — description metrics + fixed-tree phylo gain
# ======================================================================
if HAS_FT2:
    rule encode_ft2_family:
        """Encode structures and align FT2 tokens for one (size, family)."""
        input:
            str(OUTDIR / '.env_validated'),
        output:
            alignment=str(OUTDIR / 'ft2_production' / '{size}' /
                          '{family}' / 'ft2.aligned.fasta'),
            results=str(OUTDIR / 'ft2_production' / '{size}' /
                        '{family}' / 'results.json'),
        params:
            size=lambda w: int(w.size),
            outdir=lambda w: OUTDIR / 'ft2_production' / w.size,
        threads: THREADS
        log: str(OUTDIR / 'logs' / 'ft2_encode_{size}_{family}.log')
        shell:
            _script(
                'scripts/alphabets/production_alphabet_experiments.py',
                '--size', '{params.size}',
                '--family', '{wildcards.family}',
                '--families', FAMILIES_DIR,
                '--outdir', '{params.outdir}',
                '--device', 'cpu',
                '--threads', '{threads}',
                '--cohort', _cohort_path,
                '>{log} 2>&1',
            )

# ======================================================================
# PHASE 2 — 3Di BASELINE (Foldmason + ML tree)
# ======================================================================
if '3Di' in TARGET_ALPHABETS:
    rule three_di_baseline:
        """Foldmason on family PDBs → 3Di+AA co-alignment + ML tree."""
        input:
            str(OUTDIR / '.env_validated'),
            structs=lambda w: sorted(
                (FAMILIES_DIR / w.family / 'structs').glob('*.pdb')),
        output:
            three_done=str(OUTDIR / 'baselines' / '3Di' / '{family}' /
                           'completed.json'),
            alignment=str(OUTDIR / 'baselines' / '3Di' / '{family}' /
                          '3di.aligned.fasta'),
            aa_alignment=str(OUTDIR / 'baselines' / '3Di' / '{family}' /
                             'aa.aligned.fasta'),
            tree=str(OUTDIR / 'baselines' / '3Di' / '{family}' /
                     'ml.raxml.bestTree'),
            model=str(OUTDIR / 'baselines' / '3Di' / '{family}' /
                      'ml.raxml.bestModel'),
            fitted_log=str(OUTDIR / 'baselines' / '3Di' / '{family}' /
                           'ml.raxml.log'),
        params:
            outdir=lambda w: OUTDIR / 'baselines' / '3Di' / w.family,
        threads: THREADS
        log: str(OUTDIR / 'logs' / '3Di_{family}.log')
        run:
            from scripts.benchmarks.three_di_ml import three_di
            od = Path(str(self.params.outdir))
            od.mkdir(parents=True, exist_ok=True)
            three_di([str(p) for p in input.structs], od,
                     route='foldmason', threads=threads, seed=SEED)
            (od / 'completed.json').write_text(
                json.dumps({'status': 'completed'}) + '\n')

# ======================================================================
# PHASE 3 — AA NATIVE ML TREE
# ======================================================================
if 'AA' in TARGET_ALPHABETS:
    rule aa_baseline_ml:
        """RAxML-ng (LG+G+I) for AA alignment → native ML tree."""
        input:
            str(OUTDIR / '.env_validated'),
            alignment=FAMILIES_DIR / '{family}' / 'sequences.aligned.fa',
        output:
            tree=str(OUTDIR / 'baselines' / 'AA' / '{family}' /
                     'ml.raxml.bestTree'),
            model=str(OUTDIR / 'baselines' / 'AA' / '{family}' /
                      'ml.raxml.bestModel'),
            fitted_log=str(OUTDIR / 'baselines' / 'AA' / '{family}' /
                           'ml.raxml.log'),
        params:
            outdir=lambda w: OUTDIR / 'baselines' / 'AA' / w.family,
        threads: RAXML_THREADS
        log: str(OUTDIR / 'logs' / 'AA_{family}.log')
        run:
            from scripts.benchmark_common import begin_stage, complete_stage, \
                digest, infer_ml, tool_version
            aln = Path(str(input.alignment))
            dest = Path(str(self.params.outdir))
            dest.mkdir(parents=True, exist_ok=True)
            sig = {'alignment_sha256': digest(aln), 'model': 'LG+G+I',
                   'threads': RAXML_THREADS, 'seed': SEED, 'starts_each': 10,
                   'raxml': tool_version('raxml-ng')}
            if begin_stage(dest, sig):
                tf, mf = infer_ml(aln, 'LG+G+I', dest / 'ml', RAXML_THREADS)
                complete_stage(dest, sig, [tf, mf, dest / 'ml.raxml.log'],
                               execution_threads=RAXML_THREADS)

# ======================================================================
# PHASE 4 — FT2 NATIVE ML TREE
# ======================================================================
if HAS_FT2:
    rule ft2_native_ml:
        """RAxML-ng on FT2 alignment with learned substitution matrix."""
        input:
            alignment=str(OUTDIR / 'ft2_production' / '{size}' /
                          '{family}' / 'ft2.aligned.fasta'),
        output:
            tree=str(OUTDIR / 'native_ml' / 'FT2_{size}' / '{family}' /
                     'ml.raxml.bestTree'),
            model=str(OUTDIR / 'native_ml' / 'FT2_{size}' / '{family}' /
                      'ml.raxml.bestModel'),
            fitted_log=str(OUTDIR / 'native_ml' / 'FT2_{size}' / '{family}' /
                           'ml.raxml.log'),
        params:
            size=lambda w: int(w.size),
            outdir=lambda w: OUTDIR / 'native_ml' / f'FT2_{w.size}' / w.family,
        threads: RAXML_THREADS
        log: str(OUTDIR / 'logs' / 'ft2_native_{size}_{family}.log')
        run:
            from scripts.benchmark_common import begin_stage, complete_stage, \
                digest, infer_ml, tool_version
            sz = int(self.params.size)
            entry = ENTRIES[sz]
            submat = ROOT / entry['directory'] / entry['raxml']
            model_str = 'MULTI{sz}_GTR{{{sub}}}+I'.format(sz=sz, sub=submat)
            aln = Path(str(input.alignment))
            dest = Path(str(self.params.outdir))
            dest.mkdir(parents=True, exist_ok=True)
            sig = {'alignment_sha256': digest(aln), 'model': model_str,
                   'matrix_sha256': digest(submat),
                   'threads': RAXML_THREADS, 'seed': SEED, 'starts_each': 10,
                   'raxml': tool_version('raxml-ng')}
            if begin_stage(dest, sig):
                tf, mf = infer_ml(aln, model_str, dest / 'ml', RAXML_THREADS)
                complete_stage(dest, sig, [tf, mf, dest / 'ml.raxml.log'],
                               execution_threads=RAXML_THREADS)

# ======================================================================
# RESOLVERS — per-alphabet input files
# ======================================================================

def _alignment(w):
    a = w.alphabet
    if a == 'AA':
        return str(FAMILIES_DIR / w.family / 'sequences.aligned.fa')
    if a == '3Di':
        return str(OUTDIR / 'baselines' / '3Di' / w.family / '3di.aligned.fasta')
    if a.startswith('FT2_'):
        return str(OUTDIR / 'ft2_production' / a.removeprefix('FT2_') /
                   w.family / 'ft2.aligned.fasta')
    raise ValueError('Unknown alphabet: ' + a)

def _native_tree(w):
    a = w.alphabet
    if a == 'AA':
        return str(OUTDIR / 'baselines' / 'AA' / w.family / 'ml.raxml.bestTree')
    if a == '3Di':
        return str(OUTDIR / 'baselines' / '3Di' / w.family / 'ml.raxml.bestTree')
    if a.startswith('FT2_'):
        return str(OUTDIR / 'native_ml' / a / w.family / 'ml.raxml.bestTree')
    raise ValueError('Unknown alphabet: ' + a)

def _native_model(w):
    a = w.alphabet
    if a == 'AA':
        return str(OUTDIR / 'baselines' / 'AA' / w.family / 'ml.raxml.bestModel')
    if a == '3Di':
        return str(OUTDIR / 'baselines' / '3Di' / w.family / 'ml.raxml.bestModel')
    if a.startswith('FT2_'):
        return str(OUTDIR / 'native_ml' / a / w.family / 'ml.raxml.bestModel')
    raise ValueError('Unknown alphabet: ' + a)

def _native_log(w):
    a = w.alphabet
    if a == 'AA':
        return str(OUTDIR / 'baselines' / 'AA' / w.family / 'ml.raxml.log')
    if a == '3Di':
        return str(OUTDIR / 'baselines' / '3Di' / w.family / 'ml.raxml.log')
    if a.startswith('FT2_'):
        return str(OUTDIR / 'native_ml' / a / w.family / 'ml.raxml.log')
    raise ValueError('Unknown alphabet: ' + a)

def _model_str(w, native=False):
    if not native and w.alphabet in ('AA', '3Di'):
        return 'LG+G+I'
    if w.alphabet == 'AA':
        return 'LG+G+I'
    if w.alphabet == '3Di':
        mat = ROOT / 'foldtree2/config/3diphy_submats/Q.3Di.AF'
        return 'PROTGTR{{{0}}}+G+I'.format(mat)
    if w.alphabet.startswith('FT2_'):
        sz = int(w.alphabet.removeprefix('FT2_'))
        entry = ENTRIES[sz]
        submat = ROOT / entry['directory'] / entry['raxml']
        return 'MULTI{sz}_GTR{{{sub}}}+I'.format(sz=sz, sub=submat)
    raise ValueError('Unknown alphabet: ' + w.alphabet)

def _states(w):
    if w.alphabet == 'AA':
        return set('ACDEFGHIKLMNPQRSTVWY')
    if w.alphabet == '3Di':
        return set('ARNDCQEGHILKMFPSTWYV')
    if w.alphabet.startswith('FT2_'):
        from scripts.benchmark_common import MULTI_ORDER
        sz = int(w.alphabet.removeprefix('FT2_'))
        return set(MULTI_ORDER[:sz])
    raise ValueError('Unknown alphabet: ' + w.alphabet)

# ======================================================================
# PHASE 5 — PHYLOGENETIC GAIN (controlled)
# ======================================================================
if 'controlled-gain' in STAGES:
    rule phylo_gain_controlled:
        """Phylogenetic gain on AA reference topology."""
        input:
            alignment=_alignment,
            tree=lambda w: str(FAMILIES_DIR / w.family /
                               'raxml_lg_tree.raxml.bestTree'),
        output:
            csv=str(OUTDIR / 'controlled_gain' / '{alphabet}' / '{family}' /
                    'site_gain.csv'),
        params:
            model=_model_str,
            outdir=lambda w: OUTDIR / 'controlled_gain' / w.alphabet / w.family,
        threads: THREADS
        log: str(OUTDIR / 'logs' / 'controlled_gain_{alphabet}_{family}.log')
        run:
            from scripts.benchmarks.phylogenetic_information_gain import (
                DatasetSpec, analyze_alphabet_phylogenetic_info,
                parse_raxml_sitelh, run_raxml_sitelh)
            aln = Path(str(input.alignment))
            tree = Path(str(input.tree))
            model = str(self.params.model)
            st = _states(w)
            dest = Path(str(self.params.outdir))
            dest.mkdir(parents=True, exist_ok=True)
            sitelh = run_raxml_sitelh(aln, tree, model, dest / 'sitelh',
                                      'raxml-ng', threads)
            spec = DatasetSpec(w.alphabet, w.alphabet, aln, tree,
                               model, sitelh, None)
            df = analyze_alphabet_phylogenetic_info(spec,
                parse_raxml_sitelh(sitelh), st, '-', GAP_OCC_MAX)
            if df.empty:
                raise ValueError('No eligible gain columns: {}/{}'.format(
                    w.alphabet, w.family))
            df.to_csv(str(self.output.csv), index=False)

# ======================================================================
# PHASE 6 — PHYLOGENETIC GAIN (native)
# ======================================================================
if 'native-gain' in STAGES:
    rule phylo_gain_native:
        """Phylogenetic gain on each alphabet's own ML tree."""
        input:
            alignment=_alignment,
            tree=_native_tree,
        output:
            csv=str(OUTDIR / 'native_gain' / '{alphabet}' / '{family}' /
                    'site_gain.csv'),
        params:
            model=_model_str,
            outdir=lambda w: OUTDIR / 'native_gain' / w.alphabet / w.family,
        threads: THREADS
        log: str(OUTDIR / 'logs' / 'native_gain_{alphabet}_{family}.log')
        run:
            from scripts.benchmarks.phylogenetic_information_gain import (
                DatasetSpec, analyze_alphabet_phylogenetic_info,
                parse_raxml_sitelh, run_raxml_sitelh)
            aln = Path(str(input.alignment))
            tree = Path(str(input.tree))
            model = str(self.params.model)
            st = _states(w)
            dest = Path(str(self.params.outdir))
            dest.mkdir(parents=True, exist_ok=True)
            sitelh = run_raxml_sitelh(aln, tree, model, dest / 'sitelh',
                                      'raxml-ng', threads)
            spec = DatasetSpec(w.alphabet, w.alphabet, aln, tree,
                               model, sitelh, None)
            df = analyze_alphabet_phylogenetic_info(spec,
                parse_raxml_sitelh(sitelh), st, '-', GAP_OCC_MAX)
            if df.empty:
                raise ValueError('No eligible gain columns: {}/{}'.format(
                    w.alphabet, w.family))
            df.to_csv(str(self.output.csv), index=False)

# ======================================================================
# PHASE 7 — ANCESTRAL RECONSTRUCTION + UNCERTAINTY
# ======================================================================
if 'ancestral' in STAGES:
    rule ancestral_reconstruction:
        """Ancestral state reconstruction + per-site posterior uncertainty."""
        input:
            alignment=_alignment,
            tree=_native_tree,
            model_file=_native_model,
            fitted_log=_native_log,
        output:
            uncertainty=str(OUTDIR / 'ancestral' / '{alphabet}' / '{family}' /
                            'site_uncertainty.csv'),
        params:
            states=lambda w: list(_states(w)),
            outdir=lambda w: OUTDIR / 'ancestral' / w.alphabet / w.family,
        threads: THREADS
        log: str(OUTDIR / 'logs' / 'ancestral_{alphabet}_{family}.log')
        run:
            from scripts.ancestral.ancestral_state_uncertainty import (
                reconstruct, posterior_rows,
                uncertainty as _site_uncertainty)
            from scripts.benchmark_common import stationary_from_log
            import pandas as pd
            aln = Path(str(input.alignment))
            tree = Path(str(input.tree))
            model = Path(str(input.model_file))
            flog = Path(str(input.fitted_log))
            states = list(self.params.states)
            dest = Path(str(self.params.outdir))
            dest.mkdir(parents=True, exist_ok=True)
            reconstruct(str(aln), str(tree), str(model), str(flog),
                        ''.join(states), str(dest),
                        family=str(w.family), strategy=str(w.alphabet),
                        threads=THREADS)
            prior = stationary_from_log(str(flog), len(states))
            gz_path = dest / 'node_site_uncertainty.csv.gz'
            if gz_path.exists():
                df = pd.read_csv(gz_path, compression='gzip')
                site = df.groupby('site').agg({
                    'entropy_bits': 'mean',
                    'normalized_entropy': 'mean',
                    'effective_states': 'mean',
                    'kl_from_stationary_bits': 'mean',
                    'max_probability': 'mean',
                    'confidence_090': 'sum',
                    'family': 'first',
                    'alphabet': 'first',
                }).reset_index()
                site.to_csv(str(self.output.uncertainty), index=False)
            else:
                Path(str(self.output.uncertainty)).write_text(
                    'site,entropy_bits,normalized_entropy,effective_states,'
                    'kl_from_stationary_bits,max_probability,'
                    'confidence_090,family,alphabet\n')

# ======================================================================
# PHASE 8 — INFORMATION DESCRIPTORS AGGREGATE (batch per size)
# ======================================================================
if 'information' in STAGES and HAS_FT2:
    rule ft2_information_descriptors:
        """Aggregate information descriptor CSVs per FT2 size
        (entropy, Markov, MDL, MI, k-mer, positional entropy).
        Depends on all families having completed encoding for that size."""
        input:
            # Depend on per-family result files to trigger batch aggregation
            results=[str(OUTDIR / 'ft2_production' / str(s) / f / 'results.json')
                     for s in FT2_SIZES for f in ALL_FAMILIES],
        output:
            completed=str(OUTDIR / 'ft2_production' / '{size}' /
                          'completed.json'),
        params:
            size=lambda w: int(w.size),
            outdir=lambda w: OUTDIR / 'ft2_production' / w.size,
        threads: THREADS
        log: str(OUTDIR / 'logs' / 'ft2_descriptors_{size}.log')
        shell:
            _script(
                'scripts/alphabets/production_alphabet_experiments.py',
                '--size', '{params.size}',
                '--families', FAMILIES_DIR,
                '--outdir', '{params.outdir}',
                '--device', 'cpu',
                '--threads', '{threads}',
                '--cohort', _cohort_path,
                '>{log} 2>&1',
            )

    rule information_aggregate_all:
        """Cross-size information descriptor aggregation via
        alphabet_information_benchmark.py. Reads completed FT2
        production directories and 3Di/AA natives."""
        input:
            # Completed per-size descriptor batch
            completed=[str(OUTDIR / 'ft2_production' / str(s) / 'completed.json')
                       for s in FT2_SIZES],
            # 3Di baselines (for AA/3Di comparisons)
            three_di=[str(OUTDIR / 'baselines' / '3Di' / f / 'completed.json')
                      for f in ALL_FAMILIES] if '3Di' in TARGET_ALPHABETS else [],
        output:
            str(_INFORMATION_DONE),
        params:
            dirs=' '.join(str(OUTDIR / 'ft2_production' / str(s))
                          for s in FT2_SIZES),
            native_root=str(OUTDIR / 'baselines'),
        threads: THREADS
        log: str(OUTDIR / 'logs' / 'information_aggregate.log')
        shell:
            _script(
                'scripts/benchmarks/alphabet_information_benchmark.py',
                '--experiment-dirs', '{params.dirs}',
                '--outdir', _INFO_DIR,
                '--analyses', *ANALYSES,
                '--orders', '0', '1', '2', '3',
                '--folds', '5',
                '--seed', str(SEED),
                '--native-root', '{params.native_root}',
                '--permutation-controls',
                '>{log} 2>&1',
            )

# ======================================================================
# PHASE 9 — JOINT OMA COMPARISON
# ======================================================================
if HAS_FT2 and ('controlled-gain' in STAGES or 'native-gain' in STAGES):
    rule joint_oma_comparison:
        """Unified AA/3Di/FT2 comparison (controlled + native gain)."""
        input:
            *(_CONTROLLED_TARGETS if 'controlled-gain' in STAGES else []),
            *(_NATIVE_TARGETS if 'native-gain' in STAGES else []),
            *([_INFORMATION_DONE] if 'information' in STAGES else []),
        output:
            str(_JOINT_TARGETS[0]),
        params:
            sizes=' '.join(str(s) for s in FT2_SIZES),
        threads: THREADS
        log: str(OUTDIR / 'logs' / 'joint_oma_comparison.log')
        shell:
            _script(
                'scripts/benchmarks/compare_oma_representations.py',
                '--cohort', str(_cohort_path),
                '--experiment-root', str(OUTDIR / 'ft2_production'),
                '--native-root', str(OUTDIR / 'baselines'),
                '--sizes', '{params.sizes}',
                '--outdir', str(OUTDIR / 'joint_comparison'),
                '--stages',
                *(['controlled-gain'] if 'controlled-gain' in STAGES else []),
                *(['native-gain'] if 'native-gain' in STAGES else []),
                *(['information'] if 'information' in STAGES else []),
                '--threads', '{threads}',
                '>{log} 2>&1',
            )