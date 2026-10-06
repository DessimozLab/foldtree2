"""Read-only, bounded-memory summaries for the ancestral inspection notebook."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPRESENTATIONS = ['AA', '3Di', 'FT2_10', 'FT2_20', 'FT2_30', 'FT2_40', 'FT2_50']
METRICS = ['max_probability', 'normalized_entropy', 'sharpness', 'confidence_090']


def inventory(roots, family_ids):
    """Follow queue artifact pointers, including inherited pilot outputs."""
    rows, errors, seen = [], [], {}
    for root in map(Path, roots):
        if not root.exists():
            continue
        status = root / 'queue_status.json'
        jobs = json.loads(status.read_text()).get('jobs', {}) if status.exists() else {}
        for family in sorted(family_ids):
            for representation in REPRESENTATIONS:
                key = (f'AA_{family}' if representation == 'AA' else f'3Di_{family}'
                       if representation == '3Di' else f'native_{representation[4:]}_{family}')
                # Do not read artifacts currently being rewritten by an active job.
                if key in jobs and jobs[key].get('status') != 'completed':
                    continue
                # Resolve from the already loaded queue snapshot: parsing a large
                # live status file again for every family/model is needlessly slow.
                artifact = jobs.get(key, {}).get('artifacts')
                native = (Path(artifact) if isinstance(artifact, str) else
                          root / '3di' / family if representation == '3Di' else
                          root / 'native' / representation / family)
                directory = native / 'ancestral'
                marker = directory / 'completed.json'
                if not marker.exists():
                    continue
                try:
                    complete = json.loads(marker.read_text())
                    signature = json.loads((directory / 'provenance.json').read_text())
                    summary = json.loads((directory / 'summary.json').read_text())
                    if complete['signature'] != signature or signature['family'] != family:
                        raise ValueError('Completion/provenance or family mismatch')
                    expected = '3Di_foldmason' if representation == '3Di' else representation
                    if signature['strategy'] != expected:
                        raise ValueError('Representation mismatch')
                    table = directory / 'node_site_uncertainty.csv.gz'
                    if table.name not in complete['outputs'] or not table.is_file():
                        raise ValueError('Missing committed uncertainty table')
                    item = (family, representation)
                    if item in seen:
                        if seen[item] != directory.resolve():
                            raise ValueError('Conflicting completed artifacts across roots')
                        continue
                    seen[item] = directory.resolve()
                    rows.append({'family': family, 'representation': representation,
                                 'directory': str(directory), 'table': str(table),
                                 'alphabet_size': len(summary['state_order']),
                                 'n_nodes': summary['n_nodes'], 'n_sites': summary['n_sites'],
                                 'rooting': summary['rooting']})
                except (ValueError, KeyError, OSError) as error:
                    errors.append({'family': family, 'representation': representation,
                                   'directory': str(directory), 'error': str(error)})
    columns = ['family', 'representation', 'directory', 'table', 'alphabet_size',
               'n_nodes', 'n_sites', 'rooting']
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(errors)


def select_cohort(available, common=True, max_families=50):
    """Common families across *available* representations; absent models stay absent."""
    if available.empty:
        raise ValueError('No completed ancestral artifacts found. Check RESULT_ROOTS.')
    if max_families is not None and max_families < 1:
        raise ValueError('max_families must be positive or None')
    if common:
        groups = [set(group.family) for _, group in available.groupby('representation')]
        families = sorted(set.intersection(*groups))
    else:
        families = sorted(set(available.family))
    if max_families is not None:
        families = families[:max_families]
    if not families:
        raise ValueError('No common completed families yet; set COMMON_FAMILIES=False to inspect separately.')
    return available[available.family.isin(families)].copy()


def summarize_depth(selected, bins=10, chunksize=50000):
    """Pool eligible node/sites within each family/bin, never across families."""
    if bins < 1:
        raise ValueError('bins must be positive')
    columns = ['normalized_root_depth', 'normalized_tip_entropy', 'passes_occupancy',
               'max_probability', 'normalized_entropy', 'confidence_090']
    rows, coverage = [], []
    for item in selected.itertuples(index=False):
        accumulators = {False: {}, True: {}}
        total, eligible = 0, 0
        for chunk in pd.read_csv(item.table, usecols=columns, chunksize=chunksize):
            total += len(chunk)
            chunk = chunk[chunk.passes_occupancy == 1].copy()
            if chunk.empty:
                continue
            if not np.isfinite(chunk[columns[:2] + METRICS[:2]].to_numpy()).all():
                raise ValueError(f'Nonfinite eligible metrics in {item.table}')
            for name in ['normalized_root_depth', 'normalized_tip_entropy', 'max_probability', 'normalized_entropy']:
                if not chunk[name].between(-1e-6, 1 + 1e-6).all():
                    raise ValueError(f'Out-of-range {name} in {item.table}')
            eligible += len(chunk)
            chunk['depth_bin'] = np.minimum((chunk.normalized_root_depth.clip(0, 1) * bins).astype(int), bins - 1)
            chunk['entropy_bin'] = pd.cut(chunk.normalized_tip_entropy.clip(0, 1),
                                          [-1e-6, 1 / 3, 2 / 3, 1 + 1e-6], labels=['low', 'medium', 'high'])
            k = item.alphabet_size
            chunk['sharpness'] = (k * chunk.max_probability - 1) / (k - 1)
            for stratified in [False, True]:
                keys = ['depth_bin', 'entropy_bin'] if stratified else ['depth_bin']
                grouped = chunk.groupby(keys, observed=True)[METRICS].agg(['sum', 'count'])
                for key, values in grouped.iterrows():
                    key = key if isinstance(key, tuple) else (key,)
                    stored = accumulators[stratified].setdefault(key, np.zeros(len(METRICS) + 1))
                    stored[0] += values[(METRICS[0], 'count')]
                    stored[1:] += [values[(metric, 'sum')] for metric in METRICS]
        coverage.append({'family': item.family, 'representation': item.representation,
                         'total_node_sites': total, 'eligible_node_sites': eligible})
        for stratified, groups in accumulators.items():
            for key, values in groups.items():
                rows.append({'family': item.family, 'representation': item.representation,
                             'depth_bin': key[0], 'depth_midpoint': (key[0] + .5) / bins,
                             'tip_entropy_group': str(key[1]) if stratified else 'all',
                             'n_node_sites': int(values[0]),
                             **dict(zip(METRICS, values[1:] / values[0]))})
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def bootstrap_curves(family_bins, draws=1000, seed=42):
    """Resample whole families, preserving the same bootstrap draw across depth bins."""
    if family_bins.empty:
        raise ValueError('No eligible node/sites in selected families')
    if draws < 1:
        raise ValueError('draws must be positive')
    results = []
    for (representation, entropy), table in family_bins.groupby(['representation', 'tip_entropy_group']):
        families = sorted(table.family.unique())
        # Same seed/order gives paired family draws for common cohorts.
        indices = np.random.default_rng(seed).integers(0, len(families), size=(draws, len(families)))
        for metric in METRICS:
            pivot = table.pivot(index='family', columns='depth_bin', values=metric).reindex(families)
            for depth, values in pivot.items():
                a = values.to_numpy()
                sampled = a[indices]
                count = np.isfinite(sampled).sum(axis=1)
                boot = np.divide(np.nansum(sampled, axis=1), count,
                                 out=np.full(draws, np.nan), where=count > 0)
                valid = boot[np.isfinite(boot)]
                low, high = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
                midpoint = table.loc[table.depth_bin == depth, 'depth_midpoint'].iloc[0]
                results.append({'representation': representation, 'tip_entropy_group': entropy,
                                'metric': metric, 'depth_bin': depth, 'depth_midpoint': midpoint,
                                'mean': np.nanmean(a), 'lower': low, 'upper': high,
                                'n_families': int(np.isfinite(a).sum())})
    return pd.DataFrame(results)
