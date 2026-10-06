import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from ancestral_inspection import select_cohort, summarize_depth, bootstrap_curves


def test_chunking_endpoints_occupancy_and_family_weights(tmp_path):
    tables = []
    for family, repetitions, p in [('1', 100, .9), ('2', 1, .5)]:
        path = tmp_path / (family + '.csv.gz')
        rows = [{'normalized_root_depth': 1., 'normalized_tip_entropy': .8,
                 'passes_occupancy': 1, 'max_probability': p,
                 'normalized_entropy': .2, 'confidence_090': int(p >= .9)}] * repetitions
        rows.append({**rows[0], 'passes_occupancy': 0, 'max_probability': 1.})
        pd.DataFrame(rows).to_csv(path, index=False)
        tables.append({'family': family, 'representation': 'FT2_20',
                       'alphabet_size': 20, 'table': str(path)})
    bins, coverage = summarize_depth(pd.DataFrame(tables), bins=10, chunksize=7)
    assert set(bins.depth_bin) == {9}
    assert coverage.eligible_node_sites.sum() == 101
    curves = bootstrap_curves(bins, draws=50)
    row = curves[(curves.metric == 'max_probability') & (curves.tip_entropy_group == 'all')].iloc[0]
    assert np.isclose(row['mean'], .7)  # Equal-family mean, not 100:1 site weighting.
    assert row.n_families == 2
    pd.testing.assert_frame_equal(curves, bootstrap_curves(bins, draws=50))


def test_common_families_does_not_invent_missing_models():
    available = pd.DataFrame([{'family': '1', 'representation': 'AA'},
                              {'family': '2', 'representation': 'AA'},
                              {'family': '2', 'representation': 'FT2_20'}])
    selected = select_cohort(available, common=True, max_families=None)
    assert set(selected.family) == {'2'}
    assert set(selected.representation) == {'AA', 'FT2_20'}
