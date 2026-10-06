import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from benchmark_species_supermatrix import concatenate, species_labels, paired_family_comparison


def test_concatenation_pads_missing_species_and_tracks_exact_blocks():
    records, blocks = concatenate({'2': {'B_A': 'DEF'}, '1': {'A_A': 'AC', 'B_A': 'A-'}}, ['B_A', 'A_A'])
    assert records == {'A_A': 'AC---', 'B_A': 'A-DEF'}
    assert list(blocks.family) == ['1', '2']
    assert list(blocks.start) == [0, 2]
    assert list(blocks.end) == [2, 5]
    indices = np.searchsorted(blocks.end.to_numpy(), [0, 1, 2, 4], side='right')
    assert list(blocks.family.to_numpy()[indices]) == ['1', '1', '2', '2']


def test_species_copy_mapping_is_stable_and_rejects_paralogs():
    family = {'family': '1', 'taxa': [{'accession': 'P', 'label': 'P|HUMAN'}]}
    assert species_labels(family) == {'P': 'HUMAN_A'}
    family['taxa'].append({'accession': 'Q', 'label': 'Q|HUMAN'})
    with pytest.raises(ValueError, match='Multiple copies'):
        species_labels(family)


def test_bad_width_unknown_species_and_empty_species_fail():
    with pytest.raises(ValueError, match='width'):
        concatenate({'1': {'A': 'AC', 'B': 'A'}}, ['A', 'B'])
    with pytest.raises(ValueError, match='Unknown'):
        concatenate({'1': {'C': 'AC'}}, ['A'])
    with pytest.raises(ValueError, match='no observed'):
        concatenate({'1': {'A': 'AC'}}, ['A', 'B'])


def test_family_bootstrap_paired_by_family_and_reproducible():
    def table(values):
        return pd.DataFrame({'family': ['1', '2'], **{name: values for name in
            ['phylo_gain_mean', 'phylo_gain_sum', 'phylo_gain_norm_mean', 'h_tip_mean']}})
    tables = {'AA': table([2., 4.]), 'FT2_20': table([1., 3.])}
    result = paired_family_comparison(tables, draws=20)
    assert set(result.mean_difference_left_minus_right) == {1.}
    assert set(result.lower_095) == {1.}
    assert set(result.n_common_families) == {2}
    pd.testing.assert_frame_equal(result, paired_family_comparison(tables, draws=20))
