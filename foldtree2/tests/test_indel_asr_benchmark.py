"""Pairing, substitution-order and gap accounting for the indel ASR benchmark."""
import os
import shutil

import numpy as np
import pytest

from scripts.benchmark_common import read_paml
from scripts.benchmarks.compare_indel_asr import compare_family, fitted_substitution, node_keys, profile


def test_fitted_model_export_preserves_exchangeabilities(tmp_path):
    model = tmp_path / "model"
    model.write_text("MULTI4_GTR{1/2/3/4/5/6}+FU{0.1/0.2/0.3/0.4}+IU{0.2}, noname = 1-4")
    text, constant, prior = fitted_substitution(model, 4)
    matrix = tmp_path / "matrix"
    matrix.write_text(text)
    rates, pi, _ = read_paml(matrix, 4)
    np.testing.assert_array_equal(rates[np.triu_indices(4, 1)], [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(pi, prior)
    assert "+IU" not in constant
    assert constant.endswith(", noname = 1-4")
    with pytest.raises(ValueError, match="Expected one"):
        fitted_substitution(model, 5)


def test_node_pairing_ignores_labels_and_root_position(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    first.write_text("(A:1,B:1,(C:1,D:1)X:1)Y;")
    second.write_text("((A:1,B:1)OTHER:0.5,(C:1,D:1)ANOTHER:0.5)ROOT;")
    keys = node_keys(first, list("ABCD"))
    other = node_keys(second, list("ABCD"))
    assert keys["X"] == other["ANOTHER"]
    assert keys["Y"] == other["OTHER"]
    assert "ROOT" not in other
    second.write_text("(A:1,B:1,(C:1,E:1)X:1)Y;")
    with pytest.raises(ValueError, match="taxa differ"):
        node_keys(second, list("ABCD"))


def test_gap_profiles_are_missing_uncertainty_not_confidence(tmp_path):
    path = tmp_path / "profile"
    path.write_text("preamble\nSite\tNode\t0\t1\t-\n0\tN\t0.25\t0.75\t0\t\n1\tN\t0\t0\t0\t\n")
    result = profile(path, list("01"), [0.5, 0.5], {"N": "1-"}, {"N": "key"}, 2, arpip=True)
    assert result["key"][0, 1] == 0.75
    assert np.isnan(result["key"][1]).all()
    with pytest.raises(ValueError, match="unnormalized"):
        profile(path, list("01"), [0.5, 0.5], {"N": "11"}, {"N": "key"}, 2, arpip=True)
    with pytest.raises(ValueError, match="Incomplete"):
        profile(path, list("01"), [0.5, 0.5], {"N": "1--"}, {"N": "key"}, 3, arpip=True)


ARPIP = os.environ.get("ARPIP_BIN") or shutil.which("ARPIP")


@pytest.mark.skipif(not ARPIP or not shutil.which("mad") or not shutil.which("raxml-ng"),
                    reason="Real ARPIP, RAxML and MAD required")
def test_paired_benchmark_real_engines_and_cache(tmp_path):
    alignment, tree, model = [tmp_path / name for name in ("alignment.fa", "tree.nwk", "model")]
    alignment.write_text(">A\n01230123\n>B\n01-30123\n>C\n01230123\n>D\n-1230123\n")
    tree.write_text("(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);")
    model.write_text("MULTI4_GTR{1/2/3/4/5/6}+FU{0.1/0.2/0.3/0.4}+IU{0.1}, noname = 1-8")
    out = tmp_path / "output"
    summary = compare_family(alignment, tree, model, list("0123"), out, arpip=ARPIP, threads=1)
    assert summary["n_shared_nodes"] == 2
    assert summary["n_node_sites"] == 16
    assert summary["arpip_gap_sites"] + summary["present_sites"] == 16
    assert (out / "paired_node_sites.csv.gz").is_file()
    assert compare_family(alignment, tree, model, list("0123"), out, arpip=ARPIP, threads=1) == summary
    (out / "paired_node_sites.csv.gz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Completed output changed"):
        compare_family(alignment, tree, model, list("0123"), out, arpip=ARPIP, threads=1)
