"""Pairing, substitution-order and gap accounting for the indel ASR benchmark."""
import os
import shutil
import csv
import gzip

import numpy as np
import pytest

from scripts.benchmark_common import AA_ORDER, complete_stage, read_paml, write_json
from scripts.benchmarks.compare_indel_asr import compare_family, fitted_substitution, node_keys, profile
from scripts.benchmarks.join_indel_probabilities import join_probabilities
from scripts.ancestral import ancestral_state_uncertainty as asr
from scripts.benchmarks.compare_indel_asr import arpip_rows


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


def test_fitted_protein_model_export_and_rate_controls(tmp_path):
    source = tmp_path / "model"
    matrix = tmp_path / "matrix"
    expected_rates, expected_prior, _ = read_paml(
        "foldtree2/config/3diphy_submats/Q.3Di.AF", 20)
    rates = expected_rates[np.triu_indices(20, 1)]
    source.write_text("PROTGTR{" + "/".join(map(str, rates)) + "}+FU{"
                      + "/".join(map(str, expected_prior)) + "}+IU{0.1}+G4m{0.5}, noname = 1-4")
    text, constant, prior = fitted_substitution(source, 20, protein="3Di")
    matrix.write_text(text)
    actual_rates, actual_prior, _ = read_paml(matrix, 20)
    np.testing.assert_allclose(actual_rates, expected_rates)
    np.testing.assert_allclose(actual_prior, expected_prior)
    np.testing.assert_allclose(prior, expected_prior)
    assert "+IU" not in constant and "+G4m" not in constant
    source.write_text("LG+IU{0.1}+G4m{1.5}, noname = 1-4")
    text, constant, _ = fitted_substitution(source, 20, protein="AA", lg_matrix=matrix)
    assert text == matrix.read_text()
    assert constant == "LG, noname = 1-4"
    with pytest.raises(ValueError, match="LG protein"):
        fitted_substitution(source, 20, protein="AA")


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
    path.write_text("# ARPIP profile=conditional_marginal\nSite\tNode\t0\t1\t-\n"
                    "0\tN\t0.25\t0.75\t0\t\n1\tN\t0\t0\t1\t\n")
    result = profile(path, list("01"), [0.5, 0.5], {"N": "1-"}, {"N": "key"}, 2, arpip=True)
    assert result["key"][0, 1] == 0.75
    assert np.isnan(result["key"][1]).all()
    with pytest.raises(ValueError, match="posterior presence"):
        profile(path, list("01"), [0.5, 0.5], {"N": "11"}, {"N": "key"}, 2, arpip=True)
    with pytest.raises(ValueError, match="Incomplete"):
        profile(path, list("01"), [0.5, 0.5], {"N": "1--"}, {"N": "key"}, 3, arpip=True)
    path.write_text("# ARPIP profile=normalized\nSite\tNode\t0\t1\t-\n0\tN\t0.25\t0.75\t0\n")
    with pytest.raises(ValueError, match="not a conditional marginal"):
        profile(path, list("01"), [0.5, 0.5], {"N": "1"}, {"N": "key"}, 1, arpip=True)


def test_tied_mad_roots_are_not_combined_into_one_tree(tmp_path, monkeypatch):
    source = tmp_path / "input.nwk"
    source.write_text("(A:1,B:1,(C:1,D:1)X:1)Y;")
    candidates = ("((A:1,B:1)Y:0.5,(C:1,D:1)X:0.5);\n"
                  "(A:0.5,(B:1,(C:1,D:1)X:1)Y:0.5);\n")
    def fake_mad(command, log):
        from pathlib import Path
        Path(str(command[-1]) + ".rooted").write_text(candidates)
    monkeypatch.setattr(asr, "run", fake_mad)
    with pytest.raises(ValueError, match="2 tied roots"):
        asr.rooted_nodes(source, tmp_path, None)
    tree, _, rooting = asr.rooted_nodes(source, tmp_path, None, root_ties="first")
    assert len(tree.get_leaf_names()) == 4
    assert rooting == "MAD_inferred_first_of_2_tied_roots"


def test_join_character_probabilities_pairs_nodes_and_preserves_gaps(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    files = {
        "pip.arpip.tree": "((A:1,B:1)OTHER:0.5,(C:1,D:1)ANOTHER:0.5)ROOT;",
        "pip.arpip.ancestral.fasta": ">A\n00\n>B\n00\n>C\n00\n>D\n00\n"
                                     ">OTHER\n0-\n>ANOTHER\n10\n>ROOT\n00\n",
        "pip.arpip.probs": "# ARPIP profile=conditional_marginal\nSite\tNode\t0\t1\t-\n"
                           "0\tOTHER\t0.8\t0.2\t0\n1\tOTHER\t0\t0\t1\n"
                           "0\tANOTHER\t0.3\t0.7\t0\n1\tANOTHER\t0.6\t0.4\t0\n"
                           "0\tROOT\t0.5\t0.5\t0\n1\tROOT\t0.5\t0.5\t0\n",
        "submat.txt": "1\n0.5 0.5\n",
    }
    for engine in ("raxml", "raxml_constant"):
        files[f"{engine}.raxml.ancestralTree"] = "(A:1,B:1,(C:1,D:1)X:1)Y;"
        files[f"{engine}.raxml.ancestralStates.fasta"] = ">X\n10\n>Y\n00\n"
        files[f"{engine}.raxml.ancestralProbs"] = (
            "Node\tSite\tState\tp_0\tp_1\n"
            "X\t1\t1\t0.1\t0.9\nX\t2\t0\t0.7\t0.3\n"
            "Y\t1\t0\t0.55\t0.45\nY\t2\t0\t0.65\t0.35\n")
    for name, contents in files.items():
        (source / name).write_text(contents)
    signature = {"protocol": "paired_native_ft2_indel_asr_v2_conditional_marginal",
                 "states": ["0", "1"]}
    write_json(source / "provenance.json", signature)
    complete_stage(source, signature, [source / name for name in files])

    output = join_probabilities(source, tmp_path / "joined")
    with gzip.open(output, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 2 * 2
    keys = node_keys(source / "raxml.raxml.ancestralTree", list("ABCD"))
    indexed = {(r["node_id"], int(r["site"]), r["character"]): r for r in rows}
    present = indexed[keys["X"], 0, "1"]
    assert (present["arpip_probability"], present["raxml_probability"],
            present["raxml_constant_probability"]) == ("0.7", "0.9", "0.9")
    gap = indexed[keys["Y"], 1, "0"]
    assert (gap["arpip_gap"], gap["arpip_probability"], gap["raxml_probability"]) == ("1", "", "0.65")
    assert join_probabilities(source, tmp_path / "joined") == output
    output.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Completed output changed"):
        join_probabilities(source, tmp_path / "joined")
    (source / "pip.arpip.probs").write_text("corrupted")
    with pytest.raises(ValueError, match="output changed"):
        join_probabilities(source, tmp_path / "other")
    with pytest.raises(ValueError, match="separate directories"):
        join_probabilities(source, source)


ARPIP = os.environ.get("ARPIP_BIN") or shutil.which("ARPIP")
LG_MATRIX = os.environ.get("ARPIP_LG_MATRIX")


@pytest.mark.skipif(not ARPIP or not LG_MATRIX or not shutil.which("mad") or not shutil.which("raxml-ng"),
                    reason="Real engines and exported LG matrix required")
def test_lg_conditional_marginals_match_raxml_without_gaps(tmp_path):
    alignment, tree, model = [tmp_path / name for name in ("alignment.fa", "tree.nwk", "model")]
    alignment.write_text(">A description\nARND\n>B description\nARNE\n>C description\nCRND\n>D description\nCRNE\n")
    tree.write_text("(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);")
    model.write_text("LG+IU{0.1}+G4m{1.5}, noname = 1-4")
    out = tmp_path / "output"
    summary = compare_family(alignment, tree, model, list(AA_ORDER), out, arpip=ARPIP,
                             threads=1, protein="AA", lg_matrix=LG_MATRIX)
    assert summary["arpip_gap_sites"] == 0
    taxa = list("ABCD")
    pip_keys = node_keys(out / "pip.arpip.tree", taxa)
    raxml_keys = node_keys(out / "raxml_constant.raxml.ancestralTree", taxa)
    pip = {(pip_keys[name], site): values[:-1] for name, site, values in
           arpip_rows(out / "pip.arpip.probs", list(AA_ORDER)) if name in pip_keys}
    raxml = {(raxml_keys[name], site): values for name, site, values in
             asr.posterior_rows(out / "raxml_constant.raxml.ancestralProbs", list(AA_ORDER))}
    assert pip.keys() == raxml.keys()
    for key in pip:
        np.testing.assert_allclose(pip[key], raxml[key], atol=2e-5, rtol=0)


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
    joined = join_probabilities(out, tmp_path / "joined")
    with gzip.open(joined, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == summary["n_node_sites"] * 4
    assert {(int(row["site"]), row["character"]) for row in rows} == {
        (site, state) for site in range(8) for state in "0123"}
    assert compare_family(alignment, tree, model, list("0123"), out, arpip=ARPIP, threads=1) == summary
    (out / "paired_node_sites.csv.gz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Completed output changed"):
        compare_family(alignment, tree, model, list("0123"), out, arpip=ARPIP, threads=1)
