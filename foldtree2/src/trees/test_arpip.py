"""Self-checks for the ARPIP (Poisson Indel Process) ancestral runner.

Run:  python -m pytest foldtree2/src/trees/test_arpip.py
"""
import os
import shutil
from pathlib import Path

import pytest

from foldtree2.src.trees.arpip import MAX_STATES, arpip_conf, run_arpip_asr

ARPIP = os.environ.get("ARPIP_BIN") or shutil.which("ARPIP")
CHARSET = list("0123")
SUBMAT = "1.0 2.0 3.0 4.0 5.0 6.0\n0.1 0.2 0.3 0.4\n"


def _fixture(tmp_path):
    (tmp_path / "aln.fasta").write_text(
        ">seq1\n0123\n>seq2\n01-3\n>seq3\n0123\n>seq4\n-123\n")
    (tmp_path / "tree.nwk").write_text("(seq1:0.1,seq2:0.2,(seq3:0.3,seq4:0.4):0.5);\n")
    (tmp_path / "submat.txt").write_text(SUBMAT)
    return tmp_path / "aln.fasta", tmp_path / "tree.nwk", tmp_path / "submat.txt"


def _successful_runner(command):
    conf = Path(command[1].split("=", 1)[1])
    for line in conf.read_text().splitlines():
        if line.startswith("output."):
            Path(line.split("=", 1)[1]).write_text("# ARPIP profile=conditional_marginal\n")


def test_conf_uses_the_foldtree2_alphabet_and_submat(tmp_path):
    conf = arpip_conf(tmp_path / "aln.fasta", tmp_path / "tree.nwk",
                     tmp_path / "charset.txt", tmp_path / "submat.txt", "run")
    assert "alphabet=Custom(file=" in conf
    assert "model=PIP(model=GTR(file=" in conf
    assert "lambda=10.0, mu=0.01" in conf
    assert "output.ancestral.file=run.arpip.ancestral.fasta" in conf
    assert "output.node_rel.file=run.arpip.noderel" in conf
    assert "opt.ancestral.prob_profile=conditional_marginal" in conf


def test_charset_is_written_and_validated(tmp_path):
    msa, tree, submat = _fixture(tmp_path)
    outputs = run_arpip_asr(msa, tree, CHARSET, submat, tmp_path / "run",
                           runner=_successful_runner)
    assert Path(outputs["charset"]).read_text().splitlines() == CHARSET
    assert outputs["ancestral"].name == "run.arpip.ancestral.fasta"
    assert outputs["node_rel"].name == "run.arpip.noderel"
    assert all(outputs[name].is_file() for name in
               ("ancestral", "probabilities", "indelpoints", "tree", "node_rel"))

    called = []
    for bad in ([], CHARSET + ["0"], CHARSET + ["-"], ["ab"],
                ["\n"], list("ab" * 32)):
        with pytest.raises(ValueError):
            run_arpip_asr(msa, tree, bad, submat, tmp_path / "run2",
                          runner=lambda cmd: called.append(cmd))
    assert not called


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"),
                                   float("-inf"), "not-a-rate"])
@pytest.mark.parametrize("name", ["indel_lambda", "indel_mu"])
def test_indel_rates_must_be_finite_and_positive(tmp_path, rate, name):
    msa, tree, submat = _fixture(tmp_path)
    called = []
    rates = {"indel_lambda": 10.0, "indel_mu": 0.01}
    rates[name] = rate
    with pytest.raises(ValueError, match="finite positive"):
        run_arpip_asr(msa, tree, CHARSET, submat, tmp_path / "invalid",
                      runner=lambda cmd: called.append(cmd), **rates)
    assert not called


def test_missing_external_outputs_raise(tmp_path):
    msa, tree, submat = _fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="ARPIP did not produce"):
        run_arpip_asr(msa, tree, CHARSET, submat, tmp_path / "missing",
                      runner=lambda cmd: None)


def test_legacy_joint_scores_are_rejected(tmp_path):
    msa, tree, submat = _fixture(tmp_path)
    def legacy(command):
        _successful_runner(command)
        (tmp_path / "old.arpip.probs").write_text("# ARPIP profile=normalized\n")
    with pytest.raises(ValueError, match="not posterior"):
        run_arpip_asr(msa, tree, CHARSET, submat, tmp_path / "old", runner=legacy)


@pytest.mark.skipif(ARPIP is None, reason="ARPIP binary not available (set ARPIP_BIN)")
@pytest.mark.parametrize("charset", [CHARSET, list("Aa*?")])
def test_arpip_end_to_end_keeps_indels(tmp_path, charset):
    msa, tree, submat = _fixture(tmp_path)
    translation = str.maketrans("0123", "".join(charset))
    msa.write_text("\n".join(line if line.startswith(">") else line.translate(translation)
                             for line in msa.read_text().splitlines()) + "\n")
    outputs = run_arpip_asr(msa, tree, charset, submat, tmp_path / "run", arpip=ARPIP)
    records, name = {}, None
    for line in outputs["ancestral"].read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].strip()
            records[name] = ""
        else:
            records[name] += line.strip()
    assert len(records) == 7  # 4 tips + 3 internal nodes
    for name, sequence in {"seq1": "0123", "seq2": "01-3", "seq3": "0123", "seq4": "-123"}.items():
        assert records[name] == sequence.translate(translation)
    header = next(line for line in outputs["probabilities"].read_text().splitlines()
                  if line.startswith("Site"))
    assert header.split("\t") == ["Site", "Node"] + charset + ["-"]
    rows = [line.split("\t") for line in outputs["probabilities"].read_text().splitlines()
            if line.split("\t")[0].isdigit()]
    assert outputs["probabilities"].read_text().startswith("# ARPIP profile=conditional_marginal\n")
    assert len(rows) == 12
    for row in rows:
        values = [float(p) for p in row[2:] if p]
        assert sum(values) == pytest.approx(1, abs=1e-10)
        gap = records[row[1]][int(row[0])] == "-"
        assert values[-1] == int(gap)
        if gap:
            assert not any(values[:-1])
