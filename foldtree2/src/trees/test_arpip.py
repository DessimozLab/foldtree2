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


def test_conf_uses_the_foldtree2_alphabet_and_submat(tmp_path):
    conf = arpip_conf(tmp_path / "aln.fasta", tmp_path / "tree.nwk",
                     tmp_path / "charset.txt", tmp_path / "submat.txt", "run")
    assert "alphabet=Custom(file=" in conf
    assert "model=PIP(model=GTR(file=" in conf
    assert "lambda=10.0, mu=0.01" in conf
    assert "output.ancestral.file=run.arpip.ancestral.fasta" in conf


def test_charset_is_written_and_validated(tmp_path):
    msa, tree, submat = _fixture(tmp_path)
    outputs = run_arpip_asr(msa, tree, CHARSET, submat, tmp_path / "run",
                           runner=lambda cmd: None)
    assert Path(outputs["charset"]).read_text().splitlines() == CHARSET
    assert outputs["ancestral"].name == "run.arpip.ancestral.fasta"

    for bad in (CHARSET + ["0"], CHARSET + ["-"], list("ab" * 32)):
        with pytest.raises(ValueError):
            run_arpip_asr(msa, tree, bad, submat, tmp_path / "run2", runner=lambda cmd: None)


@pytest.mark.skipif(ARPIP is None, reason="ARPIP binary not available (set ARPIP_BIN)")
def test_arpip_end_to_end_keeps_indels(tmp_path):
    msa, tree, submat = _fixture(tmp_path)
    outputs = run_arpip_asr(msa, tree, CHARSET, submat, tmp_path / "run", arpip=ARPIP)
    records, name = {}, None
    for line in outputs["ancestral"].read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].strip()
            records[name] = ""
        else:
            records[name] += line.strip()
    assert len(records) == 7  # 4 tips + 3 internal nodes
    assert records["seq1"] == "0123" and records["seq2"] == "01-3"  # tips are kept as given
    header = next(line for line in outputs["probabilities"].read_text().splitlines()
                  if line.startswith("Site"))
    assert header.split("\t") == ["Site", "Node"] + CHARSET + ["-"]
    # ARPIP gives no posterior for sites reconstructed as indels: the gap state
    # has zero equilibrium frequency, so those rows are all zeros.
    rows = [line.split("\t") for line in outputs["probabilities"].read_text().splitlines()
            if line.split("\t")[0].isdigit()]
    zero_rows = sum(1 for row in rows if not any(float(p) for p in row[2:] if p))
    assert zero_rows == sum(seq.count("-") for name, seq in records.items() if name not in {"seq1", "seq2", "seq3", "seq4"})
