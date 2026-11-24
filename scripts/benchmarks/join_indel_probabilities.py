#!/usr/bin/env python3
"""Join per-character probabilities from one completed indel ASR comparison."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from ete3 import Tree

from scripts.benchmark_common import begin_stage, complete_stage, digest, read_fasta, read_paml
from scripts.benchmarks.benchmark_cohort import verify_stage
from scripts.benchmarks.compare_indel_asr import arpip_rows, node_keys, profile
from scripts.ancestral.ancestral_state_uncertainty import posterior_rows


def join_probabilities(source, outdir):
    """Write a long-form join from verified native engine outputs, without rerunning ASR."""
    source, outdir = Path(source).resolve(), Path(outdir).resolve()
    if source == outdir or source in outdir.parents or outdir in source.parents:
        raise ValueError("Join output and source must be separate directories")
    provenance = verify_stage(source)
    if provenance.get("protocol") not in {"paired_native_ft2_indel_asr_v2_conditional_marginal",
                                          "paired_native_protein_indel_asr_v1_conditional_marginal"}:
        raise ValueError("Source must use the conditional-marginal benchmark protocol")
    required = {"pip.arpip.tree", "pip.arpip.ancestral.fasta", "pip.arpip.probs", "submat.txt"}
    for engine in ("raxml", "raxml_constant"):
        required.update(f"{engine}.raxml.{suffix}" for suffix in
                        ("ancestralProbs", "ancestralTree", "ancestralStates.fasta"))
    completed = json.loads((source / "completed.json").read_text())
    if not required <= set(completed["outputs"]):
        raise ValueError("Source completion record does not cover all join inputs")
    states = provenance["states"]
    if (len(states) < 2 or len(set(states)) != len(states)
            or any(not isinstance(state, str) or len(state) != 1 for state in states)):
        raise ValueError("Invalid source alphabet")
    signature = {"protocol": "paired_indel_character_probabilities_v1",
                 "source": str(source), "source_completion_sha256": digest(source / "completed.json"),
                 "joiner_sha256": digest(__file__)}
    if not begin_stage(outdir, signature):
        return outdir / "paired_character_probabilities.csv.gz"

    tree = Tree(str(source / "pip.arpip.tree"), format=1)
    taxa = tree.get_leaf_names()
    records = read_fasta(source / "pip.arpip.ancestral.fasta", aligned=True)
    if set(records) != {node.name for node in tree.traverse()}:
        raise ValueError("ARPIP ancestral FASTA does not match its tree")
    length = len(next(iter(records.values())))
    internals = {name: seq for name, seq in records.items() if name not in taxa}
    _, prior, _ = read_paml(source / "submat.txt", len(states))
    sources = {
        "arpip": (source / "pip.arpip.probs", source / "pip.arpip.tree", internals),
        "raxml": (source / "raxml.raxml.ancestralProbs", source / "raxml.raxml.ancestralTree",
                  read_fasta(source / "raxml.raxml.ancestralStates.fasta", aligned=True)),
        "raxml_constant": (source / "raxml_constant.raxml.ancestralProbs",
                           source / "raxml_constant.raxml.ancestralTree",
                           read_fasta(source / "raxml_constant.raxml.ancestralStates.fasta", aligned=True)),
    }
    output = outdir / "paired_character_probabilities.csv.gz"
    with tempfile.TemporaryDirectory(prefix="foldtree2-join-") as tmp:
        with sqlite3.connect(str(Path(tmp) / "profiles.sqlite")) as db:
            db.execute("CREATE TABLE profiles (engine TEXT, node TEXT, site INTEGER, gap INTEGER, "
                       "probabilities BLOB, PRIMARY KEY (engine, node, site))")
            common = None
            for engine, (path, tree_path, sequences) in sources.items():
                keys = node_keys(tree_path, taxa)
                if common is None:
                    common = set(keys.values())
                elif set(keys.values()) != common:
                    raise ValueError("ASR engines changed the unrooted topology")
                if any(len(seq) != length for seq in sequences.values()):
                    raise ValueError("Ancestral sequence length differs from alignment")
                if engine != "arpip" and set(sequences) != set(keys):
                    raise ValueError("RAxML ancestral sequences do not match its tree")
                profile(path, states, prior, sequences, keys, length, arpip=engine == "arpip")
                rows = arpip_rows(path, states) if engine == "arpip" else posterior_rows(path, states)
                db.executemany(
                    "INSERT INTO profiles VALUES (?, ?, ?, ?, ?)",
                    ((engine, keys[name], site, int(sequences[name][site] == "-"),
                      np.asarray(values[:len(states)], dtype="<f8").tobytes())
                     for name, site, values in rows if name in keys),
                )
            db.commit()
            query = ("SELECT a.node, a.site, a.gap, a.probabilities, r.probabilities, c.probabilities "
                     "FROM profiles a JOIN profiles r ON r.node=a.node AND r.site=a.site AND r.engine='raxml' "
                     "JOIN profiles c ON c.node=a.node AND c.site=a.site AND c.engine='raxml_constant' "
                     "WHERE a.engine='arpip' ORDER BY a.node, a.site")
            count = 0
            with gzip.open(output, "wt", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["node_id", "site", "character", "arpip_gap", "arpip_probability",
                                 "raxml_probability", "raxml_constant_probability"])
                for node, site, gap, *blobs in db.execute(query):
                    arrays = [np.frombuffer(blob, dtype="<f8") for blob in blobs]
                    for index, character in enumerate(states):
                        writer.writerow([node, site, character, gap,
                                         "" if gap else arrays[0][index], arrays[1][index], arrays[2][index]])
                    count += 1
            if count != len(common) * length:
                raise ValueError("Incomplete paired probability coverage")
    complete_stage(outdir, signature, [output])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Completed family/representation comparison directory")
    parser.add_argument("--outdir", type=Path, required=True, help="Separate output directory")
    args = parser.parse_args()
    print(join_probabilities(args.source, args.outdir))


if __name__ == "__main__":
    main()
