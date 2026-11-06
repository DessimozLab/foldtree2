#!/usr/bin/env python3
"""Compare ARPIP and RAxML ASR on the frozen OMA native FT2 inputs.

The fitted RAxML model is the usual baseline. A second RAxML run removes
invariant sites, matching ARPIP's constant-rate substitution component.
Neither comparison is an ancestral accuracy estimate (there is no truth set).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from ete3 import Tree

from foldtree2.src.trees.ancestral import ancestral_states_to_fasta, run_ancestral
from foldtree2.src.trees.arpip import run_arpip_asr
from scripts.ancestral.ancestral_state_uncertainty import posterior_rows, rooted_nodes, uncertainty
from scripts.benchmark_common import (
    MULTI_ORDER, begin_stage, complete_stage, digest, read_fasta, tool_version, write_json,
)
from scripts.benchmarks.benchmark_cohort import DEFAULT_COHORT, load, native_directory, resolve_path, verify_stage


def fitted_substitution(path, size):
    """Export fitted upper-triangle exchangeabilities in PAML lower-triangle order."""
    text = Path(path).read_text().strip()
    match = re.fullmatch(
        rf"(MULTI{size}_GTR\{{([^}}]+)\}}\+FU\{{([^}}]+)\}})"
        r"(\+IU\{[^}]+\})?(,\s*noname\s*=\s*[^,\n]+)?", text)
    if match is None:
        raise ValueError(f"Expected one fitted MULTI{size} GTR+FU[+IU] partition: {path}")
    rates = np.asarray([float(x) for x in match[2].split("/")])
    prior = np.asarray([float(x) for x in match[3].split("/")])
    if (rates.shape != (size * (size - 1) // 2,) or prior.shape != (size,)
            or not np.isfinite(rates).all() or (rates < 0).any() or not rates.any()
            or not np.isfinite(prior).all() or (prior <= 0).any()
            or not np.isclose(prior.sum(), 1, atol=2e-5)):
        raise ValueError(f"Invalid fitted substitution parameters: {path}")
    matrix = np.zeros((size, size))
    matrix[np.triu_indices(size, 1)] = rates
    matrix += matrix.T
    rows = [" ".join(format(x, ".17g") for x in matrix[i, :i]) for i in range(1, size)]
    rows.append(" ".join(format(x, ".17g") for x in prior))
    return "\n".join(rows) + "\n", match[1] + (match[5] or ""), prior / prior.sum()


def node_keys(path, taxa):
    """Pair original nodes by incident unrooted clades, not engine-specific labels."""
    tree = Tree(Path(path).read_text(), format=1)
    if set(tree.get_leaf_names()) != set(taxa) or len(tree.get_leaf_names()) != len(taxa):
        raise ValueError(f"Tree/alignment taxa differ: {path}")
    result = {}
    for node in tree.traverse():
        if node.is_leaf():
            continue
        parts = [tuple(sorted(child.get_leaf_names())) for child in node.children]
        if not node.is_root():
            parts.append(tuple(sorted(set(taxa) - set(node.get_leaf_names()))))
        if len(parts) == 2:
            continue  # The root inserted on an edge is not a shared original node.
        if len(parts) != 3 or not node.name or node.name in result:
            raise ValueError(f"Expected labelled, bifurcating ancestral tree: {path}")
        result[node.name] = hashlib.sha256(json.dumps(sorted(parts)).encode()).hexdigest()[:20]
    if len(set(result.values())) != len(result) or len(result) != len(taxa) - 2:
        raise ValueError(f"Incomplete or ambiguous ancestral topology: {path}")
    return result


def arpip_rows(path, states):
    """ARPIP profiles are site-major and zero-based; gap rows have zero mass."""
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith("Site\tNode\t"):
                if line.rstrip().split("\t") != ["Site", "Node", *states, "-"]:
                    raise ValueError("ARPIP posterior state order differs from the fitted model")
                break
        else:
            raise ValueError("Missing ARPIP posterior header")
        for line in stream:
            if not line.strip():
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != len(states) + 3:
                raise ValueError("Malformed ARPIP posterior row")
            values = np.asarray([float(p) for p in fields[2:]])
            if not np.isfinite(values).all() or (values < 0).any() or values[-1] != 0:
                raise ValueError("Invalid ARPIP posterior values")
            yield fields[1], int(fields[0]), values[:-1]


def profile(path, states, prior, sequences, keys, length, *, arpip=False):
    """Retain entropy/max-p only; never turn zero-mass gap rows into certainty."""
    result = {key: np.full((length, 2), np.nan) for key in keys.values()}
    seen = {name: np.zeros(length, dtype=bool) for name in sequences}
    rows = arpip_rows(path, states) if arpip else posterior_rows(path, states)
    count = 0
    for name, site, probability in rows:
        if name not in seen or not 0 <= site < length or seen[name][site]:
            raise ValueError(f"Unknown, duplicate or invalid posterior node/site: {name}/{site}")
        seen[name][site] = True
        gap = sequences[name][site] == "-"
        if gap:
            if not arpip or probability.any():
                raise ValueError("Gap FASTA state disagrees with posterior mass")
            continue
        metrics = uncertainty(probability, prior)
        if name in keys:
            result[keys[name]][site] = metrics["entropy_bits"], metrics["max_probability"]
            count += 1
    if not seen or any(not sites.all() for sites in seen.values()):
        raise ValueError("Incomplete ancestral posterior coverage")
    if not count:
        raise ValueError("No non-gap ancestral posteriors")
    return result


def compare_family(alignment, tree, model, states, outdir, *, arpip, threads=2,
                   indel_lambda=10.0, indel_mu=0.01, timeout=600):
    alignment, tree, model, outdir = [Path(p).resolve() for p in (alignment, tree, model, outdir)]
    records = read_fasta(alignment, aligned=True)
    length = len(next(iter(records.values())))
    if any(set(seq) - set(states) - {"-"} for seq in records.values()):
        raise ValueError("Alignment contains states outside the declared alphabet")
    submat, constant, prior = fitted_substitution(model, len(states))
    arpip = Path(shutil.which(str(arpip)) or arpip).resolve(strict=True)
    signature = {
        "protocol": "paired_native_ft2_indel_asr_v1", "states": states,
        "inputs": {str(p): digest(p) for p in (alignment, tree, model)},
        "arpip": {"executable": str(arpip), "sha256": digest(arpip)},
        "raxml": tool_version("raxml-ng"), "mad": tool_version("mad"),
        "indel_lambda": indel_lambda, "indel_mu": indel_mu, "threads": threads,
        "timeout_seconds": timeout, "driver_sha256": digest(__file__),
    }
    if not begin_stage(outdir, signature):
        return json.loads((outdir / "summary.json").read_text())

    def execute(command):
        with (outdir / "commands.log").open("a") as log:
            log.write("\nCOMMAND " + json.dumps(list(map(str, command))) + "\n")
            log.flush()
            subprocess.run(command, cwd=outdir, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout,
                           env={**os.environ, "OMP_NUM_THREADS": str(threads)})

    rooted_nodes(tree, outdir, None)
    rooted = outdir / "rooted.nwk"
    matrix = outdir / "submat.txt"
    matrix.write_text(submat)
    constant_model = outdir / "constant.model"
    constant_model.write_text(constant + "\n")
    engines, outputs = {}, [rooted, matrix, constant_model, outdir / "commands.log"]
    for name, source in (("raxml", model), ("raxml_constant", constant_model)):
        native = run_ancestral(alignment, rooted, source, outdir / name, threads=threads,
                               overwrite=True, freeze_fitted=True, runner=execute)
        fasta = Path(ancestral_states_to_fasta(native["states"]))
        sequences = read_fasta(fasta, aligned=True)
        keys = node_keys(native["tree"], records)
        if set(sequences) != set(keys) or any(len(seq) != length for seq in sequences.values()):
            raise ValueError("RAxML ancestral FASTA dimensions differ from input")
        engines[name] = (sequences, keys, profile(native["probabilities"], states, prior,
                                                sequences, keys, length))
        outputs.extend([fasta, *native.values()])
    native = run_arpip_asr(alignment, rooted, states, matrix, outdir / "pip", arpip=arpip,
                          indel_lambda=indel_lambda, indel_mu=indel_mu, runner=execute)
    sequences = read_fasta(native["ancestral"], aligned=True)
    keys = node_keys(native["tree"], records)
    if (any(sequences.get(t) != seq for t, seq in records.items())
            or set(sequences) != {node.name for node in Tree(native["tree"].read_text(), format=1).traverse()}
            or any(len(seq) != length for seq in sequences.values())):
        raise ValueError("ARPIP did not retain all tip/internal sequences and alignment columns")
    internals = {name: seq for name, seq in sequences.items() if name not in records}
    engines["arpip"] = (internals, keys, profile(native["probabilities"], states, prior,
                                               internals, keys, length, arpip=True))
    outputs.extend(native.values())
    common = set(engines["raxml"][1].values())
    if any(set(keys.values()) != common for _, keys, _ in engines.values()):
        raise ValueError("ASR engines changed the unrooted topology")
    inverse = {engine: {key: name for name, key in keys.items()}
               for engine, (_, keys, _) in engines.items()}
    paired = outdir / "paired_node_sites.csv.gz"
    totals = {"arpip_gap_sites": 0, "present_sites": 0, "agreement_raxml": 0,
              "agreement_raxml_constant": 0, "entropy_arpip_present": 0.0,
              "entropy_raxml_present": 0.0, "entropy_raxml_constant_present": 0.0}
    with gzip.open(paired, "wt", newline="") as stream:
        fields = ["node_id", "site", "arpip_gap"]
        for engine in engines:
            fields += [engine + "_state", engine + "_entropy_bits", engine + "_max_probability"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(common):
            for site in range(length):
                row = {"node_id": key, "site": site}
                for engine, (seqs, _, metrics) in engines.items():
                    state = seqs[inverse[engine][key]][site]
                    h, maximum = metrics[key][site]
                    row.update({engine + "_state": state,
                                engine + "_entropy_bits": float(h) if np.isfinite(h) else None,
                                engine + "_max_probability": float(maximum) if np.isfinite(maximum) else None})
                gap = row["arpip_state"] == "-"
                row["arpip_gap"] = int(gap)
                totals["arpip_gap_sites"] += int(gap)
                if not gap:
                    totals["present_sites"] += 1
                    for engine in engines:
                        totals["entropy_" + engine + "_present"] += row[engine + "_entropy_bits"]
                    for engine in ("raxml", "raxml_constant"):
                        totals["agreement_" + engine] += int(row[engine + "_state"] == row["arpip_state"])
                writer.writerow(row)
    present = totals["present_sites"]
    summary = {"n_tips": len(records), "n_sites": length, "n_shared_nodes": len(common),
               "n_node_sites": length * len(common), **totals,
               "arpip_gap_fraction": totals["arpip_gap_sites"] / (length * len(common)),
               "interpretation": "Descriptive agreement/conditional uncertainty, not accuracy; "
               "ARPIP gaps have no posterior mass. RAxML constant controls for +I removal."}
    for engine in engines:
        summary["mean_entropy_" + engine + "_present"] = totals["entropy_" + engine + "_present"] / present
    for engine in ("raxml", "raxml_constant"):
        summary["agreement_" + engine + "_fraction"] = totals["agreement_" + engine] / present
    write_json(outdir / "summary.json", summary)
    complete_stage(outdir, signature, [*outputs, paired, outdir / "summary.json"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, choices=range(2, 65), default=[20, 30, 40])
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--arpip", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--indel-lambda", type=float, default=10.0)
    parser.add_argument("--indel-mu", type=float, default=0.01)
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per ASR subprocess")
    args = parser.parse_args()
    if args.threads < 1 or args.timeout < 1 or (args.max_families is not None and args.max_families < 1):
        parser.error("threads, timeout and max-families must be positive")
    if not all(np.isfinite(v) and v > 0 for v in (args.indel_lambda, args.indel_mu)):
        parser.error("indel rates must be finite and positive")
    cohort = load(args.cohort)
    families = cohort["families"][:args.max_families]
    signature = {"cohort_sha256": digest(args.cohort), "families": [f["family"] for f in families],
                 "sizes": args.sizes, "native_root": str(args.native_root.resolve()),
                 "indel_lambda": args.indel_lambda, "indel_mu": args.indel_mu,
                 "threads": args.threads, "timeout_seconds": args.timeout,
                 "arpip_sha256": digest(shutil.which(args.arpip) or args.arpip),
                 "driver_sha256": digest(__file__)}
    begin_stage(args.outdir, signature)
    rows = []
    for family in families:
        for size in args.sizes:
            ident = family["family"]
            directory = native_directory(args.native_root, f"FT2_{size}", ident)
            verify_stage(directory)
            provenance = verify_stage(directory / "ancestral")
            alignments = [resolve_path(p) for p in provenance["inputs"] if Path(p).name == "ft2.aligned.fasta"]
            if len(alignments) != 1:
                raise ValueError(f"Ambiguous native alignment provenance: {directory}")
            print(f"RUN {ident}/FT2_{size}", flush=True)
            summary = compare_family(alignments[0], directory / "ml.raxml.bestTree",
                                     directory / "ml.raxml.bestModel", list(MULTI_ORDER[:size]),
                                     args.outdir / ident / f"FT2_{size}", arpip=args.arpip,
                                     threads=args.threads, indel_lambda=args.indel_lambda,
                                     indel_mu=args.indel_mu, timeout=args.timeout)
            rows.append({"family": ident, "representation": f"FT2_{size}", **summary})
            write_json(args.outdir / "progress.json", {"status": "running", "results": rows})
            print(f"DONE {ident}/FT2_{size}: gap_fraction={summary['arpip_gap_fraction']:.4f}", flush=True)
    write_json(args.outdir / "summary.json", {"status": "completed", "results": rows})
    complete_stage(args.outdir, signature, [args.outdir / "summary.json"])


if __name__ == "__main__":
    main()
