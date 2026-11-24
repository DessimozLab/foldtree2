#!/usr/bin/env python3
"""Compare ARPIP and fitted/constant RAxML on native AA and 3Di alignments."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.benchmark_common import AA_ORDER, begin_stage, complete_stage, digest, stationary_from_log, write_json
from scripts.benchmarks.benchmark_cohort import DEFAULT_COHORT, load, native_directory, resolve_path, verify_stage
from scripts.benchmarks.compare_indel_asr import compare_family, fitted_substitution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--lg-matrix", type=Path, required=True, help="Bio++ LG matrix in PAML order")
    parser.add_argument("--arpip", required=True)
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.threads < 1 or args.timeout < 1 or (args.max_families is not None and args.max_families < 1):
        parser.error("threads, timeout and max-families must be positive")
    families = load(args.cohort)["families"][:args.max_families]
    signature = {"protocol": "paired_native_protein_indel_cohort_v1",
                 "cohort_sha256": digest(args.cohort), "families": [f["family"] for f in families],
                 "representations": ["AA", "3Di"], "native_root": str(args.native_root.resolve()),
                 "arpip_sha256": digest(shutil.which(args.arpip) or args.arpip),
                 "lg_matrix": str(args.lg_matrix.resolve()), "lg_matrix_sha256": digest(args.lg_matrix),
                 "driver_sha256": digest(__file__),
                 "comparison_sha256": digest(ROOT / "scripts/benchmarks/compare_indel_asr.py"),
                 "runner_sha256": digest(ROOT / "foldtree2/src/trees/arpip.py"),
                 "indel_lambda": 10.0, "indel_mu": 0.01,
                 "threads": args.threads, "timeout_seconds": args.timeout}
    if not begin_stage(args.outdir, signature):
        return
    results = []
    for family in families:
        ident = family["family"]
        for representation in ("AA", "3Di"):
            directory = native_directory(args.native_root, representation, ident)
            verify_stage(directory)
            ancestry = verify_stage(directory / "ancestral")
            inputs = ancestry["inputs"]
            alignments = [resolve_path(p) for p in inputs if Path(p).suffix in (".fa", ".fasta")]
            if len(alignments) != 1 or digest(alignments[0]) != inputs[str(alignments[0])]:
                raise ValueError(f"Missing, ambiguous or changed native alignment: {directory}")
            model, tree, log = [directory / f"ml.raxml.{suffix}" for suffix in ("bestModel", "bestTree", "log")]
            _, _, prior = fitted_substitution(model, 20, protein=representation,
                                              lg_matrix=args.lg_matrix if representation == "AA" else None)
            if not np.allclose(prior, stationary_from_log(log, 20), atol=2e-6):
                raise ValueError(f"Protein model stationary frequencies differ from RAxML: {directory}")
            print(f"RUN {ident}/{representation}", flush=True)
            result = compare_family(alignments[0], tree, model, list(AA_ORDER),
                                    args.outdir / ident / representation, arpip=args.arpip,
                                    threads=args.threads, timeout=args.timeout, protein=representation,
                                    lg_matrix=args.lg_matrix if representation == "AA" else None)
            results.append({"family": ident, "representation": representation, **result})
            write_json(args.outdir / "progress.json", {"status": "running", "results": results})
            print(f"DONE {ident}/{representation}: gap_fraction={result['arpip_gap_fraction']:.4f}", flush=True)
    write_json(args.outdir / "summary.json", {"status": "completed", "results": results})
    complete_stage(args.outdir, signature, [args.outdir / "summary.json"])


if __name__ == "__main__":
    main()
