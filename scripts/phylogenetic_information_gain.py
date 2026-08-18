#!/usr/bin/env python3
"""Compute phylogenetic information gain for AA, 3Di, and FT2 alphabets.

This script extracts the notebook workflow into a reusable CLI:
1) Parse alignments and per-site tree log-likelihoods (.raxml.siteLH)
2) Compute IID baseline from global background frequencies
3) Compute phylogenetic gain: phylo_gain = loglik_tree - loglik_iid
4) Compute normalized gain: phylo_gain_norm = phylo_gain / (entropy_tip + eps)
5) Optionally compute pairwise cross-alphabet MI on aligned columns

Input is provided through a JSON spec file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


AA_STATES: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class AlignmentRecord:
    record_id: str
    seq: str


@dataclass
class DatasetSpec:
    name: str
    kind: str
    alignment: Path
    tree: Path
    model: Optional[str]
    sitelh_file: Optional[Path]
    states: Optional[str]


def read_alignment_file(alignment_file: Path) -> List[AlignmentRecord]:
    records: List[AlignmentRecord] = []
    seq_id: Optional[str] = None
    seq_lines: List[str] = []

    with alignment_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    records.append(AlignmentRecord(record_id=seq_id, seq="".join(seq_lines)))
                seq_id = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

    if seq_id is not None:
        records.append(AlignmentRecord(record_id=seq_id, seq="".join(seq_lines)))

    if not records:
        raise ValueError(f"No records parsed from alignment: {alignment_file}")

    lengths = {len(r.seq) for r in records}
    if len(lengths) != 1:
        raise ValueError(f"Alignment has inconsistent sequence lengths: {alignment_file}")

    return records


def infer_alphabet_from_alignment(alignment_file: Path, gap_char: str = "-") -> Set[str]:
    records = read_alignment_file(alignment_file)
    chars: Set[str] = set()
    for record in records:
        chars.update(set(record.seq))
    chars.discard(gap_char)
    return chars


def parse_states_override(states_value: str) -> Set[str]:
    states_path = Path(states_value)
    if states_path.exists():
        states: Set[str] = set()
        with states_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                s = line.strip()
                if not s:
                    continue
                states.update(list(s))
        return states
    return set(states_value)


def resolve_valid_states(spec: DatasetSpec, gap_char: str) -> Set[str]:
    kind = spec.kind.lower()
    if spec.states:
        states = parse_states_override(spec.states)
    elif kind == "aa":
        states = AA_STATES
    elif kind in {"3di", "ft2", "custom"}:
        states = infer_alphabet_from_alignment(spec.alignment, gap_char=gap_char)
    else:
        raise ValueError(f"Unsupported dataset kind '{spec.kind}' for dataset '{spec.name}'")

    states.discard(gap_char)
    if not states:
        raise ValueError(f"No valid states found for dataset '{spec.name}'")
    return states


def parse_raxml_sitelh(sitelh_file: Path) -> np.ndarray:
    lines = sitelh_file.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid site-likelihood file format: {sitelh_file}")

    data_line = lines[1].strip().split()
    if len(data_line) < 2:
        raise ValueError(f"Invalid site-likelihood data line: {sitelh_file}")

    return np.array([float(x) for x in data_line[1:]], dtype=float)


def run_raxml_sitelh(
    alignment_file: Path,
    tree_file: Path,
    model: str,
    output_prefix: Path,
    raxml_path: str,
    threads: int,
) -> Path:
    cmd = [
        raxml_path,
        "--force",
        "--redo",
        "--evaluate",
        "--msa",
        str(alignment_file),
        "--model",
        model,
        "--tree",
        str(tree_file),
        "--sitelh",
        "--threads",
        str(threads),
        "--prefix",
        str(output_prefix),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "RAxML site-likelihood run failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    sitelh_file = Path(f"{output_prefix}.raxml.siteLH")
    if not sitelh_file.exists():
        raise FileNotFoundError(f"Expected RAxML site-likelihood output not found: {sitelh_file}")
    return sitelh_file


def compute_tip_state_frequencies(column: str, valid_states: Set[str], gap_char: str = "-") -> Dict[str, float]:
    valid_chars = [c for c in column if c in valid_states and c != gap_char]
    n_valid = len(valid_chars)
    if n_valid == 0:
        return {}
    counts = Counter(valid_chars)
    return {state: count / n_valid for state, count in counts.items()}


def compute_tip_entropy(frequencies: Dict[str, float], eps: float = 1e-12) -> float:
    h = 0.0
    for p in frequencies.values():
        if p > eps:
            h -= p * np.log2(p)
    return float(h)


def compute_column_tip_stats(column: str, valid_states: Set[str], gap_char: str = "-") -> Dict[str, float]:
    n_total = len(column)
    n_gaps = sum(1 for c in column if c == gap_char)
    gap_frac = n_gaps / n_total if n_total > 0 else 0.0

    freqs = compute_tip_state_frequencies(column, valid_states, gap_char)
    h_tip = compute_tip_entropy(freqs)
    n_observed = len(freqs)
    n_valid = sum(1 for c in column if c in valid_states and c != gap_char)

    return {
        "entropy_tip": h_tip,
        "n_observed_states": n_observed,
        "n_valid_taxa": n_valid,
        "gap_fraction": gap_frac,
    }


def compute_global_background_frequencies(
    alignment_columns: Sequence[str],
    valid_states: Set[str],
    gap_char: str = "-",
    pseudocount: float = 1e-6,
) -> Dict[str, float]:
    all_counts = Counter()
    for col in alignment_columns:
        for c in col:
            if c in valid_states and c != gap_char:
                all_counts[c] += 1

    total = sum(all_counts.values())
    n_states = len(valid_states)
    denom = total + pseudocount * n_states

    return {state: (all_counts.get(state, 0) + pseudocount) / denom for state in valid_states}


def compute_iid_log_likelihood(
    column: str,
    background_freqs: Dict[str, float],
    valid_states: Set[str],
    gap_char: str = "-",
) -> float:
    log_lik = 0.0
    for c in column:
        if c in valid_states and c != gap_char:
            p = background_freqs.get(c, 1e-12)
            log_lik += np.log(max(p, 1e-12))
    return float(log_lik)


def analyze_alphabet_phylogenetic_info(
    spec: DatasetSpec,
    site_likelihoods: np.ndarray,
    valid_states: Set[str],
    gap_char: str,
    gap_occ_max: float,
) -> pd.DataFrame:
    records = read_alignment_file(spec.alignment)
    aln_length = len(records[0].seq)
    columns = ["".join(rec.seq[i] for rec in records) for i in range(aln_length)]
    bg_freqs = compute_global_background_frequencies(columns, valid_states, gap_char)

    rows: List[Dict[str, object]] = []
    for col_idx, col in enumerate(columns):
        if col_idx >= len(site_likelihoods):
            continue

        tip_stats = compute_column_tip_stats(col, valid_states, gap_char)
        if tip_stats["gap_fraction"] > gap_occ_max:
            continue

        loglik_tree = float(site_likelihoods[col_idx])
        loglik_iid = compute_iid_log_likelihood(col, bg_freqs, valid_states, gap_char)
        phylo_gain = loglik_tree - loglik_iid
        phylo_gain_norm = phylo_gain / (float(tip_stats["entropy_tip"]) + 1e-6)

        rows.append(
            {
                "alphabet": spec.name,
                "kind": spec.kind,
                "column_index": col_idx,
                "alignment_column": col,
                "entropy_tip": float(tip_stats["entropy_tip"]),
                "n_observed_states": int(tip_stats["n_observed_states"]),
                "n_valid_taxa": int(tip_stats["n_valid_taxa"]),
                "gap_fraction": float(tip_stats["gap_fraction"]),
                "loglik_tree": loglik_tree,
                "loglik_iid": loglik_iid,
                "phylo_gain": phylo_gain,
                "phylo_gain_norm": phylo_gain_norm,
            }
        )

    return pd.DataFrame(rows)


def compute_pairwise_mi(
    col_x: str,
    col_y: str,
    states_x: Set[str],
    states_y: Set[str],
    gap_char: str = "-",
    pseudocount: float = 1e-6,
) -> Tuple[float, float, int]:
    valid_pairs: List[Tuple[str, str]] = []
    for cx, cy in zip(col_x, col_y):
        if cx in states_x and cy in states_y and cx != gap_char and cy != gap_char:
            valid_pairs.append((cx, cy))

    n_valid = len(valid_pairs)
    if n_valid < 2:
        return 0.0, 0.0, n_valid

    joint_counts = Counter(valid_pairs)
    x_counts = Counter(p[0] for p in valid_pairs)
    y_counts = Counter(p[1] for p in valid_pairs)

    nx, ny = len(states_x), len(states_y)
    denom = n_valid + pseudocount * nx * ny

    p_x = {s: (x_counts.get(s, 0) + pseudocount * ny) / denom for s in states_x}
    p_y = {s: (y_counts.get(s, 0) + pseudocount * nx) / denom for s in states_y}

    mi = 0.0
    for (sx, sy), count in joint_counts.items():
        p_xy = (count + pseudocount) / denom
        p_x_val = p_x.get(sx, 1e-12)
        p_y_val = p_y.get(sy, 1e-12)
        if p_xy > 1e-12:
            mi += p_xy * np.log2(p_xy / (p_x_val * p_y_val))

    h_y = 0.0
    for s in states_y:
        p = (y_counts.get(s, 0) + pseudocount * nx) / denom
        if p > 1e-12:
            h_y -= p * np.log2(p)

    nmi = mi / h_y if h_y > 1e-12 else 0.0
    return float(mi), float(nmi), n_valid


def compute_cross_alphabet_mi(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    states_x: Set[str],
    states_y: Set[str],
    alphabet_x: str,
    alphabet_y: str,
    gap_char: str = "-",
) -> pd.DataFrame:
    common_cols = set(df_x["column_index"]) & set(df_y["column_index"])
    rows: List[Dict[str, object]] = []

    for col_idx in common_cols:
        col_x = df_x[df_x["column_index"] == col_idx]["alignment_column"].values[0]
        col_y = df_y[df_y["column_index"] == col_idx]["alignment_column"].values[0]
        mi, nmi, n_valid = compute_pairwise_mi(col_x, col_y, states_x, states_y, gap_char)
        rows.append(
            {
                "column_index": int(col_idx),
                "alphabet_x": alphabet_x,
                "alphabet_y": alphabet_y,
                "mi": mi,
                "nmi": nmi,
                "n_valid_joint": int(n_valid),
            }
        )

    return pd.DataFrame(rows)


def compute_summary_stats(df: pd.DataFrame, alphabet_name: str) -> Dict[str, float]:
    return {
        "alphabet": alphabet_name,
        "n_columns": int(len(df)),
        "h_tip_mean": float(df["entropy_tip"].mean()),
        "h_tip_median": float(df["entropy_tip"].median()),
        "h_tip_std": float(df["entropy_tip"].std()),
        "loglik_tree_sum": float(df["loglik_tree"].sum()),
        "loglik_tree_mean": float(df["loglik_tree"].mean()),
        "loglik_iid_sum": float(df["loglik_iid"].sum()),
        "loglik_iid_mean": float(df["loglik_iid"].mean()),
        "phylo_gain_sum": float(df["phylo_gain"].sum()),
        "phylo_gain_mean": float(df["phylo_gain"].mean()),
        "phylo_gain_median": float(df["phylo_gain"].median()),
        "phylo_gain_std": float(df["phylo_gain"].std()),
        "phylo_gain_norm_mean": float(df["phylo_gain_norm"].mean()),
        "phylo_gain_norm_median": float(df["phylo_gain_norm"].median()),
        "n_states_mean": float(df["n_observed_states"].mean()),
        "n_states_max": float(df["n_observed_states"].max()),
    }


def load_spec_file(spec_path: Path) -> List[DatasetSpec]:
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    datasets = data.get("datasets", data)
    if not isinstance(datasets, list):
        raise ValueError("Spec file must be a list or contain a top-level 'datasets' list")

    specs: List[DatasetSpec] = []
    for item in datasets:
        name = item["name"]
        kind = item["kind"]
        alignment = Path(item["alignment"]).expanduser().resolve()
        tree = Path(item["tree"]).expanduser().resolve()
        model = item.get("model")
        sitelh_raw = item.get("sitelh_file")
        sitelh_file = Path(sitelh_raw).expanduser().resolve() if sitelh_raw else None
        states = item.get("states")

        specs.append(
            DatasetSpec(
                name=name,
                kind=kind,
                alignment=alignment,
                tree=tree,
                model=model,
                sitelh_file=sitelh_file,
                states=states,
            )
        )
    return specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute phylogenetic information gain for AA, 3Di, and FT2 datasets"
    )
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to JSON spec describing datasets",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for per-alphabet and combined results",
    )
    parser.add_argument(
        "--raxml-path",
        default="raxml-ng",
        help="Path to raxml-ng executable",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Threads passed to raxml-ng --threads",
    )
    parser.add_argument(
        "--gap-occ-max",
        type=float,
        default=0.3,
        help="Maximum allowed gap fraction per column",
    )
    parser.add_argument(
        "--gap-char",
        default="-",
        help="Gap character used in alignments",
    )
    parser.add_argument(
        "--skip-cross-mi",
        action="store_true",
        help="Skip pairwise cross-alphabet MI computation",
    )
    parser.add_argument(
        "--no-run-raxml",
        action="store_true",
        help="Do not run RAxML; require sitelh_file in spec",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    specs = load_spec_file(args.spec.expanduser().resolve())
    if not specs:
        raise ValueError("Spec contains no datasets")

    per_alphabet: Dict[str, pd.DataFrame] = {}
    states_by_name: Dict[str, Set[str]] = {}

    for spec in specs:
        if not spec.alignment.exists():
            raise FileNotFoundError(f"Alignment not found for {spec.name}: {spec.alignment}")
        if not spec.tree.exists():
            raise FileNotFoundError(f"Tree not found for {spec.name}: {spec.tree}")

        valid_states = resolve_valid_states(spec, gap_char=args.gap_char)
        states_by_name[spec.name] = valid_states

        sitelh_file = spec.sitelh_file
        if sitelh_file is None:
            if args.no_run_raxml:
                raise ValueError(
                    f"Dataset '{spec.name}' has no sitelh_file and --no-run-raxml was set"
                )
            if not spec.model:
                raise ValueError(
                    f"Dataset '{spec.name}' needs either sitelh_file or model to run RAxML"
                )
            prefix = outdir / f"{spec.name}_sitelh"
            sitelh_file = run_raxml_sitelh(
                alignment_file=spec.alignment,
                tree_file=spec.tree,
                model=spec.model,
                output_prefix=prefix,
                raxml_path=args.raxml_path,
                threads=args.threads,
            )
        else:
            if not sitelh_file.exists():
                raise FileNotFoundError(f"siteLH file not found for {spec.name}: {sitelh_file}")

        site_likelihoods = parse_raxml_sitelh(sitelh_file)
        df = analyze_alphabet_phylogenetic_info(
            spec=spec,
            site_likelihoods=site_likelihoods,
            valid_states=valid_states,
            gap_char=args.gap_char,
            gap_occ_max=args.gap_occ_max,
        )

        per_alphabet[spec.name] = df
        df.to_csv(outdir / f"phylo_info_{spec.name}.csv", index=False)

    combined_df = pd.concat(list(per_alphabet.values()), ignore_index=True)
    combined_df.to_csv(outdir / "phylo_info_combined.csv", index=False)

    summary_rows = [compute_summary_stats(df, name) for name, df in per_alphabet.items()]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "phylo_info_summary.csv", index=False)

    pairwise_mi_frames: List[pd.DataFrame] = []
    if not args.skip_cross_mi and len(per_alphabet) > 1:
        for a_name, b_name in combinations(per_alphabet.keys(), 2):
            mi_df = compute_cross_alphabet_mi(
                df_x=per_alphabet[a_name],
                df_y=per_alphabet[b_name],
                states_x=states_by_name[a_name],
                states_y=states_by_name[b_name],
                alphabet_x=a_name,
                alphabet_y=b_name,
                gap_char=args.gap_char,
            )
            if not mi_df.empty:
                pairwise_mi_frames.append(mi_df)
                mi_df.to_csv(outdir / f"cross_mi_{a_name}_vs_{b_name}.csv", index=False)

    if pairwise_mi_frames:
        all_mi_df = pd.concat(pairwise_mi_frames, ignore_index=True)
        all_mi_df.to_csv(outdir / "cross_alphabet_mi_all_pairs.csv", index=False)

    key_results = {
        "alphabets": {
            name: {
                "n_columns": int(summary_df[summary_df["alphabet"] == name]["n_columns"].iloc[0]),
                "phylo_gain_sum": float(
                    summary_df[summary_df["alphabet"] == name]["phylo_gain_sum"].iloc[0]
                ),
                "phylo_gain_mean": float(
                    summary_df[summary_df["alphabet"] == name]["phylo_gain_mean"].iloc[0]
                ),
                "entropy_tip_mean": float(
                    summary_df[summary_df["alphabet"] == name]["h_tip_mean"].iloc[0]
                ),
            }
            for name in per_alphabet
        }
    }

    with (outdir / "phylo_info_key_results.json").open("w", encoding="utf-8") as handle:
        json.dump(key_results, handle, indent=2)

    print("Done.")
    print(f"Output directory: {outdir}")
    for name in per_alphabet:
        print(f"  - phylo_info_{name}.csv")
    print("  - phylo_info_combined.csv")
    print("  - phylo_info_summary.csv")
    if pairwise_mi_frames:
        print("  - cross_alphabet_mi_all_pairs.csv")
    print("  - phylo_info_key_results.json")


if __name__ == "__main__":
    main()
