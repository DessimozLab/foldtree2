#!/usr/bin/env python3
"""Optuna-driven search for FoldTree2 alphabets that maximize phylogenetic information gain.

The workflow is intentionally scaffolded around the existing project modules:
1. Train a FoldTree2 model with the Lightning trainer.
2. Create a substitution matrix with makesubmat.py.
3. Run phylogenetic-information analysis with the existing script.
4. Score each trial by the phylogenetic gain delta from the benchmark comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import optuna


class DryRunTrial:
    def __init__(self, number: int) -> None:
        self.number = number
        self.user_attrs = {}

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value


@dataclass
class OptunaSearchConfig:
    study_name: str = "foldtree2_phylo_optuna"
    output_dir: str = "./models/phylo_optuna"
    dataset: str = "./structs_training_mk3.h5"
    structures_dir: str = "./alphafold_benchmark"
    benchmark_alignment: str = "./foldtree2/notebooks/benchmarks/Information_benchmark/aa_supermatrix.fasta"
    benchmark_tree: str = "./foldtree2/notebooks/benchmarks/Information_benchmark/aa_astral_tree.nwk"
    benchmark_sitelh: Optional[str] = None
    training_epochs: int = 3
    batch_size: int = 8
    num_embeddings: int = 20
    embedding_dim: int = 64
    hidden_size: int = 128
    seed: int = 0
    sampler_seed: int = 42
    trials: int = 10
    storage: Optional[str] = None
    device: Optional[str] = None
    dry_run: bool = False
    extra_train_args: Optional[list[str]] = None
    extra_submat_args: Optional[list[str]] = None
    extra_phylo_args: Optional[list[str]] = None


def parse_args() -> OptunaSearchConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", default="foldtree2_phylo_optuna")
    parser.add_argument("--output-dir", default="./models/phylo_optuna")
    parser.add_argument("--dataset", default="./structs_training_mk3.h5")
    parser.add_argument("--structures-dir", default="./alphafold_benchmark")
    parser.add_argument("--benchmark-alignment", default="./foldtree2/notebooks/benchmarks/Information_benchmark/aa_supermatrix.fasta")
    parser.add_argument("--benchmark-tree", default="./foldtree2/notebooks/benchmarks/Information_benchmark/aa_astral_tree.nwk")
    parser.add_argument("--benchmark-sitelh", default=None)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--training-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-embeddings", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them")
    parser.add_argument("--extra-train-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--extra-submat-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--extra-phylo-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    return OptunaSearchConfig(
        study_name=args.study_name,
        output_dir=args.output_dir,
        dataset=args.dataset,
        structures_dir=args.structures_dir,
        benchmark_alignment=args.benchmark_alignment,
        benchmark_tree=args.benchmark_tree,
        benchmark_sitelh=args.benchmark_sitelh,
        trials=args.trials,
        training_epochs=args.training_epochs,
        batch_size=args.batch_size,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        seed=args.seed,
        sampler_seed=args.sampler_seed,
        storage=args.storage,
        device=args.device,
        dry_run=args.dry_run,
        extra_train_args=args.extra_train_args,
        extra_submat_args=args.extra_submat_args,
        extra_phylo_args=args.extra_phylo_args,
    )


def build_trial_command(cfg: OptunaSearchConfig, trial_number: int) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "foldtree2" / "learn_lightning.py"
    submat_script = repo_root / "foldtree2" / "makesubmat.py"
    phylo_script = repo_root / "scripts" / "phylogenetic_information_gain.py"

    trial_dir = Path(cfg.output_dir) / f"trial_{trial_number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(train_script),
        "--dataset", str(Path(cfg.dataset).expanduser().resolve()),
        "--output-dir", str(trial_dir / "models"),
        "--model-name", f"{cfg.study_name}_trial_{trial_number:04d}",
        "--epochs", str(cfg.training_epochs),
        "--batch-size", str(cfg.batch_size),
        "--num-embeddings", str(cfg.num_embeddings),
        "--embedding-dim", str(cfg.embedding_dim),
        "--hidden-size", str(cfg.hidden_size),
        "--seed", str(cfg.seed + trial_number),
        "--gpus", "0",
        "--mixed-precision",
    ]
    if cfg.device:
        cmd.extend(["--device", cfg.device])
    if cfg.extra_train_args:
        cmd.extend(cfg.extra_train_args)

    submat_cmd = [
        sys.executable,
        str(submat_script),
        "--modelname", f"{cfg.study_name}_trial_{trial_number:04d}",
        "--modeldir", str(trial_dir / "models"),
        "--datadir", str(Path(cfg.structures_dir).expanduser().resolve()),
        "--encode_alns",
        "--submat", str(trial_dir / "models" / f"{cfg.study_name}_trial_{trial_number:04d}_submat.txt"),
        "--mafftmat", str(trial_dir / "models" / f"{cfg.study_name}_trial_{trial_number:04d}_mafftmat.mtx"),
    ]
    if cfg.extra_submat_args:
        submat_cmd.extend(cfg.extra_submat_args)

    phylo_cmd = [
        sys.executable,
        str(phylo_script),
        "--spec", str(trial_dir / "phylo_spec.json"),
        "--outdir", str(trial_dir / "phylo_results"),
        "--no-run-raxml",
    ]
    if cfg.extra_phylo_args:
        phylo_cmd.extend(cfg.extra_phylo_args)

    return cmd + ["--__submat_cmd", json.dumps(submat_cmd)] + ["--__phylo_cmd", json.dumps(phylo_cmd)]


def make_phylospec(trial_dir: Path, cfg: OptunaSearchConfig) -> Path:
    spec_path = trial_dir / "phylo_spec.json"
    spec = {
        "datasets": [
            {
                "name": "AA",
                "kind": "aa",
                "alignment": str(Path(cfg.benchmark_alignment).expanduser().resolve()),
                "tree": str(Path(cfg.benchmark_tree).expanduser().resolve()),
                "model": "LG+G+I",
                "sitelh_file": str(Path(cfg.benchmark_sitelh).expanduser().resolve()) if cfg.benchmark_sitelh else None,
            },
            {
                "name": "FT2",
                "kind": "ft2",
                "alignment": str(trial_dir / "ft2_supermatrix.fasta"),
                "tree": str(trial_dir / "ft2_astral_tree.nwk"),
                "model": "MULTI20_GTR{...}+I",
                "sitelh_file": str(trial_dir / "ft2_supermatrix_site_likelihood.raxml.siteLH"),
            },
        ]
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec_path


def run_trial(cfg: OptunaSearchConfig, trial: optuna.Trial) -> float:
    trial_output_dir = Path(cfg.output_dir) / f"trial_{trial.number:04d}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)
    make_phylospec(trial_output_dir, cfg)

    if cfg.dry_run:
        print(f"Dry run for trial {trial.number}:")
        print("  train:", " ".join([str(Path(__file__).resolve().parents[1] / 'foldtree2' / 'learn_lightning.py')] + [
            "--dataset", str(Path(cfg.dataset).expanduser().resolve()),
            "--output-dir", str(trial_output_dir / "models"),
            "--model-name", f"{cfg.study_name}_trial_{trial.number:04d}",
            "--epochs", str(cfg.training_epochs),
            "--batch-size", str(cfg.batch_size),
            "--num-embeddings", str(cfg.num_embeddings),
            "--embedding-dim", str(cfg.embedding_dim),
            "--hidden-size", str(cfg.hidden_size),
            "--seed", str(cfg.seed + trial.number),
            "--gpus", "0",
            "--mixed-precision",
        ]))
        trial.set_user_attr("dry_run", True)
        return 0.0

    train_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "foldtree2" / "learn_lightning.py"),
        "--dataset", str(Path(cfg.dataset).expanduser().resolve()),
        "--output-dir", str(trial_output_dir / "models"),
        "--model-name", f"{cfg.study_name}_trial_{trial.number:04d}",
        "--epochs", str(cfg.training_epochs),
        "--batch-size", str(cfg.batch_size),
        "--num-embeddings", str(cfg.num_embeddings),
        "--embedding-dim", str(cfg.embedding_dim),
        "--hidden-size", str(cfg.hidden_size),
        "--seed", str(cfg.seed + trial.number),
        "--gpus", "0",
        "--mixed-precision",
    ]
    if cfg.device:
        train_cmd.extend(["--device", cfg.device])
    if cfg.extra_train_args:
        train_cmd.extend(cfg.extra_train_args)

    train_result = subprocess.run(train_cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
    if train_result.returncode != 0:
        raise RuntimeError(train_result.stderr[-4000:])

    submat_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "foldtree2" / "makesubmat.py"),
        "--modelname", f"{cfg.study_name}_trial_{trial.number:04d}",
        "--modeldir", str(trial_output_dir / "models"),
        "--datadir", str(Path(cfg.structures_dir).expanduser().resolve()),
        "--encode_alns",
        "--submat", str(trial_output_dir / "models" / f"{cfg.study_name}_trial_{trial.number:04d}_submat.txt"),
        "--mafftmat", str(trial_output_dir / "models" / f"{cfg.study_name}_trial_{trial.number:04d}_mafftmat.mtx"),
    ]
    if cfg.extra_submat_args:
        submat_cmd.extend(cfg.extra_submat_args)

    submat_result = subprocess.run(submat_cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
    if submat_result.returncode != 0:
        raise RuntimeError(submat_result.stderr[-4000:])

    phylo_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "phylogenetic_information_gain.py"),
        "--spec", str(trial_output_dir / "phylo_spec.json"),
        "--outdir", str(trial_output_dir / "phylo_results"),
        "--no-run-raxml",
    ]
    if cfg.extra_phylo_args:
        phylo_cmd.extend(cfg.extra_phylo_args)

    phylo_result = subprocess.run(phylo_cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
    if phylo_result.returncode != 0:
        raise RuntimeError(phylo_result.stderr[-4000:])

    summary_path = trial_output_dir / "phylo_results" / "phylo_info_summary.csv"
    if not summary_path.exists():
        raise RuntimeError("Phylogenetic analysis output not found")

    summary_df = __import__("pandas").read_csv(summary_path)
    aa_row = summary_df[summary_df["alphabet"] == "AA"].iloc[0]
    ft2_row = summary_df[summary_df["alphabet"] == "FT2"].iloc[0]
    score = float(ft2_row["phylo_gain_sum"] - aa_row["phylo_gain_sum"])
    trial.set_user_attr("summary_path", str(summary_path))
    trial.set_user_attr("score", score)
    return -score


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=cfg.sampler_seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, storage=cfg.storage, load_if_exists=True)

    if cfg.dry_run:
        for trial_idx in range(cfg.trials):
            trial = DryRunTrial(number=trial_idx)
            run_trial(cfg, trial)
        return

    study.optimize(lambda trial: run_trial(cfg, trial), n_trials=cfg.trials, catch=(RuntimeError,))

    best = study.best_trial
    with (output_dir / "best_trial.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_value": best.value,
                "best_params": best.params,
                "best_attrs": dict(best.user_attrs),
            },
            handle,
            indent=2,
        )
    print(json.dumps({"best_value": best.value, "best_params": best.params}, indent=2))


if __name__ == "__main__":
    main()
