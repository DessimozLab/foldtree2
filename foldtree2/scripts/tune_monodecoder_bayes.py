import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

import optuna
from optuna.trial import TrialState


def ensure_storage_parent(storage: Optional[str]) -> None:
    if not storage or not storage.startswith("sqlite:///"):
        return

    parsed = urlparse(storage)
    raw_path = unquote(parsed.path)
    # SQLAlchemy semantics:
    #   sqlite:///relative/path.db  -> relative to CWD
    #   sqlite:////abs/path.db      -> absolute path
    if parsed.netloc:
        db_path = Path(f"/{parsed.netloc}{raw_path}")
    elif raw_path.startswith("//"):
        db_path = Path(raw_path[1:])
    elif raw_path.startswith("/"):
        db_path = Path.cwd() / raw_path.lstrip("/")
    else:
        db_path = Path.cwd() / raw_path
    db_path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter tuning for learn_monodecoder.py using Optuna."
    )
    parser.add_argument("--dataset", required=True, help="Path to the HDF5 training dataset.")
    parser.add_argument("--output-dir", default="./models/tuning", help="Directory for tuning artifacts.")
    parser.add_argument("--study-name", default="monodecoder_bayes", help="Optuna study name.")
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URL, e.g. sqlite:///study.db")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per trial.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per trial.")
    parser.add_argument("--device", default=None, help="Training device, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for search reproducibility.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split passed to trainer.")
    parser.add_argument("--objective-aa-coef", type=float, default=0.5, help="AA loss coefficient in scalar objective.")
    parser.add_argument("--objective-edge-coef", type=float, default=0.5, help="Contact loss coefficient in scalar objective.")
    parser.add_argument("--objective-vq-coef", type=float, default=0.0, help="Optional VQ coefficient in scalar objective.")
    parser.add_argument("--final-val-samples", type=int, default=0, help="0 uses full validation export, >0 uses quick validation subset.")
    parser.add_argument("--max-residues", type=int, default=None, help="Optional max residue filter passed through.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Base learning rate for all trials.")
    parser.add_argument("--num-embeddings", type=int, default=None, help="Number of codebook embeddings.")

    parser.add_argument("--embedding-dim", type=int, default=128, help="Encoder embedding dimension.")
    parser.add_argument("--sampler-seed", type=int, default=42, help="Seed for the Bayesian sampler.")
    parser.add_argument("--pruner-startup-trials", type=int, default=5, help="Startup trials before pruning kicks in.")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[], help="Extra args forwarded to learn_monodecoder.py after '--'.")
    return parser.parse_args()


def build_trial_command(args: argparse.Namespace, trial: optuna.Trial, metrics_path: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    trainer_path = repo_root / "foldtree2" / "learn_monodecoder.py"

    if args.num_embeddings is not None:
        encoder_nembeddings = args.num_embeddings
    else:
        encoder_nembeddings = trial.suggest_categorical("encoder_nembeddings", [10,15,20,25,30,35,40])

    encoder_hidden = trial.suggest_categorical("encoder_hidden_size", [96, 128, 160, 192, 256, 320])
    # Keep d_model divisible by nheads=10 used by the sequence transformer.
    sequence_hidden = trial.suggest_categorical("sequence_hidden_size", [160, 320])
    aa_hidden = trial.suggest_categorical("aa_decoder_hidden_size", [64, 96, 128, 160, 192, 256, 320])
    geometry_hidden = trial.suggest_categorical("geometry_cnn_hidden_size", [96, 128, 160, 192, 256, 320])
    base_hidden = trial.suggest_categorical("hidden_size", [128, 160, 192, 256, 320])

    edgeweight = trial.suggest_float("edgeweight", 0.02, 2.0, log=True)
    xweight = trial.suggest_float("xweight", 0.02, 2.0, log=True)
    vqweight = trial.suggest_float("vqweight", 1e-4, 0.2, log=True)

    aa_dropout = trial.suggest_float("aa_decoder_dropout", 0.0, 0.20)
    geometry_dropout = trial.suggest_float("geometry_cnn_dropout", 0.0, 0.20)
    grad_accum = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])



    trial_output_dir = Path(args.output_dir) / f"trial_{trial.number:04d}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(trainer_path),
        "--dataset", args.dataset,
        "--output-dir", str(trial_output_dir),
        "--model-name", f"{args.study_name}_trial_{trial.number:04d}",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--val-split", str(args.val_split),
        "--seed", str(args.seed + trial.number),
        "--val-seed", str(args.seed + trial.number),
        "--hidden-size", str(base_hidden),
        "--encoder-hidden-size", str(encoder_hidden),
        "--sequence-hidden-size", str(sequence_hidden),
        "--aa-decoder-hidden-size", str(aa_hidden),
        "--geometry-cnn-hidden-size", str(geometry_hidden),
        "--aa-decoder-dropout", str(aa_dropout),
        "--geometry-cnn-dropout", str(geometry_dropout),
        "--edgeweight", str(edgeweight),
        "--xweight", str(xweight),
        "--vqweight", str(vqweight),
        "--logitweight", "0.0",
        "--gradient-accumulation-steps", str(grad_accum),
        "--num-embeddings", str(encoder_nembeddings),
        "--embedding-dim", str(args.embedding_dim),
        "--metrics-output", str(metrics_path),
        "--final-val-samples", str(args.final_val_samples),
        "--no-use-geometry-transformer",
        "--no-geometry-output-ss",
        "--no-geometry-output-angles",
        "--no-geometry-cnn-output-edge-logits",
        "--sequence-use-cnn-decoder",
        "--use-geometry-cnn",
    ]

    if args.device:
        cmd.extend(["--device", args.device])
    if args.max_residues is not None:
        cmd.extend(["--max-residues", str(args.max_residues)])
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def objective_factory(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        metrics_path = output_dir / f"trial_{trial.number:04d}_metrics.json"
        cmd = build_trial_command(args, trial, metrics_path)

        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            trial.set_user_attr("stdout_tail", result.stdout[-4000:])
            trial.set_user_attr("stderr_tail", result.stderr[-4000:])
            raise RuntimeError(f"Trial {trial.number} failed with exit code {result.returncode}")

        if not metrics_path.exists():
            trial.set_user_attr("stdout_tail", result.stdout[-4000:])
            raise RuntimeError(f"Trial {trial.number} completed without metrics output")

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        val = metrics["validation"]
        aa_loss = float(val["aa_loss"])
        edge_loss = float(val["edge_loss"])
        vq_loss = float(val["vq_loss"])
        objective_value = (
            args.objective_aa_coef * aa_loss
            + args.objective_edge_coef * edge_loss
            + args.objective_vq_coef * vq_loss
        )

        trial.set_user_attr("validation_metrics", val)
        trial.set_user_attr("metrics_path", str(metrics_path))
        trial.set_user_attr("train_metrics", metrics.get("train", {}))
        trial.set_user_attr("objective_value", objective_value)
        trial.set_user_attr("stdout_tail", result.stdout[-4000:])
        return objective_value

    return objective


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_storage_parent(args.storage)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=args.pruner_startup_trials)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective_factory(args), n_trials=args.trials, catch=(RuntimeError,))

    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed_trials:
        summary = {
            "study_name": study.study_name,
            "best_value": None,
            "best_params": {},
            "best_validation_metrics": {},
            "best_metrics_path": None,
            "n_trials": len(study.trials),
            "n_completed_trials": 0,
            "n_failed_trials": sum(1 for trial in study.trials if trial.state == TrialState.FAIL),
        }
        summary_path = output_dir / f"{args.study_name}_best.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"No completed trials. Summary written to {summary_path}")
        return

    best = study.best_trial
    summary = {
        "study_name": study.study_name,
        "best_value": best.value,
        "best_params": best.params,
        "best_validation_metrics": best.user_attrs.get("validation_metrics", {}),
        "best_metrics_path": best.user_attrs.get("metrics_path"),
        "n_trials": len(study.trials),
    }

    summary_path = output_dir / f"{args.study_name}_best.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Best-trial summary written to {summary_path}")


if __name__ == "__main__":
    main()
