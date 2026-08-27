#!/usr/bin/env python3
"""Submit the existing production Lightning job through NeMo-Run.

NeMo-Run owns Slurm submission and experiment metadata here; the training
process remains the repository's tested Lightning entry point. This keeps the
PyTorch-Geometric SE3 model independent of Megatron model-parallel APIs while
allowing the job to be launched like other NeMo experiments.
"""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    env = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE for --env, got {value!r}")
        key, item = value.split("=", 1)
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid environment variable name: {key!r}")
        env[key] = item
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit the production SE3 Lightning launcher with NeMo-Run"
    )
    parser.add_argument("--name", default="ft2-prod-se3", help="NeMo-Run experiment name")
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "/users/dmoi/foldtree2/"))
    parser.add_argument("--script", default="alps/train_production_geometry_se3_lightning.sh")
    parser.add_argument("--account", default=os.environ.get("SLURM_ACCOUNT", "a0117"))
    parser.add_argument("--partition", default=os.environ.get("SLURM_PARTITION"))
    parser.add_argument("--nodes", type=int, default=int(os.environ.get("NODES", "1")))
    parser.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("GPUS_PER_NODE", "4")))
    parser.add_argument("--time", default=os.environ.get("WALLTIME", "08:00:00"))
    parser.add_argument(
        "--job-dir",
        default=os.environ.get("NEMO_RUN_JOB_DIR", "/capstor/store/cscs/swissai/a0117/nemo-run"),
    )
    parser.add_argument("--detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the resolved submission settings without importing or submitting NeMo-Run",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override passed to the Slurm job; repeatable",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).expanduser()
    script = Path(args.script)
    if not script.is_absolute():
        script = project_root / script

    env = {
        "PROJECT_ROOT": str(project_root),
        # The existing launcher owns the four-GPU Lightning process.
        "DEVICES": str(args.gpus_per_node),
        "STRATEGY": os.environ.get("STRATEGY", "auto"),
        "PYTHONUNBUFFERED": "1",
    }
    env.update(parse_env_overrides(args.env))

    command = f"cd {shlex.quote(str(project_root))} && bash {shlex.quote(str(script))}"
    print("NeMo-Run submission")
    print(f"  experiment: {args.name}")
    print(f"  command: {command}")
    print(f"  nodes: {args.nodes}")
    print(f"  gpus_per_node: {args.gpus_per_node}")
    print(f"  account: {args.account}")
    print(f"  partition: {args.partition or '<cluster default>'}")
    print(f"  time: {args.time}")
    print(f"  env: {env}")

    if args.print_only:
        return 0

    try:
        import nemo_run as run
    except ImportError as exc:
        raise SystemExit(
            "NeMo-Run is required for this launcher. Install it in the submission "
            "environment with `pip install nemo-run`."
        ) from exc

    executor_kwargs = {
        "account": args.account,
        "nodes": args.nodes,
        "ntasks_per_node": 1,
        "gpus_per_node": args.gpus_per_node,
        "time": args.time,
        "job_dir": args.job_dir,
        "tunnel": run.LocalTunnel(job_dir=args.job_dir),
        "env_vars": env,
    }
    if args.partition:
        executor_kwargs["partition"] = args.partition

    executor = run.SlurmExecutor(**executor_kwargs)
    task = run.Script(command)
    with run.Experiment(args.name) as experiment:
        experiment.add(task, executor=executor, name="training")
        experiment.run(detach=args.detach)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
