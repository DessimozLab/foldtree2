# Geometry experiment tracking

Open the workstation UI at **http://127.0.0.1:5000**.

Direct training-run tables (explicitly select MLflow 3's model-training view):

- [SE(3) runs](http://localhost:5000/#/experiments/4/runs?workflowType=machine-learning)
- [Transformer geometry runs](http://localhost:5000/#/experiments/5/runs?workflowType=machine-learning)

Two MLflow experiments separate the model families:

- `foldtree2/se3`: cache-only SE(3) Lightning training, revised pilots, and downloaded cluster learning curves.
- `foldtree2/transformer_geometry`: distance/contact heads, the spherical contact objective screen, and coordinate-transformer TensorBoard runs. TensorBoard runs explicitly configured with `use_se3` or `train_se3_only` are assigned to SE(3).

Run names are the original relative log directory paths. Filter `tags.foldtree2.stage` to hide `smoke` runs. `foldtree2.source` links each MLflow run to its original local records.

## Start, stop, and import

From the repository root:

```bash
.venv-mlflow/bin/python -m scripts.geometry_mlflow start
.venv-mlflow/bin/python -m scripts.geometry_mlflow status
.venv-mlflow/bin/python -m scripts.geometry_mlflow stop
```

The workstation already has an MLflow service on port **5000**. `start` reuses it and launches an importer that polls every 30 seconds. `stop` stops the importer; it does not stop the existing MLflow service or affect its other experiments. If no service is listening, `start` can launch a project-owned MLflow server on localhost. The importer runs independently of the terminal; it does not automatically restart after a workstation reboot. Logs and PID files are in `.mlflow/`.

### Access using the workstation IP

The runs page also works at [172.26.141.223:5000](http://172.26.141.223:5000/#/experiments/5/runs?startTime=ALL&workflowType=machine_learning&compareRunsMode=TABLE).
MLflow 3 requires that browser origin to be explicitly allowed. Without it, the page loads but its POST requests return `403 Cross-origin request blocked`, displayed by the UI as a request/`INTERNAL_ERROR` message.

The shared service has been restarted with its original command, working directory, and environment, plus:

```bash
export MLFLOW_SERVER_CORS_ALLOWED_ORIGINS=http://172.26.141.223:5000
```

Keep this environment variable when manually restarting the shared MLflow server. The original service starts from `/home/dmoi` with `/home/dmoi/miniforge3/bin/mlflow server --port 5000`; its working directory selects the existing tracking database. The recovered launch arguments and allowed origin are recorded in `.mlflow/shared-server-launch.json`, and its background output is in `.mlflow/shared-server.log`. The loopback address and existing workstation-IP forwarding remain in place. Requests from localhost continue to work.

For a one-time import while the watcher is stopped:

```bash
.venv-mlflow/bin/python -m scripts.geometry_mlflow sync
```

An exclusive lock prevents concurrent importers. Check `.mlflow/sync-errors.json` and `.mlflow/watcher.log` for individual import failures. Partial JSON writes are retried on the next poll. A trailing incomplete JSONL event is left pending until its line is complete.

## What gets tracked

The importer reads `history.json`, `events.jsonl`, and TensorBoard scalar files from existing geometry run directories. It also imports `learning_curve` arrays from saved cluster summaries under `docs/`. It skips frozen caches, copied source directories, and checkpoints.

It logs configuration parameters, training/validation curves, final and best-checkpoint summary metrics where available, and JSON/YAML/Markdown/PNG diagnostics up to 5 MB each. Nonfinite scalar metrics are omitted. Credential-named fields are removed from configurations and structured artifacts. Model weights and dataset caches stay in their original locations.

For the contact objective screen, compare `objective/variant`, `objective/contact_weight`, and `model/spherical_dim`. These are the actual per-case settings; the shared command-line arguments also appear separately. Legacy YAML object tags are represented by descriptive placeholders, never instantiated.

Each directory maps to one MLflow run, persisted in `.mlflow/sync-state-<server-hash>.json`. State is separate for each tracking server. Restarting tracking resumes incremental imports. The original source directory is also stored in an MLflow tag for recovery. TensorBoard metrics keep their original names and steps. History-based validation metrics use `val/`; event-based SE(3) metrics use optimizer updates when the original records provide them. Source configurations and provenance remain available as artifacts.

Explicit completion and failure records set `FINISHED` and `FAILED`. A log with no terminal record and no activity for 24 hours is marked `KILLED`, with `foldtree2.status_inferred=true`. This is an inactivity inference, not evidence that training crashed. New training events reactivate the same run. Do not reuse a source directory for an unrelated experiment: configurations are immutable parameters within a run.

No training code or loss is changed. This preserves the source-hash checks used by resumable SE(3) runs. Existing scripts automatically appear as they write their normal logs. For cluster training, copy its result logs into the existing `runs/` or evidence directories on this workstation; the importer cannot see logs that only exist remotely.

## Storage and installation

The importer connects to `http://127.0.0.1:5000` by default. Run metadata and artifacts go to the existing server's configured stores, alongside its other experiments. Back up that server's stores as well as `.mlflow/` (import cursors). If the server is unavailable, imports retry on subsequent polls and the original training logs remain available. Override the destination with `MLFLOW_TRACKING_URI` when needed.

The initial setup also created a project-local SQLite backfill under `.mlflow/tracking.db`; the active configuration uses the existing HTTP server on port 5000. The local backfill is retained as a redundant copy. Both `.mlflow/` and `.venv-mlflow/` are git-ignored.

The tracking environment is separate from the `foldtree2` CUDA conda environment. To recreate it with Python 3.10 or newer:

```bash
python3 -m venv .venv-mlflow
.venv-mlflow/bin/python -m pip install -r configs/requirements-mlflow.txt
.venv-mlflow/bin/python -m unittest foldtree2.tests.test_geometry_mlflow
```

MLflow is pinned to 2.22.5 for this setup. The explicit SQLite/artifact settings avoid relying on version-dependent defaults. See the [MLflow tracking server documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) for the storage architecture.
