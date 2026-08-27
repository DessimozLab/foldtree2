# NeMo-Run Alps launcher

`launch_production_geometry_se3_nemo.py` submits the existing production
Lightning job through NeMo-Run's `SlurmExecutor`. NeMo-Run provides the
experiment/submission layer; the PyTorch-Geometric SE3 model and Lightning
loss stack remain in `foldtree2/learn_production_geometry_se3_lightning.py`.

Install NeMo-Run in the login/submission environment, not necessarily in the
training environment:

```bash
pip install nemo-run
```

Submit from an Alps login node:

```bash
python alps/launch_production_geometry_se3_nemo.py \
  --project-root /users/dmoi/foldtree2 \
  --account a0117 \
  --gpus-per-node 4 \
  --time 08:00:00 \
  --slurm-environment pygmk3 \
  --env DATASET=/capstor/store/cscs/swissai/a0117/structalnfinal.h5 \
  --env EPOCHS=100 \
  --env BATCH_SIZE=1
```

Use `--print-only` to inspect the resolved command and resources without
submitting. Job-specific settings can be supplied repeatedly with `--env
KEY=VALUE`; these become environment variables consumed by the existing Alps
launcher. The launcher requests one Slurm task with all GPUs on the node,
matching the current Lightning configuration. The same `--job-dir` is also the
NeMo-Run metadata home, so set it when inspecting the experiment later:

```bash
NEMORUN_HOME=/capstor/store/cscs/swissai/a0117/nemo-run \
  nemo experiment logs ft2-prod-se3_1787859357
```

Replace the experiment ID with the one printed by the submit command.
Alternatively, inspect the Slurm output files below
`/capstor/store/cscs/swissai/a0117/nemo-run/experiments/ft2-prod-se3/`.


python alps/launch_production_geometry_se3_nemo.py \
  --project-root /users/dmoi/foldtree2 \
  --account a0117 \
  --gpus-per-node 4 \
  --time 08:00:00 \
  --env DATASET=/capstor/store/cscs/swissai/a0117/structalnfinal.h5 \
  --env EPOCHS=100 \
  --env BATCH_SIZE=1 \
  --print-only
