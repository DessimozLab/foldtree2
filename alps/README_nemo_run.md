# NeMo-Run Alps launcher

`launch_production_geometry_se3_nemo.py` submits the existing production
Lightning job through NeMo-Run's `SlurmExecutor`. NeMo-Run provides the
experiment/submission layer; the PyTorch-Geometric SE3 model and Lightning
loss stack remain in `foldtree2/learn_production_geometry_se3_lightning.py`.

For the staged geometry refiner, submit
`alps/train_production_staged_transformer_geometry_scaled.sh` directly with
`sbatch`. It requests four GH200 GPUs and four Slurm tasks, which is required
when Lightning is configured with `--devices 4`. Do not use a NeMo executor
with `ntasks-per-node=1` for this script.

## Staged geometry run

The staged trainer freezes the production encoder and geometry decoder and
optimizes only the staged refiner. Frozen outputs are evaluated under
`torch.no_grad()`, but their forward passes are still repeated for each
structure. The complete staged loss stack is enabled by the Alps script:
stage coordinate, quaternion, angle, coarse CA, coarse backbone, C, CB, N,
backbone FAPE, and backbone angle losses. Validation runs over the complete
validation split unless `LIMIT_VAL_BATCHES` is set.

Launch the scaled run with:

```bash
sbatch alps/train_production_staged_transformer_geometry_scaled.sh
```

Useful overrides are supplied as environment variables, for example:

```bash
sbatch --export=ALL,DATASET=/capstor/store/cscs/swissai/a0117/structalnfinal.h5,EPOCHS=100,BATCH_SIZE=1,TARGET_EFFECTIVE_BATCH_SIZE=64 alps/train_production_staged_transformer_geometry_scaled.sh
```

The script writes Slurm output and a timestamped run log, and checkpoints to
`/capstor/store/cscs/swissai/a0117/chkpts/results/geometry/` by default.

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
