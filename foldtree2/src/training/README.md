# Training helpers

Splits, distributed training, optimizers and diagnostics for the `learn_*` pipelines.

[Source map](../README.md) | [Project](../../../README.md) |
[Training guide](../../../docs/training.md) |
[Lightning notebook](../../notebooks/experiments/test_lightning.ipynb)

- [training_protocol](./training_protocol.py): deterministic held-out splits
  (`held_out_split`). Use this for reproducible train/validation partitions.
- [ddp_training](./ddp_training.py): DDP training wrapper. No active importers;
  kept for reference. For multi-GPU Lightning runs use the CUDA accelerator with
  the requested device count and a DDP strategy.
- [lion_optimizer](./lion_optimizer.py): Lion optimizer; optional dependency for
  selected `learn_*` runs.
- [visualization](./visualization.py): plots reconstructions and metrics.
  Diagnostics only; figures are not held-out validation.

## When to use this

Use `training_protocol` for deterministic, reproducible splits across
experiments; record the split state in checkpoint provenance. `ddp_training` is
retained for reference but not currently imported — drive multi-GPU runs from the
Lightning entry points instead. `lion_optimizer` is opt-in; `visualization`
produces diagnostic figures, not validation metrics. Entry points are the
top-level `learn_*` scripts, e.g.
[learn_lightning](../../learn_lightning.py),
[learn_geometry_lightning](../../learn_geometry_lightning.py) and
[learn_monodecoder](../../learn_monodecoder.py); interactive runs start from the
[lightning notebook](../../notebooks/experiments/test_lightning.ipynb). See the
[training guide](../../../docs/training.md).