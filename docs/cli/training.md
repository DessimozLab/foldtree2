# Training and Bayesian tuning

[CLI index](README.md) | [Training guide](../training.md) |
[Training notebooks](../../foldtree2/notebooks/experiments/README.md) |
[Encoder and decoder modules](../../foldtree2/src/README.md)

Use [training.md](../training.md) for graph prerequisites, a minimal invocation,
the held-out split and matrix preparation. Main alphabet trainers are:

```bash
python -m foldtree2.learn_monodecoder --help
python -m foldtree2.learn_lightning --help
tune-monodecoder-bayes --help
```

The first is the original trainer; Lightning supports distributed execution.
Their device/resume/optimizer flags differ: do not copy arguments between them
without checking help. Training is unnecessary for users with a complete bundle.

Bayesian tuning requires an HDF5 dataset and uses Optuna to launch bounded
`learn_monodecoder` trials:

```bash
tune-monodecoder-bayes --dataset /path/to/training.h5 \
  --output-dir models/tuning --trials 2 --epochs 1 --device cpu
```

This is an interface smoke workload, not a recommended training budget. Fix
dataset versions, model/sampler seeds, held-out proteins and hardware in reports.
Keep tuning decisions separate from final held-out evaluation.

Geometry/refiner trainers are research workflows, not equivalent alphabet
training recipes. Start with [staged geometry](../staged_geometry_training.md)
and [SE(3) validation](../se3_validation.md); optional GotenNet/OpenFold dependencies
and cluster environments are not supplied by the base tree-inference package.
