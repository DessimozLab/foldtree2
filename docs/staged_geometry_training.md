# Staged Geometry Training

## Training path

`foldtree2/learn_production_staged_transformer_geometry.py` trains a
`StagedTransformerRefiner` from frozen production outputs. The production
encoder supplies the local latent and discrete codebook vectors. The frozen
geometry decoder supplies the three-dimensional bottleneck used both as the
initial coordinate sketch and for the dot-product contact graph. The staged
refiner predicts three successive coordinate/frame updates.

The encoder and production geometry decoder are forced to evaluation mode and
run under `torch.no_grad()`. Only the staged refiner is passed to AdamW. The
trainer prints the trainable/frozen parameter counts and logs the staged
gradient norm so a run cannot silently optimize nothing.

## Loss stack

The staged loss stack can be configured from the command line or a YAML/JSON
config. The validated full stack used for the current experiments is:

- staged CA, quaternion, and angle losses at all three refinement stages
- coarse CA loss
- coarse backbone frame loss
- coarse backbone atom loss
- coarse C, CB, and N losses
- coarse backbone FAPE loss
- coarse backbone angle loss

The original decoder frame FAPE, decoder quaternion-geodesic, and decoder
angle losses are disabled in staged-only runs. The production encoder and
geometry decoder remain frozen.

## Local checks

The following checks were run with the production 40-character encoder and
geometry decoder pair:

| Refiner | Structures | Epochs | Train loss | Validation loss |
| --- | ---: | ---: | ---: | ---: |
| hidden 32, 2 layers, 2 heads | 128 | 10 | about 100 to 44 | about 89 to 44 |
| hidden 32, 2 layers, 2 heads | 32 | 30 | about 100 to 43 | about 89 to 44 |
| hidden 16, 2 layers, 2 heads | 128 | 2 | 91.6 to 59.7 | 73.7 to 49.6 |

All completed checks had finite losses and nonzero staged gradients. The
hidden-16 run has its last checkpoint at
`/tmp/foldtree2_staged_learning_128_hidden16/last.ckpt`.

The attempted 2,048-structure workstation run did not complete because the
local execution environment terminated the long-running process. Its partial
loss trace still showed active gradients and decreasing coarse C/CB/N losses.
This is why the several-thousand-structure run should be executed on Alps.

## Scaling notes

Freezing removes activation storage and backward compute, but it does not
remove the encoder and geometry decoder forward passes. At scale, the next
performance improvement is to precompute and cache the frozen latent vectors,
codebook vectors, decoder bottleneck, and contact inputs. Until that cache is
implemented, use the scaled Alps launcher and monitor data-loader and forward
throughput separately from refiner compute.

For direct Alps submission:

```bash
sbatch alps/train_production_staged_transformer_geometry_scaled.sh
```

The launcher requests one Slurm task per GPU (`ntasks-per-node=4`) and uses
Lightning DDP with the complete validation split. It defaults to the
production 40-character model pair and the `bottleneck` coordinate source.
