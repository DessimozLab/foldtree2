# Structure-token models

Neural architectures that encode protein structures into learned discrete tokens
and predict geometry from them.

[Source map](../README.md) | [Project](../../../README.md) |
[Training guide](../../../docs/training.md) |
[SE(3) validation](../../../docs/se3_validation.md)

- [encoder](./encoder.py): `mk1_Encoder`, the SE(3)-equivariant encoder, plus
  feature and FASTA loaders. The main entry point for continuous embeddings and
  discrete code indices.
- [quantizers](./quantizers.py): VQ codebooks and training penalties producing the
  discrete structure tokens.
- [layers](./layers.py): core neural building blocks with graph/channel/mask
  contracts shared across encoders and decoders.
- [chebconv](./chebconv.py): `StableChebConv` spectral graph convolution; the graph
  operator used by the encoder.
- [xsatransformer](./xsatransformer.py): cross-structure-attention (XSA)
  transformer blocks.
- [dynamictan](./dynamictan.py): experimental normalization architecture; choices
  belong in checkpoint provenance.
- [manifold_hyper_connections](./manifold_hyper_connections.py): experimental
  residual/connection architecture; choices belong in checkpoint provenance.
- [rigid_utils](./rigid_utils.py): quaternion and rotation algebra; preserve frame
  direction and quaternion convention across components.
- [mono_decoders](./mono_decoders.py): `MultiMonoDecoder` family — AA, contact,
  angle and geometry heads and the multi-task composition. The main
  sequence-and-geometry decoder.
- [folding_refiner](./folding_refiner.py): quaternion-frame iterative folding
  refinement on top of decoder geometry.
- [geometry_seeds](./geometry_seeds.py): coordinate seeds that initialize the
  refinement stack.
- [se3_struct_decoder](./se3_struct_decoder.py): SE(3) denoiser/refiner stack.
  Experimental; pairs with the versioned SE(3) loss protocols.
- [sidechain_refiner](./sidechain_refiner.py): experimental masked side-chain
  prediction. Not validated; not a production folding system.
- [decoder_feature_fusion](./decoder_feature_fusion.py): fuses decoder features.
  Imports top-level validation/training modules, so it couples to the training
  entry points rather than the base install.
- [pairwise](./pairwise.py): complex/pairwise models; needs optional `faiss`. Not
  part of the base inference install. See the [complex notebooks](../../notebooks/complexes/README.md).
- [test_chebconv_rework](./test_chebconv_rework.py): developer scratch/regression.
  Excluded from installed builds; do not import from package code.

## When to use this

Start from `encoder` for representation learning and `mono_decoders` for
sequence-and-geometry reconstruction. Refinement (`folding_refiner`,
`geometry_seeds`, `se3_struct_decoder`) layers on top; `sidechain_refiner` is
experimental and unvalidated. Entry points are the top-level
[learn_lightning](../../learn_lightning.py),
[learn_geometry_lightning](../../learn_geometry_lightning.py),
[train_se3_lightning](../../train_se3_lightning.py) and
[train_se3_decoder_fusion](../../train_se3_decoder_fusion.py) scripts.
Interactive starting points live under
[notebooks/experiments](../../notebooks/experiments/README.md):
[monodecoder training](../../notebooks/experiments/test_monodecoders.ipynb),
[configured training](../../notebooks/experiments/test_monodecoders_config.ipynb),
[folding exploration](<../../notebooks/experiments/test_monodecoders _folding.ipynb>),
[frozen refiner](../../notebooks/experiments/train_frozen_encoder_geometry_refiner.ipynb),
[geometry-learning inspection](../../notebooks/experiments/inspect_frozen_encoder_geometry_learning.ipynb),
[lightning experiments](../../notebooks/experiments/test_lightning.ipynb) and
[embedding inspection](../../notebooks/experiments/test_embedding.ipynb). See
[SE(3) validation](../../../docs/se3_validation.md) and
[staged geometry training](../../../docs/staged_geometry_training.md) before
making reconstruction-quality claims.