# Losses and geometry conventions

[Source map](../README.md) | [Training](../../../docs/training.md) |
[Conversion notebook](../../notebooks/experiments/test_fapeloss.ipynb)

- [fape](fape.py): active lightweight quaternion/rotation conversion, local-frame
  FAPE, coordinate/CA objectives and lDDT helpers. Match the decoder's frame and
  coordinate representation; use validity masks for missing atoms/residues.
- [losses](losses.py): multi-task reconstruction, distograms, AA/secondary
  structure, contact and regularization objectives.
- [se3_ablation](se3_ablation.py): opt-in soft-FAPE/local-bond experiments;
  historical evaluation FAPE remains separate.
- [se3_balanced](se3_balanced.py): versioned SE(3) objectives; weights and geometric
  terms are protocol choices, not interchangeable evaluation scores.
- [openfold_loss](openfold_loss.py): optional upstream-style research code requiring
  OpenFold and ml_collections; not imported by the base tree workflow.

Do not compare clamped evaluation FAPE, soft training FAPE and normalized
distance objectives as if they had identical units or gradient behavior.
Check finite values/gradients and the exact masked reduction, including empty
masks and missing targets. Preserve the loss version, weights and coordinate
scale in experiment provenance. See [SE(3) validation](../../../docs/se3_validation.md)
and its dated follow-ups before making reconstruction-quality claims.
