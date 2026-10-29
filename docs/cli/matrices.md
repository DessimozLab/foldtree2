# makesubmat: model-specific substitution matrices

[CLI index](README.md) | [Training and matrix protocol](../training.md#matrices-from-prepared-references) |
[Manual notebook](../../foldtree2/notebooks/experiments/test_makesubmat_manual.ipynb) |
[Implementation](../../foldtree2/makesubmat.py)

Required inputs are a trained encoder, compatible reference graphs and structural
reference alignments with matching accessions. `--modelname` is the full encoder
filename stem, not an arbitrary label. See the training guide for the reference
directory layout and a complete invocation:

```bash
makesubmat --help
```

`--download_structs`, `--convert_to_pyg`, and `--align_structs` are separate,
potentially expensive preparation stages. Foldseek is required for creating
reference alignments, not for tree inference with an existing bundle.

Outputs include `_mafftmat.mtx`, `_submat.txt`, `_pair_counts.pkl` and optional
history/plots. Build in staging and validate dimensions, state order, pair counts,
coverage and convergence before promotion. A failed gate can leave diagnostic
files; their existence is not acceptance.

The [production protocol](../production_alphabet_readiness.md) uses sustained
EMA convergence, a minimum evidence count and an explicit FT2-50 exception.
Preserve exception sidecars and hashes in every downstream result; do not
reinterpret a nonconverged matrix as converged.
