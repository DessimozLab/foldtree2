# Graph conversion and encoded FASTA

[CLI index](README.md) | [Source map](../../foldtree2/src/README.md) |
[Graph notebook](../../foldtree2/notebooks/monomer_graph_trainingdata.ipynb)

## pdbs-to-graphs

```bash
pdbs-to-graphs /path/to/pdbs /path/to/training.h5 --verbose
```

Uses [pdbgraphmk2](../../foldtree2/src/graphs/pdbgraphmk2.py), not the legacy ProDy
converter. The properties CSV is packaged and resolved independently of the
working directory. Consult `pdbs-to-graphs --help` for multiprocessing controls.
Inspect conversion errors and populated HDF5 groups; a nonempty file alone does
not establish graph coverage. Check schema/feature compatibility with the model,
particularly when reusing historical HDF5 datasets. Do not regenerate running
training jobs' datasets in place.

## foldcomp2fasta

```bash
foldcomp2fasta /path/to/encoder.pt /path/to/foldcomp_db \
  /path/to/encoded.fasta --device cpu --max-structures 10
```

The database argument is its basename, without `.lookup`. Remove the smoke limit
only for a deliberately scoped full run. Optional `--ids-file` selects IDs.
This streams Foldcomp structures through graph conversion and the trained encoder;
it does not align them or infer a tree. Output symbols are structural tokens,
**not amino acids**. Keep the encoder's exact character replacement convention.

See the [Foldcomp notebook](../../foldtree2/notebooks/experiments/test_foldcompconversion.ipynb)
for chunk/batch experiments and the
[representation guide](../representation_conversion_guide.md) for geometry
conversions. Both routes need a trusted checkpoint with matching graph features.
