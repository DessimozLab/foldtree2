# Source module map

[Project](../../README.md) | [CLI guides](../../docs/cli/README.md) |
[Notebook catalogue](../notebooks/README.md) | [Experiments](../../docs/experiments.md)

## Follow the structure-to-tree path

1. [ft2treebuilder](../ft2treebuilder.py) orchestrates preprocessing, encoding,
   alignment, inference and optional rooting/ancestral analysis.
2. [pdbgraphmk2](pdbgraphmk2.py) converts PDB/Foldcomp structures to heterogeneous
   residue graphs; [config_paths](config_paths.py) resolves packaged properties.
3. [encoder](encoder.py) builds continuous embeddings and discrete code indices;
   [quantizers](quantizers.py) implements codebooks and training penalties.
4. [makesubmat](../makesubmat.py) estimates model-specific matrices from reference
   alignments. [Native wrappers](../mafft_wrappers.py) invoke packaged helpers.
5. [ancestral](ancestral.py) shares native marginal reconstruction/FASTA export
   between the tree CLI and benchmark scripts.

Graph schemas, codebooks, maps and matrices form one compatibility boundary.
Equal feature widths or alphabet sizes alone do not establish compatibility.
Preserve trusted bundle hashes and exact state order. Never strip arbitrary
whitespace from structural-token FASTA.

## Model and geometry modules

| Modules | Contract / role | Interactive starting point |
| --- | --- | --- |
| [mono_decoders](mono_decoders.py) | AA/contact/angle/geometry heads and multi-task composition; consume named geometry representations. | [Alphabet training](../notebooks/experiments/test_monodecoders.ipynb) |
| [layers](layers.py), [chebconv](chebconv.py), [xsatransformer](xsatransformer.py) | Neural building blocks with graph/channel/mask contracts. | [Configured training](../notebooks/experiments/test_monodecoders_config.ipynb) |
| [losses](losses/README.md) | Geometry and reconstruction objectives, units/masks and optional upstream losses. | [Conversion/FAPE checks](../notebooks/experiments/test_fapeloss.ipynb) |
| [rigid_utils](rigid_utils.py), [folding_refiner](folding_refiner.py) | Quaternion/frame algebra and iterative folding; preserve frame direction and quaternion convention. | [Geometry exploration](<../notebooks/experiments/test_monodecoders _folding.ipynb>) |
| [se3_struct_decoder](se3_struct_decoder.py), [geometry_seeds](geometry_seeds.py) | Experimental GotenNet/staged refiners and coordinate seeds; optional dependencies and versioned loss protocols. | [Frozen refiner](../notebooks/experiments/train_frozen_encoder_geometry_refiner.ipynb), [validation](../../docs/se3_validation.md) |
| [sidechain_refiner](sidechain_refiner.py) | Experimental masked side-chain predictions, not a validated production folding system. | [Geometry experiments](../../docs/staged_geometry_training.md) |
| [dynamictan](dynamictan.py), [manifold_hyper_connections](manifold_hyper_connections.py) | Experimental normalization/residual architectures; choices belong in checkpoint provenance. | [Geometry learning inspection](../notebooks/experiments/inspect_frozen_encoder_geometry_learning.ipynb) |
| [training_protocol](training_protocol.py), [ddp_training](ddp_training.py), [lion_optimizer](lion_optimizer.py) | Deterministic splits, distributed training and optional optimizer. | [Lightning experiments](../notebooks/experiments/test_lightning.ipynb) |
| [visualization](visualization.py) | Plot supplied reconstructions and metrics; figures are diagnostics, not held-out validation. | [Embedding inspection](../notebooks/experiments/test_embedding.ipynb) |

## Data, benchmarks and historical research

| Modules | Status / use |
| --- | --- |
| [tracks](tracks.py) | Residue-aligned confidence/contact/burial diagnostic tracks; not trained codebook states. |
| [download_utils](download_utils.py), [AFDB_tools](AFDB_tools.py) | Dataset/reference downloads; explicit network/storage side effects. AFDB_tools is historical. |
| [treescore](treescore.py) | Taxonomy/tree drawing and scoring; frozen taxa/taxonomy required. See [TCS benchmark](../../docs/curnagl_tcs.md). |
| [pairwise](pairwise.py) | Experimental complex/pairwise models; see [complex notebooks](../notebooks/complexes/README.md). |
| [pdbgraph](pdbgraph.py), [pdbgraph_polars](pdbgraph_polars.py) | Historical/alternative graph builders. Do not swap their feature schemas into mk2 production inference. |
| [foldseek2tree](foldseek2tree.py), [foldtree](foldtree.py) | Historical FoldTree distance pipeline, not the learned-alphabet CLI. foldtree retains an unavailable legacy corecut import. |
| [test_chebconv_rework](test_chebconv_rework.py) | Developer scratch regression, excluded from runtime distributions. |

## Documentation conventions

Every Python module in this directory and its loss subpackage has a purpose
docstring. Public interfaces should describe non-obvious input/output shapes,
geometry units/frame direction, masks, device behavior and side effects. Follow
existing NumPy/Google docstring style within a module. Inline comments explain
numerical/protocol constraints, not obvious assignments. Preserve upstream
copyright/license notices.

When adding a module, update this map and its notebook/CLI links. Document
experimental dependencies explicitly rather than adding them implicitly to
the base inference install. See the [maintenance checklist](../../docs/documentation_maintenance.md).
