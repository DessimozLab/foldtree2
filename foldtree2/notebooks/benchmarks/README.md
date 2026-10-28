# Benchmark notebooks

[Notebook catalogue / prerequisites](../README.md) |
[Experiment scripts and protocols](../../../docs/experiments.md) |
[Final results checklist](../../../docs/final_benchmarks.md) |
[Source map](../../src/README.md)

| Notebook | Inputs and intended analysis | Reproducible route / outputs |
| --- | --- | --- |
| [alphabet_Information_content_benchmark](alphabet_Information_content_benchmark.ipynb) | Family sequences/tokens from multiple model sizes; usage, entropy, Markov rates, MDL proxies, MI and discrimination controls. | [Information script](../../../scripts/benchmarks/alphabet_information_benchmark.py), [shared metrics](../../../scripts/alphabets/alphabet_information_metrics.py); CSV/provenance outputs. |
| [treelikelihood_info_theory_benchmark](treelikelihood_info_theory_benchmark.ipynb) | OMA markers, AA/structural alignments and reference species tree; phylogenetic information and concatenation. | [Joint comparison](../../../scripts/benchmarks/compare_oma_representations.py) and [species supermatrix](../../../scripts/benchmarks/benchmark_species_supermatrix.py); native and controlled protocols remain separate. |
| [mini_tcs_benchmark](mini_tcs_benchmark.ipynb) | Family structures/metadata/taxonomy -> inferred trees and taxonomic consistency comparisons. | [Curnagl TCS workflow](../../../docs/curnagl_tcs.md), [tree scores](../../src/treescore.py). Requires frozen matched taxa and taxonomy. |
| [setup_alphafold_benchmark](setup_alphafold_benchmark.ipynb) | Prepare a dual-method AlphaFold-related benchmark with external datasets/tools. | Research setup, not evidence of a completed comparison. Freeze inputs and report actual coverage in the final checklist. |

The workstation's frozen 499-family OMA marker cohort is **not** the cluster
TCS OMA cohort. Likewise the repaired CATH subsets and new genus-unique CATH
families are distinct datasets. Never pool them without identifying the cohort.

Use AA, published-model 3Di and all five FT2 alphabets for the final integrated
comparison. FT2-50 retains an explicit matrix-nonconvergence acceptance exception.
Report per-family inclusion, failures, uncertainty and units. Do not compare raw
likelihood totals across different alphabet sizes/native alignment widths as
an accuracy ranking. Completion markers must match their requested scope.
