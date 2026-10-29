# Ancestral state reconstruction helpers

Streaming node/site uncertainty and read-only inspection summaries for ancestral-state reconstruction over the foldtree2 trees.

[Scripts home](../README.md) | [Project](../../README.md) | [Source modules](../../foldtree2/src/README.md) | [SE3 toolkit](../se3/README.md)

Ancestral-state reconstruction here is the per-site posterior over a fixed
foldtree2 tree, used downstream to budget the phylogenetic-information gain
across nodes and sites. This package streams that posterior and renders
read-only inspection summaries; the reconstruction itself lives in the trees
module, not in these drivers, so no checkpoint or tree state is mutated from
this package.

## Modules
- [ancestral_state_uncertainty.py](./ancestral_state_uncertainty.py): stream per-node, per-site posterior uncertainty over a fixed tree.
- [ancestral_inspection.py](./ancestral_inspection.py): read-only summaries for the ancestral notebook; no state mutation.

## Workflow
The core ancestral-state reconstruction implementation lives in
[foldtree2.src.trees.ancestral](../../foldtree2/src/trees/ancestral.py); these
scripts are thin drivers over that module, not standalone implementations.
`ancestral_state_uncertainty.py` streams the per-node, per-site posterior over a
fixed tree and cohort, emitting the uncertainty table the downstream
phylogenetic-information analysis consumes. `ancestral_inspection.py` produces
the read-only summaries the ancestral notebook renders — per-site marginal
counts, node-wise entropy and cohort-level aggregates — without mutating any
checkpoint or tree state. Experiment context and dated runs are recorded in
[docs/experiments.md](../../docs/experiments.md).

## Caveats
Both drivers are read-only with respect to the tree and cohort inputs; do not
use them to mutate checkpoints or rewrite tree files. Match the tree topology,
site filter and cohort manifest against the pinned provenance before comparing
uncertainty across runs — the per-site posterior is conditional on the fixed
tree, so a topology or filter change silently shifts the entropy budget.
Outputs write under `runs/` (gitignored); stream large cohorts in site blocks
rather than loading the full posterior into memory at once. The uncertainty
table is keyed by the tree's internal node ids and the cohort's site order, so
any re-ordering of either makes a previously written table uncomparable — pin
both in provenance before streaming.

## Status
Both drivers are read-only and compose entirely on the trees module; no
checkpoint or tree mutation happens here. The uncertainty stream is the
maintained entry point for downstream information analysis, and
`ancestral_inspection.py` is the notebook companion — run it first on any new
tree to confirm the node and site indexing before streaming the full posterior.
Keep the tree topology stable across a comparison: a re-rooted or pruned tree
rekeys every node id, so its uncertainty table is not comparable to the
pre-pruning one without an explicit re-index.