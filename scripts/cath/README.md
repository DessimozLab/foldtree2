# CATH family cohort preparation

Generate, retain and supervise the CATH family cohorts that feed the TCS benchmark.

[Scripts home](../README.md) | [Project](../../README.md) | [Source modules](../../foldtree2/src/README.md) | [SE3 toolkit](../se3/README.md)

CATH families are the structural-domain cohort that anchors the TCS benchmark:
each family contributes a set of homologous folds whose sequence-structure
alignment the distance and contact heads train against. This package stages the
family library, retains a filtered subset for a benchmark cohort, and
supervises the generation run that emits the cohort manifest the downstream TCS
entry point reads by hash.

## Modules
- [generate_cath_families.py](./generate_cath_families.py): generate CATH family cohorts from the source library.
- [retain_cath_subsets.py](./retain_cath_subsets.py): retain filtered CATH subsets for a benchmark cohort.
- [supervise_cath_generation.py](./supervise_cath_generation.py): supervise the CATH generation run end-to-end.

## Workflow
These produce the CATH cohorts consumed by the TCS benchmark; see
[curnagl_tcs.md](../../docs/curnagl_tcs.md) for how a generated cohort flows
into the Curnagl pipeline. Generation is staged by `generate_cath_families.py`,
filtered down by `retain_cath_subsets.py`, and watched by
`supervise_cath_generation.py`; the supervised run emits the cohort manifest the
TCS entry point reads. The retain step is where the cohort size and family
coverage are pinned, so the downstream TCS benchmark sees a fixed family roster
rather than a moving target.

## Caveats
The shared CATH helpers live at the scripts root —
[audit_cath_chains.py](../audit_cath_chains.py) (chain-level audit) and
[rebuild_cath_families.py](../rebuild_cath_families.py) (family-input rebuild) —
imported by these scripts, not duplicated here. Prefer composing on those rather
than re-deriving the chain selection or family reconstruction. Cohorts write
under `runs/` (gitignored); pin the CATH release and domain filter in
provenance before retaining a subset, since retain decisions are what fix the
cohort for the benchmark. The chain audit (`audit_cath_chains.py`) flags
non-singleton chains and structural outliers; clear its warnings before
retaining, or the TCS benchmark will inherit the noise into its distance targets.

## Status
Generation is the current entry point; retain and supervise compose on it and on
the shared root helpers. A retained cohort is immutable once its manifest is
pinned — regenerate and re-label rather than editing a retained subset in place,
since the TCS benchmark reads the manifest by hash and a silent overwrite breaks
the pinned evidence copy. When a CATH release ships new families, generate a
fresh cohort rather than folding the new chains into a retained one, so the
benchmark cohort and its pinned evidence stay reconciled. Document the family
coverage and domain filter in the cohort manifest so a later retain pass can
reproduce the roster exactly.