# Keeping documentation current

[Back to README](../README.md)

Update docs in the same change that adds/changes a user-facing experiment. When
a long job finishes later, make an evidence-backed follow-up. A job exiting or
an output file appearing is not enough to mark scientific work complete.

## Adding or changing an experiment

1. Add the entrypoint/purpose to the [experiment guide](experiments.md). Identify
   notebook origins and any differences; do not imply an entire notebook was ported.
2. Document inputs/layout/identifiers, artifact selection, portable smoke/full
   commands, device/thread controls, measured runtime/storage if available,
   output tables, metric units, and limitations.
3. Explain restart/provenance/cache behavior, logs, completion evidence, and
   failures. Separate smoke outputs from full results and optional downloads
   from required inputs.
4. Describe splits, seeds, baselines, and controls. Change the protocol identifier
   when methodology changes; never silently pool old/new protocols.
5. Update README navigation, manuscript methods if science changed, and release
   requirements if this benchmark becomes mandatory.
6. Validate help/flags and a representative smoke run. State exactly what was
   tested; do not invent compatibility, performance numbers, or download URLs.

## When an experiment completes

1. Inspect aggregate/per-family completion records, logs, and provenance. Confirm
   expected sizes/counts and no smoke limit or partial upstream sweep. Record
   exclusions and failed/skipped stages.
2. Record source revision, protocol, command/config, dataset version/checksums,
   seeds/splits, hardware/software, matched artifact hashes (including character
   maps), results location, and completion date.
   If the working tree was dirty, preserve a source snapshot including untracked
   scripts; the base commit hash alone does not identify the executed code.
3. Add a concise results record with measured outcomes, units, counts, uncertainty
   where computed, and limitations. Link tables/logs without committing large
   raw outputs or private data. Local paths are evidence, not public downloads.
4. Update [model status](user_guide.md) only after bundle validation, selected-pair
   evaluation where applicable, and convergence pass. Experiment completion alone
   does not promote a model.
5. Update the [release checklist](production_alphabet_readiness.md); keep external
   and unrun requirements pending. Distinguish pending, smoke-tested, complete,
   validated, and failed.
6. Retain old checksums/results when artifacts change, document which protocol
   produced each set, and rerun affected comparisons. Never silently overwrite
   reported numbers or mix matrices from different generations.

## Results record template

For results ready to report, add `docs/results/<date>-<experiment>-<protocol>.md`.
Do not create a completion record for an active job.

```text
Experiment and protocol:
Status: pending | smoke-tested | complete | validated | failed
Completion date and source revision:
Command/configuration and seeds/splits:
Dataset version/checksums; requested versus completed counts:
Alphabet sizes and checkpoint/matrix/character-map hashes:
Hardware/software versions:
Outputs/completion/provenance records:
Results with units and uncertainty (if computed):
Exclusions, failures, limitations, remaining work:
Validation checks performed:
```

## Documentation checks

- Run `python scripts/check_documentation.py` for local links and example flags.
- Run `git diff --check`; inspect changes for stale claims or unrelated edits.
- Check CLI help and smoke commands in fresh directories. Do not launch full
  training/download jobs simply to test documentation examples.
- Validate installation changes in isolated environments and record blockers.
- Clear stale notebook outputs and validate edited cells; disclose notebooks
  that were not rerun in full.

## Validation log

Documentation refresh checks on the Linux workstation:

- Fresh conda environment: Python 3.10.21, torch 2.14.0+cpu, PyG 2.8.0.post1.
  Installation completed and `pip check` passed. CLI import required the
  activated-environment C++ runtime workaround in the installation guide.
- Installed-package help passed outside the checkout for tree inference, graph
  conversion, matrix building, and both documented trainers. Experiment/preparation
  help checks also passed in the existing workstation environment.
- Four-structure CPU tree inference succeeded with a preserved 30-character
  bundle in both environments. The fresh installed package produced a tree and
  alignment with all four expected identifiers and equal aligned sequence lengths.
  These used retained matrices, not the pending corrected-matrix production sweep.
- Missing `colour`/TensorBoard trainer dependencies were added to packaging;
  Python minimum metadata was aligned to 3.9. Explicit CPU device selection now
  overrides checkpoint CUDA attributes and has a regression test.
- Ten documentation/device/counting/convergence regressions passed. Local-link
  and documented-flag checks passed, including repaired manuscript-guide links.
- Temporary installation logs and exact dependency versions are in
  `/tmp/foldtree2-doc-install-TFvpWO/`; inference logs, copied inputs/artifacts,
  and fresh outputs are in `/tmp/foldtree2-doc-inference-jWIHz0/`. Temporary paths
  are diagnostic evidence, not permanent downloadable benchmark artifacts.

Production training/matrix jobs remain separate and unfinished. No full training,
full experiment sweep, full notebook rerun, external-machine benchmark, or new
GPU/platform compatibility claim follows from these documentation checks.

Safe-source-staging follow-up: four additional regression tests passed, including
80 GiB sparse inputs excluded before copying, symlink handling, preflight limits,
and stale-destination rejection. The real checkout stages 79 files (27,040,605
bytes) without datasets, model bundles, notebooks, or tests. Staged runtime
Python parses after excluding the invalid legacy scratch experiment. Local and
CI build commands now share the stager. An offline wheel build from the staged
source passed; its archive includes required tools/configuration and excludes
datasets/models/tests. Dependency/platform/conda-installation
repairs are not established by this source-filtering change.
