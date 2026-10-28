# Data acquisition

Dataset and reference downloads for structure and tree pipelines.

[Source map](../README.md) | [Project](../../../README.md) |
[Experiments](../../../docs/experiments.md)

- [download_utils](./download_utils.py): downloads with verification. Explicit
  network and storage side effects; use this for fetching reference datasets.
- [AFDB_tools](./AFDB_tools.py): AlphaFold Database fetching. Historical; prefer
  `download_utils` for new acquisition paths.
- [tracks](./tracks.py): residue-aligned confidence, contact and burial
  diagnostic tracks. No active importers; kept for reference, not trained
  codebook states.

## When to use this

Use `download_utils` for any new dataset or reference download; it verifies on
fetch and is the supported acquisition path. `AFDB_tools` is a historical
AlphaFold-DB fetcher retained for older notebooks. `tracks` provides
residue-aligned diagnostic tracks and is reference-only — it is not imported by
current training or inference and its values are not codebook states. Both
`download_utils` and `AFDB_tools` have explicit network and storage side
effects, so run them in an environment with outbound access and configured
cache locations. See [experiments](../../../docs/experiments.md) for configured
acquisition runs.