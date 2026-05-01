# Conda CI Workflows

FoldTree2 now uses a two-workflow conda pipeline with clear responsibilities.

## 1) Validation Workflow (PR + Main)

- File: `test-conda-build.yml`
- Name in Actions UI: `Conda Validation`
- Triggers:
  - Pull requests to `main` and `develop` (when conda/package files change)
  - Pushes to `main` and `master` (same path filters)
  - Manual dispatch

### Purpose

- Build the conda package and fail fast on packaging regressions.
- Verify no large unwanted artifacts are included (`.h5`, `.ipynb`, `.pkl`, `.pt`, `.pth`).
- Enforce expected package size bounds.
- Check that essential package files are present.

## 2) Release Workflow (Tags)

- File: `build-conda-package.yml`
- Name in Actions UI: `Conda Release Build`
- Triggers:
  - Version tag push (`v*`)
  - Manual dispatch

### Purpose

- Build the release `.conda` package.
- Upload package artifact to GitHub Actions.
- Attach package to the GitHub release.
- Optionally upload to Anaconda Cloud if `ANACONDA_TOKEN` is configured.

## Why This Setup

- Avoids duplicate conda builds across multiple workflows.
- Keeps fast validation separate from release publishing.
- Makes Actions history easier to read and troubleshoot.

## Release Usage

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

## Optional Anaconda Upload

Set repository secret `ANACONDA_TOKEN` to enable upload during release builds.
