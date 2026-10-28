# Replacing main and publishing a release

[Project](../README.md) | [Installation](installation.md) |
[Final benchmarks](final_benchmarks.md) |
[Documentation maintenance](documentation_maintenance.md)

Supported initial target: **Linux x86_64**, CPU-first inference, with optional
CUDA configured separately. Do not label the current bundled executables as
macOS/Windows/ARM compatible. No workflow should publish merely because dev was
pushed or a PR opened.

## Distribution contract

- [pyproject.toml](../pyproject.toml) is the single Python metadata/dependency/
  entry-point source. [setup.py](../setup.py) contains build hooks only.
- Local wheels carry `py3-none-linux_x86_64`, not `py3-none-any`: the package
  includes native programs even though its own modules are Python. Before PyPI
  upload, auditwheel validates/repairs to `py3-none-manylinux_2_34_x86_64`.
  The bundled MAFFT helpers require glibc >=2.34; older Linux versions would
  need rebuilt binaries, not a renamed wheel. The conda recipe records the same
  minimum via its `__glibc` requirement.
- [Conda recipe](../conda-recipe/meta.yaml) reads the project version from TOML
  and builds a `linux-64` package. Pip and conda install the same console scripts
  and package-local resources. MAFFT is an external pip prerequisite and a conda
  runtime dependency.
- The named channel also needs the [PyDSSP companion recipe](../conda-pydssp/meta.yaml),
  pinned to the upstream 0.9.1 wheel and SHA-256 with its MIT license. It is built,
  tested and uploaded alongside FoldTree2; no runtime pip/network bootstrap is
  hidden in the conda installation.
- Models, datasets, notebooks, tests, caches and legacy invalid scratch code stay
  out of runtime artifacts. Code installation does not supply a usable pretrained
  bundle automatically.

## Local checks

Use isolated temporary environments; do not change active training environments.
Build from the allowlisted staged source, not a data-filled checkout:

```bash
stage_parent=$(mktemp -d /tmp/foldtree2-release-XXXXXX)
python scripts/stage_conda_source.py --destination "$stage_parent/source"
python -m build "$stage_parent/source" --outdir "$stage_parent/dist"
python scripts/check_distribution.py "$stage_parent"/dist/*
python -m auditwheel repair --plat manylinux_2_34_x86_64 \
  --wheel-dir "$stage_parent/release" "$stage_parent"/dist/*.whl
cp "$stage_parent"/dist/*.tar.gz "$stage_parent/release/"
python scripts/check_distribution.py --release "$stage_parent"/release/*
python -m twine check "$stage_parent"/release/*
python scripts/check_documentation.py
python -m unittest foldtree2.tests.test_documentation foldtree2.tests.test_conda_staging foldtree2.tests.test_distribution
bash scripts/test_conda_packaging.sh --croot "$stage_parent/conda-build" --cli-smoke
```

Build tooling (`build`, `twine`, `auditwheel`, `conda-build`) belongs in a development/build
environment. The wheel must be built from the sdist as well as directly from
staging; `python -m build` does this by default. The conda helper runs recipe tests
and a fresh-environment CLI check. It expects a fresh build root and inspects the
actual platform archive rather than assuming `noarch`.

Then install each artifact **outside the checkout**, run `pip check` for pip,
exercise every console entry point/native helper, convert a small PDB sample and
build a four-structure tree with a trusted matched bundle. Check tree/alignment
identifiers and nonempty populated graphs, not just command exit status.

Run the relevant repository regression tests and broader release suite. Module
docstrings may change, but public import/class paths used by notebooks, scripts
and pickled checkpoints must stay compatible. Do not reorganize serialized
classes without stable re-exports and checkpoint-load tests.

## Verified local preparation

The development working tree passed the following checks on the Linux x86_64
workstation. This is local engineering validation, not a published release or
a completed scientific benchmark:

- Full CPU regression suite: **211 passed, 1 skipped, 1 xfailed**. GPU execution
  was deliberately disabled to avoid interfering with ongoing work; this does
  not establish GPU compatibility.
- Built the sdist and a wheel from that sdist, checked archive contents and
  Twine metadata, and used auditwheel to validate/repair the publication wheel
  to `py3-none-manylinux_2_34_x86_64`.
- Installed the wheel outside the checkout; `pip check`, all six Python CLI
  entry points, native tools, token conversion, graph conversion, four-structure
  tree inference and MAD rooting passed. Python was 3.10.21, torch
  2.14.0+cpu and PyG 2.8.0.post1. This isolated wheel installation inherited an
  existing CPU dependency environment: **fresh pip dependency resolution was
  not rerun locally**. The Python Distribution workflow performs that check.
- Built both conda packages and passed their recipe tests. A fresh environment
  installed `foldtree2=0.1.0=py310h556b467_0` from the local channel with dependency
  solving, rather than installing the archive in dependency-skipping explicit
  mode. Python was 3.10.22, torch 2.6.0 and PyG 2.8.0.post1.
- In that fresh conda environment, all six Python CLIs, RAxML, MAFFT, token
  conversion and MAD passed outside the checkout. Four sample PDBs produced
  four populated graphs with 857 node features. CPU inference using an existing
  matched 30-state full-object checkpoint bundle produced an alignment, ML tree
  and rooted tree containing exactly the four expected identifiers; aligned
  sequences had equal lengths. This also exercises legacy checkpoint paths.
- Documentation link/CLI checks covered **50 documents and 65 examples** with
  no errors; the strict Sphinx build passed without warnings. Notebooks were
  catalogued, not rerun in full.
- After moving the companion recipe out of the main recipe directory, safe
  staging, independent recipe rendering and the layout regression passed.
  The already-built runtime packages were unchanged; a full build from that
  final layout still needs CI confirmation on the reviewed revision.

MAD rooted inferred trees successfully, but its bundled binary failed on an
integer-only synthetic branch-length fixture with
`ValueError: Format specifier missing precision`. The smoke checks use decimal
branch lengths. This native-tool limitation is not silently treated as success.

No remote CI run, registry upload, full training run or new full benchmark sweep
is established by these checks. The remaining go/no-go items below still apply.

## Publishing setup (requires maintainer configuration)

1. Choose an **Anaconda account or organization** that you control. Configure
   environment `conda` with variable `ANACONDA_OWNER` and secret `ANACONDA_TOKEN`,
   scoped to package upload. The public install channel is then
   `https://conda.anaconda.org/OWNER` (default `main` label). Do not assume the
   GitHub organization name is an Anaconda account.
2. Configure PyPI Trusted Publishing for this repository and
   `.github/workflows/python-publish.yml`, environment `pypi`. Confirm project-name
   ownership and choose an unused release version.
3. Add required reviewers to both GitHub environments. Protect main and require
   passing Python Distribution, Conda Validation and documentation checks.
4. Update project version once; the recipe inherits it. Confirm the intended
   `vVERSION` tag matches artifact metadata. Do not overwrite published versions.
5. After explicit approval, integrate dev into main (the earlier plan proposed a
   squash; preserve the full reviewed dirty/untracked work intentionally), tag
   that reviewed revision and publish a GitHub Release.
6. Release publication triggers validated PyPI and conda builds. Their protected
   publish jobs upload only after build/install checks. A manual conda workflow
   dispatch builds artifacts **without publishing**.
7. Verify `pip install foldtree2==VERSION` and
   `conda create -n ft2 -c OWNER -c conda-forge -c bioconda foldtree2=VERSION`
   from clean machines/environments, then run the published-artifact inference
   smoke. Record package hashes and dependency versions.

The conda workflow fails explicitly if owner/token are missing or upload fails.
Only the versioned PyDSSP dependency may be explicitly reused with
`--skip-existing`; a FoldTree2 release never silently skips or replaces an
existing artifact. Bump the companion recipe's build number when its recipe
changes. The publisher uses the tested pre-1.14 Anaconda client CLI.
Bioconda submission is a separate future review, not part of this named-channel
release path.

## Licensing

The project license does not replace third-party licenses. The RAxML directory
includes its [upstream license](../foldtree2/raxml-ng/LICENSE.txt), which is
included in staged sources and pip archives. Rigid/OpenFold-derived modules
retain upstream attribution. Before public distribution:

- Verify redistribution terms/source obligations for the custom RAxML-NG binary,
  MAD binary and MAFFT conversion helpers, including any required source offers,
  notices and exact binary provenance. Existing binary presence is not proof
  that these obligations have been met.
- Include required license/notice files for each bundled tool; MAD/helper
  licensing provenance is not established by this preparation.
- Confirm data/model redistribution rights and model cards separately.

Unresolved licensing is a **publication blocker**, not a reason to remove notices
or label the whole aggregate MIT-only.

## Final go/no-go

- [ ] Reviewed source snapshot includes intended experiments/configs; unrelated
  dirty work remains preserved and is not accidentally swept into a release.
- [ ] Full regression suite and artifact-specific fresh installation/inference
  pass on the supported Linux target; record exact versions.
- [ ] All five trusted model bundles are hosted at real, versioned public URLs
  with hashes, licenses, compatibility metadata and FT2-50's acceptance caveat.
- [ ] Required [final results](final_benchmarks.md) are compiled, with failures
  and pending research clearly separated from release claims.
- [ ] Native-tool redistribution obligations are resolved.
- [ ] Registry ownership, credentials, protected environments and unused version
  are confirmed; CI succeeds on the reviewed revision.
- [ ] Maintainer explicitly approves main integration, tagging and publication.

This checklist deliberately does not claim main is ready merely because a wheel
can be built. No merge, commit, tag or registry upload is performed by the local
preparation commands.
