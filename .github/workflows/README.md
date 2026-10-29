# Distribution and documentation workflows

[Project](../../README.md) | [Release procedure](../../docs/releasing.md) |
[Installation](../../docs/installation.md)

| Workflow | Triggers | Contract |
| --- | --- | --- |
| [Python Distribution](python-publish.yml) | PRs to main/dev, relevant dev pushes, published release. | Build staged-compatible sdist/wheel, validate archives/metadata, fresh CPU wheel install and CLI/native checks. Only release events enter protected `pypi` publishing. |
| [Conda Validation](test-conda-build.yml) | PRs to main/dev, relevant main/dev pushes, manual. | Shared allowlisted stager, Linux-64 build, recipe tests, archive checks and fresh CLI install. No publishing. |
| [Conda Release Build](build-conda-package.yml) | Published release or manual. | Validate version/build/install; save real Linux-64 artifact. Only release events enter protected `conda` publishing and attach artifacts. Manual dispatch is build-only. |
| [Documentation](docs.yml) | See workflow path/branch filters. | Offline local links/CLI flags and Sphinx HTML. |

Conda builds are platform-specific, never `noarch`: the runtime includes ELF
executables. The shared [build helper](../../scripts/test_conda_packaging.sh)
supports current `.conda` archives (ZIP containing compressed tar payloads) and
legacy `.tar.bz2`; it does not treat `.conda` as gzip tar.

Configure `ANACONDA_OWNER` and `ANACONDA_TOKEN` in the protected `conda`
environment. Missing settings or failed uploads fail the job. Configure PyPI
Trusted Publishing in the `pypi` environment. Neither workflow bypasses review,
publishes on a dev push, or forces replacement of an existing registry artifact.
See the [release gates](../../docs/releasing.md#final-go-no-go) before creating
any release event.
