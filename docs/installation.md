# Installation and troubleshooting

[Back to README](../README.md)

## Supported release target

Linux x86_64 with glibc >=2.34 is the initial supported platform. Bundled RAxML-NG, MAD and token
helpers are native ELF executables; macOS, Windows and ARM are unsupported.
Release wheels are audited/repaired to `manylinux_2_34_x86_64`; local intermediate
wheels use `linux_x86_64` and must not be uploaded directly to PyPI.
Python 3.10 is the installation-test target; CUDA is optional for inference.

## Registry installations after publication

The development branch is not automatically published. Confirm the release
version and the maintainer's actual Anaconda owner before using these routes:

```bash
# PyPI: create/activate an isolated Python environment and install MAFFT separately.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install foldtree2
python -m pip check

# Conda: replace OWNER with the release's documented Anaconda account.
conda create -n foldtree2-cpu --override-channels \
  -c OWNER -c conda-forge -c bioconda python=3.10 foldtree2
conda activate foldtree2-cpu
foldtree2 --help
```

The named channel must contain both FoldTree2 (`linux-64`) and the companion
PyDSSP (`noarch`) package. PyDSSP is not currently available in conda-forge/
bioconda; the checked-in companion recipe preserves legacy notebook imports.
Do not substitute an assumed GitHub organization/channel name. Before publication,
install from a checkout or locally built artifacts instead. Registry ownership,
model hosting and publication gates are in the [release procedure](releasing.md).

## CPU-first installation

Run from a checkout. Use a new environment rather than changing one running jobs:

```bash
conda create -n foldtree2-cpu -c conda-forge -c bioconda python=3.10 pip mafft
conda activate foldtree2-cpu
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
python -m pip check
```

On this Linux workstation, a fresh conda/PyTorch combination required the conda
C++ runtime to take precedence over the system copy. If `foldtree2 --help`
fails with a `CXXABI`/`GLIBCXX` error while importing SQLite/ETE3, set this only
in the activated environment and retry:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
foldtree2 --help
```

Do not replace system libraries or add this globally to unrelated environments.
The validation log records this workaround as part of the tested installation.

Python 3.10 is the fresh-install target; package metadata requires at least
Python 3.9 for the training argument parsers. Dependency compatibility still
depends on the versions resolved for your Python/platform. Install the CPU torch wheel
before FoldTree2 to avoid inadvertently downloading a CUDA stack. See
[PyTorch installation](https://pytorch.org/get-started/locally/) and
[PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
for platform requirements and any additional extension wheels your workload needs.

The historical `foldtree2.yml` is a workstation recipe, not the portable default:
it pins Python 3.9 and attempts a different CUDA wheel than the observed runtime.
`environment-dev.yml` is a developer/build environment with older CUDA choices.
Neither should be treated as a verified fresh installation recipe.

## CUDA installation

Create the same conda environment with a distinct name, such as `foldtree2-gpu`.
Before `python -m pip install .`, install the torch wheel selected by the
[official PyTorch selector](https://pytorch.org/get-started/locally/) for your OS
and NVIDIA driver. Do not install the CPU wheel first or assume one CUDA
toolkit/driver combination works everywhere. Then check:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Use `--device cuda:0` for inference/experiments after this succeeds. Training
device arguments differ by trainer; consult its `--help`. When running through
a restricted sandbox, repeat the check in a normal terminal before concluding
that GPUs are unavailable.

## Installation checks

```bash
python -c "import torch, torch_geometric, foldtree2; print(torch.__version__, torch_geometric.__version__)"
foldtree2 --help
pdbs-to-graphs --help
makesubmat --help
command -v mafft
command -v raxml-ng
command -v hex2maffttext
command -v maffttext2hex
raxml-ng --version
```

FoldTree2 supplies RAxML-NG, MAD, and MAFFT token-conversion wrappers using
native files shipped in the package. MAFFT is separate. Foldseek is additionally
required if generating structural reference alignments, not for inference with
prepared models. Native compatibility is distinct from Python installation.

## Validation status

The existing Linux workstation uses Python 3.9.23, torch 2.8.0+cu128, and
torch-geometric 2.6.1; CUDA visibility was confirmed outside the sandbox.
This is an observed environment, not a fresh-install guarantee. Fresh CPU
installation and small inference checks are recorded in the
[documentation validation log](documentation_maintenance.md#validation-log).
Windows/macOS execution and GPU installation on other hardware are unverified.

Graph conversion uses `foldtree2.src.graphs.pdbgraphmk2.PDB2PyG` for PDB and
Foldcomp inputs. The `pdbs-to-graphs` entry point is identical in pip and conda;
its `--mp` option uses the mk2 multiprocessing pool. Legacy FoldX/ProDy and
custom-distance options are not supported for PDB conversion via `encode_pdbs`.
Existing precomputed training HDF5 datasets are not regenerated by this change.

## Safe conda build sources

The local build helper and both conda CI workflows use a shared allowlisted
stager. It copies Python runtime code, required configuration/native tools,
packaging metadata, and the recipe. Datasets, models, notebooks, tests, caches,
and symlinks are excluded before copying. Individual files over 50 MiB or a
selected payload over 100 MiB fail preflight; old/nonempty staging directories
are rejected rather than overwritten. The non-runtime legacy scratch file
`scaling_experiment.py`, which contains invalid Python, is excluded too; its
original source is left untouched.

To inspect selection without copying or building:

```bash
python scripts/stage_conda_source.py --destination /tmp/foldtree2-source-preview --dry-run
```

To stage into a fresh temporary directory:

```bash
stage_parent=$(mktemp -d /tmp/foldtree2-source-XXXXXX)
python scripts/stage_conda_source.py --destination "$stage_parent/source"
```

The JSON output identifies the staged recipe and byte/file counts. Conda should
build that recipe, not directly run `conda build conda-recipe` in a data-filled
checkout: `source.path: ..` otherwise copies the checkout before the build script
runs. `.conda_build_ignore` is not relied on to enforce filtering.

The build helper now builds the companion dependency, uses conda-forge's
`pytorch_geometric` package name, runs recipe tests, discovers Linux-64 artifacts,
inspects the real archive format and can install into a fresh environment with
`--cli-smoke`. Pip and conda share the Python metadata/entrypoints and package-local
native resources. These checks must pass on the release revision; staging alone
is not a successful installation test. Staging only reads runtime/build sources
and does not touch datasets, production models or active training/matrix jobs.
For local installation, use the generated local channel and a package spec.
Passing a `.conda` filename directly to `conda install` uses explicit-archive mode
and can omit dependencies; the smoke helper deliberately resolves through the
local channel instead.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Command not found | Activate the correct environment, install from the checkout, and check `command -v`. |
| Native executable cannot run | Check OS/architecture and permissions. `--raxmlpath` can select a compatible RAxML-NG with the required multistate support. |
| `CXXABI`/`GLIBCXX` import error on Linux | Try the activated-environment C++ runtime workaround above; do not overwrite system libraries. |
| Missing model/map | Acquire the complete trusted bundle; pip does not supply models. See the [user guide](user_guide.md). |
| Feature-width/missing-track error | Use matching graph schemas; do not slice or pad arbitrary features. |
| CUDA unavailable/out of memory | Check driver/device visibility; use CPU, a free GPU, or smaller training batches. Do not stop another user's processes. |
| Fewer than four usable structures | Check the quoted glob, identifiers, backbone coordinates, and conversion errors. |
| Empty HDF5 groups | Confirm conversion completed and groups contain nodes/features; file existence is insufficient. |
| Output provenance differs | Use a fresh experiment directory for changed artifacts, protocol, or smoke limits. |
| Matrix convergence fails | Inspect history/coverage and add reference evidence; do not bypass the production gate. |

Report the command, complete error/log, OS, Python/torch/PyG versions, model
checksums, and input count. Avoid sharing private datasets without permission.
