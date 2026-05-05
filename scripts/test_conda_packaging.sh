#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RECIPE_DIR="${ROOT_DIR}/conda-recipe"
CROOT="${CROOT:-/tmp/cb}"
MIN_SIZE_MB=10
MAX_SIZE_MB=50
RUN_SMOKE_INSTALL=0
RUN_CLI_SMOKE=0
CLI_SMOKE_PYTHON_VERSION="3.10"
CLI_SMOKE_SAMPLE_PDB_DIR=""
CLI_SMOKE_SAMPLE_MAX_FILES=1
CLI_SMOKE_AAPROPCSV="${ROOT_DIR}/foldtree2/config/aaindex1.csv"
KEEP_TMP=0

WORK_DIR=""
PAYLOAD_DIR=""
STAGE_DIR=""
STAGED_RECIPE_DIR=""

log() {
  printf "[conda-test] %s\n" "$*"
}

fail() {
  printf "[conda-test] ERROR: %s\n" "$*" >&2
  exit 1
}

cleanup() {
  if [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" && "${KEEP_TMP}" -eq 0 ]]; then
    rm -rf "${WORK_DIR}"
  fi
}
trap cleanup EXIT

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Builds and validates the FoldTree2 conda package locally.

Options:
  --recipe-dir PATH      Conda recipe directory (default: ${RECIPE_DIR})
  --croot PATH           Conda build root directory (default: ${CROOT})
  --min-size-mb N        Minimum allowed package size in MB (default: ${MIN_SIZE_MB})
  --max-size-mb N        Maximum allowed package size in MB (default: ${MAX_SIZE_MB})
  --smoke-install        Create temporary env and install package for import smoke test
  --cli-smoke            Run CLI smoke tests in a fresh env using scripts/smoke_test_conda_cli.sh
  --cli-smoke-python VER Python version for CLI smoke env (default: ${CLI_SMOKE_PYTHON_VERSION})
  --sample-pdb-dir PATH  Optional sample .pdb directory for lightweight data smoke test
  --sample-max-files N   Number of sample .pdb files for data smoke (default: ${CLI_SMOKE_SAMPLE_MAX_FILES})
  --aapropcsv PATH       Amino acid CSV for pdbs-to-graphs data smoke (default: ${CLI_SMOKE_AAPROPCSV})
  --keep-tmp             Keep temporary extraction/install directory for debugging
  -h, --help             Show this help text
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recipe-dir)
      RECIPE_DIR="$2"
      shift 2
      ;;
    --croot)
      CROOT="$2"
      shift 2
      ;;
    --min-size-mb)
      MIN_SIZE_MB="$2"
      shift 2
      ;;
    --max-size-mb)
      MAX_SIZE_MB="$2"
      shift 2
      ;;
    --smoke-install)
      RUN_SMOKE_INSTALL=1
      shift
      ;;
    --cli-smoke)
      RUN_CLI_SMOKE=1
      shift
      ;;
    --cli-smoke-python)
      CLI_SMOKE_PYTHON_VERSION="$2"
      shift 2
      ;;
    --sample-pdb-dir)
      CLI_SMOKE_SAMPLE_PDB_DIR="$2"
      shift 2
      ;;
    --sample-max-files)
      CLI_SMOKE_SAMPLE_MAX_FILES="$2"
      shift 2
      ;;
    --aapropcsv)
      CLI_SMOKE_AAPROPCSV="$2"
      shift 2
      ;;
    --keep-tmp)
      KEEP_TMP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

require_cmd conda
require_cmd find
require_cmd awk
require_cmd stat
require_cmd tar
require_cmd unzip

[[ -d "${RECIPE_DIR}" ]] || fail "Recipe directory does not exist: ${RECIPE_DIR}"

WORK_DIR="$(mktemp -d -t foldtree2-conda-test-XXXXXX)"
STAGE_DIR="${WORK_DIR}/stage"
mkdir -p "${STAGE_DIR}"

# Conda copies source to its own work dir before build. Stage a slim source tree so
# very large training data files are never copied in the first place.
log "Preparing slim source tree"
cp -a "${RECIPE_DIR}" "${STAGE_DIR}/"

for path in foldtree2 models/production; do
  if [[ -e "${ROOT_DIR}/${path}" ]]; then
    mkdir -p "${STAGE_DIR}/$(dirname "${path}")"
    cp -a "${ROOT_DIR}/${path}" "${STAGE_DIR}/${path}"
  fi
done

# Keep this directory present even when no production model file exists yet.
mkdir -p "${STAGE_DIR}/models/production"

for file in README.md LICENSE.txt pyproject.toml setup.py MANIFEST.in .conda_build_ignore; do
  if [[ -f "${ROOT_DIR}/${file}" ]]; then
    cp -a "${ROOT_DIR}/${file}" "${STAGE_DIR}/"
  fi
done

RECIPE_BASENAME="$(basename "${RECIPE_DIR}")"
STAGED_RECIPE_DIR="${STAGE_DIR}/${RECIPE_BASENAME}"
[[ -d "${STAGED_RECIPE_DIR}" ]] || fail "Staged recipe directory missing: ${STAGED_RECIPE_DIR}"

log "Building package with conda build"
conda build "${STAGED_RECIPE_DIR}" \
  --croot "${CROOT}" \
  --prefix-length 80 \
  --no-test

mapfile -t PKG_CANDIDATES < <(find "${CROOT}" -type f \( -name 'foldtree2-*.conda' -o -name 'foldtree2-*.tar.bz2' \) | sort)
[[ ${#PKG_CANDIDATES[@]} -gt 0 ]] || fail "No foldtree2 package found under ${CROOT}"

PACKAGE="${PKG_CANDIDATES[${#PKG_CANDIDATES[@]}-1]}"
SIZE_BYTES="$(stat -c%s "${PACKAGE}")"
SIZE_MB=$(( SIZE_BYTES / 1024 / 1024 ))

log "Package: ${PACKAGE}"
log "Size: ${SIZE_MB}MB"

if (( SIZE_MB < MIN_SIZE_MB )); then
  fail "Package too small: ${SIZE_MB}MB < ${MIN_SIZE_MB}MB"
fi

if (( SIZE_MB > MAX_SIZE_MB )); then
  fail "Package too large: ${SIZE_MB}MB > ${MAX_SIZE_MB}MB"
fi

PAYLOAD_DIR="${WORK_DIR}/payload"
mkdir -p "${PAYLOAD_DIR}"

extract_tar_to_payload() {
  local tar_path="$1"
  case "${tar_path}" in
    *.tar.zst)
      tar --zstd -xf "${tar_path}" -C "${PAYLOAD_DIR}"
      ;;
    *.tar.bz2)
      tar -xjf "${tar_path}" -C "${PAYLOAD_DIR}"
      ;;
    *.tar.gz)
      tar -xzf "${tar_path}" -C "${PAYLOAD_DIR}"
      ;;
    *.tar.xz)
      tar -xJf "${tar_path}" -C "${PAYLOAD_DIR}"
      ;;
    *)
      fail "Unsupported tar format: ${tar_path}"
      ;;
  esac
}

if [[ "${PACKAGE}" == *.conda ]]; then
  log "Extracting .conda package"
  OUTER_DIR="${WORK_DIR}/outer"
  mkdir -p "${OUTER_DIR}"
  unzip -q "${PACKAGE}" -d "${OUTER_DIR}"

  PKG_TAR="$(find "${OUTER_DIR}" -maxdepth 1 -type f -name 'pkg-*.tar.*' | head -n1)"
  [[ -n "${PKG_TAR}" ]] || fail "Could not locate payload tar inside ${PACKAGE}"
  extract_tar_to_payload "${PKG_TAR}"
else
  log "Extracting .tar.bz2 package"
  extract_tar_to_payload "${PACKAGE}"
fi

count_pattern() {
  local pattern="$1"
  find "${PAYLOAD_DIR}" -type f -name "${pattern}" | wc -l | awk '{print $1}'
}

count_pattern_excluding_production_models() {
  local pattern="$1"
  find "${PAYLOAD_DIR}" -type f -name "${pattern}" \
    ! -path '*/models/production/*' | wc -l | awk '{print $1}'
}

list_unwanted_files() {
  find "${PAYLOAD_DIR}" -type f \
    \( -name '*.h5' -o -name '*.ipynb' -o -name '*.pkl' -o -name '*.pt' -o -name '*.pth' \) \
    ! -path '*/models/production/*'
}

list_large_files() {
  find "${PAYLOAD_DIR}" -type f -size +100M
}

H5_COUNT="$(count_pattern '*.h5')"
IPYNB_COUNT="$(count_pattern '*.ipynb')"
PKL_COUNT="$(count_pattern_excluding_production_models '*.pkl')"
PT_COUNT="$(count_pattern_excluding_production_models '*.pt')"
PTH_COUNT="$(count_pattern_excluding_production_models '*.pth')"
TOTAL_UNWANTED=$(( H5_COUNT + IPYNB_COUNT + PKL_COUNT + PT_COUNT + PTH_COUNT ))

log "Verification results:"
log "  .h5 files: ${H5_COUNT}"
log "  .ipynb files: ${IPYNB_COUNT}"
log "  .pkl files outside models/production: ${PKL_COUNT}"
log "  .pt files outside models/production: ${PT_COUNT}"
log "  .pth files outside models/production: ${PTH_COUNT}"

if (( TOTAL_UNWANTED > 0 )); then
  log "Unwanted files found:"
  list_unwanted_files
  fail "Package contains unwanted artifacts"
fi

LARGE_COUNT="$(list_large_files | wc -l | awk '{print $1}')"
log "  files >100MB: ${LARGE_COUNT}"
if (( LARGE_COUNT > 0 )); then
  log "Oversized files found:"
  list_large_files
  fail "Package contains oversized files (>100MB)"
fi

if ! find "${PAYLOAD_DIR}" -path '*/site-packages/foldtree2/__init__.py' | grep -q .; then
  fail "Missing required file foldtree2/__init__.py in package payload"
fi

if ! find "${PAYLOAD_DIR}" -path '*/site-packages/foldtree2/src/encoder.py' | grep -q .; then
  fail "Missing required file foldtree2/src/encoder.py in package payload"
fi

if ! find "${PAYLOAD_DIR}" -path '*/share/foldtree2/raxml-ng/raxml-ng' | grep -q .; then
  fail "Missing required bundled binary raxml-ng"
fi

if ! find "${PAYLOAD_DIR}" -path '*/share/foldtree2/mafft_tools/hex2maffttext' | grep -q .; then
  fail "Missing required bundled MAFFT helper hex2maffttext"
fi

if ! find "${PAYLOAD_DIR}" -path '*/share/foldtree2/models/production*' | grep -q .; then
  fail "Missing required production model directory under share/foldtree2/models/production"
fi

if (( RUN_SMOKE_INSTALL == 1 )); then
  log "Running optional smoke install test"
  SMOKE_ENV="${WORK_DIR}/smoke-env"
  conda create -y -p "${SMOKE_ENV}" python=3.10 >/dev/null
  conda install -y -p "${SMOKE_ENV}" "${PACKAGE}" >/dev/null
  "${SMOKE_ENV}/bin/python" - <<'PY'
import foldtree2
from foldtree2.src import encoder
print('smoke-import-ok')
PY
fi

if (( RUN_CLI_SMOKE == 1 )); then
  CLI_SMOKE_SCRIPT="${ROOT_DIR}/scripts/smoke_test_conda_cli.sh"
  [[ -x "${CLI_SMOKE_SCRIPT}" ]] || fail "CLI smoke script missing or not executable: ${CLI_SMOKE_SCRIPT}"

  log "Running optional CLI smoke test"
  CLI_ARGS=(
    --package "${PACKAGE}"
    --python-version "${CLI_SMOKE_PYTHON_VERSION}"
    --sample-max-files "${CLI_SMOKE_SAMPLE_MAX_FILES}"
    --aapropcsv "${CLI_SMOKE_AAPROPCSV}"
  )

  if [[ -n "${CLI_SMOKE_SAMPLE_PDB_DIR}" ]]; then
    CLI_ARGS+=(--sample-pdb-dir "${CLI_SMOKE_SAMPLE_PDB_DIR}")
  fi

  "${CLI_SMOKE_SCRIPT}" "${CLI_ARGS[@]}"
fi

log "All conda packaging checks passed"
if (( KEEP_TMP == 1 )); then
  log "Temporary directory kept at: ${WORK_DIR}"
fi
