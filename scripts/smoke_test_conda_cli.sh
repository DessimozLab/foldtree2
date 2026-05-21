#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PACKAGE_PATH=""
CROOT="${CROOT:-/tmp/cb}"
PYTHON_VERSION="3.10"
KEEP_TMP=0
SAMPLE_PDB_DIR=""
SAMPLE_MAX_FILES=1
AAPROPCSV="${ROOT_DIR}/foldtree2/config/aaindex1.csv"

WORK_DIR=""
ENV_PREFIX=""

log() {
  printf "[cli-smoke] %s\n" "$*"
}

fail() {
  printf "[cli-smoke] ERROR: %s\n" "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Create a fresh conda environment, install a built foldtree2 package, and run
command-line smoke checks.

Options:
  --package PATH         Path to package (.conda or .tar.bz2). If omitted, auto-detect from --croot.
  --croot PATH           Conda build root for auto-detection (default: ${CROOT})
  --python-version VER   Python version for fresh env (default: ${PYTHON_VERSION})
  --sample-pdb-dir PATH  Optional directory containing sample .pdb files for a lightweight data smoke test
  --sample-max-files N   Number of sample .pdb files to use (default: ${SAMPLE_MAX_FILES})
  --aapropcsv PATH       Amino-acid properties CSV for pdbs-to-graphs (default: ${AAPROPCSV})
  --keep-tmp             Keep temporary env/files for debugging
  -h, --help             Show this help text
EOF
}

cleanup() {
  if [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" && "${KEEP_TMP}" -eq 0 ]]; then
    rm -rf "${WORK_DIR}"
  fi
}
trap cleanup EXIT

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --package)
      PACKAGE_PATH="$2"
      shift 2
      ;;
    --croot)
      CROOT="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --sample-pdb-dir)
      SAMPLE_PDB_DIR="$2"
      shift 2
      ;;
    --sample-max-files)
      SAMPLE_MAX_FILES="$2"
      shift 2
      ;;
    --aapropcsv)
      AAPROPCSV="$2"
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
require_cmd sort
require_cmd head

if [[ -z "${PACKAGE_PATH}" ]]; then
  mapfile -t PKG_CANDIDATES < <(find "${CROOT}" -type f \( -name 'foldtree2-*.conda' -o -name 'foldtree2-*.tar.bz2' \) | sort)
  [[ ${#PKG_CANDIDATES[@]} -gt 0 ]] || fail "No foldtree2 package found under ${CROOT}; pass --package explicitly"
  PACKAGE_PATH="${PKG_CANDIDATES[${#PKG_CANDIDATES[@]}-1]}"
fi

[[ -f "${PACKAGE_PATH}" ]] || fail "Package not found: ${PACKAGE_PATH}"

WORK_DIR="$(mktemp -d -t foldtree2-cli-smoke-XXXXXX)"
ENV_PREFIX="${WORK_DIR}/env"

log "Creating fresh conda environment"
conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" >/dev/null

log "Installing package: ${PACKAGE_PATH}"
conda install -y -p "${ENV_PREFIX}" "${PACKAGE_PATH}" -c conda-forge -c pytorch -c bioconda >/dev/null

run_cmd() {
  local label="$1"
  shift
  log "Running: ${label}"
  conda run -p "${ENV_PREFIX}" "$@" >/dev/null
}

run_cmd "foldtree2 --about" foldtree2 --about
run_cmd "pdbs-to-graphs --help" pdbs-to-graphs --help
run_cmd "makesubmat --help" makesubmat --help
run_cmd "ft2treebuilder --help" ft2treebuilder --help

if [[ -n "${SAMPLE_PDB_DIR}" ]]; then
  [[ -d "${SAMPLE_PDB_DIR}" ]] || fail "Sample directory not found: ${SAMPLE_PDB_DIR}"
  [[ -f "${AAPROPCSV}" ]] || fail "Amino-acid properties CSV not found: ${AAPROPCSV}"

  mapfile -t SAMPLE_FILES < <(find "${SAMPLE_PDB_DIR}" -maxdepth 1 -type f -name '*.pdb' | sort | head -n "${SAMPLE_MAX_FILES}")
  [[ ${#SAMPLE_FILES[@]} -gt 0 ]] || fail "No .pdb files found in ${SAMPLE_PDB_DIR}"

  STAGED_PDB_DIR="${WORK_DIR}/sample_pdbs"
  mkdir -p "${STAGED_PDB_DIR}"
  for pdb in "${SAMPLE_FILES[@]}"; do
    cp "${pdb}" "${STAGED_PDB_DIR}/"
  done

  OUT_H5="${WORK_DIR}/sample_graphs.h5"
  log "Running lightweight data smoke with ${#SAMPLE_FILES[@]} sample PDB file(s)"
  conda run -p "${ENV_PREFIX}" pdbs-to-graphs "${STAGED_PDB_DIR}" "${OUT_H5}" --aapropcsv "${AAPROPCSV}" >/dev/null
  [[ -s "${OUT_H5}" ]] || fail "Expected output not created by pdbs-to-graphs: ${OUT_H5}"
fi

log "All CLI smoke tests passed"
if [[ "${KEEP_TMP}" -eq 1 ]]; then
  log "Temporary files kept at: ${WORK_DIR}"
fi
