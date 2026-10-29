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

PACKAGE_CHANNEL="$(cd "$(dirname "${PACKAGE_PATH}")/.." && pwd)"
PACKAGE_NAME="$(basename "${PACKAGE_PATH}")"
[[ "${PACKAGE_NAME}" == foldtree2-* ]] || fail "Expected a foldtree2 package: ${PACKAGE_NAME}"
PACKAGE_STEM="${PACKAGE_NAME%.conda}"
PACKAGE_STEM="${PACKAGE_STEM%.tar.bz2}"
PACKAGE_STEM="${PACKAGE_STEM#foldtree2-}"
PACKAGE_SPEC="foldtree2=${PACKAGE_STEM%-*}=${PACKAGE_STEM##*-}"

# Direct archive installation is explicit mode and does not solve dependencies.
log "Creating fresh environment from local channel: ${PACKAGE_SPEC}"
conda create -y -p "${ENV_PREFIX}" --override-channels \
  -c "file://${PACKAGE_CHANNEL}" -c conda-forge -c bioconda \
  "python=${PYTHON_VERSION}" "${PACKAGE_SPEC}"

run_cmd() {
  local label="$1"
  shift
  log "Running: ${label}"
  (cd "${WORK_DIR}" && conda run --no-capture-output -p "${ENV_PREFIX}" "$@")
}

run_cmd "foldtree2 --about" foldtree2 --about
run_cmd "pdbs-to-graphs --help" pdbs-to-graphs --help
run_cmd "makesubmat --help" makesubmat --help
run_cmd "ft2treebuilder --help" ft2treebuilder --help
run_cmd "foldcomp2fasta --help" foldcomp2fasta --help
run_cmd "tune-monodecoder-bayes --help" tune-monodecoder-bayes --help
run_cmd "raxml-ng --version" raxml-ng --version
run_cmd "mafft --version" mafft --version

printf '>smoke\n01 02 03\n' > "${WORK_DIR}/tokens.hex"
conda run --no-capture-output -p "${ENV_PREFIX}" hex2maffttext < "${WORK_DIR}/tokens.hex" > "${WORK_DIR}/tokens.text"
conda run --no-capture-output -p "${ENV_PREFIX}" maffttext2hex < "${WORK_DIR}/tokens.text" > "${WORK_DIR}/tokens.roundtrip"
grep -q '01 02 03' "${WORK_DIR}/tokens.roundtrip" || fail "Token conversion round trip failed"
printf '((A:0.101,B:0.203):0.307,(C:0.409,D:0.503):0.607);\n' > "${WORK_DIR}/tree.nwk"
run_cmd "MAD rooting" mad "${WORK_DIR}/tree.nwk"
[[ -s "${WORK_DIR}/tree.nwk.rooted" ]] || fail "MAD did not produce a rooted tree"

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
