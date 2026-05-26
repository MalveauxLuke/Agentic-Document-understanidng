#!/usr/bin/env bash
#SBATCH --job-name=install-mmlb-doc
#SBATCH --account=grp_vgupt140
#SBATCH --partition=lightwork
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/install_mmlongbench_doc.py" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "${SBATCH_DIR}/.." && pwd)"
fi

cd "${PROJECT_ROOT}"

SLEUTH_ENV_NAME="${SLEUTH_ENV_NAME:-$HOME/mamba-envs/sleuth-static}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "${SLEUTH_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN="${SLEUTH_ENV_NAME}/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

if [[ -z "${PYTHON_BIN:-}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "[mmlb-install] ERROR: Could not find a usable Python interpreter." >&2
  echo "[mmlb-install] Set PYTHON_BIN=/path/to/python before sbatch." >&2
  exit 1
fi

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
if [[ -d "${SLEUTH_ENV_NAME}" ]]; then
  export PATH="${SLEUTH_ENV_NAME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${SLEUTH_ENV_NAME}/lib:${SLEUTH_ENV_NAME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export MMLONGBENCH_DOC_DIR="${MMLONGBENCH_DOC_DIR:-/scratch/$USER/agenticdocai/data/MMLongBench-Doc}"
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-}"
LIMIT_DOCS="${LIMIT_DOCS:-}"
FORCE="${FORCE:-0}"
MANIFEST_ONLY="${MANIFEST_ONLY:-0}"

echo "[mmlb-install] project root: ${PROJECT_ROOT}"
echo "[mmlb-install] target: ${MMLONGBENCH_DOC_DIR}"
echo "[mmlb-install] python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V

cmd=(
  "${PYTHON_BIN}" scripts/install_mmlongbench_doc.py
  --target-dir "${MMLONGBENCH_DOC_DIR}"
)

if [[ -n "${LIMIT_EXAMPLES}" ]]; then
  cmd+=(--limit-examples "${LIMIT_EXAMPLES}")
fi

if [[ -n "${LIMIT_DOCS}" ]]; then
  cmd+=(--limit-docs "${LIMIT_DOCS}")
fi

if [[ "${FORCE}" == "1" || "${FORCE}" == "true" || "${FORCE}" == "TRUE" ]]; then
  cmd+=(--force)
fi

if [[ "${MANIFEST_ONLY}" == "1" || "${MANIFEST_ONLY}" == "true" || "${MANIFEST_ONLY}" == "TRUE" ]]; then
  cmd+=(--manifest-only)
fi

echo "[mmlb-install] command:"
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
