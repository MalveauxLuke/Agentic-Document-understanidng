#!/usr/bin/env bash
#SBATCH --job-name=sleuth-sol
#SBATCH --account=grp_vgupt140
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -G a100:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/run_sleuth_baseline.py" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

cd "${PROJECT_ROOT}"

SLEUTH_ENV_NAME="${SLEUTH_ENV_NAME:-$HOME/mamba-envs/sleuth-static}"
PYTHON_BIN="${PYTHON_BIN:-${SLEUTH_ENV_NAME}/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[sleuth-sol] ERROR: Python not found or not executable at ${PYTHON_BIN}" >&2
  echo "[sleuth-sol] Set SLEUTH_ENV_NAME=/path/to/env or PYTHON_BIN=/path/to/python before sbatch." >&2
  exit 1
fi

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="${SLEUTH_ENV_NAME}/bin:${PATH}"
export LD_LIBRARY_PATH="${SLEUTH_ENV_NAME}/lib:${SLEUTH_ENV_NAME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

export HF_HOME="${HF_HOME:-/scratch/$USER/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"

: "${PDF:?Set PDF=/path/to/input.pdf before submitting.}"
: "${QUESTION:?Set QUESTION='your question' before submitting.}"
: "${OUT_DIR:=/scratch/$USER/agenticdocai/runs/sol_${SLURM_JOB_ID}}"

"${PYTHON_BIN}" - <<'PY'
import sys
import pydantic
import torch
import torchvision
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import ColPaliForRetrieval, ColPaliProcessor
import qwen_vl_utils

print("[sleuth-sol] dependency preflight ok")
print(f"[sleuth-sol] executable: {sys.executable}")
print(f"[sleuth-sol] torch: {torch.__version__}")
print(f"[sleuth-sol] torchvision: {torchvision.__version__}")
print(f"[sleuth-sol] cuda available: {torch.cuda.is_available()}")
PY

echo "[sleuth-sol] project root: ${PROJECT_ROOT}"
echo "[sleuth-sol] pdf: ${PDF}"
echo "[sleuth-sol] question: ${QUESTION}"
echo "[sleuth-sol] out dir: ${OUT_DIR}"
echo "[sleuth-sol] python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V

"${PYTHON_BIN}" scripts/run_sleuth_baseline.py \
  --pdf "$PDF" \
  --question "$QUESTION" \
  --out-dir "$OUT_DIR" \
  --mode sol
