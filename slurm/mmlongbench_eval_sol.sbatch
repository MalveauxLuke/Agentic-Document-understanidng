#!/usr/bin/env bash
#SBATCH --job-name=sleuth-mmlb-eval
#SBATCH --account=grp_vgupt140
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -G a100:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/run_mmlongbench_eval.py" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "${SBATCH_DIR}/.." && pwd)"
fi

cd "${PROJECT_ROOT}"

SLEUTH_ENV_NAME="${SLEUTH_ENV_NAME:-$HOME/mamba-envs/sleuth-static}"
PYTHON_BIN="${PYTHON_BIN:-${SLEUTH_ENV_NAME}/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[mmlb-eval] ERROR: Python not found or not executable at ${PYTHON_BIN}" >&2
  echo "[mmlb-eval] Set SLEUTH_ENV_NAME=/path/to/env or PYTHON_BIN=/path/to/python before sbatch." >&2
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

METHOD="${METHOD:-sleuth}"
MODE="${MODE:-sol}"
DATA_DIR="${DATA_DIR:-.}"
LIMIT="${LIMIT:-3}"
CATEGORY="${CATEGORY:-}"
TOP_K="${TOP_K:-5}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
RETRIEVER="${RETRIEVER:-vidore/colpali-v1.3-hf}"
RENDER_DPI="${RENDER_DPI:-144}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/agenticdocai/runs/mmlongbench_eval_${METHOD}_${SLURM_JOB_ID:-manual}}"

case "${METHOD}" in
  base|sleuth) ;;
  *)
    echo "[mmlb-eval] ERROR: METHOD must be 'base' or 'sleuth', got '${METHOD}'." >&2
    exit 1
    ;;
esac

case "${MODE}" in
  mock|local|sol) ;;
  *)
    echo "[mmlb-eval] ERROR: MODE must be 'mock', 'local', or 'sol', got '${MODE}'." >&2
    exit 1
    ;;
esac

if [[ ! -e "${DATA_DIR}" ]]; then
  echo "[mmlb-eval] ERROR: DATA_DIR does not exist: ${DATA_DIR}" >&2
  echo "[mmlb-eval] Set DATA_DIR=/path/to/MMLongBench-Doc before sbatch." >&2
  exit 1
fi

"${PYTHON_BIN}" - <<PY
import sys
import fitz
import pydantic
import yaml
print("[mmlb-eval] core dependency preflight ok")
print(f"[mmlb-eval] executable: {sys.executable}")
print(f"[mmlb-eval] python: {sys.version.split()[0]}")
if "${MODE}" != "mock":
    import torch
    import torchvision
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers import ColPaliForRetrieval, ColPaliProcessor
    import qwen_vl_utils
    print("[mmlb-eval] heavy dependency preflight ok")
    print(f"[mmlb-eval] torch: {torch.__version__}")
    print(f"[mmlb-eval] torchvision: {torchvision.__version__}")
    print(f"[mmlb-eval] cuda available: {torch.cuda.is_available()}")
PY

echo "[mmlb-eval] project root: ${PROJECT_ROOT}"
echo "[mmlb-eval] data dir: ${DATA_DIR}"
echo "[mmlb-eval] method: ${METHOD}"
echo "[mmlb-eval] mode: ${MODE}"
echo "[mmlb-eval] model: ${MODEL}"
echo "[mmlb-eval] retriever: ${RETRIEVER}"
echo "[mmlb-eval] top_k: ${TOP_K}"
echo "[mmlb-eval] temperature: ${TEMPERATURE}"
echo "[mmlb-eval] limit: ${LIMIT}"
echo "[mmlb-eval] category: ${CATEGORY:-<none>}"
echo "[mmlb-eval] out dir: ${OUT_DIR}"
echo "[mmlb-eval] python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V

cmd=(
  "${PYTHON_BIN}" scripts/run_mmlongbench_eval.py
  --data_dir "${DATA_DIR}"
  --method "${METHOD}"
  --mode "${MODE}"
  --model "${MODEL}"
  --retriever "${RETRIEVER}"
  --top_k "${TOP_K}"
  --temperature "${TEMPERATURE}"
  --output_dir "${OUT_DIR}"
  --render_dpi "${RENDER_DPI}"
)

if [[ -n "${LIMIT}" && "${LIMIT}" != "ALL" && "${LIMIT}" != "all" && "${LIMIT}" != "NONE" && "${LIMIT}" != "none" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

if [[ -n "${CATEGORY}" ]]; then
  cmd+=(--category "${CATEGORY}")
fi

echo "[mmlb-eval] command:"
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

echo "[mmlb-eval] done"
echo "[mmlb-eval] metrics:"
cat "${OUT_DIR}/metrics.json"
echo
echo "[mmlb-eval] first prediction:"
head -n 1 "${OUT_DIR}/predictions.jsonl" || true
