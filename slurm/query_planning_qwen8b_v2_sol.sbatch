#!/usr/bin/env bash
#SBATCH --job-name=qwen8b-query-plan-v2
#SBATCH --account=grp_vgupt140
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -G a100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SBATCH_DIR}/.." && pwd)}}"
cd "${PROJECT_ROOT}"

SLEUTH_ENV_NAME="${SLEUTH_ENV_NAME:-$HOME/mamba-envs/sleuth-static}"
PYTHON_BIN="${PYTHON_BIN:-${SLEUTH_ENV_NAME}/bin/python}"
QUESTIONS_JSON="${QUESTIONS_JSON:-${PROJECT_ROOT}/colpali_v1_canvas_test_corpus/question_list.json}"
PROMPT_FILE="${PROMPT_FILE:-${PROJECT_ROOT}/prompts/query_planning_v2_exact.txt}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/agenticdocai/analysis/query_planning_qwen8b_v2_${SLURM_JOB_ID:-manual}}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
LIMIT="${LIMIT:-20}"
QIDS="${QIDS:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[query-planning-v2] ERROR: Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${QUESTIONS_JSON}" ]]; then
  echo "[query-planning-v2] ERROR: questions JSON not found: ${QUESTIONS_JSON}" >&2
  exit 1
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[query-planning-v2] ERROR: exact prompt not found: ${PROMPT_FILE}" >&2
  exit 1
fi

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="${SLEUTH_ENV_NAME}/bin:${PATH}"
export LD_LIBRARY_PATH="${SLEUTH_ENV_NAME}/lib:${SLEUTH_ENV_NAME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TMPDIR}" "${OUT_DIR}"

"${PYTHON_BIN}" - <<'PY'
import os
import torch
import transformers
print("[query-planning-v2] dependency preflight ok")
print(f"[query-planning-v2] torch: {torch.__version__}")
print(f"[query-planning-v2] transformers: {transformers.__version__}")
print(f"[query-planning-v2] cuda available: {torch.cuda.is_available()}")
print(f"[query-planning-v2] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
PY

cmd=(
  "${PYTHON_BIN}" scripts/run_query_planning_batch.py
  --questions-json "${QUESTIONS_JSON}"
  --prompt-file "${PROMPT_FILE}"
  --output-dir "${OUT_DIR}"
  --model "${MODEL}"
  --device cuda
  --dtype bfloat16
  --limit "${LIMIT}"
)
[[ -n "${QIDS}" ]] && cmd+=(--qid "${QIDS}")

echo "[query-planning-v2] model: ${MODEL}"
echo "[query-planning-v2] mode: non-thinking instruct"
echo "[query-planning-v2] questions: ${QUESTIONS_JSON}"
echo "[query-planning-v2] prompt: ${PROMPT_FILE}"
echo "[query-planning-v2] output: ${OUT_DIR}"
echo '[query-planning-v2] parameters: {"temperature":0,"top_p":1,"max_tokens":200,"frequency_penalty":0,"presence_penalty":0}'
printf '[query-planning-v2] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
echo "[query-planning-v2] done: ${OUT_DIR}"
