#!/usr/bin/env bash
#SBATCH --job-name=sleuth-colpali-patches
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
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/export_colpali_query_patch_scores.py" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "${SBATCH_DIR}/.." && pwd)"
fi

cd "${PROJECT_ROOT}"

SLEUTH_ENV_NAME="${SLEUTH_ENV_NAME:-$HOME/mamba-envs/sleuth-static}"
PYTHON_BIN="${PYTHON_BIN:-${SLEUTH_ENV_NAME}/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[colpali-patches] ERROR: Python not found or not executable at ${PYTHON_BIN}" >&2
  echo "[colpali-patches] Set SLEUTH_ENV_NAME=/path/to/env or PYTHON_BIN=/path/to/python before sbatch." >&2
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

CORPUS_JSON="${CORPUS_JSON:-/scratch/$USER/agenticdocai/analysis/colpali_crop_test_corpus/crop_colpali_test_corpus.json}"
CACHE_DIR="${CACHE_DIR:-/scratch/$USER/agenticdocai/cache/mmlongbench_doc}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/agenticdocai/analysis/colpali_query_patch_scores_${SLURM_JOB_ID:-manual}}"
RETRIEVER="${RETRIEVER:-vidore/colpali-v1.3-hf}"
RENDER_DPI="${RENDER_DPI:-144}"
TOP_K="${TOP_K:-5}"
DEVICE="${DEVICE:-cuda}"
QIDS="${QIDS:-}"
QID_FILE="${QID_FILE:-}"
MAX_CASES="${MAX_CASES:-}"
TOP_PATCHES_PER_TOKEN="${TOP_PATCHES_PER_TOKEN:-10}"
COPY_PAGE_IMAGES="${COPY_PAGE_IMAGES:-1}"
INCLUDE_FULL_SIMILARITY_MATRIX="${INCLUDE_FULL_SIMILARITY_MATRIX:-1}"

if [[ ! -f "${CORPUS_JSON}" ]]; then
  echo "[colpali-patches] ERROR: CORPUS_JSON does not exist: ${CORPUS_JSON}" >&2
  echo "[colpali-patches] Generate it first, for example:" >&2
  echo "  ${PYTHON_BIN} scripts/select_colpali_crop_test_corpus.py \\" >&2
  echo "    --data_dir /scratch/\\$USER/agenticdocai/data/MMLongBench-Doc \\" >&2
  echo "    --cache_dir /scratch/\\$USER/agenticdocai/cache/mmlongbench_doc \\" >&2
  echo "    --output_dir /scratch/\\$USER/agenticdocai/analysis/colpali_crop_test_corpus \\" >&2
  echo "    --retriever vidore/colpali-v1.3-hf --top_k 5 --render_dpi 144 --per-category 4 --max-cases 25" >&2
  exit 1
fi

if [[ ! -d "${CACHE_DIR}" ]]; then
  echo "[colpali-patches] ERROR: CACHE_DIR does not exist: ${CACHE_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<PY
import os
import sys
import numpy
import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor
print("[colpali-patches] dependency preflight ok")
print(f"[colpali-patches] executable: {sys.executable}")
print(f"[colpali-patches] python: {sys.version.split()[0]}")
print(f"[colpali-patches] numpy: {numpy.__version__}")
print(f"[colpali-patches] torch: {torch.__version__}")
print(f"[colpali-patches] cuda available: {torch.cuda.is_available()}")
print(f"[colpali-patches] cuda device count: {torch.cuda.device_count()}")
print(f"[colpali-patches] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
PY

echo "[colpali-patches] project root: ${PROJECT_ROOT}"
echo "[colpali-patches] corpus json: ${CORPUS_JSON}"
echo "[colpali-patches] cache dir: ${CACHE_DIR}"
echo "[colpali-patches] out dir: ${OUT_DIR}"
echo "[colpali-patches] retriever: ${RETRIEVER}"
echo "[colpali-patches] render_dpi: ${RENDER_DPI}"
echo "[colpali-patches] top_k: ${TOP_K}"
echo "[colpali-patches] device: ${DEVICE}"
echo "[colpali-patches] qids: ${QIDS:-<none>}"
echo "[colpali-patches] qid file: ${QID_FILE:-<none>}"
echo "[colpali-patches] max cases: ${MAX_CASES:-<none>}"
echo "[colpali-patches] copy page images: ${COPY_PAGE_IMAGES}"
echo "[colpali-patches] include full similarity matrix: ${INCLUDE_FULL_SIMILARITY_MATRIX}"
echo "[colpali-patches] top patches per token: ${TOP_PATCHES_PER_TOKEN}"
echo "[colpali-patches] python: ${PYTHON_BIN}"

cmd=(
  "${PYTHON_BIN}" scripts/export_colpali_query_patch_scores.py
  --corpus_json "${CORPUS_JSON}"
  --cache_dir "${CACHE_DIR}"
  --output_dir "${OUT_DIR}"
  --retriever "${RETRIEVER}"
  --render_dpi "${RENDER_DPI}"
  --top_k "${TOP_K}"
  --device "${DEVICE}"
  --top-patches-per-token "${TOP_PATCHES_PER_TOKEN}"
)

if [[ -n "${QIDS}" ]]; then
  cmd+=(--qid "${QIDS}")
fi

if [[ -n "${QID_FILE}" ]]; then
  cmd+=(--qid-file "${QID_FILE}")
fi

if [[ -n "${MAX_CASES}" ]]; then
  cmd+=(--max-cases "${MAX_CASES}")
fi

if [[ "${COPY_PAGE_IMAGES}" == "0" || "${COPY_PAGE_IMAGES}" == "false" || "${COPY_PAGE_IMAGES}" == "FALSE" ]]; then
  cmd+=(--no-copy-page-images)
fi

if [[ "${INCLUDE_FULL_SIMILARITY_MATRIX}" == "0" || "${INCLUDE_FULL_SIMILARITY_MATRIX}" == "false" || "${INCLUDE_FULL_SIMILARITY_MATRIX}" == "FALSE" ]]; then
  cmd+=(--no-full-similarity-matrix)
fi

echo "[colpali-patches] command:"
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

echo "[colpali-patches] done"
echo "[colpali-patches] output: ${OUT_DIR}"
