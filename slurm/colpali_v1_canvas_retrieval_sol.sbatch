#!/usr/bin/env bash
#SBATCH --job-name=colpali-v1-canvas
#SBATCH --account=grp_vgupt140
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -G a100:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SBATCH_DIR}/.." && pwd)}}"
cd "${PROJECT_ROOT}"

SLEUTH_ENV_NAME="${SLEUTH_ENV_NAME:-$HOME/mamba-envs/sleuth-static}"
PYTHON_BIN="${PYTHON_BIN:-${SLEUTH_ENV_NAME}/bin/python}"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/colpali_v1_canvas_test_corpus}"
CACHE_DIR="${CACHE_DIR:-/scratch/$USER/agenticdocai/cache/mmlongbench_doc}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/agenticdocai/analysis/colpali_v1_canvas_retrieval_${SLURM_JOB_ID:-manual}}"
RETRIEVER="${RETRIEVER:-vidore/colpali-v1.3-hf}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-5}"
COLPALI_BATCH_SIZE="${COLPALI_BATCH_SIZE:-2}"
RENDER_DPI="${RENDER_DPI:-144}"
MODALITIES="${MODALITIES:-Chart,Table,Figure}"
QIDS="${QIDS:-}"
MAX_CASES="${MAX_CASES:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[v1-retrieval] ERROR: Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${INPUT_DIR}/question_list.json" ]]; then
  echo "[v1-retrieval] ERROR: missing ${INPUT_DIR}/question_list.json" >&2
  echo "[v1-retrieval] Pull the repository bundle or set INPUT_DIR explicitly." >&2
  exit 1
fi
if [[ ! -f "${INPUT_DIR}/region_canvas_visualizations/region_views.json" ]]; then
  echo "[v1-retrieval] ERROR: missing V1 canvas metadata under ${INPUT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${CACHE_DIR}/documents/dpi_${RENDER_DPI}" ]]; then
  echo "[v1-retrieval] ERROR: missing rendered document cache: ${CACHE_DIR}/documents/dpi_${RENDER_DPI}" >&2
  exit 1
fi

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="${SLEUTH_ENV_NAME}/bin:${PATH}"
export LD_LIBRARY_PATH="${SLEUTH_ENV_NAME}/lib:${SLEUTH_ENV_NAME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TMPDIR}" "${OUT_DIR}"

cmd=(
  "${PYTHON_BIN}" scripts/run_colpali_v1_canvas_retrieval.py
  --input_dir "${INPUT_DIR}"
  --output_dir "${OUT_DIR}"
  --cache_dir "${CACHE_DIR}"
  --retriever "${RETRIEVER}"
  --render_dpi "${RENDER_DPI}"
  --device "${DEVICE}"
  --top_k "${TOP_K}"
  --colpali_batch_size "${COLPALI_BATCH_SIZE}"
  --modality "${MODALITIES}"
)
[[ -n "${QIDS}" ]] && cmd+=(--qid "${QIDS}")
[[ -n "${MAX_CASES}" ]] && cmd+=(--max_cases "${MAX_CASES}")

echo "[v1-retrieval] project root: ${PROJECT_ROOT}"
echo "[v1-retrieval] input: ${INPUT_DIR}"
echo "[v1-retrieval] shared cache: ${CACHE_DIR}"
echo "[v1-retrieval] output: ${OUT_DIR}"
echo "[v1-retrieval] wall time: 01:00:00"
printf '[v1-retrieval] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
echo "[v1-retrieval] done: ${OUT_DIR}"
