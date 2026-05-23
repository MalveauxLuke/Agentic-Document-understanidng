#!/usr/bin/env bash
#SBATCH --job-name=sleuth-mmlb-doc
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

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
else
  SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SBATCH_DIR}/.." && pwd)}"
fi

cd "${PROJECT_ROOT}"

module load mamba/latest
source activate "${SLEUTH_ENV_NAME:-sleuth-static}"

export HF_HOME="${HF_HOME:-/scratch/$USER/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"

PDF="${PDF:-samples/mmlongbench_doc/a4f3ced0696009fec3179f493e4f28c4.pdf}"
QUESTION="${QUESTION:-WHAT IS USCA CASE NUMBER?}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/agenticdocai/runs/mmlongbench_doc_sol_${SLURM_JOB_ID}}"

echo "[sleuth-sol] project root: ${PROJECT_ROOT}"
echo "[sleuth-sol] pdf: ${PDF}"
echo "[sleuth-sol] question: ${QUESTION}"
echo "[sleuth-sol] out dir: ${OUT_DIR}"
echo "[sleuth-sol] python: $(which python)"
python -V

python scripts/run_sleuth_baseline.py \
  --pdf "${PDF}" \
  --question "${QUESTION}" \
  --out-dir "${OUT_DIR}" \
  --mode sol

echo "[sleuth-sol] done"
echo "[sleuth-sol] final answer:"
cat "${OUT_DIR}/final/final_answer.json"
