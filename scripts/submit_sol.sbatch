#!/usr/bin/env bash
#SBATCH --job-name=sleuth-sol
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -G 1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module load mamba/latest
source activate sleuth-static

export HF_HOME="${HF_HOME:-/scratch/$USER/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"

: "${PDF:?Set PDF=/path/to/input.pdf before submitting.}"
: "${QUESTION:?Set QUESTION='your question' before submitting.}"
: "${OUT_DIR:=/scratch/$USER/agenticdocai/runs/sol_${SLURM_JOB_ID}}"

python scripts/run_sleuth_baseline.py \
  --pdf "$PDF" \
  --question "$QUESTION" \
  --out-dir "$OUT_DIR" \
  --mode sol
