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

# Compatibility wrapper for the one-line command:
#   sbatch slurm/gdpo_debug_upstream.sbatch
#
# This repository does not contain a GDPO/VERL experiment. This file is kept
# only as a convenience alias for launching the static SLEUTH baseline on the
# included MMLongBench-Doc sample.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/sleuth_mmlongbench_doc_sol.sbatch" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

if [[ ! -f "${PROJECT_ROOT}/slurm/sleuth_mmlongbench_doc_sol.sbatch" ]]; then
  echo "[sleuth-sol] ERROR: Could not find ${PROJECT_ROOT}/slurm/sleuth_mmlongbench_doc_sol.sbatch" >&2
  echo "[sleuth-sol] Submit from the repo root or set PROJECT_ROOT=/path/to/Agentic-Document-understanidng." >&2
  exit 1
fi

export SLEUTH_ENV_NAME="${SLEUTH_ENV_NAME:-$HOME/mamba-envs/sleuth-static}"
export PROJECT_ROOT
exec bash "${PROJECT_ROOT}/slurm/sleuth_mmlongbench_doc_sol.sbatch" "$@"
