#!/usr/bin/env bash
#SBATCH --job-name=sleuth-mmlb-base
#SBATCH --account=grp_vgupt140
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -G a100:2
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/mmlongbench_eval_sol.sbatch" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

if [[ ! -f "${PROJECT_ROOT}/slurm/mmlongbench_eval_sol.sbatch" ]]; then
  echo "[mmlb-eval] ERROR: Could not find ${PROJECT_ROOT}/slurm/mmlongbench_eval_sol.sbatch" >&2
  echo "[mmlb-eval] Submit from the repo root or set PROJECT_ROOT=/path/to/Agentic-Document-understanidng." >&2
  exit 1
fi

export METHOD=base
export PROJECT_ROOT
exec bash "${PROJECT_ROOT}/slurm/mmlongbench_eval_sol.sbatch" "$@"
