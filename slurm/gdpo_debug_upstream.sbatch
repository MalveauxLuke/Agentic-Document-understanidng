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

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SBATCH_DIR}/sleuth_mmlongbench_doc_sol.sbatch" "$@"
