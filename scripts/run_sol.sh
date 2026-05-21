#!/usr/bin/env bash
set -euo pipefail

python scripts/run_sleuth_baseline.py \
  --pdf "${1:-data/example.pdf}" \
  --question "${2:-According to Table II, which datasets have exactly three methods?}" \
  --out-dir "${3:-runs/sol_test}" \
  --mode sol
