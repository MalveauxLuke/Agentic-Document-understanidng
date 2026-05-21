#!/usr/bin/env bash
set -euo pipefail

python scripts/run_sleuth_baseline.py \
  --pdf "${1:-data/example.pdf}" \
  --question "${2:-What is the main result?}" \
  --out-dir "${3:-runs/mock_test}" \
  --mode mock \
  --retriever dummy \
  --llm mock
