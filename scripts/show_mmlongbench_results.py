#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sleuth.evaluation.reporting import print_results_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a compact MMLongBench-Doc run summary.")
    parser.add_argument("output_dir", help="Evaluation output directory.")
    parser.add_argument("--max-examples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_results_summary(args.output_dir, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
