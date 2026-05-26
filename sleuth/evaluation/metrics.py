from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sleuth.evaluation.categories import CANONICAL_CATEGORIES


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    score_sum = sum(float(item.get("score", 0.0)) for item in predictions)
    correct = sum(1 for item in predictions if float(item.get("score", 0.0)) > 0.0)
    return {
        "total": total,
        "average_score": score_sum / total if total else 0.0,
        "accuracy": score_sum / total if total else 0.0,
        "correct": correct,
        "count_correct": correct,
        "count_total": total,
        "correctness_accuracy": correct / total if total else 0.0,
    }


def compute_metrics_by_category(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    categories = list(CANONICAL_CATEGORIES)
    for prediction in predictions:
        for category in prediction.get("categories", []):
            if category not in categories:
                categories.append(category)

    for category in categories:
        subset = [prediction for prediction in predictions if category in prediction.get("categories", [])]
        total = len(subset)
        score_sum = sum(float(item.get("score", 0.0)) for item in subset)
        correct = sum(1 for item in subset if float(item.get("score", 0.0)) > 0.0)
        rows.append(
            {
                "category": category,
                "total": total,
                "correct": correct,
                "accuracy": score_sum / total if total else 0.0,
                "average_score": score_sum / total if total else 0.0,
            }
        )
    return rows


def save_metrics_by_category_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "total", "correct", "accuracy", "average_score"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
