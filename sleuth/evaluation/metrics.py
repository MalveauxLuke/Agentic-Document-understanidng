from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sleuth.evaluation.categories import CANONICAL_CATEGORIES


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    score_sum = sum(float(item.get("score", 0.0)) for item in predictions)
    correct = sum(1 for item in predictions if float(item.get("score", 0.0)) > 0.0)
    raw_score_sum = sum(float(item.get("raw_score", 0.0)) for item in predictions)
    raw_correct = sum(1 for item in predictions if float(item.get("raw_score", 0.0)) > 0.0)
    gold_diagnostic_rows = [item for item in predictions if item.get("gold_hit_at_k") is not None]
    gold_hits = sum(1 for item in gold_diagnostic_rows if item.get("gold_hit_at_k"))
    gold_all_hits = sum(1 for item in gold_diagnostic_rows if item.get("gold_all_hit_at_k"))
    gold_recall_values = [
        float(item.get("gold_page_recall_at_k", 0.0))
        for item in gold_diagnostic_rows
        if item.get("gold_page_recall_at_k") is not None
    ]
    clue_rows = [item for item in predictions if item.get("clue_hit_gold") is not None]
    clue_hits = sum(1 for item in clue_rows if item.get("clue_hit_gold"))
    clue_any_rows = [item for item in predictions if item.get("clue_has_any_evidence") is not None]
    clue_any_hits = sum(1 for item in clue_any_rows if item.get("clue_has_any_evidence"))
    verification_rows = [item for item in predictions if item.get("verification_retained_gold") is not None]
    verification_hits = sum(1 for item in verification_rows if item.get("verification_retained_gold"))
    verification_rejected_count = sum(int(item.get("verification_rejected_count") or 0) for item in predictions)
    verification_uncertain_count = sum(int(item.get("verification_uncertain_count") or 0) for item in predictions)
    verification_full_page_fallback_count = sum(
        int(item.get("verification_full_page_fallback_count") or 0) for item in predictions
    )
    core_decision_rows = [item for item in predictions if item.get("core_decision_mode")]
    thinking_core_decisions = sum(1 for item in core_decision_rows if item.get("core_decision_mode") == "thinking")
    changed = sum(1 for item in predictions if item.get("raw_model_answer") != item.get("extracted_answer"))
    metrics = {
        "total": total,
        "average_score": score_sum / total if total else 0.0,
        "accuracy": score_sum / total if total else 0.0,
        "correct": correct,
        "count_correct": correct,
        "count_total": total,
        "correctness_accuracy": correct / total if total else 0.0,
        "raw_average_score": raw_score_sum / total if total else 0.0,
        "raw_correct": raw_correct,
        "raw_correctness_accuracy": raw_correct / total if total else 0.0,
        "answer_extraction_changed": changed,
        "answer_extraction_changed_rate": changed / total if total else 0.0,
        "gold_hit_at_k_count": gold_hits,
        "gold_hit_at_k_total": len(gold_diagnostic_rows),
        "gold_hit_at_k_rate": gold_hits / len(gold_diagnostic_rows) if gold_diagnostic_rows else 0.0,
        "gold_all_hit_at_k_count": gold_all_hits,
        "gold_all_hit_at_k_total": len(gold_diagnostic_rows),
        "gold_all_hit_at_k_rate": gold_all_hits / len(gold_diagnostic_rows) if gold_diagnostic_rows else 0.0,
        "gold_page_recall_at_k_average": (
            sum(gold_recall_values) / len(gold_recall_values) if gold_recall_values else 0.0
        ),
        "clue_hit_gold_count": clue_hits,
        "clue_hit_gold_total": len(clue_rows),
        "clue_hit_gold_rate": clue_hits / len(clue_rows) if clue_rows else 0.0,
        "clue_has_any_evidence_count": clue_any_hits,
        "clue_has_any_evidence_total": len(clue_any_rows),
        "clue_has_any_evidence_rate": clue_any_hits / len(clue_any_rows) if clue_any_rows else 0.0,
        "screening_retained_gold_count": 0,
        "screening_retained_gold_total": 0,
        "screening_retained_gold_rate": 0.0,
        "verification_retained_gold_count": verification_hits,
        "verification_retained_gold_total": len(verification_rows),
        "verification_retained_gold_rate": (
            verification_hits / len(verification_rows) if verification_rows else 0.0
        ),
        "verification_rejected_count": verification_rejected_count,
        "verification_uncertain_count": verification_uncertain_count,
        "verification_full_page_fallback_count": verification_full_page_fallback_count,
        "core_decision_thinking_count": thinking_core_decisions,
        "core_decision_thinking_total": len(core_decision_rows),
        "core_decision_thinking_rate": (
            thinking_core_decisions / len(core_decision_rows) if core_decision_rows else 0.0
        ),
    }
    return metrics


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
