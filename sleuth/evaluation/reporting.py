from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _load_category_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def build_results_summary(output_dir: str | Path, max_examples: int = 5) -> str:
    out_dir = Path(output_dir)
    metrics = _load_json(out_dir / "metrics.json")
    run_config = _load_json(out_dir / "run_config.json")
    category_rows = _load_category_rows(out_dir / "metrics_by_category.csv")
    predictions = _load_jsonl(out_dir / "predictions.jsonl", limit=max_examples)
    failures = _load_jsonl(out_dir / "failed_examples.jsonl", limit=max_examples)

    lines = [
        "MMLongBench-Doc Results",
        "=" * 25,
        f"Output directory: {out_dir}",
    ]

    if run_config:
        lines.extend(
            [
                f"Method: {run_config.get('method')}",
                f"Mode: {run_config.get('mode')}",
                f"Model: {run_config.get('model')}",
                f"Retriever: {run_config.get('actual_retriever') or run_config.get('retriever')}",
                f"Top-k: {run_config.get('top_k')}",
                f"Limit: {run_config.get('limit')}",
                f"Category filter: {run_config.get('category') or 'None'}",
            ]
        )

    if metrics:
        lines.extend(
            [
                "",
                "Overall",
                f"  attempted: {metrics.get('attempted', metrics.get('count_total', 0))}",
                f"  scored: {metrics.get('total', 0)}",
                f"  failed: {metrics.get('failed', 0)}",
                f"  average_score: {_format_float(metrics.get('average_score', 0.0))}",
                f"  count_correct: {metrics.get('count_correct', metrics.get('correct', 0))}/{metrics.get('count_total', metrics.get('total', 0))}",
                f"  correctness_accuracy: {_format_float(metrics.get('correctness_accuracy', 0.0))}",
                f"  raw_average_score: {_format_float(metrics.get('raw_average_score', 0.0))}",
                f"  raw_correctness_accuracy: {_format_float(metrics.get('raw_correctness_accuracy', 0.0))}",
                f"  retrieval_hit_at_k: {metrics.get('gold_hit_at_k_count', 0)}/{metrics.get('gold_hit_at_k_total', 0)} ({_format_float(metrics.get('gold_hit_at_k_rate', 0.0))})",
                f"  retrieval_all_hit_at_k: {metrics.get('gold_all_hit_at_k_count', 0)}/{metrics.get('gold_all_hit_at_k_total', 0)} ({_format_float(metrics.get('gold_all_hit_at_k_rate', 0.0))})",
                f"  retrieval_page_recall_at_k: {_format_float(metrics.get('gold_page_recall_at_k_average', 0.0))}",
                f"  clue_has_any_evidence: {metrics.get('clue_has_any_evidence_count', 0)}/{metrics.get('clue_has_any_evidence_total', 0)} ({_format_float(metrics.get('clue_has_any_evidence_rate', 0.0))})",
                f"  clue_hit_gold: {metrics.get('clue_hit_gold_count', 0)}/{metrics.get('clue_hit_gold_total', 0)} ({_format_float(metrics.get('clue_hit_gold_rate', 0.0))})",
                f"  verification_retained_gold: {metrics.get('verification_retained_gold_count', 0)}/{metrics.get('verification_retained_gold_total', 0)} ({_format_float(metrics.get('verification_retained_gold_rate', 0.0))})",
                f"  verification_rejected: {metrics.get('verification_rejected_count', 0)}",
                f"  verification_uncertain: {metrics.get('verification_uncertain_count', 0)}",
                f"  verification_full_page_fallback: {metrics.get('verification_full_page_fallback_count', 0)}",
                f"  core_decision_thinking: {metrics.get('core_decision_thinking_count', 0)}/{metrics.get('core_decision_thinking_total', 0)} ({_format_float(metrics.get('core_decision_thinking_rate', 0.0))})",
                f"  answer_extractor: {metrics.get('answer_extractor', 'unknown')}",
                f"  paper_comparable_scoring: {metrics.get('paper_comparable_scoring', False)}",
            ]
        )

    if category_rows:
        lines.extend(["", "By Category", "  category       total  correct  avg_score"])
        for row in category_rows:
            lines.append(
                f"  {row.get('category', ''):<13} "
                f"{row.get('total', ''):>5}  "
                f"{row.get('correct', ''):>7}  "
                f"{_format_float(row.get('average_score', 0.0)):>9}"
            )

    if predictions:
        lines.extend(["", f"Sample Predictions (first {len(predictions)})"])
        for item in predictions:
            lines.extend(
                [
                    f"  - question_id: {item.get('question_id')}",
                    f"    document_id: {item.get('document_id')}",
                    f"    category: {', '.join(item.get('categories', [])) or item.get('category')}",
                    f"    score: {_format_float(item.get('score', 0.0))}",
                    f"    raw_score: {_format_float(item.get('raw_score', 0.0))}",
                    f"    ground_truth: {item.get('ground_truth_answer')}",
                    f"    model_answer: {item.get('model_answer')}",
                    f"    raw_model_answer: {item.get('raw_model_answer')}",
                    f"    core_decision_mode: {item.get('core_decision_mode')} difficulty={item.get('difficulty_level')}",
                    f"    retrieved_pages: {item.get('retrieved_page_indices')} display={item.get('retrieved_display_page_numbers')}",
                    f"    gold_pages: {item.get('gold_evidence_pages')} display={item.get('gold_display_page_numbers')}",
                    f"    failure_label: {item.get('failure_label')}",
                ]
            )

    if failures:
        lines.extend(["", f"Failures (first {len(failures)})"])
        for item in failures:
            lines.extend(
                [
                    f"  - question_id: {item.get('question_id')}",
                    f"    document_id: {item.get('doc_id') or item.get('document_id')}",
                    f"    error: {item.get('errors')}",
                ]
            )

    lines.extend(
        [
            "",
            "Files",
            f"  predictions: {out_dir / 'predictions.jsonl'}",
            f"  metrics: {out_dir / 'metrics.json'}",
            f"  categories: {out_dir / 'metrics_by_category.csv'}",
            f"  failures: {out_dir / 'failed_examples.jsonl'}",
            f"  config: {out_dir / 'run_config.json'}",
        ]
    )
    return "\n".join(lines)


def print_results_summary(output_dir: str | Path, max_examples: int = 5) -> None:
    print(build_results_summary(output_dir, max_examples=max_examples))
