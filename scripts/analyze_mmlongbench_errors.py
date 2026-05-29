#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def _load_predictions(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _short(text: Any, limit: int) -> str:
    value = str(text or "").replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _print_failure_summary(predictions: list[dict[str, Any]]) -> None:
    wrong = [item for item in predictions if float(item.get("score", 0.0)) == 0.0]
    by_label = collections.Counter(item.get("failure_label", "unknown") for item in wrong)
    by_category: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for item in wrong:
        for category in item.get("categories", []) or [item.get("category", "unknown")]:
            by_category[item.get("failure_label", "unknown")][category] += 1

    print("Failure Labels")
    print("==============")
    print(dict(by_label))
    print()
    print("Failure Labels By Category")
    print("==========================")
    for label, categories in by_category.items():
        print(f"{label}: {dict(categories)}")


def _print_retrieval_summary(predictions: list[dict[str, Any]]) -> None:
    rows = [item for item in predictions if item.get("gold_hit_at_k") is not None]
    if not rows:
        return
    hit = sum(1 for item in rows if item.get("gold_hit_at_k"))
    all_hit = sum(1 for item in rows if item.get("gold_all_hit_at_k"))
    recall = sum(float(item.get("gold_page_recall_at_k") or 0.0) for item in rows) / len(rows)
    print()
    print("Retrieval")
    print("=========")
    print(f"gold_hit_at_k: {hit}/{len(rows)} ({hit / len(rows):.4f})")
    print(f"gold_all_hit_at_k: {all_hit}/{len(rows)} ({all_hit / len(rows):.4f})")
    print(f"gold_page_recall_at_k_average: {recall:.4f}")


def _print_clue_summary(predictions: list[dict[str, Any]]) -> None:
    rows = [item for item in predictions if item.get("clue_hit_gold") is not None]
    if not rows:
        return
    hit = sum(1 for item in rows if item.get("clue_hit_gold"))
    any_evidence = sum(1 for item in rows if item.get("clue_has_any_evidence"))
    non_gold = sum(1 for item in rows if item.get("clue_non_gold_pages_with_evidence"))
    print()
    print("Clue Discovery")
    print("==============")
    print(f"clue_hit_gold: {hit}/{len(rows)} ({hit / len(rows):.4f})")
    print(f"clue_has_any_evidence: {any_evidence}/{len(rows)} ({any_evidence / len(rows):.4f})")
    print(f"clue_non_gold_evidence_cases: {non_gold}/{len(rows)} ({non_gold / len(rows):.4f})")


def _print_examples(predictions: list[dict[str, Any]], label: str | None, limit: int, show_clues: bool) -> None:
    wrong = [item for item in predictions if float(item.get("score", 0.0)) == 0.0]
    if label:
        wrong = [item for item in wrong if item.get("failure_label") == label]
    print()
    print(f"Examples ({label or 'all wrong'}, first {min(limit, len(wrong))})")
    print("=" * 80)
    for item in wrong[:limit]:
        print()
        print(f"QID {item.get('question_id')} | {item.get('category')} | {item.get('failure_label')}")
        print(f"Q: {_short(item.get('question'), 300)}")
        print(f"GT: {item.get('ground_truth_answer')}")
        print(f"Ans: {item.get('model_answer')}")
        print(f"Raw: {_short(item.get('raw_model_answer'), 300)}")
        print(f"Retrieved: {item.get('retrieved_page_indices')} display={item.get('retrieved_display_page_numbers')}")
        print(
            f"Gold: {item.get('gold_evidence_pages')} "
            f"display={item.get('gold_display_page_numbers')} "
            f"source={item.get('source_evidence_pages')}"
        )
        print(
            f"Gold retrieved: {item.get('gold_retrieved_pages')} "
            f"display={item.get('gold_retrieved_display_page_numbers')}"
        )
        print(f"Gold recall: {item.get('gold_page_recall_at_k')}")
        print(
            f"Clue pages: {item.get('clue_pages_with_evidence')} "
            f"display={item.get('clue_display_page_numbers_with_evidence')}"
        )
        print(
            f"Clue gold pages: {item.get('clue_gold_pages_with_evidence')} "
            f"display={item.get('clue_gold_display_page_numbers_with_evidence')}"
        )
        print(
            f"Clue non-gold pages: {item.get('clue_non_gold_pages_with_evidence')} "
            f"display={item.get('clue_non_gold_display_page_numbers_with_evidence')}"
        )

        if not show_clues:
            continue
        for clue in item.get("clue_output", []):
            page_index = clue.get("page_index")
            gold = page_index in set(item.get("gold_evidence_pages", []))
            relevant = clue.get("has_relevant_evidence")
            evidence_items = clue.get("evidence_items", [])
            print(f"  Page {page_index} gold={gold} relevant={relevant} items={len(evidence_items)}")
            if clue.get("page_summary"):
                print(f"    summary: {_short(clue.get('page_summary'), 240)}")
            if clue.get("key_insights"):
                print(f"    insights: {_short(clue.get('key_insights'), 240)}")
            for evidence in evidence_items[:3]:
                print(f"    evidence: {_short(evidence.get('content'), 240)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MMLongBench prediction errors.")
    parser.add_argument("output_dir", help="Evaluation output directory containing predictions.jsonl")
    parser.add_argument("--label", default=None, help="Only print examples for one failure label")
    parser.add_argument("--limit", type=int, default=10, help="Maximum examples to print")
    parser.add_argument("--show-clues", action="store_true", help="Print clue summaries and evidence snippets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = _load_predictions(Path(args.output_dir))
    _print_failure_summary(predictions)
    _print_retrieval_summary(predictions)
    _print_clue_summary(predictions)
    _print_examples(predictions, args.label, args.limit, args.show_clues)


if __name__ == "__main__":
    main()
