#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sleuth.evaluation.categories import CANONICAL_CATEGORIES
from sleuth.evaluation.dataset import MMLongBenchExample, load_mmlongbench_examples
from sleuth.evaluation.qid_filter import resolve_qids


DEFAULT_TARGET_CATEGORIES = ["Chart", "Table", "Figure", "Pure-text", "Layout"]


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _dpi_key(render_dpi: int) -> str:
    return f"dpi_{int(render_dpi)}"


def _retrieval_cache_path(
    cache_dir: Path,
    example: MMLongBenchExample,
    retriever_name: str,
    top_k: int,
    render_dpi: int,
) -> Path:
    key = _short_hash(
        "\n".join(
            [
                str(example.doc_id),
                str(example.question),
                str(retriever_name),
                str(top_k),
                str(render_dpi),
            ]
        )
    )
    return cache_dir / "retrieval" / _dpi_key(render_dpi) / f"{_safe_id(example.doc_id)}_{key}.json"


def _load_cached_retrieval(
    cache_dir: Path,
    example: MMLongBenchExample,
    retriever_name: str,
    top_k: int,
    render_dpi: int,
) -> list[dict[str, Any]] | None:
    path = _retrieval_cache_path(cache_dir, example, retriever_name, top_k, render_dpi)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cached_document_pages(
    cache_dir: Path,
    doc_id: str,
    render_dpi: int,
) -> list[dict[str, Any]] | None:
    for candidate_dir in (
        cache_dir / "documents" / _dpi_key(render_dpi) / _safe_id(doc_id),
        cache_dir / "documents" / _safe_id(doc_id),
    ):
        pages_json = candidate_dir / "pages.json"
        if pages_json.exists():
            return json.loads(pages_json.read_text(encoding="utf-8"))
    return None


def _display_pages(pages: list[int]) -> list[int]:
    return [page + 1 for page in pages]


def _page_index(page: dict[str, Any]) -> int:
    return int(page["page_index"])


def _page_score(page: dict[str, Any]) -> float:
    return float(page.get("score", 0.0))


def _retrieved_payload(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, page in enumerate(retrieved, start=1):
        page_index = _page_index(page)
        rows.append(
            {
                "rank": rank,
                "page_index": page_index,
                "display_page": page_index + 1,
                "score": _page_score(page),
                "reason": page.get("reason"),
            }
        )
    return rows


def _page_image_paths(
    cache_dir: Path,
    example: MMLongBenchExample,
    render_dpi: int,
    page_indices: list[int],
) -> dict[str, str]:
    pages = _load_cached_document_pages(cache_dir, example.doc_id, render_dpi)
    if not pages:
        return {}
    by_index = {int(page["page_index"]): str(page["image_path"]) for page in pages}
    return {str(page_index): by_index[page_index] for page_index in page_indices if page_index in by_index}


def _candidate_record(
    example: MMLongBenchExample,
    retrieved: list[dict[str, Any]],
    cache_dir: Path,
    render_dpi: int,
) -> dict[str, Any]:
    retrieved_pages = [_page_index(page) for page in retrieved]
    gold_pages = sorted(set(example.evidence_pages))
    retrieved_set = set(retrieved_pages)
    gold_rank = {
        _page_index(page): rank
        for rank, page in enumerate(retrieved, start=1)
        if _page_index(page) in set(gold_pages)
    }
    return {
        "question_id": example.question_id,
        "row_index": example.row_index,
        "doc_id": example.doc_id,
        "doc_type": example.doc_type,
        "question": example.question,
        "answer": example.answer,
        "answer_format": example.answer_format,
        "categories": example.categories,
        "evidence_sources": example.evidence_sources,
        "gold_pages": gold_pages,
        "gold_display_pages": _display_pages(gold_pages),
        "source_evidence_pages": example.source_evidence_pages,
        "retrieved_pages": retrieved_pages,
        "retrieved_display_pages": _display_pages(retrieved_pages),
        "retrieved": _retrieved_payload(retrieved),
        "gold_ranks": {str(page): gold_rank.get(page) for page in gold_pages},
        "all_gold_retrieved": bool(gold_pages) and set(gold_pages).issubset(retrieved_set),
        "gold_page_count": len(gold_pages),
        "page_image_paths": _page_image_paths(cache_dir, example, render_dpi, sorted(set(gold_pages + retrieved_pages))),
    }


def _selection_score(record: dict[str, Any]) -> tuple[int, int, int, str]:
    categories = set(record.get("categories") or [])
    visual_bonus = int(bool(categories & {"Chart", "Table", "Figure", "Layout"}))
    multi_page_bonus = int(record.get("gold_page_count", 0) > 1)
    best_gold_rank = min((rank for rank in (record.get("gold_ranks") or {}).values() if rank is not None), default=999)
    return (-multi_page_bonus, -visual_bonus, best_gold_rank, str(record.get("question_id")))


def select_balanced(
    candidates: list[dict[str, Any]],
    target_categories: list[str],
    per_category: int,
    max_cases: int | None,
) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        for category in record.get("categories") or ["None"]:
            by_category[category].append(record)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category in target_categories:
        rows = sorted(by_category.get(category, []), key=_selection_score)
        for record in rows[:per_category]:
            qid = str(record["question_id"])
            if qid in seen:
                continue
            seen.add(qid)
            copy = dict(record)
            copy["selected_for_category"] = category
            selected.append(copy)

    if max_cases is not None and len(selected) < max_cases:
        for record in sorted(candidates, key=_selection_score):
            qid = str(record["question_id"])
            if qid in seen:
                continue
            seen.add(qid)
            copy = dict(record)
            copy["selected_for_category"] = "fill"
            selected.append(copy)
            if len(selected) >= max_cases:
                break

    if max_cases is not None:
        selected = selected[:max_cases]
    return selected


def write_markdown(path: Path, selected: list[dict[str, Any]], candidates: list[dict[str, Any]], missing_count: int) -> None:
    lines = [
        "# ColPali Crop Retrieval Test Corpus",
        "",
        f"- selected cases: {len(selected)}",
        f"- all-gold-retrieved candidates: {len(candidates)}",
        f"- examples missing cached retrieval: {missing_count}",
        "",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        grouped[str(record.get("selected_for_category", "unknown"))].append(record)

    for category, rows in grouped.items():
        lines.extend([f"## {category}", ""])
        for record in rows:
            lines.extend(
                [
                    f"### QID {record['question_id']} | {', '.join(record.get('categories') or [])}",
                    "",
                    f"**Question:** {record['question']}",
                    "",
                    f"**Answer:** {record['answer']}",
                    "",
                    f"**Doc:** `{record['doc_id']}`",
                    "",
                    f"**Gold pages:** {record['gold_pages']} display={record['gold_display_pages']}",
                    "",
                    f"**Retrieved pages:** {record['retrieved_pages']} display={record['retrieved_display_pages']}",
                    "",
                    f"**Evidence sources:** {record.get('evidence_sources')}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select diverse MMLongBench cases where cached ColPali top-k retrieved every gold evidence page."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/colpali_crop_test_corpus")
    parser.add_argument("--retriever", default="vidore/colpali-v1.3-hf")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--render_dpi", type=int, default=144)
    parser.add_argument("--category", action="append", default=None, help="Restrict source categories; repeatable")
    parser.add_argument("--qid", action="append", default=None, help="Restrict question ids; repeatable/comma-compatible")
    parser.add_argument("--qid-file", default=None)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument(
        "--target-category",
        action="append",
        default=None,
        help="Category to balance for; default Chart, Table, Figure, Pure-text, Layout",
    )
    parser.add_argument("--evidence-page-base", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    qids = resolve_qids(args.qid, args.qid_file)
    category_filters = set(args.category or [])
    target_categories = args.target_category or DEFAULT_TARGET_CATEGORIES
    invalid_targets = [category for category in target_categories if category not in CANONICAL_CATEGORIES]
    if invalid_targets:
        raise ValueError(f"Unknown target categories: {invalid_targets}")

    examples = load_mmlongbench_examples(
        args.data_dir,
        evidence_page_base=args.evidence_page_base,
        qids=qids,
    )
    candidates: list[dict[str, Any]] = []
    missing_retrieval: list[dict[str, Any]] = []
    partial_retrieval: list[dict[str, Any]] = []

    for example in examples:
        if category_filters and not (set(example.categories) & category_filters):
            continue
        retrieved = _load_cached_retrieval(cache_dir, example, args.retriever, args.top_k, args.render_dpi)
        if retrieved is None:
            missing_retrieval.append({"question_id": example.question_id, "doc_id": example.doc_id, "question": example.question})
            continue
        record = _candidate_record(example, retrieved, cache_dir, args.render_dpi)
        if record["all_gold_retrieved"]:
            candidates.append(record)
        else:
            partial_retrieval.append(record)

    selected = select_balanced(candidates, target_categories, args.per_category, args.max_cases)
    manifest = {
        "data_dir": str(Path(args.data_dir)),
        "cache_dir": str(cache_dir),
        "retriever": args.retriever,
        "top_k": args.top_k,
        "render_dpi": args.render_dpi,
        "evidence_page_base": args.evidence_page_base,
        "qids": sorted(qids) if qids else None,
        "category_filters": sorted(category_filters) if category_filters else None,
        "target_categories": target_categories,
        "example_count": len(examples),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "missing_retrieval_count": len(missing_retrieval),
        "partial_retrieval_count": len(partial_retrieval),
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "all_gold_retrieved_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    (output_dir / "selected_corpus.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    (output_dir / "missing_retrieval.json").write_text(json.dumps(missing_retrieval, indent=2), encoding="utf-8")
    (output_dir / "partial_retrieval.json").write_text(json.dumps(partial_retrieval, indent=2), encoding="utf-8")
    write_markdown(output_dir / "selected_corpus.md", selected, candidates, len(missing_retrieval))
    print(json.dumps(manifest, indent=2))
    print(f"[select-corpus] wrote {output_dir}")


if __name__ == "__main__":
    main()
