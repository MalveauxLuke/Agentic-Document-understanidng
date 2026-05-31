#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sleuth.evaluation.cache_paths import (
    colpali_embedding_cache_path,
    document_cache_dir,
    load_or_build_document_pages,
    retrieval_cache_path,
)
from sleuth.evaluation.dataset import MMLongBenchExample, load_mmlongbench_examples
from sleuth.evaluation.qid_filter import resolve_qids
from sleuth.retrieval.colpali_retriever import ColPaliRetriever
from sleuth.utils.file_utils import ensure_dir
from sleuth.utils.json_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm the MMLongBench-Doc render, ColPali embedding, and retrieval cache.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--retriever", default="vidore/colpali-v1.3-hf")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--render_dpi", type=int, default=144)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--category", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qid", action="append", default=None, help="Only cache one question id; repeatable and comma-compatible")
    parser.add_argument("--qid-file", default=None, help="JSON/list/text file of question ids to cache")
    parser.add_argument("--evidence-page-base", choices=["auto", "zero", "one", "0-based", "1-based"], default="auto")
    return parser.parse_args()


def _group_by_document(examples: list[MMLongBenchExample]) -> dict[str, list[MMLongBenchExample]]:
    grouped: dict[str, list[MMLongBenchExample]] = defaultdict(list)
    for example in examples:
        grouped[example.doc_id].append(example)
    return dict(grouped)


def main() -> None:
    args = parse_args()
    cache_dir = ensure_dir(args.cache_dir)
    qids = resolve_qids(args.qid, args.qid_file)
    examples = load_mmlongbench_examples(
        args.data_dir,
        limit=args.limit,
        category=args.category,
        evidence_page_base=args.evidence_page_base,
        qids=qids,
    )
    if not examples:
        raise ValueError("No MMLongBench-Doc examples matched the requested filters.")

    grouped = _group_by_document(examples)
    retriever = ColPaliRetriever(
        model_name_or_path=args.retriever,
        device=args.device,
        top_k_default=args.top_k,
    )

    stats = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": args.data_dir,
        "cache_dir": str(cache_dir),
        "retriever": args.retriever,
        "top_k": args.top_k,
        "render_dpi": args.render_dpi,
        "device": args.device,
        "category": args.category,
        "limit": args.limit,
        "qids": sorted(qids) if qids else None,
        "qid_file": args.qid_file,
        "document_count": len(grouped),
        "query_count": len(examples),
        "render_cache_hits": 0,
        "render_cache_misses": 0,
        "embedding_cache_hits": 0,
        "embedding_cache_misses": 0,
        "retrieval_cache_hits": 0,
        "retrieval_cache_misses": 0,
        "failures": [],
    }

    for doc_number, (doc_id, doc_examples) in enumerate(grouped.items(), start=1):
        first = doc_examples[0]
        print(f"[cache-colpali] {doc_number}/{len(grouped)} doc={doc_id} queries={len(doc_examples)}", flush=True)
        pages_json = document_cache_dir(cache_dir, doc_id, args.render_dpi) / "pages.json"
        embedding_path = colpali_embedding_cache_path(cache_dir, args.retriever, args.render_dpi, doc_id)
        render_hit = pages_json.exists()
        embedding_hit = embedding_path.exists()
        try:
            pages = load_or_build_document_pages(first.pdf_path, doc_id, cache_dir, args.render_dpi)
            if render_hit:
                stats["render_cache_hits"] += 1
            else:
                stats["render_cache_misses"] += 1

            retriever.index_with_cache(pages, embedding_path)
            if embedding_hit:
                stats["embedding_cache_hits"] += 1
            else:
                stats["embedding_cache_misses"] += 1

            for example in doc_examples:
                retrieval_path = retrieval_cache_path(cache_dir, example, args.retriever, args.top_k, args.render_dpi)
                if retrieval_path.exists():
                    stats["retrieval_cache_hits"] += 1
                    continue
                retrieved_pages = retriever.retrieve(example.question, top_k=args.top_k)
                save_json(retrieval_path, retrieved_pages)
                stats["retrieval_cache_misses"] += 1
        except Exception as exc:
            failure = {
                "document_id": doc_id,
                "question_ids": [example.question_id for example in doc_examples],
                "error": f"{type(exc).__name__}: {exc}",
            }
            stats["failures"].append(failure)
            print(f"[cache-colpali] ERROR doc={doc_id}: {failure['error']}", file=sys.stderr, flush=True)

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest_dir = ensure_dir(cache_dir / "manifests")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = manifest_dir / f"colpali_cache_{stamp}.json"
    latest_path = cache_dir / "colpali_cache_latest.json"
    save_json(manifest_path, stats)
    save_json(latest_path, stats)
    print(json.dumps(stats, indent=2))
    print(f"[cache-colpali] manifest: {manifest_path}")
    if stats["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
