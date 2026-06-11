#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sleuth.retrieval.colpali_retriever import ColPaliRetriever
from sleuth.evaluation.cache_paths import colpali_embedding_cache_path, load_cached_document_pages
from sleuth.schemas import DocumentPage


DEFAULT_MODALITIES = ("Chart", "Table", "Figure")
V1_METHOD = "reading_order"
V1_VIEW_TYPE = "V1_anchor_plus_1_below"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_values(values: list[str] | None) -> set[str]:
    parsed: set[str] = set()
    for value in values or []:
        parsed.update(part.strip() for part in value.split(",") if part.strip())
    return parsed


def _resolve_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("No candidate path exists: " + ", ".join(str(path) for path in candidates))


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return cleaned.strip("_") or "item"


def _embedding_cache_name(question: dict[str, Any], candidates: list[dict[str, Any]], model: str) -> str:
    payload = {
        "question_id": str(question["question_id"]),
        "model": model,
        "candidate_paths": [str(item["image_path"]) for item in candidates],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return f"qid_{_safe_name(str(question['question_id']))}_{digest}.pt"


def load_visual_questions(question_list_path: Path, modalities: set[str], qids: set[str]) -> list[dict[str, Any]]:
    questions = _read_json(question_list_path)
    selected = []
    for question in questions:
        qid = str(question["question_id"])
        question_modalities = {str(value) for value in question.get("modalities", [])}
        if qids and qid not in qids:
            continue
        if not question_modalities.intersection(modalities):
            continue
        selected.append(question)
    return selected


def build_candidates(
    question: dict[str, Any],
    input_dir: Path,
    region_views: list[dict[str, Any]],
    full_document_pages: list[DocumentPage] | None = None,
    require_full_page_files: bool = True,
) -> list[dict[str, Any]]:
    qid = str(question["question_id"])
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    if full_document_pages:
        full_page_rows = [(int(page.page_index) + 1, Path(page.image_path)) for page in full_document_pages]
    else:
        full_page_rows = [
            (int(display_page), input_dir / "cases" / qid / "pages" / f"page_{int(display_page):03d}.png")
            for display_page in question["top5_display_pages"]
        ]

    for display_page, unresolved_page_path in full_page_rows:
        page_path = (
            _resolve_existing(unresolved_page_path)
            if require_full_page_files
            else unresolved_page_path.resolve()
        )
        candidate = {
            "candidate_id": f"qid_{qid}_page_{int(display_page):03d}_full_page",
            "representation_type": "full_page",
            "display_page": int(display_page),
            "anchor_block_id": None,
            "anchor_type": None,
            "included_block_ids": [],
            "included_block_types": [],
            "image_path": str(page_path),
        }
        candidates.append(candidate)
        seen.add((candidate["candidate_id"], str(page_path)))

    matching_views = sorted(
        (
            view
            for view in region_views
            if str(view.get("question_id")) == qid
            and view.get("method") == V1_METHOD
            and view.get("view_type") == V1_VIEW_TYPE
            and int(view.get("display_page", -1)) in {int(page) for page in question["top5_display_pages"]}
            and view.get("anchor_type") in {"figure", "table"}
        ),
        key=lambda view: (int(view["display_page"]), str(view["anchor_block_id"])),
    )
    canvas_dir = input_dir / "region_canvas_visualizations" / "canvases"
    for view in matching_views:
        exported_path = Path(str(view["output_image_path"]))
        canvas_path = _resolve_existing(canvas_dir / exported_path.name, exported_path)
        candidate_id = (
            f"qid_{qid}_page_{int(view['display_page']):03d}_"
            f"{view['anchor_type']}_{_safe_name(str(view['anchor_block_id']))}_v1_below"
        )
        key = (candidate_id, str(canvas_path))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "representation_type": "v1_anchor_plus_1_below",
                "display_page": int(view["display_page"]),
                "anchor_block_id": view["anchor_block_id"],
                "anchor_type": view["anchor_type"],
                "included_block_ids": view.get("included_block_ids", []),
                "included_block_types": view.get("included_block_types", []),
                "image_path": str(canvas_path),
            }
        )
    return candidates


def aggregate_page_rankings(scored_candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in scored_candidates:
        by_page[int(candidate["display_page"])].append(candidate)

    pages = []
    for display_page, representations in by_page.items():
        ordered = sorted(representations, key=lambda item: float(item["score"]), reverse=True)
        winner = ordered[0]
        pages.append(
            {
                "display_page": display_page,
                "score": winner["score"],
                "winning_candidate_id": winner["candidate_id"],
                "winning_representation_type": winner["representation_type"],
                "winning_anchor_block_id": winner["anchor_block_id"],
                "winning_anchor_type": winner["anchor_type"],
                "winning_image_path": winner["image_path"],
                "representation_count": len(ordered),
                "all_representations": ordered,
            }
        )
    return sorted(pages, key=lambda item: float(item["score"]), reverse=True)[:top_k]


def score_question(
    retriever: ColPaliRetriever,
    question: dict[str, Any],
    candidates: list[dict[str, Any]],
    full_document_pages: list[DocumentPage],
    full_page_cache_path: Path,
    canvas_cache_path: Path,
    top_k: int,
) -> dict[str, Any]:
    full_candidates = [item for item in candidates if item["representation_type"] == "full_page"]
    canvas_candidates = [item for item in candidates if item["representation_type"] != "full_page"]
    if len(full_candidates) != len(full_document_pages):
        raise ValueError("Full-page candidate count does not match cached document page count")

    full_page_cache_hit = full_page_cache_path.exists()
    retriever.index_with_cache(full_document_pages, full_page_cache_path)
    full_embeddings = retriever.image_embeddings

    canvas_cache_hit = canvas_cache_path.exists()
    canvas_pages = [DocumentPage(page_index=index, image_path=item["image_path"]) for index, item in enumerate(canvas_candidates)]
    canvas_embeddings = None
    if canvas_pages:
        retriever.index_with_cache(canvas_pages, canvas_cache_path)
        canvas_embeddings = retriever.image_embeddings

    combined_embeddings = full_embeddings
    if canvas_embeddings is not None:
        combined_embeddings = retriever.torch.cat([full_embeddings, canvas_embeddings], dim=0)
    combined_candidates = full_candidates + canvas_candidates
    retriever.document_pages = [
        DocumentPage(page_index=index, image_path=item["image_path"]) for index, item in enumerate(combined_candidates)
    ]
    retriever.image_embeddings = combined_embeddings
    retrieved = retriever.retrieve(str(question["question"]), top_k=len(candidates))

    scored = []
    for rank, result in enumerate(retrieved, start=1):
        candidate = dict(combined_candidates[result.page_index])
        candidate.update({"representation_rank": rank, "score": float(result.score)})
        scored.append(candidate)

    page_ranking = aggregate_page_rankings(scored, top_k)
    gold = {int(page) for page in question.get("gold_display_pages", [])}
    ranked_pages = {int(item["display_page"]) for item in page_ranking}
    return {
        "question_id": str(question["question_id"]),
        "question": question["question"],
        "answer": question.get("answer"),
        "modalities": question.get("modalities", []),
        "gold_display_pages": sorted(gold),
        "original_colpali_top5_display_pages": question.get("top5_display_pages", []),
        "candidate_count": len(candidates),
        "full_page_candidate_count": sum(item["representation_type"] == "full_page" for item in candidates),
        "v1_canvas_candidate_count": sum(item["representation_type"] != "full_page" for item in candidates),
        "full_page_embedding_cache_path": str(full_page_cache_path),
        "full_page_embedding_cache_hit": full_page_cache_hit,
        "canvas_embedding_cache_path": str(canvas_cache_path),
        "canvas_embedding_cache_hit": canvas_cache_hit,
        "top_k_page_ranking": page_ranking,
        "top_k_display_pages": [item["display_page"] for item in page_ranking],
        "all_gold_in_top_k": gold.issubset(ranked_pages),
        "missing_gold_display_pages": sorted(gold - ranked_pages),
        "representation_ranking": scored,
    }


def _copy_ranked_assets(results: list[dict[str, Any]], output_dir: Path) -> None:
    assets = output_dir / "ranked_images"
    assets.mkdir(parents=True, exist_ok=True)
    for result in results:
        qid_dir = assets / _safe_name(result["question_id"])
        qid_dir.mkdir(parents=True, exist_ok=True)
        for rank, page in enumerate(result["top_k_page_ranking"], start=1):
            source = Path(page["winning_image_path"])
            destination = qid_dir / f"rank_{rank:02d}_{source.name}"
            if not destination.exists():
                shutil.copy2(source, destination)
            page["viewer_image_path"] = str(destination.relative_to(output_dir))


def _write_markdown(results: list[dict[str, Any]], output_dir: Path) -> None:
    lines = ["# ColPali V1 Canvas Retrieval", ""]
    for result in results:
        lines.extend(
            [
                f"## QID {result['question_id']}",
                "",
                result["question"],
                "",
                f"Gold pages: {result['gold_display_pages']}",
                f"Top-{len(result['top_k_page_ranking'])}: {result['top_k_display_pages']}",
                f"All gold retained: {result['all_gold_in_top_k']}",
                "",
                "| Rank | Page | Score | Winning representation | Anchor |",
                "|---:|---:|---:|---|---|",
            ]
        )
        for rank, page in enumerate(result["top_k_page_ranking"], start=1):
            lines.append(
                f"| {rank} | {page['display_page']} | {page['score']:.6f} | "
                f"{page['winning_representation_type']} | {page['winning_anchor_block_id'] or ''} |"
            )
        lines.append("")
    (output_dir / "rankings.md").write_text("\n".join(lines), encoding="utf-8")


def _write_html(results: list[dict[str, Any]], output_dir: Path) -> None:
    sections = []
    for result in results:
        cards = []
        for rank, page in enumerate(result["top_k_page_ranking"], start=1):
            cards.append(
                f"""<article class="card">
                <h3>#{rank} · Page {page['display_page']}</h3>
                <img loading="lazy" src="{html.escape(page.get('viewer_image_path', ''))}" alt="ranked page">
                <p><b>Score:</b> {page['score']:.6f}</p>
                <p><b>Winner:</b> {html.escape(page['winning_representation_type'])}</p>
                <p><b>Anchor:</b> {html.escape(str(page['winning_anchor_block_id'] or 'none'))}</p>
                <details><summary>All representations for this page</summary><pre>{html.escape(json.dumps(page['all_representations'], indent=2))}</pre></details>
                </article>"""
            )
        sections.append(
            f"""<section>
            <h2>QID {html.escape(result['question_id'])}</h2>
            <p class="question">{html.escape(result['question'])}</p>
            <p>Gold pages: {result['gold_display_pages']} · Ranked pages: {result['top_k_display_pages']} · All gold retained: <b>{result['all_gold_in_top_k']}</b></p>
            <p>Candidates: {result['full_page_candidate_count']} full pages + {result['v1_canvas_candidate_count']} V1 canvases</p>
            <div class="grid">{''.join(cards)}</div>
            </section>"""
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>ColPali V1 Canvas Retrieval</title>
    <style>
    body{{font:14px system-ui;margin:0;background:#f3f5f7;color:#18202a}}main{{max-width:1500px;margin:auto;padding:24px}}
    section{{background:white;margin:0 0 24px;padding:20px;border:1px solid #d8dee6;border-radius:8px}}
    .question{{font-size:17px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}}
    .card{{border:1px solid #d8dee6;padding:12px;border-radius:6px;min-width:0}}img{{width:100%;height:360px;object-fit:contain;background:white;border:1px solid #e4e8ee}}
    pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}}h1,h2,h3{{letter-spacing:0}}
    </style></head><body><main><h1>ColPali: Full Pages + V1 Anchor/Below Canvases</h1>{''.join(sections)}</main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ColPali retrieval over full pages and V1 anchor-plus-below canvases.")
    parser.add_argument("--input_dir", default="outputs/docling_colpali_top5_corpus")
    parser.add_argument("--output_dir", default="outputs/colpali_v1_canvas_retrieval")
    parser.add_argument("--cache_dir", default=None, help="Shared rendered-page and full-document ColPali cache")
    parser.add_argument("--render_dpi", type=int, default=144)
    parser.add_argument("--retriever", default="vidore/colpali-v1.3-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--colpali_batch_size", type=int, default=2)
    parser.add_argument("--modality", action="append", default=None, help="Visual modality; repeatable/comma-compatible")
    parser.add_argument("--qid", action="append", default=None, help="Question id; repeatable/comma-compatible")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true", help="Validate and export candidate sets without loading ColPali")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    modalities = _parse_values(args.modality) or set(DEFAULT_MODALITIES)
    qids = _parse_values(args.qid)

    questions = load_visual_questions(input_dir / "question_list.json", modalities, qids)
    if args.max_cases is not None:
        questions = questions[: max(0, args.max_cases)]
    region_views = _read_json(input_dir / "region_canvas_visualizations" / "region_views.json")
    prepared = []
    missing_document_caches = []
    for question in questions:
        full_document_pages = None
        if cache_dir is not None:
            full_document_pages = load_cached_document_pages(
                str(question["doc_id"]), cache_dir, args.render_dpi
            )
        if not full_document_pages:
            missing_document_caches.append(str(question["question_id"]))
        candidates = build_candidates(
            question,
            input_dir,
            region_views,
            full_document_pages,
            require_full_page_files=not args.prepare_only,
        )
        prepared.append(
            {
                "question": question,
                "candidates": candidates,
                "full_document_page_count": len(full_document_pages or []),
                "using_full_document_cache": bool(full_document_pages),
            }
        )
    _write_json(output_dir / "prepared_candidates.json", prepared)

    if args.prepare_only:
        manifest = {
            "mode": "prepare_only",
            "question_count": len(prepared),
            "candidate_count": sum(len(item["candidates"]) for item in prepared),
            "modalities": sorted(modalities),
            "missing_full_document_cache_qids": missing_document_caches,
        }
        _write_json(output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
        return

    if cache_dir is None:
        raise SystemExit("--cache_dir is required unless --prepare-only is used")
    if missing_document_caches:
        raise SystemExit(
            "Missing cached full-document pages for QIDs: " + ", ".join(missing_document_caches)
        )

    retriever = ColPaliRetriever(
        model_name_or_path=args.retriever,
        device=args.device,
        top_k_default=args.top_k,
        index_batch_size=args.colpali_batch_size,
    )
    results = []
    failures = []
    for index, item in enumerate(prepared, start=1):
        question = item["question"]
        print(f"[v1-retrieval] {index}/{len(prepared)} qid={question['question_id']} candidates={len(item['candidates'])}", flush=True)
        try:
            full_document_pages = load_cached_document_pages(
                str(question["doc_id"]), cache_dir, args.render_dpi
            )
            if not full_document_pages:
                raise FileNotFoundError(f"Missing full-document page cache for {question['doc_id']}")
            full_page_cache_path = colpali_embedding_cache_path(
                cache_dir, args.retriever, args.render_dpi, str(question["doc_id"])
            )
            canvas_candidates = [
                candidate for candidate in item["candidates"] if candidate["representation_type"] != "full_page"
            ]
            canvas_cache_path = cache_dir / "colpali_v1_canvas_embeddings" / _embedding_cache_name(
                question, canvas_candidates, args.retriever
            )
            result = score_question(
                retriever,
                question,
                item["candidates"],
                full_document_pages,
                full_page_cache_path,
                canvas_cache_path,
                args.top_k,
            )
            results.append(result)
        except Exception as exc:
            failure = {"question_id": str(question["question_id"]), "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"[v1-retrieval] ERROR {failure}", flush=True)

    _copy_ranked_assets(results, output_dir)
    _write_json(output_dir / "rankings.json", results)
    _write_markdown(results, output_dir)
    _write_html(results, output_dir)
    manifest = {
        "started_and_finished_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "retriever": args.retriever,
        "device": args.device,
        "cache_dir": str(cache_dir),
        "render_dpi": args.render_dpi,
        "top_k": args.top_k,
        "colpali_batch_size": args.colpali_batch_size,
        "modalities": sorted(modalities),
        "question_count": len(prepared),
        "completed_count": len(results),
        "failure_count": len(failures),
        "candidate_count": sum(item["candidate_count"] for item in results),
        "gold_retained_count": sum(bool(item["all_gold_in_top_k"]) for item in results),
        "failures": failures,
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
