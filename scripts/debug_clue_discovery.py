#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sleuth.agents.clue_discovery import ClueDiscoveryAgent
from sleuth.config import get_nested, load_config
from sleuth.evaluation.cache_paths import load_cached_document_pages, load_cached_retrieval, safe_id
from sleuth.evaluation.dataset import MMLongBenchExample, load_mmlongbench_examples
from sleuth.evaluation.qid_filter import resolve_qids
from sleuth.instructions.instruction_loader import read_markdown_file
from sleuth.instructions.prompt_loader import get_prompt_section, load_agent_prompt_markdown
from sleuth.llm.mock_client import MockClient
from sleuth.llm.qwen_vl_client import QwenVLClient
from sleuth.schemas import RetrievedPage
from sleuth.utils.file_utils import ensure_dir, write_text
from sleuth.utils.json_utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only Clue Discovery on cached MMLongBench page images.")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--run-dir", default=None, help="Existing eval run directory; data/cache defaults are read from run_config.json")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--qid", action="append", default=None, help="Question id; repeatable and comma-compatible")
    parser.add_argument("--qid-file", default=None)
    parser.add_argument("--page", action="append", default=None, help="Internal 0-based page index; repeatable and comma-compatible")
    parser.add_argument("--display-page", action="append", default=None, help="Human 1-based page number; repeatable and comma-compatible")
    parser.add_argument("--use-retrieved-pages", action="store_true")
    parser.add_argument("--gold-pages", action="store_true")
    parser.add_argument("--mode", choices=["mock", "local", "sol"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--retriever", default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--render_dpi", type=int, default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent-prompts-md", default="agent_prompts.md")
    parser.add_argument("--sol-instructions-md", default="sol_instructions.md")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--region-refinement", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--evidence-page-base", choices=["auto", "zero", "one", "0-based", "1-based"], default=None)
    return parser.parse_args()


def _load_run_config(run_dir: str | None) -> dict[str, Any]:
    if not run_dir:
        return {}
    config_path = Path(run_dir) / "run_config.json"
    return load_json(config_path) if config_path.exists() else {}


def _load_predictions(run_dir: str | None) -> dict[str, dict[str, Any]]:
    if not run_dir:
        return {}
    predictions_path = Path(run_dir) / "predictions.jsonl"
    if not predictions_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                rows[str(item.get("question_id"))] = item
    return rows


def _split_int_values(values: list[str] | None) -> list[int]:
    parsed: list[int] = []
    for value in values or []:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece:
                parsed.append(int(piece))
    return parsed


def _build_llm_client(args: argparse.Namespace, config: dict[str, Any], mode: str, model_name: str, max_new_tokens: int):
    if mode == "mock":
        return MockClient()
    return QwenVLClient(
        model_name_or_path=model_name,
        device=args.device or get_nested(config, ["model", "device"], "cuda"),
        dtype=args.dtype or get_nested(config, ["model", "dtype"], "bfloat16"),
        max_new_tokens=max_new_tokens,
    )


def _retrieved_from_prediction(predictions: dict[str, dict[str, Any]], example: MMLongBenchExample) -> list[RetrievedPage] | None:
    item = predictions.get(str(example.question_id))
    if not item:
        return None
    raw_pages = item.get("retrieved_pages") or []
    pages: list[RetrievedPage] = []
    for raw in raw_pages:
        if isinstance(raw, dict):
            validator = getattr(RetrievedPage, "model_validate", None)
            pages.append(validator(raw) if callable(validator) else RetrievedPage.parse_obj(raw))
    return pages or None


def _selected_page_indices(
    example: MMLongBenchExample,
    args: argparse.Namespace,
    cache_dir: Path,
    retriever_name: str,
    top_k: int,
    render_dpi: int,
    predictions: dict[str, dict[str, Any]],
) -> list[int]:
    selected = set(_split_int_values(args.page))
    selected.update(page - 1 for page in _split_int_values(args.display_page))
    if args.gold_pages:
        selected.update(example.evidence_pages)
    if args.use_retrieved_pages:
        retrieved = load_cached_retrieval(cache_dir, example, retriever_name, top_k, render_dpi)
        if retrieved is None:
            retrieved = _retrieved_from_prediction(predictions, example)
        if retrieved is None:
            raise FileNotFoundError(
                "Missing cached retrieval for "
                f"QID {example.question_id}. Run scripts/cache_mmlongbench_colpali.py first or pass --run-dir "
                "for a run with retrieved_pages in predictions.jsonl."
            )
        selected.update(page.page_index for page in retrieved)
    if not selected:
        selected.update(example.evidence_pages)
    return sorted(index for index in selected if index >= 0)


def _write_case_html(rows: list[dict[str, Any]], output_dir: Path) -> None:
    blocks = []
    for row in rows:
        clue = row.get("clue", {})
        image_rel = html.escape(row.get("image_rel", ""))
        evidence_items = clue.get("evidence_items") or []
        evidence_html = "".join(
            "<li>"
            f"<b>{html.escape(str(item.get('evidence_type', '')))}</b>: "
            f"{html.escape(str(item.get('content', '')))}"
            f"<br><small>{html.escape(str(item.get('location', '')))}</small>"
            "</li>"
            for item in evidence_items
            if isinstance(item, dict)
        )
        blocks.append(
            f"""
            <section>
              <h2>QID {html.escape(str(row['question_id']))} | Page {row['display_page']} internal {row['page_index']}</h2>
              <p><b>Question:</b> {html.escape(str(row['question']))}</p>
              <p><b>Gold:</b> <code>{html.escape(str(row['answer']))}</code></p>
              <a href="{image_rel}"><img src="{image_rel}" /></a>
              <h3>Summary</h3>
              <p>{html.escape(str(clue.get('page_summary', '')))}</p>
              <h3>Insights</h3>
              <p>{html.escape(str(clue.get('key_insights', '')))}</p>
              <h3>Evidence Items</h3>
              <ul>{evidence_html or '<li>No evidence items.</li>'}</ul>
              <p><a href="{html.escape(row['case_rel'])}/raw_output.txt">raw output</a> |
                 <a href="{html.escape(row['case_rel'])}/prompt.txt">prompt</a> |
                 <a href="{html.escape(row['case_rel'])}/clue.json">clue json</a></p>
            </section>
            """
        )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Clue Discovery Debug</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; line-height: 1.35; }}
    section {{ border-top: 2px solid #ddd; padding-top: 20px; margin-top: 24px; }}
    img {{ max-width: min(100%, 1050px); border: 1px solid #ddd; }}
    code {{ background: #f6f6f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Clue Discovery Debug</h1>
  {''.join(blocks)}
</body>
</html>
"""
    write_text(output_dir / "index.html", html_text)


def main() -> None:
    args = parse_args()
    run_config = _load_run_config(args.run_dir)
    config = load_config(args.config)

    data_dir = args.data_dir or run_config.get("data_dir")
    if not data_dir:
        raise ValueError("Pass --data_dir, or pass --run-dir for a run whose run_config.json records data_dir.")
    cache_dir = Path(args.cache_dir or run_config.get("cache_dir") or (Path(args.run_dir) / "cache" if args.run_dir else "cache"))
    render_dpi = int(args.render_dpi or run_config.get("render_dpi") or get_nested(config, ["pipeline", "render_dpi"], 144))
    retriever_name = str(args.retriever or run_config.get("actual_retriever") or run_config.get("retriever") or get_nested(config, ["retriever", "name_or_path"], "vidore/colpali-v1.3-hf"))
    top_k = int(args.top_k or run_config.get("top_k") or get_nested(config, ["retriever", "top_k"], 5))
    mode = args.mode or run_config.get("mode") or "mock"
    model_name = str(args.model or run_config.get("model") or get_nested(config, ["model", "name_or_path"], "Qwen/Qwen3-VL-8B-Instruct"))
    evidence_page_base = args.evidence_page_base or run_config.get("evidence_page_base") or get_nested(config, ["evaluation", "evidence_page_base"], "auto")
    qids = resolve_qids(args.qid, args.qid_file)
    predictions = _load_predictions(args.run_dir)
    if qids is None and predictions:
        qids = set(predictions)

    examples = load_mmlongbench_examples(
        data_dir,
        qids=qids,
        evidence_page_base=str(evidence_page_base),
    )
    if not examples:
        raise ValueError("No MMLongBench-Doc examples matched the requested filters.")

    max_tokens = get_nested(config, ["model", "max_new_tokens"], {})
    prompt_markdown = load_agent_prompt_markdown(args.agent_prompts_md)
    sol_instruction_text = read_markdown_file(args.sol_instructions_md) if mode == "sol" else None
    client = _build_llm_client(
        args,
        config,
        mode=mode,
        model_name=model_name,
        max_new_tokens=int(max_tokens.get("clue_discovery", 3072)),
    )
    region_refinement = args.region_refinement or str(get_nested(config, ["pipeline", "region_refinement"], "fallback"))
    clue_agent = ClueDiscoveryAgent(
        llm_client=client,
        agent_prompt_text=get_prompt_section(prompt_markdown, "clue_discovery"),
        sol_instruction_text=sol_instruction_text,
        temperature=args.temperature,
        max_new_tokens=int(max_tokens.get("clue_discovery", 3072)),
        region_refinement=region_refinement,
    )

    output_dir = ensure_dir(args.output_dir)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for example in examples:
        try:
            pages = load_cached_document_pages(example.doc_id, cache_dir, render_dpi, legacy_cache_dirs=[Path(args.run_dir) / "cache"] if args.run_dir else None)
            if pages is None:
                raise FileNotFoundError(
                    f"Missing cached rendered pages for doc {example.doc_id} at {cache_dir}. "
                    "Run scripts/cache_mmlongbench_colpali.py first."
                )
            page_lookup = {page.page_index: page for page in pages}
            page_indices = _selected_page_indices(example, args, cache_dir, retriever_name, top_k, render_dpi, predictions)
            for page_index in page_indices:
                page = page_lookup.get(page_index)
                if page is None:
                    failures.append({"question_id": example.question_id, "page_index": page_index, "error": "page index not found"})
                    continue
                clue = clue_agent.run(example.question, page)
                case_dir = ensure_dir(output_dir / f"qid_{safe_id(example.question_id)}" / f"page_{page_index:04d}")
                save_json(case_dir / "clue.json", clue)
                write_text(case_dir / "raw_output.txt", clue.raw_output or "")
                write_text(case_dir / "prompt.txt", clue.prompt_used or "")
                image_target = case_dir / f"page_{page_index:04d}{Path(page.image_path).suffix or '.png'}"
                if not image_target.exists():
                    shutil.copy2(page.image_path, image_target)
                meta = {
                    "question_id": example.question_id,
                    "row_index": example.row_index,
                    "document_id": example.doc_id,
                    "question": example.question,
                    "answer": example.answer,
                    "page_index": page_index,
                    "display_page": page_index + 1,
                    "image_path": page.image_path,
                    "copied_image_path": str(image_target),
                    "mode": mode,
                    "model": model_name if mode != "mock" else "MockClient",
                    "cache_dir": str(cache_dir),
                    "render_dpi": render_dpi,
                    "retriever": retriever_name,
                    "top_k": top_k,
                    "region_refinement": region_refinement,
                }
                save_json(case_dir / "meta.json", meta)
                rows.append(
                    {
                        **meta,
                        "case_rel": str(case_dir.relative_to(output_dir)),
                        "image_rel": str(image_target.relative_to(output_dir)),
                        "clue": clue.model_dump() if hasattr(clue, "model_dump") else clue.dict(),
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "question_id": example.question_id,
                    "document_id": example.doc_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    save_json(output_dir / "run_config.json", {
        "data_dir": str(data_dir),
        "run_dir": args.run_dir,
        "cache_dir": str(cache_dir),
        "qids": sorted(qids) if qids else None,
        "qid_file": args.qid_file,
        "use_retrieved_pages": args.use_retrieved_pages,
        "gold_pages": args.gold_pages,
        "pages": _split_int_values(args.page),
        "display_pages": _split_int_values(args.display_page),
        "mode": mode,
        "model": model_name,
        "retriever": retriever_name,
        "top_k": top_k,
        "render_dpi": render_dpi,
        "region_refinement": region_refinement,
    })
    save_json(output_dir / "cases.json", rows)
    save_json(output_dir / "failures.json", failures)
    _write_case_html(rows, output_dir)
    print(f"Wrote clue debug report: {output_dir / 'index.html'}")
    print(f"Cases: {len(rows)}")
    print(f"Failures: {len(failures)}")
    if failures and not rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
