#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
import html
import json
import os
import shutil
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - exercised only in minimal envs
    Image = None
    ImageDraw = None


FAILURE_ORDER = {
    "retrieval_miss": 0,
    "clue_miss": 1,
    "screening_drop": 2,
    "final_wrong": 3,
    "scoring_or_extraction_mismatch": 4,
}


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _display_page(page_index: int) -> int:
    return page_index + 1


def _short(text: Any, limit: int = 240) -> str:
    value = str(text or "").replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    return _load_json(path) if path.exists() else {}


def resolve_run_dir(run: str, run_root: Path) -> Path:
    candidate = Path(run).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if run.isdigit():
        matches = sorted(run_root.expanduser().glob(f"*{run}*"))
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError(
        f"Could not resolve run '{run}'. Pass a run directory, or set --run-root to the directory containing job {run}."
    )


def _document_pages(run_dir: Path, document_id: str) -> dict[int, str]:
    run_config = _run_config(run_dir)
    render_dpi = int(run_config.get("render_dpi") or 144)
    cache_dirs = []
    if run_config.get("cache_dir"):
        cache_dirs.append(Path(str(run_config["cache_dir"])))
    cache_dirs.append(run_dir / "cache")

    candidates = []
    for cache_dir in cache_dirs:
        candidates.extend(
            [
                cache_dir / "documents" / f"dpi_{render_dpi}" / _safe_id(document_id) / "pages.json",
                cache_dir / "documents" / _safe_id(document_id) / "pages.json",
            ]
        )
    pages_json = next((path for path in candidates if path.exists()), None)
    if pages_json is None:
        return {}
    pages: dict[int, str] = {}
    for page in _load_json(pages_json):
        try:
            pages[int(page["page_index"])] = str(page["image_path"])
        except (KeyError, TypeError, ValueError):
            continue
    return pages


def _copy_page_image(
    source_path: str | None,
    assets_dir: Path,
    case_id: str,
    role: str,
    page_index: int,
) -> str | None:
    if not source_path:
        return None
    source = Path(source_path)
    if not source.exists():
        return None
    case_dir = assets_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix or ".png"
    target = case_dir / f"{role}_page_{page_index:04d}{suffix}"
    if not target.exists():
        shutil.copy2(source, target)
    return str(target.relative_to(assets_dir.parent))


def _make_contact_sheet(image_rows: list[dict[str, Any]], target_path: Path, thumb_width: int = 360) -> str | None:
    if Image is None or ImageDraw is None or not image_rows:
        return None
    report_dir = target_path.parents[2]
    thumbs = []
    for row in image_rows:
        path = row.get("copied_path")
        if not path:
            continue
        image_path = report_dir / path
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            scale = thumb_width / max(image.width, 1)
            thumb_height = max(1, int(image.height * scale))
            thumb = image.resize((thumb_width, thumb_height))
        label = str(row.get("label", ""))
        label_height = 42
        canvas = Image.new("RGB", (thumb_width, thumb_height + label_height), "white")
        draw = ImageDraw.Draw(canvas)
        outline = row.get("outline", "black")
        draw.rectangle((0, 0, thumb_width - 1, thumb_height - 1), outline=outline, width=6)
        canvas.paste(thumb, (0, 0))
        draw.text((8, thumb_height + 8), label[:90], fill="black")
        thumbs.append(canvas)
    if not thumbs:
        return None

    gap = 18
    columns = min(3, len(thumbs))
    rows = (len(thumbs) + columns - 1) // columns
    cell_w = thumb_width
    cell_h = max(thumb.height for thumb in thumbs)
    sheet = Image.new("RGB", (columns * cell_w + (columns - 1) * gap, rows * cell_h + (rows - 1) * gap), "white")
    for index, thumb in enumerate(thumbs):
        row = index // columns
        col = index % columns
        sheet.paste(thumb, (col * (cell_w + gap), row * (cell_h + gap)))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target_path)
    return str(target_path.relative_to(report_dir))


def _page_status(item: dict[str, Any], page_index: int) -> dict[str, Any]:
    gold_pages = set(item.get("gold_evidence_pages") or item.get("evidence_pages") or [])
    clue_pages = set(item.get("clue_pages_with_evidence") or [])
    retained_pages = set(item.get("screening_retained_page_indices") or [])
    clue_by_page = {clue.get("page_index"): clue for clue in item.get("clue_output", [])}
    screen_by_page = {screen.get("page_index"): screen for screen in item.get("page_screening_output", [])}
    clue = clue_by_page.get(page_index) or {}
    screen = screen_by_page.get(page_index) or {}
    return {
        "is_gold": page_index in gold_pages,
        "has_clue": page_index in clue_pages,
        "kept_visual": page_index in retained_pages,
        "clue_items": len(clue.get("evidence_items") or []),
        "clue_summary": clue.get("page_summary"),
        "clue_insights": clue.get("key_insights"),
        "screen_relevance": screen.get("relevance"),
        "screen_reasoning": screen.get("reasoning"),
    }


def _diagnose(item: dict[str, Any]) -> list[str]:
    gold_pages = set(item.get("gold_evidence_pages") or item.get("evidence_pages") or [])
    retrieved_pages = [page.get("page_index") for page in item.get("retrieved_pages", [])]
    retrieved_set = {page for page in retrieved_pages if page is not None}
    clue_gold_pages = set(item.get("clue_gold_pages_with_evidence") or [])
    retained_gold = bool(item.get("screening_retained_gold"))
    label = item.get("failure_label")
    notes: list[str] = []

    if not gold_pages:
        notes.append("No gold evidence pages are annotated, so retrieval recall cannot explain this case directly.")
    else:
        missed = sorted(gold_pages - retrieved_set)
        hit = sorted(gold_pages & retrieved_set)
        if not hit:
            notes.append(
                "ColPali top-K retrieved no gold page. This is a true retrieval miss before SLEUTH agents can help."
            )
        elif missed:
            notes.append(
                f"ColPali found some gold pages {hit} but missed required gold pages {missed}; multi-page questions may be capped by partial recall."
            )
        else:
            notes.append("ColPali included all annotated gold pages in top-K; later stages or answer synthesis caused the failure.")

    if label == "retrieval_miss":
        notes.append("Primary failure label is retrieval_miss: inspect the retrieved distractor pages and consider higher K or feedback retrieval.")
    elif label == "clue_miss":
        if clue_gold_pages:
            notes.append("Gold pages had clue evidence, but diagnostics still label clue_miss; inspect scoring/extraction and final reasoning.")
        else:
            notes.append("Gold page retrieval succeeded, but Clue Discovery did not record evidence on a gold page.")
    elif label == "screening_drop":
        notes.append("A relevant visual page appears to have been filtered out before Core Decision received images.")
    elif label == "final_wrong":
        notes.append("Evidence reached the final stage; the error is likely answer synthesis, counting, formatting, or visual value reading.")
    elif label == "scoring_or_extraction_mismatch":
        notes.append("Raw answer scored better than extracted answer; inspect answer extraction rather than ColPali.")

    if not retained_gold and gold_pages & retrieved_set:
        notes.append("At least one gold page was retrieved but no gold visual page was retained by Page Screening.")
    return notes


def _retrieval_rows(item: dict[str, Any], pages: dict[int, str], assets_dir: Path, case_id: str) -> list[dict[str, Any]]:
    rows = []
    for rank, retrieved in enumerate(item.get("retrieved_pages", []), start=1):
        page_index = int(retrieved.get("page_index"))
        status = _page_status(item, page_index)
        role = f"rank_{rank:02d}"
        copied = _copy_page_image(pages.get(page_index), assets_dir, case_id, role, page_index)
        rows.append(
            {
                "rank": rank,
                "page_index": page_index,
                "display_page": _display_page(page_index),
                "score": retrieved.get("score"),
                "copied_path": copied,
                "label": (
                    f"rank {rank} | p{_display_page(page_index)} | score {float(retrieved.get('score', 0.0)):.3f} "
                    f"| {'gold' if status['is_gold'] else 'non-gold'}"
                ),
                "outline": "green" if status["is_gold"] else "gray",
                **status,
            }
        )
    return rows


def _gold_rows(item: dict[str, Any], pages: dict[int, str], assets_dir: Path, case_id: str) -> list[dict[str, Any]]:
    retrieved = {row.get("page_index") for row in item.get("retrieved_pages", [])}
    rows = []
    for page_index in sorted(item.get("gold_evidence_pages") or item.get("evidence_pages") or []):
        copied = _copy_page_image(pages.get(page_index), assets_dir, case_id, "gold", page_index)
        status = _page_status(item, page_index)
        rows.append(
            {
                "page_index": page_index,
                "display_page": _display_page(page_index),
                "retrieved": page_index in retrieved,
                "copied_path": copied,
                "label": f"gold p{_display_page(page_index)} | {'retrieved' if page_index in retrieved else 'missed'}",
                "outline": "green" if page_index in retrieved else "red",
                **status,
            }
        )
    return rows


def _prepare_cases(
    predictions: list[dict[str, Any]],
    run_dir: Path,
    report_dir: Path,
    label: str | None,
    qids: set[str] | None,
    limit: int | None,
    include_correct: bool,
) -> list[dict[str, Any]]:
    rows = predictions if include_correct else [item for item in predictions if float(item.get("score", 0.0)) == 0.0]
    if label:
        rows = [item for item in rows if item.get("failure_label") == label]
    if qids:
        rows = [item for item in rows if str(item.get("question_id")) in qids]
    rows = sorted(
        rows,
        key=lambda item: (
            FAILURE_ORDER.get(str(item.get("failure_label")), 99),
            str(item.get("document_id")),
            int(item.get("row_index", 0)),
        ),
    )
    if limit is not None:
        rows = rows[:limit]

    assets_dir = report_dir / "assets"
    cases: list[dict[str, Any]] = []
    for case_number, item in enumerate(rows, start=1):
        case_id = f"case_{case_number:03d}_qid_{_safe_id(str(item.get('question_id')))}"
        pages = _document_pages(run_dir, str(item.get("document_id")))
        retrieved_rows = _retrieval_rows(item, pages, assets_dir, case_id)
        gold_rows = _gold_rows(item, pages, assets_dir, case_id)
        contact_rows = retrieved_rows + [row for row in gold_rows if not row.get("retrieved")]
        contact_sheet = _make_contact_sheet(contact_rows, assets_dir / case_id / "contact_sheet.jpg")
        cases.append(
            {
                "case_id": case_id,
                "item": item,
                "diagnosis": _diagnose(item),
                "retrieved_rows": retrieved_rows,
                "gold_rows": gold_rows,
                "contact_sheet": contact_sheet,
                "missing_page_images": not pages,
            }
        )
    return cases


def _summary_counts(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = [item for item in predictions if float(item.get("score", 0.0)) == 0.0]
    by_label = collections.Counter(item.get("failure_label", "unknown") for item in wrong)
    retrievable = [item for item in predictions if item.get("gold_hit_at_k") is not None]
    return {
        "total": len(predictions),
        "wrong": len(wrong),
        "failure_labels": dict(by_label),
        "gold_hit_at_k": sum(1 for item in retrievable if item.get("gold_hit_at_k")),
        "gold_all_hit_at_k": sum(1 for item in retrievable if item.get("gold_all_hit_at_k")),
        "retrieval_total": len(retrievable),
    }


def _write_markdown(report_dir: Path, run_dir: Path, cases: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    lines = [
        f"# ColPali Failure Analysis",
        "",
        f"Run: `{run_dir}`",
        f"Total predictions: {summary['total']}",
        f"Wrong predictions: {summary['wrong']}",
        f"Failure labels: `{summary['failure_labels']}`",
        (
            "Retrieval hit/all-hit: "
            f"{summary['gold_hit_at_k']}/{summary['retrieval_total']} / "
            f"{summary['gold_all_hit_at_k']}/{summary['retrieval_total']}"
        ),
        "",
    ]
    for case in cases:
        item = case["item"]
        lines.extend(
            [
                f"## QID {item.get('question_id')} | {item.get('category')} | {item.get('failure_label')}",
                "",
                f"Question: {_short(item.get('question'), 500)}",
                f"Ground truth: `{item.get('ground_truth_answer')}`",
                f"Answer: `{item.get('model_answer')}`",
                f"Raw answer: `{_short(item.get('raw_model_answer'), 500)}`",
                "",
            ]
        )
        if case.get("contact_sheet"):
            lines.append(f"![contact sheet]({case['contact_sheet']})")
            lines.append("")
        lines.append("Diagnosis:")
        lines.extend(f"- {note}" for note in case["diagnosis"])
        lines.append("")
        lines.append("| Rank | Page | Score | Gold | Clue | Kept | Image |")
        lines.append("| --- | ---: | ---: | --- | --- | --- | --- |")
        for row in case["retrieved_rows"]:
            image = f"[png]({row['copied_path']})" if row.get("copied_path") else "missing"
            lines.append(
                f"| {row['rank']} | {row['display_page']} | {float(row.get('score') or 0.0):.4f} | "
                f"{row['is_gold']} | {row['has_clue']} | {row['kept_visual']} | {image} |"
            )
        lines.append("")
        if case["gold_rows"]:
            lines.append("Gold pages:")
            for row in case["gold_rows"]:
                image = f" [png]({row['copied_path']})" if row.get("copied_path") else ""
                lines.append(
                    f"- Page {row['display_page']} internal `{row['page_index']}`: "
                    f"{'retrieved' if row['retrieved'] else 'missed'}{image}"
                )
            lines.append("")
    path = report_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _html_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        image_html = (
            f'<a href="{html.escape(row["copied_path"])}"><img src="{html.escape(row["copied_path"])}" /></a>'
            if row.get("copied_path")
            else "<span class='missing'>missing</span>"
        )
        table_rows.append(
            "<tr>"
            f"<td>{row.get('rank', '')}</td>"
            f"<td>{row.get('display_page')}</td>"
            f"<td>{float(row.get('score') or 0.0):.4f}</td>"
            f"<td>{row.get('is_gold')}</td>"
            f"<td>{row.get('has_clue')}</td>"
            f"<td>{row.get('kept_visual')}</td>"
            f"<td>{html.escape(_short(row.get('clue_summary'), 140))}</td>"
            f"<td>{html.escape(str(row.get('screen_relevance') or ''))}</td>"
            f"<td>{image_html}</td>"
            "</tr>"
        )
    return "\n".join(table_rows)


def _write_html(report_dir: Path, run_dir: Path, cases: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    case_blocks = []
    for case in cases:
        item = case["item"]
        diagnosis = "\n".join(f"<li>{html.escape(note)}</li>" for note in case["diagnosis"])
        contact = ""
        if case.get("contact_sheet"):
            contact = (
                f'<a href="{html.escape(case["contact_sheet"])}">'
                f'<img class="sheet" src="{html.escape(case["contact_sheet"])}" /></a>'
            )
        gold_lines = []
        for row in case["gold_rows"]:
            link = f' <a href="{html.escape(row["copied_path"])}">image</a>' if row.get("copied_path") else ""
            gold_lines.append(
                f"<li>Page {row['display_page']} internal {row['page_index']}: "
                f"{'retrieved' if row['retrieved'] else 'missed'}{link}</li>"
            )
        case_blocks.append(
            f"""
            <section class="case">
              <h2>QID {html.escape(str(item.get('question_id')))} | {html.escape(str(item.get('category')))} | {html.escape(str(item.get('failure_label')))}</h2>
              <p><b>Question:</b> {html.escape(_short(item.get('question'), 900))}</p>
              <p><b>Ground truth:</b> <code>{html.escape(str(item.get('ground_truth_answer')))}</code></p>
              <p><b>Answer:</b> <code>{html.escape(str(item.get('model_answer')))}</code></p>
              <p><b>Raw answer:</b> <code>{html.escape(_short(item.get('raw_model_answer'), 900))}</code></p>
              <p><b>Core:</b> d={html.escape(str(item.get('difficulty_level')))} mode={html.escape(str(item.get('core_decision_mode')))} switched={html.escape(str(item.get('difficulty_model_switching_used')))}</p>
              {contact}
              <h3>Diagnosis</h3>
              <ul>{diagnosis}</ul>
              <h3>Retrieved ColPali Top-K</h3>
              <table>
                <thead><tr><th>Rank</th><th>Page</th><th>Score</th><th>Gold</th><th>Clue</th><th>Kept</th><th>Clue Summary</th><th>Screen</th><th>Image</th></tr></thead>
                <tbody>{_html_table(case["retrieved_rows"])}</tbody>
              </table>
              <h3>Gold Pages</h3>
              <ul>{''.join(gold_lines) or '<li>No gold pages annotated.</li>'}</ul>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>ColPali Failure Analysis</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; line-height: 1.35; }}
    code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 4px; }}
    .case {{ border-top: 2px solid #ddd; padding-top: 24px; margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
    th {{ background: #f3f3f3; text-align: left; }}
    td img {{ max-width: 170px; max-height: 220px; display: block; }}
    .sheet {{ max-width: min(100%, 1150px); border: 1px solid #ddd; }}
    .missing {{ color: #a00; }}
  </style>
</head>
<body>
  <h1>ColPali Failure Analysis</h1>
  <p><b>Run:</b> <code>{html.escape(str(run_dir))}</code></p>
  <p><b>Total predictions:</b> {summary['total']} &nbsp; <b>Wrong:</b> {summary['wrong']}</p>
  <p><b>Failure labels:</b> <code>{html.escape(str(summary['failure_labels']))}</code></p>
  <p><b>Retrieval hit/all-hit:</b> {summary['gold_hit_at_k']}/{summary['retrieval_total']} / {summary['gold_all_hit_at_k']}/{summary['retrieval_total']}</p>
  {''.join(case_blocks)}
</body>
</html>
"""
    path = report_dir / "index.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    default_run_root = Path(f"/scratch/{os.environ.get('USER', '$USER')}/agenticdocai/runs")
    parser = argparse.ArgumentParser(
        description="Generate a visual ColPali retrieval failure report for an MMLongBench eval run."
    )
    parser.add_argument("run", help="Run directory or numeric Slurm job id, e.g. 53765133")
    parser.add_argument("--run-root", default=str(default_run_root), help="Directory containing eval runs")
    parser.add_argument("--out-dir", default=None, help="Report output directory")
    parser.add_argument("--label", default=None, help="Only include one failure label")
    parser.add_argument("--qid", action="append", default=None, help="Only include a question id; repeatable")
    parser.add_argument("--limit", type=int, default=30, help="Maximum cases to include")
    parser.add_argument("--include-correct", action="store_true", help="Include correct cases too")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run, Path(args.run_root))
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")
    report_dir = Path(args.out_dir) if args.out_dir else run_dir / "colpali_failure_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_jsonl(predictions_path)
    cases = _prepare_cases(
        predictions,
        run_dir=run_dir,
        report_dir=report_dir,
        label=args.label,
        qids=set(args.qid) if args.qid else None,
        limit=args.limit,
        include_correct=args.include_correct,
    )
    summary = _summary_counts(predictions)
    html_path = _write_html(report_dir, run_dir, cases, summary)
    md_path = _write_markdown(report_dir, run_dir, cases, summary)
    print(f"Wrote HTML report: {html_path}")
    print(f"Wrote Markdown report: {md_path}")
    print(f"Cases included: {len(cases)}")


if __name__ == "__main__":
    main()
