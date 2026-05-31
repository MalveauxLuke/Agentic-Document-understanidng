from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 160), color).save(path)


def test_colpali_failure_report_copies_images_and_writes_html(tmp_path):
    run_dir = tmp_path / "mmlongbench_eval_sleuth_123"
    pages_dir = run_dir / "cache" / "documents" / "doc_pdf" / "pages"
    page_paths = []
    for index, color in enumerate(["white", "lightgray", "pink"]):
        path = pages_dir / f"page_{index + 1:04d}.png"
        _write_image(path, color)
        page_paths.append(path)

    pages_json = run_dir / "cache" / "documents" / "doc_pdf" / "pages.json"
    pages_json.parent.mkdir(parents=True, exist_ok=True)
    pages_json.write_text(
        json.dumps(
            [
                {"page_index": index, "image_path": str(path)}
                for index, path in enumerate(page_paths)
            ]
        ),
        encoding="utf-8",
    )
    prediction = {
        "question_id": "42",
        "row_index": 0,
        "document_id": "doc.pdf",
        "question": "Which page has the answer?",
        "ground_truth_answer": "page three",
        "model_answer": "No answers found!",
        "raw_model_answer": "No answers found!",
        "score": 0.0,
        "category": "Chart",
        "failure_label": "retrieval_miss",
        "gold_evidence_pages": [2],
        "evidence_pages": [2],
        "retrieved_pages": [
            {"page_index": 0, "score": 8.1},
            {"page_index": 1, "score": 7.9},
        ],
        "clue_output": [],
        "page_screening_output": [],
        "gold_hit_at_k": False,
        "gold_all_hit_at_k": False,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")

    report_dir = tmp_path / "report"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_colpali_failures.py"),
            str(run_dir),
            "--out-dir",
            str(report_dir),
            "--limit",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Wrote HTML report" in completed.stdout
    html = (report_dir / "index.html").read_text(encoding="utf-8")
    markdown = (report_dir / "summary.md").read_text(encoding="utf-8")
    assert "retrieval_miss" in html
    assert "ColPali top-K retrieved no gold page" in html
    assert "rank_01_page_0000.png" in html
    assert "gold_page_0002.png" in markdown
    assert (report_dir / "assets" / "case_001_qid_42" / "rank_01_page_0000.png").exists()
    assert (report_dir / "assets" / "case_001_qid_42" / "gold_page_0002.png").exists()


def test_colpali_failure_report_reads_external_dpi_cache_from_run_config(tmp_path):
    run_dir = tmp_path / "mmlongbench_eval_sleuth_456"
    cache_dir = tmp_path / "shared_cache"
    pages_dir = cache_dir / "documents" / "dpi_144" / "doc_pdf" / "pages"
    page_path = pages_dir / "page_0001.png"
    _write_image(page_path, "white")
    pages_json = cache_dir / "documents" / "dpi_144" / "doc_pdf" / "pages.json"
    pages_json.parent.mkdir(parents=True, exist_ok=True)
    pages_json.write_text(json.dumps([{"page_index": 0, "image_path": str(page_path)}]), encoding="utf-8")

    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps({"cache_dir": str(cache_dir), "render_dpi": 144}),
        encoding="utf-8",
    )
    prediction = {
        "question_id": "99",
        "row_index": 0,
        "document_id": "doc.pdf",
        "question": "Which page?",
        "ground_truth_answer": "one",
        "model_answer": "wrong",
        "raw_model_answer": "wrong",
        "score": 0.0,
        "category": "Chart",
        "failure_label": "final_wrong",
        "gold_evidence_pages": [0],
        "retrieved_pages": [{"page_index": 0, "score": 1.0}],
        "clue_output": [],
        "page_screening_output": [],
        "gold_hit_at_k": True,
        "gold_all_hit_at_k": True,
    }
    (run_dir / "predictions.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")

    report_dir = tmp_path / "report_external"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_colpali_failures.py"),
            str(run_dir),
            "--out-dir",
            str(report_dir),
            "--limit",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert (report_dir / "assets" / "case_001_qid_99" / "rank_01_page_0000.png").exists()
