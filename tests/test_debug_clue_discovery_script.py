from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_debug_clue_discovery_uses_cached_retrieval_and_skips_scoring(tmp_path):
    cache_dir = tmp_path / "cache"
    eval_dir = tmp_path / "eval"
    debug_dir = tmp_path / "debug"
    qid_file = tmp_path / "qids.json"
    qid_file.write_text(json.dumps(["949"]), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_mmlongbench_eval.py"),
            "--data_dir",
            str(ROOT),
            "--method",
            "sleuth",
            "--mode",
            "mock",
            "--qid",
            "949",
            "--top_k",
            "1",
            "--output_dir",
            str(eval_dir),
            "--cache_dir",
            str(cache_dir),
            "--render_dpi",
            "72",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "debug_clue_discovery.py"),
            "--data_dir",
            str(ROOT),
            "--cache_dir",
            str(cache_dir),
            "--qid-file",
            str(qid_file),
            "--use-retrieved-pages",
            "--mode",
            "mock",
            "--retriever",
            "dummy",
            "--top_k",
            "1",
            "--render_dpi",
            "72",
            "--output_dir",
            str(debug_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Wrote clue debug report" in completed.stdout
    assert (debug_dir / "index.html").exists()
    cases = json.loads((debug_dir / "cases.json").read_text(encoding="utf-8"))
    assert len(cases) == 1
    assert (debug_dir / cases[0]["case_rel"] / "clue.json").exists()
    assert not (debug_dir / "predictions.jsonl").exists()
    assert not (debug_dir / "metrics.json").exists()
