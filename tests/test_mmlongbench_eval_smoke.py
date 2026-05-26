import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mmlongbench_eval_mock_smoke(tmp_path):
    output_dir = tmp_path / "mmlongbench_eval_mock"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_mmlongbench_eval.py"),
        "--data_dir",
        str(ROOT),
        "--method",
        "sleuth",
        "--mode",
        "mock",
        "--limit",
        "1",
        "--top_k",
        "1",
        "--output_dir",
        str(output_dir),
        "--render_dpi",
        "72",
    ]

    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)

    assert "Output directory" in completed.stdout
    assert "MMLongBench-Doc Results" in completed.stdout
    assert "By Category" in completed.stdout
    for filename in [
        "predictions.jsonl",
        "metrics.json",
        "metrics_by_category.csv",
        "failed_examples.jsonl",
        "run_config.json",
    ]:
        assert (output_dir / filename).exists()

    prediction_lines = (output_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(prediction_lines) == 1
    prediction = json.loads(prediction_lines[0])
    assert prediction["document_id"].endswith(".pdf")
    assert prediction["retrieved_page_indices"] == [0]
    assert prediction["clue_output"]
    assert prediction["page_screening_output"]
    assert prediction["difficulty_output"]
    assert prediction["final_prompt"]

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["attempted"] == 1
    assert metrics["total"] == 1
    assert metrics["failed"] == 0
