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
    assert prediction["retrieved_display_page_numbers"] == [1]
    assert prediction["raw_model_answer"]
    assert prediction["extracted_answer"]
    assert "raw_score" in prediction
    assert "gold_hit_at_k" in prediction
    assert prediction["source_evidence_pages"] == [1]
    assert prediction["gold_evidence_pages"] == [0]
    assert prediction["gold_display_page_numbers"] == [1]
    assert "gold_all_hit_at_k" in prediction
    assert "gold_page_recall_at_k" in prediction
    assert "gold_retrieved_pages" in prediction
    assert "gold_retrieved_display_page_numbers" in prediction
    assert "gold_missed_pages" in prediction
    assert "gold_missed_display_page_numbers" in prediction
    assert "clue_has_any_evidence" in prediction
    assert "clue_gold_pages_with_evidence" in prediction
    assert "clue_gold_display_page_numbers_with_evidence" in prediction
    assert "clue_non_gold_pages_with_evidence" in prediction
    assert "clue_non_gold_display_page_numbers_with_evidence" in prediction
    assert "clue_missed_gold_pages" in prediction
    assert "clue_missed_gold_display_page_numbers" in prediction
    assert "failure_label" in prediction
    assert prediction["clue_output"]
    assert prediction["page_screening_output"]
    assert prediction["difficulty_output"]
    assert prediction["difficulty_level"] == 0
    assert prediction["core_decision_mode"] == "instruct"
    assert prediction["core_decision_model"] == "MockClient"
    assert prediction["difficulty_model_switching_used"] is False
    assert prediction["final_prompt"]

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["attempted"] == 1
    assert metrics["total"] == 1
    assert metrics["failed"] == 0
    assert "raw_average_score" in metrics
    assert "gold_hit_at_k_rate" in metrics
    assert "gold_all_hit_at_k_rate" in metrics
    assert "gold_page_recall_at_k_average" in metrics
    assert "clue_has_any_evidence_rate" in metrics
    assert "answer_extractor_model" in metrics
    assert "answer_extractor_base_url" in metrics
    assert metrics["pipeline_cache_version"]
    assert metrics["region_refinement"] == "fallback"
    assert metrics["agent_source_fingerprint"]
    assert metrics["thinking_model"] is None
    assert metrics["difficulty_model_switching_enabled"] is False
    assert metrics["core_decision_thinking_max_tokens"] == 4096
    assert metrics["core_decision_thinking_count"] == 0
    assert metrics["core_decision_thinking_total"] == 1

    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert "answer_extractor_model" in run_config
    assert "answer_extractor_base_url" in run_config
    assert run_config["pipeline_cache_version"] == metrics["pipeline_cache_version"]
    assert run_config["region_refinement"] == metrics["region_refinement"]
    assert run_config["agent_source_fingerprint"] == metrics["agent_source_fingerprint"]
    assert run_config["thinking_model"] is None
    assert run_config["difficulty_model_switching_enabled"] is False
    assert run_config["core_decision_thinking_max_tokens"] == 4096
