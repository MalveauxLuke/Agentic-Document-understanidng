from sleuth.evaluation.metrics import compute_metrics, compute_metrics_by_category
from sleuth.evaluation.scoring import eval_score


def test_eval_score_int_and_float():
    assert eval_score("12", "12", "Int") == 1.0
    assert eval_score("12", "12.0", "Int") == 1.0
    assert eval_score("0.25", "25%", "Float") == 1.0
    assert eval_score("10.0", "10.05", "Float") == 1.0


def test_eval_score_string_none_and_list():
    assert eval_score("21-13199", "21-13199", "Str") == 1.0
    assert eval_score("No answers found!", "No answers found!", "None") == 1.0
    assert eval_score("Not answerable", "No answers found!", "None") == 1.0
    assert eval_score("unanswerable", "insufficient information", "None") == 1.0
    assert eval_score("Not answerable", "42", "None") == 0.0
    assert eval_score("['alpha', 'beta']", "['beta', 'alpha']", "List") == 1.0
    assert eval_score("['alpha', 'beta']", "['alpha']", "List") == 0.0


def test_metrics_overall_and_by_category():
    predictions = [
        {"score": 1.0, "categories": ["Table"]},
        {"score": 0.5, "categories": ["Table", "Figure"]},
        {"score": 0.0, "categories": ["None"]},
    ]

    metrics = compute_metrics(predictions)
    by_category = {row["category"]: row for row in compute_metrics_by_category(predictions)}

    assert metrics["total"] == 3
    assert metrics["correct"] == 2
    assert metrics["accuracy"] == 0.5
    assert by_category["Table"]["total"] == 2
    assert by_category["Table"]["correct"] == 2
    assert by_category["Figure"]["accuracy"] == 0.5
    assert by_category["None"]["correct"] == 0
