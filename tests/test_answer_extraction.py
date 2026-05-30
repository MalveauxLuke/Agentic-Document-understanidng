import pytest

from sleuth.evaluation.answer_extraction import (
    HeuristicAnswerExtractor,
    NoneAnswerExtractor,
    OpenAICompatibleAnswerExtractor,
    build_answer_extractor,
)


def test_heuristic_extracts_json_answer():
    extractor = HeuristicAnswerExtractor()
    result = extractor.extract("Q?", '{"answer": "Latinos interviewed by cellphone"}', "Str")
    assert result.extracted_answer == "Latinos interviewed by cellphone"
    assert result.paper_comparable is False


def test_heuristic_extracts_answer_line():
    extractor = HeuristicAnswerExtractor()
    result = extractor.extract("Q?", "Reasoning text.\nAnswer: Less well-off", "Str")
    assert result.extracted_answer == "Less well-off"


def test_none_extractor_keeps_clean_raw_answer():
    extractor = NoneAnswerExtractor()
    result = extractor.extract("Q?", "No answers found!", "None")
    assert result.extracted_answer == "No answers found!"


def test_heuristic_extracts_list_answer_as_atomic_items():
    extractor = HeuristicAnswerExtractor()
    result = extractor.extract("Q?", "Answer: White, 10 percentage points", "List")
    assert result.extracted_answer == "['White', '10 percentage points']"


def test_heuristic_strips_visible_thinking_blocks():
    extractor = HeuristicAnswerExtractor()
    result = extractor.extract("Q?", "<think>private reasoning</think>\nAnswer: Europe", "Str")
    assert result.extracted_answer == "Europe"


def test_sol_auto_requires_api_answer_extractor_key(monkeypatch):
    for env_name in [
        "ANSWER_EXTRACTOR_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    with pytest.raises(EnvironmentError, match="SOL mode requires"):
        build_answer_extractor("auto", mode="sol")


def test_sol_auto_defaults_to_gpt_41_mini(monkeypatch):
    for env_name in [
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANSWER_EXTRACTOR_MODEL",
        "ANSWER_EXTRACTOR_BASE_URL",
        "DEEPSEEK_BASE_URL",
        "OPENAI_BASE_URL",
    ]:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("ANSWER_EXTRACTOR_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "not-used")
    monkeypatch.setenv("OPENAI_MODEL", "not-used-either")

    extractor = build_answer_extractor("auto", mode="sol")

    assert isinstance(extractor, OpenAICompatibleAnswerExtractor)
    assert extractor.model == "gpt-4.1-mini"
    assert extractor.base_url == "https://api.openai.com/v1"
