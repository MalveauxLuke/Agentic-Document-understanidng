from sleuth.evaluation.answer_extraction import HeuristicAnswerExtractor, NoneAnswerExtractor


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
