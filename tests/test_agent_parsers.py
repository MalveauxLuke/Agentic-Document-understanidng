from sleuth.agents.clue_discovery import _normalize_clue_data
from sleuth.agents.difficulty_assessment import _normalize_difficulty_data
from sleuth.agents.page_screening import _parse_screening_text


def test_clue_parser_accepts_paper_style_keys():
    data = _normalize_clue_data(
        {
            "page number": 2,
            "has relevant evidence": True,
            "evidence items": [
                {
                    "evidence type": "chart",
                    "content": "42%",
                    "location": "bar chart",
                    "relevance": "answers the question",
                    "confidence": "high",
                }
            ],
            "page summary": "summary",
            "key insights": "insight",
        },
        page_index=2,
    )
    assert data["has_relevant_evidence"] is True
    assert data["evidence_items"][0]["evidence_type"] == "chart"
    assert data["page_summary"] == "summary"
    assert data["key_insights"] == "insight"


def test_page_screening_parser_accepts_original_format():
    data = _parse_screening_text(
        "Has Chart: Yes\n"
        "Relevance: Completely Relevant\n"
        "Reasoning: The chart directly answers the question."
    )
    assert data is not None
    assert data["has_visual_element"] is True
    assert data["relevance"] == "Completely Relevant"
    assert data["keep_page"] is True


def test_page_screening_parser_accepts_none():
    data = _parse_screening_text("none")
    assert data is not None
    assert data["has_visual_element"] is False
    assert data["relevance"] == "None"
    assert data["keep_page"] is False


def test_difficulty_parser_accepts_paper_style_keys():
    data = _normalize_difficulty_data({"difficulty level": "1", "instruction set": "Compare pages."})
    assert data == {"difficulty_level": 1, "instruction_set": "Compare pages."}
