from pathlib import Path

from sleuth.instructions.prompt_loader import get_prompt_section, load_agent_prompt_markdown


ROOT = Path(__file__).resolve().parents[1]


def test_load_agent_prompts_markdown():
    text = load_agent_prompt_markdown(ROOT / "agent_prompts.md")
    assert "CLUE DISCOVERY AGENT PROMPT" in text


def test_extract_required_sections_or_safe_fallback():
    text = load_agent_prompt_markdown(ROOT / "agent_prompts.md")
    for section_name in [
        "clue_discovery",
        "page_screening",
        "difficulty_assessment",
        "core_decision_text",
        "core_decision_visual",
    ]:
        section = get_prompt_section(text, section_name)
        assert section.strip()
        assert "Fallback prompt section" in section or len(section) < len(text)
