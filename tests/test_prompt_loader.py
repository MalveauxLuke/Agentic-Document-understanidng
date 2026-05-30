from pathlib import Path

from sleuth.agents.prompts import (
    build_clue_discovery_prompt,
    build_core_decision_prompt,
    build_evidence_verification_crop_prompt,
    build_page_screening_prompt,
)
from sleuth.instructions.prompt_loader import get_prompt_section, load_agent_prompt_markdown


ROOT = Path(__file__).resolve().parents[1]


def test_load_agent_prompts_markdown():
    text = load_agent_prompt_markdown(ROOT / "agent_prompts.md")
    assert "Clue Discovery Agent" in text


def test_extract_required_sections_or_safe_fallback():
    text = load_agent_prompt_markdown(ROOT / "agent_prompts.md")
    for section_name in [
        "clue_discovery",
        "page_screening",
        "evidence_verification_crop",
        "evidence_verification_full_page",
        "difficulty_assessment",
        "core_decision_text",
        "core_decision_visual",
    ]:
        section = get_prompt_section(text, section_name)
        assert section.strip()
        assert "Fallback prompt section" in section or len(section) < len(text)


def test_prompt_builder_does_not_add_removed_noise():
    text = load_agent_prompt_markdown(ROOT / "agent_prompts.md")
    section = get_prompt_section(text, "clue_discovery")
    prompt = build_clue_discovery_prompt(
        question="What changed?",
        page_index=3,
        page_text="This extracted text should not be appended.",
        agent_prompt_text=section,
        sol_instruction_text="This SOL text should not be prepended.",
    )
    assert "SOL-SPECIFIC INSTRUCTIONS" not in prompt
    assert "This extracted text should not be appended." not in prompt
    assert "Return valid JSON only. Do not wrap" not in prompt
    assert "What changed?" in prompt
    assert "Page Number: 4" in prompt
    assert "crop_region" in prompt


def test_page_screening_prompt_uses_display_page_number():
    text = load_agent_prompt_markdown(ROOT / "agent_prompts.md")
    section = get_prompt_section(text, "page_screening")
    prompt = build_page_screening_prompt(
        question="Which chart?",
        page_index=0,
        agent_prompt_text=section,
    )
    assert "Page Number: 1" in prompt


def test_verification_prompt_uses_display_page_number_and_crop_metadata():
    text = load_agent_prompt_markdown(ROOT / "agent_prompts.md")
    section = get_prompt_section(text, "evidence_verification_crop")
    prompt = build_evidence_verification_crop_prompt(
        question="Which value?",
        page_index=0,
        crop_hint="upper_left",
        crop_location="upper_left bbox=(0, 0, 50, 50)",
        evidence_type="chart",
        content='Value "42"',
        location="chart",
        relevance="direct",
        confidence="high",
        agent_prompt_text=section,
    )
    assert "Page Number: 1" in prompt
    assert "Crop Hint: upper_left" in prompt
    assert 'Value \\"42\\"' in prompt


def test_core_visual_prompt_uses_display_page_numbers():
    text_prompt = "Text EVIDENCE: {evidence_summary}"
    visual_prompt = "{visual_evidence_section}\nEVIDENCE: {evidence_summary}"
    prompt = build_core_decision_prompt(
        question="Which page?",
        instruction_set="Use visual pages.",
        evidence_summary="Evidence.",
        retained_page_indices=[0, 3],
        retained_image_paths=["/tmp/page_0001.png", "/tmp/page_0004.png"],
        text_agent_prompt_text=text_prompt,
        visual_agent_prompt_text=visual_prompt,
    )
    assert "Page Number 1" in prompt
    assert "Page Number 4" in prompt
    assert "Page index" not in prompt
