"""Prompt builders for the static SLEUTH-style baseline.

The runtime prompts are loaded from agent_prompts.md. This module only fills
placeholders; it does not prepend SOL instructions or append implementation
constraints.
"""

from __future__ import annotations

import json


def display_page_number(page_index: int) -> int:
    return page_index + 1


def format_sol_instruction_block(sol_instruction_text: str | None) -> str:
    return ""


def _fill_placeholders(text: str, values: dict[str, str]) -> str:
    rendered = text
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _json_string_fragment(value: object) -> str:
    return json.dumps("" if value is None else str(value))[1:-1]


def _compose_prompt(agent_prompt_text: str, sol_instruction_text: str | None = None) -> str:
    return agent_prompt_text.strip()


def build_clue_discovery_prompt(
    question: str,
    page_index: int,
    page_text: str | None,
    agent_prompt_text: str,
    sol_instruction_text: str | None = None,
) -> str:
    page_number = str(display_page_number(page_index))
    prompt = _fill_placeholders(
        agent_prompt_text,
        {
            "question": question,
            "page_num": page_number,
            "page_number": page_number,
            "page num": page_number,
            "page number": page_number,
        },
    )
    return _compose_prompt(prompt, sol_instruction_text)


def build_page_screening_prompt(
    question: str,
    page_index: int,
    agent_prompt_text: str,
    sol_instruction_text: str | None = None,
) -> str:
    page_number = str(display_page_number(page_index))
    prompt = _fill_placeholders(
        agent_prompt_text,
        {
            "question": question,
            "page_num": page_number,
            "page_number": page_number,
            "page num": page_number,
            "page number": page_number,
        },
    )
    return _compose_prompt(prompt, sol_instruction_text)


def build_evidence_verification_crop_prompt(
    question: str,
    page_index: int,
    crop_hint: str,
    crop_location: str,
    evidence_type: str,
    content: str,
    location: str,
    relevance: str,
    confidence: str,
    agent_prompt_text: str,
    sol_instruction_text: str | None = None,
) -> str:
    page_number = str(display_page_number(page_index))
    prompt = _fill_placeholders(
        agent_prompt_text,
        {
            "question": question,
            "page_num": page_number,
            "page_number": page_number,
            "crop_hint": crop_hint,
            "crop_location": crop_location,
            "evidence_type": _json_string_fragment(evidence_type),
            "content": _json_string_fragment(content),
            "location": _json_string_fragment(location),
            "relevance": _json_string_fragment(relevance),
            "confidence": _json_string_fragment(confidence),
        },
    )
    return _compose_prompt(prompt, sol_instruction_text)


def build_evidence_verification_full_page_prompt(
    question: str,
    page_index: int,
    crop_hint: str,
    crop_problem: str | None,
    crop_verifier_output: str,
    evidence_type: str,
    content: str,
    location: str,
    relevance: str,
    confidence: str,
    agent_prompt_text: str,
    sol_instruction_text: str | None = None,
) -> str:
    page_number = str(display_page_number(page_index))
    prompt = _fill_placeholders(
        agent_prompt_text,
        {
            "question": question,
            "page_num": page_number,
            "page_number": page_number,
            "crop_hint": crop_hint,
            "crop_problem": _json_string_fragment(crop_problem or ""),
            "crop_verifier_output": crop_verifier_output,
            "evidence_type": _json_string_fragment(evidence_type),
            "content": _json_string_fragment(content),
            "location": _json_string_fragment(location),
            "relevance": _json_string_fragment(relevance),
            "confidence": _json_string_fragment(confidence),
        },
    )
    return _compose_prompt(prompt, sol_instruction_text)


def build_difficulty_prompt(
    question: str,
    evidence_summary: str,
    agent_prompt_text: str,
    sol_instruction_text: str | None = None,
) -> str:
    prompt = _fill_placeholders(
        agent_prompt_text,
        {
            "question": question,
            "evidence_summary": evidence_summary,
            "evidence summary": evidence_summary,
        },
    )
    return _compose_prompt(prompt, sol_instruction_text)


def build_core_decision_text_prompt(
    question: str,
    instruction_set: str,
    evidence_summary: str,
    num_pages: int,
    agent_prompt_text: str,
    sol_instruction_text: str | None = None,
) -> str:
    prompt = _fill_placeholders(
        agent_prompt_text,
        {
            "question": question,
            "instruction_set": instruction_set,
            "instruction set": instruction_set,
            "evidence_summary": evidence_summary,
            "evidence summary": evidence_summary,
            "num_pages": str(num_pages),
            "num pages": str(num_pages),
        },
    )
    return _compose_prompt(prompt, sol_instruction_text)


def build_core_decision_visual_prompt(
    question: str,
    instruction_set: str,
    evidence_summary: str,
    visual_evidence_section: str,
    num_pages: int,
    agent_prompt_text: str,
    sol_instruction_text: str | None = None,
) -> str:
    prompt = _fill_placeholders(
        agent_prompt_text,
        {
            "question": question,
            "instruction_set": instruction_set,
            "instruction set": instruction_set,
            "evidence_summary": evidence_summary,
            "evidence summary": evidence_summary,
            "visual_evidence_section": visual_evidence_section,
            "visual evidence section": visual_evidence_section,
            "num_pages": str(num_pages),
            "num pages": str(num_pages),
        },
    )
    return _compose_prompt(prompt, sol_instruction_text)


def build_core_decision_prompt(
    question: str,
    instruction_set: str,
    evidence_summary: str,
    retained_page_indices: list[int],
    retained_image_paths: list[str],
    text_agent_prompt_text: str,
    visual_agent_prompt_text: str,
    sol_instruction_text: str | None = None,
) -> str:
    if retained_image_paths:
        page_lines = "\n".join(
            f"- Page Number {display_page_number(page_index)}" for page_index in retained_page_indices
        )
        visual_evidence_section = (
            "The following page images are provided as visual evidence:\n"
            f"{page_lines}\n\n"
            "The actual images will be passed to the multimodal model separately."
        )
        return build_core_decision_visual_prompt(
            question=question,
            instruction_set=instruction_set,
            evidence_summary=evidence_summary,
            visual_evidence_section=visual_evidence_section,
            num_pages=len(retained_page_indices),
            agent_prompt_text=visual_agent_prompt_text,
            sol_instruction_text=sol_instruction_text,
        )
    return build_core_decision_text_prompt(
        question=question,
        instruction_set=instruction_set,
        evidence_summary=evidence_summary,
        num_pages=len(retained_page_indices),
        agent_prompt_text=text_agent_prompt_text,
        sol_instruction_text=sol_instruction_text,
    )
