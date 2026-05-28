"""Prompt builders for the static SLEUTH-style baseline.

The runtime prompts are loaded from agent_prompts.md. This module only fills
placeholders; it does not prepend SOL instructions or append implementation
constraints.
"""

from __future__ import annotations


def format_sol_instruction_block(sol_instruction_text: str | None) -> str:
    return ""


def _fill_placeholders(text: str, values: dict[str, str]) -> str:
    rendered = text
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _compose_prompt(agent_prompt_text: str, sol_instruction_text: str | None = None) -> str:
    return agent_prompt_text.strip()


def build_clue_discovery_prompt(
    question: str,
    page_index: int,
    page_text: str | None,
    agent_prompt_text: str,
    sol_instruction_text: str | None = None,
) -> str:
    prompt = _fill_placeholders(
        agent_prompt_text,
        {
            "question": question,
            "page_num": str(page_index),
            "page_number": str(page_index),
            "page num": str(page_index),
            "page number": str(page_index),
        },
    )
    return _compose_prompt(prompt, sol_instruction_text)


def build_page_screening_prompt(
    question: str,
    page_index: int,
    agent_prompt_text: str,
    sol_instruction_text: str | None = None,
) -> str:
    prompt = _fill_placeholders(
        agent_prompt_text,
        {
            "question": question,
            "page_num": str(page_index),
            "page_number": str(page_index),
            "page num": str(page_index),
            "page number": str(page_index),
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
        page_lines = "\n".join(f"- Page index {page_index}" for page_index in retained_page_indices)
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
