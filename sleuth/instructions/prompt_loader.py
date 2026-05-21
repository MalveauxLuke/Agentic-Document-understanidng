from __future__ import annotations

import warnings
from pathlib import Path

from sleuth.instructions.instruction_loader import read_markdown_file


SECTION_NAMES = {
    "clue_discovery": ("CLUE DISCOVERY AGENT PROMPT",),
    "page_screening": ("PAGE SCREENING AGENT PROMPT",),
    "difficulty_assessment": ("DIFFICULTY ASSESSMENT AGENT PROMPT",),
    "core_decision_text": ("CORE DECISION AGENT PROMPT", "TEXT EVIDENCE ONLY"),
    "core_decision_visual": ("CORE DECISION AGENT PROMPT", "WITH VISUALS"),
}


def load_agent_prompt_markdown(path: str | Path) -> str:
    return read_markdown_file(path)


def _line_matches(line: str, tokens: tuple[str, ...]) -> bool:
    upper = line.strip().upper()
    return all(token in upper for token in tokens)


def _all_section_starts(lines: list[str]) -> list[tuple[int, str]]:
    starts: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        for name, tokens in SECTION_NAMES.items():
            if _line_matches(line, tokens):
                starts.append((idx, name))
    starts.sort(key=lambda item: item[0])
    return starts


def _strip_implementation_preamble(section: str) -> str:
    marker = "Use this prompt content:"
    if marker in section:
        return section.split(marker, 1)[1].strip()
    return section.strip()


def get_prompt_section(markdown_text: str, section_name: str) -> str:
    if section_name not in SECTION_NAMES:
        raise KeyError(f"Unknown prompt section name: {section_name}")

    lines = markdown_text.splitlines()
    starts = _all_section_starts(lines)
    start_indices = {name: idx for idx, name in starts}
    if section_name not in start_indices:
        warnings.warn(
            f"Could not find prompt section '{section_name}'. Falling back to full prompt markdown.",
            RuntimeWarning,
            stacklevel=2,
        )
        return f"Fallback prompt section for {section_name}.\n\n{markdown_text.strip()}"

    start = start_indices[section_name]
    later_starts = [idx for idx, _ in starts if idx > start]
    end = min(later_starts) if later_starts else len(lines)
    section = "\n".join(lines[start:end]).strip()
    cleaned = _strip_implementation_preamble(section)
    if not cleaned:
        warnings.warn(
            f"Prompt section '{section_name}' was empty. Falling back to full prompt markdown.",
            RuntimeWarning,
            stacklevel=2,
        )
        return f"Fallback prompt section for {section_name}.\n\n{markdown_text.strip()}"
    return cleaned
