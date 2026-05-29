from __future__ import annotations

import json
import re

from sleuth.agents._helpers import validate_model
from sleuth.agents.prompts import build_clue_discovery_prompt
from sleuth.llm.base import LLMClient
from sleuth.schemas import ClueDiscoveryOutput, DocumentPage
from sleuth.utils.json_utils import extract_json_from_text


def _value(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _normalize_clue_data(data: dict, page_index: int) -> dict:
    normalized = {
        "page_index": page_index,
        "has_relevant_evidence": _value(data, "has_relevant_evidence", "has relevant evidence"),
        "evidence_items": _value(data, "evidence_items", "evidence items") or [],
        "page_summary": _value(data, "page_summary", "page summary") or "",
        "key_insights": _value(data, "key_insights", "key insights") or "",
    }
    items = normalized["evidence_items"]
    if not isinstance(items, list):
        items = []
    normalized_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized_items.append(
            {
                "page_index": page_index,
                "evidence_type": _value(item, "evidence_type", "evidence type") or "",
                "content": _value(item, "content") or "",
                "location": _value(item, "location") or "",
                "relevance": _value(item, "relevance") or "",
                "confidence": _value(item, "confidence") or "",
            }
        )
    normalized["evidence_items"] = normalized_items
    if normalized["has_relevant_evidence"] is None:
        normalized["has_relevant_evidence"] = bool(normalized_items)
    return normalized


def _decode_jsonish_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace('\\"', '"').replace("\\n", "\n").strip()


def _extract_string_field(text: str, *keys: str) -> str | None:
    for key in keys:
        pattern = rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"'
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return _decode_jsonish_string(match.group(1)).strip()
    return None


def _extract_bool_field(text: str, *keys: str) -> bool | None:
    for key in keys:
        pattern = rf'"{re.escape(key)}"\s*:\s*(true|false)'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
    return None


def _salvage_evidence_items(raw_output: str, page_index: int) -> list[dict]:
    item_blocks = re.findall(
        r'\{[^{}]*"content"\s*:\s*"(?:\\.|[^"\\])*"[^{}]*\}',
        raw_output,
        flags=re.DOTALL,
    )
    items: list[dict] = []
    for block in item_blocks:
        content = _extract_string_field(block, "content")
        if not content:
            continue
        items.append(
            {
                "page_index": page_index,
                "evidence_type": _extract_string_field(block, "evidence_type", "evidence type") or "raw",
                "content": content,
                "location": _extract_string_field(block, "location") or "Salvaged from unparsed clue output.",
                "relevance": _extract_string_field(block, "relevance")
                or "Recovered from invalid or truncated Clue Discovery JSON.",
                "confidence": _extract_string_field(block, "confidence") or "low",
            }
        )

    if items:
        return items

    for match in re.finditer(r'"content"\s*:\s*"((?:\\.|[^"\\])*)"', raw_output, flags=re.DOTALL):
        content = _decode_jsonish_string(match.group(1)).strip()
        if content:
            items.append(
                {
                    "page_index": page_index,
                    "evidence_type": "raw",
                    "content": content,
                    "location": "Salvaged from unparsed clue output.",
                    "relevance": "Recovered from invalid or truncated Clue Discovery JSON.",
                    "confidence": "low",
                }
            )
    return items


def _salvage_clue_data(raw_output: str | None, page_index: int) -> dict | None:
    if not raw_output:
        return None

    items = _salvage_evidence_items(raw_output, page_index)
    page_summary = _extract_string_field(raw_output, "page_summary", "page summary") or ""
    key_insights = _extract_string_field(raw_output, "key_insights", "key insights") or ""
    explicit_relevance = _extract_bool_field(raw_output, "has_relevant_evidence", "has relevant evidence")
    if not items and not page_summary and not key_insights and explicit_relevance is None:
        return None

    if items and not page_summary:
        page_summary = "Salvaged partial clue discovery output because the JSON was invalid or truncated."

    return {
        "page_index": page_index,
        "has_relevant_evidence": bool(items) or bool(explicit_relevance),
        "evidence_items": items,
        "page_summary": page_summary,
        "key_insights": key_insights,
    }


class ClueDiscoveryAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        agent_prompt_text: str,
        sol_instruction_text: str | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = 3072,
    ) -> None:
        self.llm_client = llm_client
        self.agent_prompt_text = agent_prompt_text
        self.sol_instruction_text = sol_instruction_text
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def _fallback(self, page_index: int, raw_output: str | None, prompt_used: str) -> ClueDiscoveryOutput:
        salvaged = _salvage_clue_data(raw_output, page_index)
        if salvaged is not None:
            salvaged["raw_output"] = raw_output
            salvaged["prompt_used"] = prompt_used
            return validate_model(ClueDiscoveryOutput, salvaged)
        return ClueDiscoveryOutput(
            page_index=page_index,
            has_relevant_evidence=False,
            evidence_items=[],
            page_summary="Failed to parse clue discovery output.",
            key_insights="",
            raw_output=raw_output,
            prompt_used=prompt_used,
        )

    def run(self, question: str, page: DocumentPage) -> ClueDiscoveryOutput:
        prompt = build_clue_discovery_prompt(
            question=question,
            page_index=page.page_index,
            page_text=page.text,
            agent_prompt_text=self.agent_prompt_text,
            sol_instruction_text=self.sol_instruction_text,
        )
        try:
            raw_output = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                images=[page.image_path],
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            data = extract_json_from_text(raw_output)
            if data is None:
                return self._fallback(page.page_index, raw_output, prompt)

            data = _normalize_clue_data(data, page.page_index)
            data["raw_output"] = raw_output
            data["prompt_used"] = prompt
            return validate_model(ClueDiscoveryOutput, data)
        except Exception as exc:
            return self._fallback(page.page_index, f"Agent failure: {exc}", prompt)
