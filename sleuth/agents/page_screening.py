from __future__ import annotations

from sleuth.agents._helpers import validate_model
from sleuth.agents.prompts import build_page_screening_prompt
from sleuth.llm.base import LLMClient
from sleuth.schemas import DocumentPage, PageScreeningOutput
from sleuth.utils.json_utils import extract_json_from_text


def _parse_screening_text(text: str) -> dict | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.lower() == "none":
        return {
            "has_visual_element": False,
            "relevance": "None",
            "reasoning": "The page was marked as none.",
            "keep_page": False,
        }

    fields: dict[str, str] = {}
    for line in stripped.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()

    has_chart = fields.get("has chart")
    relevance = fields.get("relevance")
    reasoning = fields.get("reasoning", "")
    if has_chart is None and relevance is None:
        return None

    has_visual = str(has_chart).strip().lower() in {"yes", "true", "y"}
    if not has_visual and not relevance:
        relevance = "None"
    relevance = _normalize_relevance(relevance or "None")
    return {
        "has_visual_element": has_visual,
        "relevance": relevance,
        "reasoning": reasoning,
        "keep_page": relevance in {"Completely Relevant", "Relevant"},
    }


def _normalize_relevance(relevance: str) -> str:
    value = relevance.strip()
    allowed = {"Completely Relevant", "Relevant", "Irrelevant", "None"}
    if value in allowed:
        return value
    lower = value.lower()
    if lower == "none":
        return "None"
    if "completely" in lower and "relevant" in lower:
        return "Completely Relevant"
    if lower == "relevant" or lower.startswith("relevant"):
        return "Relevant"
    if "irrelevant" in lower:
        return "Irrelevant"
    return "Irrelevant"


class PageScreeningAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        agent_prompt_text: str,
        sol_instruction_text: str | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = 512,
    ) -> None:
        self.llm_client = llm_client
        self.agent_prompt_text = agent_prompt_text
        self.sol_instruction_text = sol_instruction_text
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def _fallback(self, page_index: int, raw_output: str | None, prompt_used: str) -> PageScreeningOutput:
        return PageScreeningOutput(
            page_index=page_index,
            has_visual_element=False,
            relevance="Irrelevant",
            reasoning="Failed to parse page screening output.",
            keep_page=False,
            raw_output=raw_output,
            prompt_used=prompt_used,
        )

    def run(self, question: str, page: DocumentPage) -> PageScreeningOutput:
        prompt = build_page_screening_prompt(
            question=question,
            page_index=page.page_index,
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
                data = _parse_screening_text(raw_output)
            if data is None:
                return self._fallback(page.page_index, raw_output, prompt)

            relevance = _normalize_relevance(str(data.get("relevance", "Irrelevant")))
            data["page_index"] = page.page_index
            data["relevance"] = relevance
            data.setdefault("has_visual_element", relevance != "None")
            data.setdefault("reasoning", "")
            data["keep_page"] = bool(data.get("keep_page", relevance in {"Completely Relevant", "Relevant"}))
            if relevance not in {"Completely Relevant", "Relevant"}:
                data["keep_page"] = False
            data["raw_output"] = raw_output
            data["prompt_used"] = prompt
            return validate_model(PageScreeningOutput, data)
        except Exception as exc:
            return self._fallback(page.page_index, f"Agent failure: {exc}", prompt)
