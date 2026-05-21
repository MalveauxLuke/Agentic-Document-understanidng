from __future__ import annotations

from sleuth.agents._helpers import validate_model
from sleuth.agents.prompts import build_clue_discovery_prompt
from sleuth.llm.base import LLMClient
from sleuth.schemas import ClueDiscoveryOutput, DocumentPage
from sleuth.utils.json_utils import extract_json_from_text


class ClueDiscoveryAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        agent_prompt_text: str,
        sol_instruction_text: str | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = 1024,
    ) -> None:
        self.llm_client = llm_client
        self.agent_prompt_text = agent_prompt_text
        self.sol_instruction_text = sol_instruction_text
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def _fallback(self, page_index: int, raw_output: str | None, prompt_used: str) -> ClueDiscoveryOutput:
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

            data["page_index"] = page.page_index
            items = data.get("evidence_items") or []
            if not isinstance(items, list):
                items = []
            normalized_items = []
            for item in items:
                if isinstance(item, dict):
                    item = dict(item)
                    item.setdefault("page_index", page.page_index)
                    normalized_items.append(item)
            data["evidence_items"] = normalized_items
            data.setdefault("has_relevant_evidence", bool(normalized_items))
            data.setdefault("page_summary", "")
            data.setdefault("key_insights", "")
            data["raw_output"] = raw_output
            data["prompt_used"] = prompt
            return validate_model(ClueDiscoveryOutput, data)
        except Exception as exc:
            return self._fallback(page.page_index, f"Agent failure: {exc}", prompt)
