from __future__ import annotations

from sleuth.agents._helpers import validate_model
from sleuth.agents.prompts import build_difficulty_prompt
from sleuth.llm.base import LLMClient
from sleuth.schemas import DifficultyOutput, EvidenceContext
from sleuth.utils.json_utils import extract_json_from_text


class DifficultyAssessmentAgent:
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

    def _fallback(self, raw_output: str | None, prompt_used: str) -> DifficultyOutput:
        return DifficultyOutput(
            difficulty_level=0,
            instruction_set="Use ordinary direct extraction from the provided evidence.",
            raw_output=raw_output,
            prompt_used=prompt_used,
        )

    def run(self, question: str, evidence_context: EvidenceContext) -> DifficultyOutput:
        prompt = build_difficulty_prompt(
            question=question,
            evidence_summary=evidence_context.evidence_summary,
            agent_prompt_text=self.agent_prompt_text,
            sol_instruction_text=self.sol_instruction_text,
        )
        try:
            raw_output = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                images=None,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            data = extract_json_from_text(raw_output)
            if data is None:
                return self._fallback(raw_output, prompt)

            if data.get("difficulty_level") not in (0, 1):
                data["difficulty_level"] = 0
            data.setdefault("instruction_set", "Use ordinary direct extraction from the provided evidence.")
            data["raw_output"] = raw_output
            data["prompt_used"] = prompt
            return validate_model(DifficultyOutput, data)
        except Exception as exc:
            return self._fallback(f"Agent failure: {exc}", prompt)
