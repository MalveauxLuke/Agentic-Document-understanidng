from __future__ import annotations

from sleuth.agents._helpers import validate_model
from sleuth.agents.prompts import (
    build_core_decision_text_prompt,
    build_core_decision_visual_prompt,
    display_page_number,
)
from sleuth.llm.base import LLMClient
from sleuth.schemas import DifficultyOutput, EvidenceContext, FinalAnswer
from sleuth.utils.json_utils import extract_json_from_text


class CoreDecisionAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        text_agent_prompt_text: str,
        visual_agent_prompt_text: str,
        sol_instruction_text: str | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = 512,
    ) -> None:
        self.llm_client = llm_client
        self.text_agent_prompt_text = text_agent_prompt_text
        self.visual_agent_prompt_text = visual_agent_prompt_text
        self.sol_instruction_text = sol_instruction_text
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def _fallback(self, raw_output: str | None, prompt_used: str) -> FinalAnswer:
        return FinalAnswer(
            answer="No answers found!",
            evidence_references=[],
            raw_output=raw_output,
            prompt_used=prompt_used,
        )

    def _answer_from_text(self, raw_output: str | None, prompt_used: str) -> FinalAnswer:
        answer = (raw_output or "").strip()
        if not answer:
            answer = "No answers found!"
        return FinalAnswer(
            answer=answer,
            evidence_references=[],
            raw_output=raw_output,
            prompt_used=prompt_used,
        )

    def _build_prompt(self, question: str, evidence_context: EvidenceContext, difficulty_output: DifficultyOutput) -> str:
        if evidence_context.retained_image_paths:
            page_lines = "\n".join(
                f"- Page Number {display_page_number(page_index)}"
                for page_index in evidence_context.retained_page_indices
            )
            visual_evidence_section = (
                "The following page images are provided as visual evidence:\n"
                f"{page_lines}\n\n"
                "The actual images will be passed to the multimodal model separately."
            )
            return build_core_decision_visual_prompt(
                question=question,
                instruction_set=difficulty_output.instruction_set,
                evidence_summary=evidence_context.evidence_summary,
                visual_evidence_section=visual_evidence_section,
                num_pages=len(evidence_context.retrieved_pages),
                agent_prompt_text=self.visual_agent_prompt_text,
                sol_instruction_text=self.sol_instruction_text,
            )
        return build_core_decision_text_prompt(
            question=question,
            instruction_set=difficulty_output.instruction_set,
            evidence_summary=evidence_context.evidence_summary,
            num_pages=len(evidence_context.retrieved_pages),
            agent_prompt_text=self.text_agent_prompt_text,
            sol_instruction_text=self.sol_instruction_text,
        )

    def run(
        self,
        question: str,
        evidence_context: EvidenceContext,
        difficulty_output: DifficultyOutput,
    ) -> FinalAnswer:
        prompt = self._build_prompt(question, evidence_context, difficulty_output)
        images = evidence_context.retained_image_paths or None
        try:
            raw_output = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                images=images,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            data = extract_json_from_text(raw_output)
            if data is None:
                return self._answer_from_text(raw_output, prompt)

            data.setdefault("answer", "No answers found!")
            references = data.get("evidence_references", data.get("evidence references")) or []
            data["evidence_references"] = references if isinstance(references, list) else []
            data["raw_output"] = raw_output
            data["prompt_used"] = prompt
            return validate_model(FinalAnswer, data)
        except Exception as exc:
            return self._fallback(f"Agent failure: {exc}", prompt)
