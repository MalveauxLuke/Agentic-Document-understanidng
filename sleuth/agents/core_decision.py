from __future__ import annotations

from sleuth.agents._helpers import validate_model
from sleuth.agents.prompts import (
    build_core_decision_text_prompt,
    build_core_decision_visual_prompt,
    display_page_number,
)
from sleuth.evaluation.answer_extraction import strip_thinking_blocks
from sleuth.llm.base import LLMClient
from sleuth.schemas import DifficultyOutput, EvidenceContext, FinalAnswer
from sleuth.utils.json_utils import extract_json_from_text


def _client_model_name(client: LLMClient) -> str:
    return str(
        getattr(client, "model_name_or_path", None)
        or getattr(client, "model", None)
        or client.__class__.__name__
    )


class CoreDecisionAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        text_agent_prompt_text: str,
        visual_agent_prompt_text: str,
        sol_instruction_text: str | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = 512,
        thinking_llm_client: LLMClient | None = None,
        difficulty_model_switching_enabled: bool = False,
        thinking_max_new_tokens: int | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.thinking_llm_client = thinking_llm_client
        self.text_agent_prompt_text = text_agent_prompt_text
        self.visual_agent_prompt_text = visual_agent_prompt_text
        self.sol_instruction_text = sol_instruction_text
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.difficulty_model_switching_enabled = difficulty_model_switching_enabled
        self.thinking_max_new_tokens = thinking_max_new_tokens or max_new_tokens

    def _select_client(self, difficulty_output: DifficultyOutput) -> tuple[LLMClient, str, bool, int | None]:
        if self.difficulty_model_switching_enabled and difficulty_output.difficulty_level == 1:
            if self.thinking_llm_client is None:
                raise RuntimeError("Difficulty model switching is enabled, but no Thinking Core Decision client exists.")
            return self.thinking_llm_client, "thinking", True, self.thinking_max_new_tokens
        return self.llm_client, "instruct", False, self.max_new_tokens

    def _fallback(
        self,
        raw_output: str | None,
        prompt_used: str,
        model_name: str,
        mode: str,
        switching_used: bool,
    ) -> FinalAnswer:
        return FinalAnswer(
            answer="No answers found!",
            evidence_references=[],
            raw_output=raw_output,
            prompt_used=prompt_used,
            core_decision_model=model_name,
            core_decision_mode=mode,
            difficulty_model_switching_used=switching_used,
        )

    def _answer_from_text(
        self,
        raw_output: str | None,
        prompt_used: str,
        model_name: str,
        mode: str,
        switching_used: bool,
    ) -> FinalAnswer:
        answer = strip_thinking_blocks(raw_output or "").strip()
        if not answer:
            answer = "No answers found!"
        return FinalAnswer(
            answer=answer,
            evidence_references=[],
            raw_output=raw_output,
            prompt_used=prompt_used,
            core_decision_model=model_name,
            core_decision_mode=mode,
            difficulty_model_switching_used=switching_used,
        )

    def _build_prompt(self, question: str, evidence_context: EvidenceContext, difficulty_output: DifficultyOutput) -> str:
        if evidence_context.retained_image_paths:
            visual_rows = evidence_context.retained_visual_evidence or [
                {
                    "page_index": page_index,
                    "display_page_number": display_page_number(page_index),
                    "crop_region": "full_page",
                    "crop_location": "full original page",
                    "is_crop": False,
                }
                for page_index in evidence_context.retained_page_indices
            ]
            page_lines = "\n".join(
                "- Page Number {page} | {kind} | crop_region={region} | location={location}".format(
                    page=row.get("display_page_number") or display_page_number(int(row.get("page_index", 0))),
                    kind="crop image" if row.get("is_crop") else "full page image",
                    region=row.get("crop_region", "full_page"),
                    location=row.get("crop_location", ""),
                )
                for row in visual_rows
            )
            visual_evidence_section = (
                "The following verified images are provided as visual evidence:\n"
                f"{page_lines}\n\n"
                "The actual images will be passed to the multimodal model separately in this order."
            )
            return build_core_decision_visual_prompt(
                question=question,
                instruction_set=difficulty_output.instruction_set,
                evidence_summary=evidence_context.evidence_summary,
                visual_evidence_section=visual_evidence_section,
                num_pages=len(evidence_context.retained_image_paths),
                agent_prompt_text=self.visual_agent_prompt_text,
                sol_instruction_text=self.sol_instruction_text,
            )
        return build_core_decision_text_prompt(
            question=question,
            instruction_set=difficulty_output.instruction_set,
            evidence_summary=evidence_context.evidence_summary,
            num_pages=len(evidence_context.clue_outputs),
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
        llm_client, mode, switching_used, max_new_tokens = self._select_client(difficulty_output)
        model_name = _client_model_name(llm_client)
        try:
            raw_output = llm_client.chat(
                [{"role": "user", "content": prompt}],
                images=images,
                temperature=self.temperature,
                max_new_tokens=max_new_tokens,
            )
            data = extract_json_from_text(raw_output)
            if data is None:
                return self._answer_from_text(raw_output, prompt, model_name, mode, switching_used)

            data.setdefault("answer", "No answers found!")
            data["answer"] = strip_thinking_blocks(str(data["answer"])).strip() or "No answers found!"
            references = data.get("evidence_references", data.get("evidence references")) or []
            data["evidence_references"] = references if isinstance(references, list) else []
            data["raw_output"] = raw_output
            data["prompt_used"] = prompt
            data["core_decision_model"] = model_name
            data["core_decision_mode"] = mode
            data["difficulty_model_switching_used"] = switching_used
            return validate_model(FinalAnswer, data)
        except Exception as exc:
            return self._fallback(f"Agent failure: {exc}", prompt, model_name, mode, switching_used)
