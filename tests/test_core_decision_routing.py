from __future__ import annotations

import pytest

from sleuth.agents.core_decision import CoreDecisionAgent
from sleuth.llm.base import LLMClient
from sleuth.schemas import DifficultyOutput, EvidenceContext


TEXT_PROMPT = (
    "QUERY: {question}\n"
    "STRATEGIC INSTRUCTIONS Γd:\n{instruction_set}\n"
    "EVIDENCE (from {num_pages}pages):\n{evidence_summary}\n"
    "YOUR ANSWER:"
)
VISUAL_PROMPT = TEXT_PROMPT + "\n{visual_evidence_section}"


class RecordingClient(LLMClient):
    def __init__(self, model_name: str, response: str) -> None:
        self.model_name_or_path = model_name
        self.response = response
        self.calls: list[dict] = []

    def chat(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "images": images,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
        )
        return self.response


def _context(retained_images: list[str] | None = None) -> EvidenceContext:
    retained_images = retained_images or []
    return EvidenceContext(
        question="Which region wins?",
        retrieved_pages=[],
        clue_outputs=[],
        page_screening_outputs=[],
        retained_page_indices=[0] if retained_images else [],
        retained_image_paths=retained_images,
        evidence_summary="Page Number 1:\n- Europe has the most participants.",
    )


def _difficulty(level: int) -> DifficultyOutput:
    return DifficultyOutput(difficulty_level=level, instruction_set="Use the provided evidence.")


def test_difficulty_one_routes_core_decision_to_thinking_client():
    instruct = RecordingClient("Qwen/Qwen3-VL-8B-Instruct", "instruct answer")
    thinking = RecordingClient("Qwen/Qwen3-VL-8B-Thinking", "<think>private</think>\nEurope")
    agent = CoreDecisionAgent(
        instruct,
        TEXT_PROMPT,
        VISUAL_PROMPT,
        max_new_tokens=512,
        thinking_llm_client=thinking,
        difficulty_model_switching_enabled=True,
        thinking_max_new_tokens=4096,
    )

    answer = agent.run("Which region wins?", _context(["/tmp/page_0001.png"]), _difficulty(1))

    assert not instruct.calls
    assert len(thinking.calls) == 1
    assert thinking.calls[0]["max_new_tokens"] == 4096
    assert thinking.calls[0]["images"] == ["/tmp/page_0001.png"]
    assert answer.answer == "Europe"
    assert answer.raw_output == "<think>private</think>\nEurope"
    assert answer.core_decision_model == "Qwen/Qwen3-VL-8B-Thinking"
    assert answer.core_decision_mode == "thinking"
    assert answer.difficulty_model_switching_used is True


def test_difficulty_zero_stays_on_instruct_client():
    instruct = RecordingClient("Qwen/Qwen3-VL-8B-Instruct", '{"answer": "No lean"}')
    thinking = RecordingClient("Qwen/Qwen3-VL-8B-Thinking", "wrong path")
    agent = CoreDecisionAgent(
        instruct,
        TEXT_PROMPT,
        VISUAL_PROMPT,
        max_new_tokens=512,
        thinking_llm_client=thinking,
        difficulty_model_switching_enabled=True,
        thinking_max_new_tokens=4096,
    )

    answer = agent.run("Which group?", _context(), _difficulty(0))

    assert len(instruct.calls) == 1
    assert not thinking.calls
    assert instruct.calls[0]["max_new_tokens"] == 512
    assert answer.answer == "No lean"
    assert answer.core_decision_model == "Qwen/Qwen3-VL-8B-Instruct"
    assert answer.core_decision_mode == "instruct"
    assert answer.difficulty_model_switching_used is False


def test_switching_enabled_without_thinking_client_fails_loudly():
    instruct = RecordingClient("Qwen/Qwen3-VL-8B-Instruct", "unused")
    agent = CoreDecisionAgent(
        instruct,
        TEXT_PROMPT,
        VISUAL_PROMPT,
        difficulty_model_switching_enabled=True,
    )

    with pytest.raises(RuntimeError, match="no Thinking Core Decision client"):
        agent.run("Which group?", _context(), _difficulty(1))
