from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from sleuth.agents.evidence_verification import EvidenceVerificationAgent
from sleuth.llm.base import LLMClient
from sleuth.schemas import ClueDiscoveryOutput, DocumentPage, EvidenceItem


CROP_PROMPT = """
Return valid JSON only:
{
  "verification_stage": "crop",
  "verification_status": "faithful/corrected/rejected/uncertain",
  "needs_full_page": true/false,
  "faithful_evidence": {"content": ""}
}
"""

FULL_PROMPT = """
Return valid JSON only:
{
  "verification_stage": "full_page",
  "verification_status": "faithful/corrected/rejected/uncertain",
  "needs_full_page": false,
  "faithful_evidence": {"content": ""}
}
"""


class QueueClient(LLMClient):
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def chat(self, messages, images=None, temperature=0.1, max_new_tokens=None):
        self.calls.append({"messages": messages, "images": images, "max_new_tokens": max_new_tokens})
        return json.dumps(self.responses.pop(0))


def _page(tmp_path: Path) -> DocumentPage:
    image_path = tmp_path / "page_0001.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    return DocumentPage(page_index=0, image_path=str(image_path))


def _clue(crop_region: str) -> ClueDiscoveryOutput:
    return ClueDiscoveryOutput(
        page_index=0,
        has_relevant_evidence=True,
        evidence_items=[
            EvidenceItem(
                page_index=0,
                evidence_type="figure",
                content="Proposed evidence.",
                location="right side",
                relevance="direct",
                confidence="medium",
                crop_region=crop_region,
            )
        ],
        page_summary="summary",
        key_insights="insight",
    )


def test_generated_crop_needs_full_page_triggers_fallback(tmp_path: Path):
    client = QueueClient(
        [
            {
                "verification_status": "faithful",
                "needs_full_page": True,
                "crop_problem": "legend is cut off",
                "visible_evidence": "",
                "comparison": "",
                "faithful_evidence": {
                    "evidence type": "figure",
                    "content": "partial",
                    "location": "crop",
                    "relevance": "partial",
                    "confidence": "low",
                },
            },
            {
                "verification_status": "corrected",
                "needs_full_page": False,
                "visible_evidence": "Correct full-page evidence.",
                "comparison": "Corrected.",
                "faithful_evidence": {
                    "evidence type": "figure",
                    "content": "Correct full-page evidence.",
                    "location": "full page",
                    "relevance": "direct",
                    "confidence": "high",
                },
            },
        ]
    )
    agent = EvidenceVerificationAgent(client, CROP_PROMPT, FULL_PROMPT)

    outputs = agent.run("Question?", _page(tmp_path), _clue("right_half"))

    assert len(outputs) == 1
    assert len(client.calls) == 2
    assert outputs[0].verification_stage == "full_page"
    assert outputs[0].used_full_page_fallback is True
    assert outputs[0].verification_status == "corrected"
    assert outputs[0].faithful_evidence.content == "Correct full-page evidence."


def test_original_page_input_does_not_trigger_fallback_even_if_requested(tmp_path: Path):
    client = QueueClient(
        [
            {
                "verification_status": "uncertain",
                "needs_full_page": True,
                "crop_problem": "uncertain",
                "visible_evidence": "",
                "comparison": "",
                "faithful_evidence": {
                    "evidence type": "text",
                    "content": "",
                    "location": "page",
                    "relevance": "partial",
                    "confidence": "low",
                },
            }
        ]
    )
    page = _page(tmp_path)
    agent = EvidenceVerificationAgent(client, CROP_PROMPT, FULL_PROMPT)

    outputs = agent.run("Question?", page, _clue("not_applicable"))

    assert len(outputs) == 1
    assert len(client.calls) == 1
    assert outputs[0].verification_stage == "crop"
    assert outputs[0].input_image_path == page.image_path
    assert outputs[0].used_full_page_fallback is False
