from __future__ import annotations

from pathlib import Path

from PIL import Image

from sleuth.agents.clue_discovery import ClueDiscoveryAgent
from sleuth.llm.base import LLMClient
from sleuth.schemas import DocumentPage


class RegionFallbackClient(LLMClient):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = None,
    ) -> str:
        assert messages[0]["content"]
        image_path = images[0] if images else ""
        self.calls.append(image_path)
        if "_region_crops" not in image_path:
            return (
                '{"page number": 1, "has relevant evidence": false, '
                '"evidence items": [], "page summary": "No whole-page evidence.", '
                '"key insights": ""}'
            )
        if "middle_third" in image_path:
            return (
                '{"page number": 1, "has relevant evidence": true, "evidence items": ['
                '{"evidence type": "chart", "content": "Middle crop value is 42.", '
                '"location": "bar label in crop", "relevance": "A partial clue.", '
                '"confidence": "high"}], "page summary": "Middle crop has the clue.", '
                '"key insights": "Use the middle crop value."}'
            )
        return (
            '{"page number": 1, "has relevant evidence": false, '
            '"evidence items": [], "page summary": "No crop evidence.", '
            '"key insights": ""}'
        )


def test_whole_page_no_evidence_triggers_region_fallback(tmp_path: Path):
    image_path = tmp_path / "page_0001.png"
    Image.new("RGB", (90, 90), "white").save(image_path)
    client = RegionFallbackClient()
    prompt = "Question: {question}\nPage Number: {page_num}\nReturn JSON."
    agent = ClueDiscoveryAgent(client, prompt, region_refinement="fallback")

    output = agent.run("What value is shown?", DocumentPage(page_index=0, image_path=str(image_path)))

    assert len(client.calls) == 4
    assert client.calls[0] == str(image_path)
    assert any("_region_crops" in call and "middle_third" in call for call in client.calls)
    assert output.has_relevant_evidence is True
    assert output.page_index == 0
    assert output.evidence_items[0].page_index == 0
    assert output.evidence_items[0].content == "Middle crop value is 42."
    assert "region crop middle_third" in output.evidence_items[0].location
    assert "Page Number: 1" in (output.prompt_used or "")
