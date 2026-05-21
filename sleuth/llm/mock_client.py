from __future__ import annotations

import json
import re

from sleuth.llm.base import LLMClient


def _message_text(messages: list[dict]) -> str:
    chunks: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
    return "\n".join(chunks)


def _page_index_from_prompt(prompt: str) -> int:
    for pattern in (r"Page Number:\s*(\d+)", r"Page index\s*(\d+)", r"page_index[\"']?\s*[:=]\s*(\d+)"):
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


class MockClient(LLMClient):
    def chat(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = None,
    ) -> str:
        prompt = _message_text(messages)
        page_index = _page_index_from_prompt(prompt)

        if '"has_visual_element"' in prompt and '"keep_page"' in prompt:
            return json.dumps(
                {
                    "has_visual_element": False,
                    "relevance": "None",
                    "reasoning": "Mock screening found no visual element.",
                    "keep_page": False,
                }
            )

        if '"difficulty_level"' in prompt and '"instruction_set"' in prompt:
            return json.dumps(
                {
                    "difficulty_level": 0,
                    "instruction_set": "Use ordinary direct extraction from the provided evidence.",
                }
            )

        if '"answer"' in prompt and '"evidence_references"' in prompt:
            return json.dumps(
                {
                    "answer": "Mock answer based on extracted evidence.",
                    "evidence_references": [
                        {
                            "page_index": 0,
                            "evidence": "Mock evidence reference.",
                        }
                    ],
                }
            )

        if '"has_relevant_evidence"' in prompt and '"evidence_items"' in prompt:
            return json.dumps(
                {
                    "page_number": page_index,
                    "has_relevant_evidence": True,
                    "evidence_items": [
                        {
                            "evidence_type": "text",
                            "content": "Mock evidence extracted from the page.",
                            "location": "mock page text",
                            "relevance": "Potentially relevant to the question.",
                            "confidence": "low",
                        }
                    ],
                    "page_summary": "Mock page summary.",
                    "key_insights": "Mock key insight.",
                }
            )

        return json.dumps({"answer": "No answers found!", "evidence_references": []})
