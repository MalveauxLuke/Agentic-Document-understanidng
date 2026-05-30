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

        if "Has Chart:" in prompt and "Relevance:" in prompt and "Reasoning:" in prompt:
            return "Has Chart: No\nRelevance: None\nReasoning: Mock screening found no visual element."

        if '"verification_status"' in prompt and '"faithful_evidence"' in prompt:
            stage = "full_page" if '"verification_stage": "full_page"' in prompt else "crop"
            return json.dumps(
                {
                    "page_number": page_index,
                    "verification_stage": stage,
                    "crop_hint": "not_applicable",
                    "verification_status": "faithful",
                    "needs_full_page": False,
                    "crop_problem": None,
                    "visible_evidence": "Mock visible evidence from the image.",
                    "comparison": "The proposed evidence is supported by the mock image.",
                    "faithful_evidence": {
                        "evidence type": "text",
                        "content": "Mock verified evidence extracted from the page.",
                        "location": "mock verified location",
                        "relevance": "direct",
                        "confidence": "low",
                    },
                    "notes": [],
                    "uncertainties": [],
                }
            )

        if (
            '"difficulty_level"' in prompt
            and '"instruction_set"' in prompt
            or '"difficulty level"' in prompt
            and '"instruction set"' in prompt
        ):
            return json.dumps(
                {
                    "difficulty level": 0,
                    "instruction set": "Use ordinary direct extraction from the provided evidence.",
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

        if (
            '"has_relevant_evidence"' in prompt
            and '"evidence_items"' in prompt
            or '"has relevant evidence"' in prompt
            and '"evidence items"' in prompt
        ):
            return json.dumps(
                {
                    "page number": page_index,
                    "has relevant evidence": True,
                    "evidence items": [
                        {
                            "evidence type": "text",
                            "content": "Mock evidence extracted from the page.",
                            "location": "mock page text",
                            "crop_region": "not_applicable",
                            "relevance": "Potentially relevant to the question.",
                            "confidence": "low",
                        }
                    ],
                    "page summary": "Mock page summary.",
                    "key insights": "Mock key insight.",
                }
            )

        if "YOUR ANSWER:" in prompt and "QUERY:" in prompt:
            return "Mock answer based on extracted evidence."

        return json.dumps({"answer": "No answers found!", "evidence_references": []})
