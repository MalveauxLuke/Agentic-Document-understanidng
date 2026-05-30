from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from sleuth.agents._helpers import validate_model
from sleuth.agents.prompts import build_clue_discovery_prompt
from sleuth.llm.base import LLMClient
from sleuth.schemas import ClueDiscoveryOutput, DocumentPage
from sleuth.utils.json_utils import extract_json_from_text


REGION_REFINEMENT_DISABLED = {"", "none", "off", "disabled", "false", "0"}


def _model_dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    if isinstance(obj, list):
        return [_model_dump(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _model_dump(value) for key, value in obj.items()}
    return obj


def _value(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _normalize_clue_data(data: dict, page_index: int) -> dict:
    normalized = {
        "page_index": page_index,
        "has_relevant_evidence": _value(data, "has_relevant_evidence", "has relevant evidence"),
        "evidence_items": _value(data, "evidence_items", "evidence items") or [],
        "page_summary": _value(data, "page_summary", "page summary") or "",
        "key_insights": _value(data, "key_insights", "key insights") or "",
    }
    items = normalized["evidence_items"]
    if not isinstance(items, list):
        items = []
    normalized_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized_items.append(
            {
                "page_index": page_index,
                "evidence_type": _value(item, "evidence_type", "evidence type") or "",
                "content": _value(item, "content") or "",
                "location": _value(item, "location") or "",
                "relevance": _value(item, "relevance") or "",
                "confidence": _value(item, "confidence") or "",
            }
        )
    normalized["evidence_items"] = normalized_items
    if normalized["has_relevant_evidence"] is None:
        normalized["has_relevant_evidence"] = bool(normalized_items)
    return normalized


def _decode_jsonish_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace('\\"', '"').replace("\\n", "\n").strip()


def _extract_string_field(text: str, *keys: str) -> str | None:
    for key in keys:
        pattern = rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"'
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return _decode_jsonish_string(match.group(1)).strip()
    return None


def _extract_bool_field(text: str, *keys: str) -> bool | None:
    for key in keys:
        pattern = rf'"{re.escape(key)}"\s*:\s*(true|false)'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
    return None


def _salvage_evidence_items(raw_output: str, page_index: int) -> list[dict]:
    item_blocks = re.findall(
        r'\{[^{}]*"content"\s*:\s*"(?:\\.|[^"\\])*"[^{}]*\}',
        raw_output,
        flags=re.DOTALL,
    )
    items: list[dict] = []
    for block in item_blocks:
        content = _extract_string_field(block, "content")
        if not content:
            continue
        items.append(
            {
                "page_index": page_index,
                "evidence_type": _extract_string_field(block, "evidence_type", "evidence type") or "raw",
                "content": content,
                "location": _extract_string_field(block, "location") or "Salvaged from unparsed clue output.",
                "relevance": _extract_string_field(block, "relevance")
                or "Recovered from invalid or truncated Clue Discovery JSON.",
                "confidence": _extract_string_field(block, "confidence") or "low",
            }
        )

    if items:
        return items

    for match in re.finditer(r'"content"\s*:\s*"((?:\\.|[^"\\])*)"', raw_output, flags=re.DOTALL):
        content = _decode_jsonish_string(match.group(1)).strip()
        if content:
            items.append(
                {
                    "page_index": page_index,
                    "evidence_type": "raw",
                    "content": content,
                    "location": "Salvaged from unparsed clue output.",
                    "relevance": "Recovered from invalid or truncated Clue Discovery JSON.",
                    "confidence": "low",
                }
            )
    return items


def _salvage_clue_data(raw_output: str | None, page_index: int) -> dict | None:
    if not raw_output:
        return None

    items = _salvage_evidence_items(raw_output, page_index)
    page_summary = _extract_string_field(raw_output, "page_summary", "page summary") or ""
    key_insights = _extract_string_field(raw_output, "key_insights", "key insights") or ""
    explicit_relevance = _extract_bool_field(raw_output, "has_relevant_evidence", "has relevant evidence")
    if not items and not page_summary and not key_insights and explicit_relevance is None:
        return None

    if items and not page_summary:
        page_summary = "Salvaged partial clue discovery output because the JSON was invalid or truncated."

    return {
        "page_index": page_index,
        "has_relevant_evidence": bool(items) or bool(explicit_relevance),
        "evidence_items": items,
        "page_summary": page_summary,
        "key_insights": key_insights,
    }


def _region_refinement_enabled(mode: str | None) -> bool:
    return (mode or "fallback").strip().lower() not in REGION_REFINEMENT_DISABLED


def _region_crop_specs(width: int, height: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    first_break = max(1, height // 3)
    second_break = max(first_break + 1, (height * 2) // 3)
    return [
        ("top_third", (0, 0, width, first_break)),
        ("middle_third", (0, first_break, width, second_break)),
        ("bottom_third", (0, second_break, width, height)),
    ]


def _region_crop_path(image_path: Path, region_name: str) -> Path:
    return image_path.parent / "_region_crops" / f"{image_path.stem}_{region_name}{image_path.suffix}"


def _build_region_crops(image_path: str) -> list[tuple[str, tuple[int, int, int, int], str]]:
    source_path = Path(image_path)
    if not source_path.exists():
        return []

    crops: list[tuple[str, tuple[int, int, int, int], str]] = []
    with Image.open(source_path) as image:
        width, height = image.size
        if width <= 1 or height <= 1:
            return []
        crop_dir = source_path.parent / "_region_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        for region_name, bbox in _region_crop_specs(width, height):
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            crop_path = _region_crop_path(source_path, region_name)
            if not crop_path.exists():
                image.crop(bbox).save(crop_path)
            crops.append((region_name, bbox, str(crop_path)))
    return crops


class ClueDiscoveryAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        agent_prompt_text: str,
        sol_instruction_text: str | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = 3072,
        region_refinement: str = "fallback",
    ) -> None:
        self.llm_client = llm_client
        self.agent_prompt_text = agent_prompt_text
        self.sol_instruction_text = sol_instruction_text
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.region_refinement = region_refinement

    def _fallback(self, page_index: int, raw_output: str | None, prompt_used: str) -> ClueDiscoveryOutput:
        salvaged = _salvage_clue_data(raw_output, page_index)
        if salvaged is not None:
            salvaged["raw_output"] = raw_output
            salvaged["prompt_used"] = prompt_used
            return validate_model(ClueDiscoveryOutput, salvaged)
        return ClueDiscoveryOutput(
            page_index=page_index,
            has_relevant_evidence=False,
            evidence_items=[],
            page_summary="Failed to parse clue discovery output.",
            key_insights="",
            raw_output=raw_output,
            prompt_used=prompt_used,
        )

    def _run_image(self, image_path: str, page_index: int, prompt: str) -> ClueDiscoveryOutput:
        try:
            raw_output = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                images=[image_path],
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            data = extract_json_from_text(raw_output)
            if data is None:
                return self._fallback(page_index, raw_output, prompt)

            data = _normalize_clue_data(data, page_index)
            data["raw_output"] = raw_output
            data["prompt_used"] = prompt
            return validate_model(ClueDiscoveryOutput, data)
        except Exception as exc:
            return self._fallback(page_index, f"Agent failure: {exc}", prompt)

    def _merge_region_outputs(
        self,
        whole_page_output: ClueDiscoveryOutput,
        region_outputs: list[tuple[str, tuple[int, int, int, int], ClueDiscoveryOutput]],
    ) -> ClueDiscoveryOutput:
        merged_items: list[dict] = []
        summary_parts: list[str] = []
        insight_parts: list[str] = []
        raw_parts = [f"WHOLE_PAGE_OUTPUT:\n{whole_page_output.raw_output or ''}".strip()]
        regions_with_items = 0

        if whole_page_output.page_summary.strip():
            summary_parts.append(whole_page_output.page_summary.strip())
        if whole_page_output.key_insights.strip():
            insight_parts.append(whole_page_output.key_insights.strip())

        for region_name, bbox, region_output in region_outputs:
            raw_parts.append(
                f"REGION_CROP {region_name} bbox={bbox}:\n{region_output.raw_output or ''}".strip()
            )
            if region_output.page_summary.strip():
                summary_parts.append(f"{region_name}: {region_output.page_summary.strip()}")
            if region_output.key_insights.strip():
                insight_parts.append(f"{region_name}: {region_output.key_insights.strip()}")
            if region_output.evidence_items:
                regions_with_items += 1
            for item in region_output.evidence_items:
                item_data = _model_dump(item)
                location = item_data.get("location", "").strip()
                item_data["page_index"] = whole_page_output.page_index
                item_data["location"] = (
                    f"region crop {region_name} bbox={bbox} on original page"
                    + (f"; {location}" if location else "")
                )
                merged_items.append(item_data)

        if not merged_items:
            return whole_page_output

        summary_prefix = f"Region fallback found evidence in {regions_with_items} deterministic crop(s)."
        merged_data = {
            "page_index": whole_page_output.page_index,
            "has_relevant_evidence": True,
            "evidence_items": merged_items,
            "page_summary": " ".join([summary_prefix, *summary_parts]).strip(),
            "key_insights": " ".join(insight_parts).strip(),
            "raw_output": "\n\n".join(raw_parts),
            "prompt_used": whole_page_output.prompt_used,
        }
        return validate_model(ClueDiscoveryOutput, merged_data)

    def _run_region_refinement(
        self,
        page: DocumentPage,
        prompt: str,
        whole_page_output: ClueDiscoveryOutput,
    ) -> ClueDiscoveryOutput:
        if not _region_refinement_enabled(self.region_refinement):
            return whole_page_output
        if whole_page_output.has_relevant_evidence or whole_page_output.evidence_items:
            return whole_page_output

        try:
            crop_specs = _build_region_crops(page.image_path)
        except Exception:
            return whole_page_output

        region_outputs: list[tuple[str, tuple[int, int, int, int], ClueDiscoveryOutput]] = []
        for region_name, bbox, crop_path in crop_specs:
            crop_output = self._run_image(crop_path, page.page_index, prompt)
            if crop_output.has_relevant_evidence or crop_output.evidence_items:
                region_outputs.append((region_name, bbox, crop_output))
        return self._merge_region_outputs(whole_page_output, region_outputs)

    def run(self, question: str, page: DocumentPage) -> ClueDiscoveryOutput:
        prompt = build_clue_discovery_prompt(
            question=question,
            page_index=page.page_index,
            page_text=page.text,
            agent_prompt_text=self.agent_prompt_text,
            sol_instruction_text=self.sol_instruction_text,
        )
        whole_page_output = self._run_image(page.image_path, page.page_index, prompt)
        return self._run_region_refinement(page, prompt, whole_page_output)
