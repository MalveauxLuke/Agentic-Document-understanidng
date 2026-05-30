from __future__ import annotations

import json
from typing import Any

from sleuth.agents._helpers import validate_model
from sleuth.agents.prompts import (
    build_evidence_verification_crop_prompt,
    build_evidence_verification_full_page_prompt,
)
from sleuth.documents.evidence_crops import EvidenceCrop, build_evidence_crop
from sleuth.llm.base import LLMClient
from sleuth.schemas import (
    ClueDiscoveryOutput,
    DocumentPage,
    EvidenceItem,
    EvidenceVerificationOutput,
)
from sleuth.utils.json_utils import extract_json_from_text


ACCEPTED_VERIFICATION_STATUSES = {"faithful", "corrected"}
VERIFICATION_STATUSES = ACCEPTED_VERIFICATION_STATUSES | {"rejected", "uncertain"}


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


def _normalize_status(value: object) -> str:
    status = str(value or "").strip().lower()
    if status in VERIFICATION_STATUSES:
        return status
    return "uncertain"


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes", "on"}


def _normalize_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _normalize_faithful_evidence(
    data: dict,
    page_index: int,
    source_item: EvidenceItem,
    crop_region: str,
) -> dict | None:
    faithful = _value(data, "faithful_evidence", "faithful evidence")
    if not isinstance(faithful, dict):
        return None
    return {
        "page_index": page_index,
        "evidence_type": _value(faithful, "evidence_type", "evidence type") or source_item.evidence_type,
        "content": _value(faithful, "content") or "",
        "location": _value(faithful, "location") or source_item.location,
        "relevance": _value(faithful, "relevance") or source_item.relevance,
        "confidence": _value(faithful, "confidence") or "low",
        "crop_region": crop_region,
    }


def _normalize_verification_data(
    data: dict,
    *,
    page_index: int,
    source_evidence_item_index: int,
    source_item: EvidenceItem,
    crop: EvidenceCrop,
    raw_output: str | None,
    prompt_used: str,
    stage: str,
    used_full_page_fallback: bool = False,
) -> dict:
    status = _normalize_status(_value(data, "verification_status", "verification status"))
    needs_full_page = _normalize_bool(_value(data, "needs_full_page", "needs full page"))
    if stage == "full_page":
        needs_full_page = False
    faithful_evidence = _normalize_faithful_evidence(
        data,
        page_index=page_index,
        source_item=source_item,
        crop_region=crop.crop_region,
    )
    if status in ACCEPTED_VERIFICATION_STATUSES and faithful_evidence is None:
        status = "uncertain"

    return {
        "page_index": page_index,
        "source_evidence_item_index": source_evidence_item_index,
        "verification_stage": stage,
        "crop_hint": _value(data, "crop_hint", "crop hint") or crop.crop_region,
        "crop_location": crop.crop_location,
        "verification_status": status,
        "needs_full_page": needs_full_page,
        "crop_problem": _value(data, "crop_problem", "crop problem"),
        "visible_evidence": _value(data, "visible_evidence", "visible evidence") or "",
        "comparison": _value(data, "comparison") or "",
        "faithful_evidence": faithful_evidence,
        "notes": _normalize_list(_value(data, "notes", "faithfulness_notes", "faithfulness notes")),
        "uncertainties": _normalize_list(_value(data, "uncertainties")),
        "input_image_path": crop.image_path,
        "crop_bbox": list(crop.bbox) if crop.bbox is not None else None,
        "used_full_page_fallback": used_full_page_fallback,
        "raw_output": raw_output,
        "prompt_used": prompt_used,
    }


class EvidenceVerificationAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        crop_prompt_text: str,
        full_page_prompt_text: str,
        sol_instruction_text: str | None = None,
        temperature: float = 0.1,
        crop_max_new_tokens: int | None = 2048,
        full_page_max_new_tokens: int | None = 2048,
    ) -> None:
        self.llm_client = llm_client
        self.crop_prompt_text = crop_prompt_text
        self.full_page_prompt_text = full_page_prompt_text
        self.sol_instruction_text = sol_instruction_text
        self.temperature = temperature
        self.crop_max_new_tokens = crop_max_new_tokens
        self.full_page_max_new_tokens = full_page_max_new_tokens

    def _fallback(
        self,
        *,
        page_index: int,
        source_evidence_item_index: int,
        source_item: EvidenceItem,
        crop: EvidenceCrop,
        raw_output: str | None,
        prompt_used: str,
        stage: str,
        message: str,
        used_full_page_fallback: bool = False,
    ) -> EvidenceVerificationOutput:
        data = {
            "page_index": page_index,
            "source_evidence_item_index": source_evidence_item_index,
            "verification_stage": stage,
            "crop_hint": crop.crop_region,
            "crop_location": crop.crop_location,
            "verification_status": "uncertain",
            "needs_full_page": False if stage == "full_page" else crop.is_generated_crop,
            "crop_problem": message,
            "visible_evidence": "",
            "comparison": "",
            "faithful_evidence": None,
            "notes": [message],
            "uncertainties": [message],
            "input_image_path": crop.image_path,
            "crop_bbox": list(crop.bbox) if crop.bbox is not None else None,
            "used_full_page_fallback": used_full_page_fallback,
            "raw_output": raw_output,
            "prompt_used": prompt_used,
        }
        _ = source_item
        return validate_model(EvidenceVerificationOutput, data)

    def _run_crop_verifier(
        self,
        question: str,
        page_index: int,
        source_evidence_item_index: int,
        item: EvidenceItem,
        crop: EvidenceCrop,
    ) -> EvidenceVerificationOutput:
        prompt = build_evidence_verification_crop_prompt(
            question=question,
            page_index=page_index,
            crop_hint=crop.crop_region,
            crop_location=crop.crop_location,
            evidence_type=item.evidence_type,
            content=item.content,
            location=item.location,
            relevance=item.relevance,
            confidence=item.confidence,
            agent_prompt_text=self.crop_prompt_text,
            sol_instruction_text=self.sol_instruction_text,
        )
        try:
            raw_output = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                images=[crop.image_path],
                temperature=self.temperature,
                max_new_tokens=self.crop_max_new_tokens,
            )
            data = extract_json_from_text(raw_output)
            if data is None:
                return self._fallback(
                    page_index=page_index,
                    source_evidence_item_index=source_evidence_item_index,
                    source_item=item,
                    crop=crop,
                    raw_output=raw_output,
                    prompt_used=prompt,
                    stage="crop",
                    message="Failed to parse crop verifier output.",
                )
            normalized = _normalize_verification_data(
                data,
                page_index=page_index,
                source_evidence_item_index=source_evidence_item_index,
                source_item=item,
                crop=crop,
                raw_output=raw_output,
                prompt_used=prompt,
                stage="crop",
            )
            return validate_model(EvidenceVerificationOutput, normalized)
        except Exception as exc:
            return self._fallback(
                page_index=page_index,
                source_evidence_item_index=source_evidence_item_index,
                source_item=item,
                crop=crop,
                raw_output=f"Agent failure: {exc}",
                prompt_used=prompt,
                stage="crop",
                message=f"Crop verifier failed: {exc}",
            )

    def _run_full_page_verifier(
        self,
        question: str,
        page: DocumentPage,
        source_evidence_item_index: int,
        item: EvidenceItem,
        crop_output: EvidenceVerificationOutput,
    ) -> EvidenceVerificationOutput:
        full_page = EvidenceCrop(
            crop_region=crop_output.crop_hint,
            image_path=page.image_path,
            crop_location="full original page",
            bbox=None,
            is_generated_crop=False,
        )
        prompt = build_evidence_verification_full_page_prompt(
            question=question,
            page_index=page.page_index,
            crop_hint=crop_output.crop_hint,
            crop_problem=crop_output.crop_problem,
            crop_verifier_output=json.dumps(_model_dump(crop_output), ensure_ascii=False),
            evidence_type=item.evidence_type,
            content=item.content,
            location=item.location,
            relevance=item.relevance,
            confidence=item.confidence,
            agent_prompt_text=self.full_page_prompt_text,
            sol_instruction_text=self.sol_instruction_text,
        )
        try:
            raw_output = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                images=[page.image_path],
                temperature=self.temperature,
                max_new_tokens=self.full_page_max_new_tokens,
            )
            data = extract_json_from_text(raw_output)
            if data is None:
                return self._fallback(
                    page_index=page.page_index,
                    source_evidence_item_index=source_evidence_item_index,
                    source_item=item,
                    crop=full_page,
                    raw_output=raw_output,
                    prompt_used=prompt,
                    stage="full_page",
                    message="Failed to parse full-page verifier output.",
                    used_full_page_fallback=True,
                )
            normalized = _normalize_verification_data(
                data,
                page_index=page.page_index,
                source_evidence_item_index=source_evidence_item_index,
                source_item=item,
                crop=full_page,
                raw_output=raw_output,
                prompt_used=prompt,
                stage="full_page",
                used_full_page_fallback=True,
            )
            return validate_model(EvidenceVerificationOutput, normalized)
        except Exception as exc:
            return self._fallback(
                page_index=page.page_index,
                source_evidence_item_index=source_evidence_item_index,
                source_item=item,
                crop=full_page,
                raw_output=f"Agent failure: {exc}",
                prompt_used=prompt,
                stage="full_page",
                message=f"Full-page verifier failed: {exc}",
                used_full_page_fallback=True,
            )

    def run(self, question: str, page: DocumentPage, clue_output: ClueDiscoveryOutput) -> list[EvidenceVerificationOutput]:
        outputs: list[EvidenceVerificationOutput] = []
        for item_index, item in enumerate(clue_output.evidence_items):
            crop = build_evidence_crop(page.image_path, item.crop_region, item_index)
            crop_output = self._run_crop_verifier(
                question=question,
                page_index=page.page_index,
                source_evidence_item_index=item_index,
                item=item,
                crop=crop,
            )
            if crop_output.needs_full_page and crop.is_generated_crop:
                outputs.append(
                    self._run_full_page_verifier(
                        question=question,
                        page=page,
                        source_evidence_item_index=item_index,
                        item=item,
                        crop_output=crop_output,
                    )
                )
            else:
                outputs.append(crop_output)
        return outputs
