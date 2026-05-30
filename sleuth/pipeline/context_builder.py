from __future__ import annotations

from sleuth.agents.prompts import display_page_number
from sleuth.schemas import (
    ClueDiscoveryOutput,
    DocumentPage,
    EvidenceContext,
    EvidenceVerificationOutput,
    PageScreeningOutput,
    RetrievedPage,
)


ACCEPTED_VERIFICATION_STATUSES = {"faithful", "corrected"}


def build_evidence_summary(clue_outputs: list[ClueDiscoveryOutput]) -> str:
    lines: list[str] = []
    for clue_output in clue_outputs:
        page_label = display_page_number(clue_output.page_index)
        if not clue_output.evidence_items:
            if clue_output.page_summary.strip() or clue_output.key_insights.strip():
                lines.append(f"Page Number {page_label}:")
                if clue_output.page_summary.strip():
                    lines.append(f"page_summary: {clue_output.page_summary.strip()}")
                if clue_output.key_insights.strip():
                    lines.append(f"key_insights: {clue_output.key_insights.strip()}")
                lines.append("")
            continue
        lines.append(f"Page Number {page_label}:")
        if clue_output.page_summary.strip():
            lines.append(f"page_summary: {clue_output.page_summary.strip()}")
        if clue_output.key_insights.strip():
            lines.append(f"key_insights: {clue_output.key_insights.strip()}")
        for item in clue_output.evidence_items:
            content = item.content.strip()
            lines.append(f"- [{item.evidence_type}, {item.confidence}] {content}")
            lines.append(f"  location: {item.location}")
            if getattr(item, "crop_region", None):
                lines.append(f"  crop_region: {item.crop_region}")
            lines.append(f"  relevance: {item.relevance}")
        lines.append("")
    summary = "\n".join(lines).strip()
    return summary or "No relevant evidence was extracted."


def build_verified_evidence_summary(verification_outputs: list[EvidenceVerificationOutput]) -> str:
    accepted = [
        output
        for output in verification_outputs
        if output.verification_status in ACCEPTED_VERIFICATION_STATUSES and output.faithful_evidence is not None
    ]
    lines: list[str] = []
    current_page: int | None = None
    for output in accepted:
        item = output.faithful_evidence
        if item is None:
            continue
        page_label = display_page_number(output.page_index)
        if current_page != output.page_index:
            if lines:
                lines.append("")
            lines.append(f"Page Number {page_label}:")
            current_page = output.page_index
        lines.append(f"- [{item.evidence_type}, {item.confidence}, {output.verification_status}] {item.content.strip()}")
        lines.append(f"  location: {item.location}")
        lines.append(f"  crop_region: {item.crop_region}")
        lines.append(f"  relevance: {item.relevance}")
    summary = "\n".join(lines).strip()
    return summary or "No verified evidence was accepted."


def build_multimodal_evidence_summary(
    clue_outputs: list[ClueDiscoveryOutput],
    page_screening_outputs: list[PageScreeningOutput],
) -> str:
    # Paper-faithful final context is C=(P,E): E is Clue evidence text, while
    # Page Screening contributes retained images P and remains in diagnostics.
    _ = page_screening_outputs
    return build_evidence_summary(clue_outputs)


def build_evidence_context(
    question: str,
    pages: list[DocumentPage],
    retrieved_pages: list[RetrievedPage],
    clue_outputs: list[ClueDiscoveryOutput],
    page_screening_outputs: list[PageScreeningOutput],
    verification_outputs: list[EvidenceVerificationOutput] | None = None,
) -> EvidenceContext:
    page_lookup = {page.page_index: page for page in pages}
    retained_visual_evidence: list[dict] = []
    if verification_outputs is not None:
        evidence_summary = build_verified_evidence_summary(verification_outputs)
        seen_images: set[str] = set()
        for output in verification_outputs:
            if output.verification_status not in ACCEPTED_VERIFICATION_STATUSES or output.faithful_evidence is None:
                continue
            image_path = output.input_image_path or page_lookup.get(output.page_index, DocumentPage(page_index=output.page_index, image_path="")).image_path
            if not image_path or image_path in seen_images:
                continue
            seen_images.add(image_path)
            retained_visual_evidence.append(
                {
                    "page_index": output.page_index,
                    "display_page_number": display_page_number(output.page_index),
                    "image_path": image_path,
                    "crop_region": output.crop_hint,
                    "crop_location": output.crop_location,
                    "verification_stage": output.verification_stage,
                    "source_evidence_item_index": output.source_evidence_item_index,
                    "is_crop": output.crop_bbox is not None,
                }
            )
        retained_page_indices = [item["page_index"] for item in retained_visual_evidence]
        retained_image_paths = [item["image_path"] for item in retained_visual_evidence]
    else:
        retained_page_indices = [
            screening.page_index for screening in page_screening_outputs if screening.keep_page
        ]
        retained_image_paths = [
            page_lookup[index].image_path
            for index in retained_page_indices
            if index in page_lookup
        ]
        retained_visual_evidence = [
            {
                "page_index": index,
                "display_page_number": display_page_number(index),
                "image_path": page_lookup[index].image_path,
                "crop_region": "full_page",
                "crop_location": "full original page",
                "verification_stage": "page_screening",
                "source_evidence_item_index": None,
                "is_crop": False,
            }
            for index in retained_page_indices
            if index in page_lookup
        ]
        evidence_summary = build_evidence_summary(clue_outputs)
    return EvidenceContext(
        question=question,
        retrieved_pages=retrieved_pages,
        clue_outputs=clue_outputs,
        page_screening_outputs=page_screening_outputs,
        verification_outputs=verification_outputs or [],
        retained_page_indices=retained_page_indices,
        retained_image_paths=retained_image_paths,
        retained_visual_evidence=retained_visual_evidence,
        evidence_summary=evidence_summary,
    )
