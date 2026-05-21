from __future__ import annotations

from sleuth.schemas import (
    ClueDiscoveryOutput,
    DocumentPage,
    EvidenceContext,
    PageScreeningOutput,
    RetrievedPage,
)


def build_evidence_summary(clue_outputs: list[ClueDiscoveryOutput]) -> str:
    lines: list[str] = []
    for clue_output in clue_outputs:
        if not clue_output.evidence_items:
            continue
        lines.append(f"Page index {clue_output.page_index}:")
        for item in clue_output.evidence_items:
            content = item.content.strip()
            lines.append(f"- [{item.evidence_type}, {item.confidence}] {content}")
            lines.append(f"  location: {item.location}")
            lines.append(f"  relevance: {item.relevance}")
        lines.append("")
    summary = "\n".join(lines).strip()
    return summary or "No relevant evidence was extracted."


def build_evidence_context(
    question: str,
    pages: list[DocumentPage],
    retrieved_pages: list[RetrievedPage],
    clue_outputs: list[ClueDiscoveryOutput],
    page_screening_outputs: list[PageScreeningOutput],
) -> EvidenceContext:
    page_lookup = {page.page_index: page for page in pages}
    retained_page_indices = [
        screening.page_index for screening in page_screening_outputs if screening.keep_page
    ]
    retained_image_paths = [
        page_lookup[index].image_path
        for index in retained_page_indices
        if index in page_lookup
    ]
    evidence_summary = build_evidence_summary(clue_outputs)
    return EvidenceContext(
        question=question,
        retrieved_pages=retrieved_pages,
        clue_outputs=clue_outputs,
        page_screening_outputs=page_screening_outputs,
        retained_page_indices=retained_page_indices,
        retained_image_paths=retained_image_paths,
        evidence_summary=evidence_summary,
    )
