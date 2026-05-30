from sleuth.pipeline.context_builder import build_evidence_context, build_multimodal_evidence_summary
from sleuth.schemas import (
    ClueDiscoveryOutput,
    DocumentPage,
    EvidenceItem,
    EvidenceVerificationOutput,
    PageScreeningOutput,
    RetrievedPage,
)


def test_evidence_summary_excludes_retained_visual_screening_reasoning():
    clue_outputs = [
        ClueDiscoveryOutput(
            page_index=1,
            has_relevant_evidence=False,
            evidence_items=[],
            page_summary="No clue evidence.",
            key_insights="",
        )
    ]
    page_screening_outputs = [
        PageScreeningOutput(
            page_index=1,
            has_visual_element=True,
            relevance="Completely Relevant",
            reasoning="The retained chart contains the comparison needed for the question.",
            keep_page=True,
        ),
        PageScreeningOutput(
            page_index=2,
            has_visual_element=True,
            relevance="Irrelevant",
            reasoning="Unrelated figure.",
            keep_page=False,
        ),
    ]

    summary = build_multimodal_evidence_summary(clue_outputs, page_screening_outputs)

    assert "Page Number 2:" in summary
    assert "No clue evidence." in summary
    assert "Retained visual evidence:" not in summary
    assert "screening_relevance: Completely Relevant" not in summary
    assert "retained chart contains the comparison" not in summary
    assert "Page Number 3:" not in summary


def test_evidence_context_keeps_retained_images_outside_text_summary():
    clue_outputs = [
        ClueDiscoveryOutput(
            page_index=0,
            has_relevant_evidence=False,
            evidence_items=[],
            page_summary="No clue evidence.",
            key_insights="",
        )
    ]
    page_screening_outputs = [
        PageScreeningOutput(
            page_index=0,
            has_visual_element=True,
            relevance="Completely Relevant",
            reasoning="The retained chart contains the comparison needed for the question.",
            keep_page=True,
        )
    ]

    context = build_evidence_context(
        question="Which chart?",
        pages=[DocumentPage(page_index=0, image_path="/tmp/page_0001.png")],
        retrieved_pages=[RetrievedPage(page_index=0, score=1.0)],
        clue_outputs=clue_outputs,
        page_screening_outputs=page_screening_outputs,
    )

    assert context.retained_page_indices == [0]
    assert context.retained_image_paths == ["/tmp/page_0001.png"]
    assert "retained chart contains" not in context.evidence_summary


def test_verified_context_uses_only_faithful_and_corrected_evidence():
    verification_outputs = [
        EvidenceVerificationOutput(
            page_index=0,
            source_evidence_item_index=0,
            verification_stage="crop",
            crop_hint="upper_left",
            crop_location="upper_left bbox=(0, 0, 50, 50)",
            verification_status="faithful",
            needs_full_page=False,
            faithful_evidence=EvidenceItem(
                page_index=0,
                evidence_type="figure",
                content="Verified chart fact.",
                location="upper-left crop",
                relevance="direct",
                confidence="high",
                crop_region="upper_left",
            ),
            input_image_path="/tmp/page_0001_crop.png",
            crop_bbox=[0, 0, 50, 50],
        ),
        EvidenceVerificationOutput(
            page_index=1,
            source_evidence_item_index=0,
            verification_stage="crop",
            crop_hint="not_applicable",
            crop_location="full original page",
            verification_status="rejected",
            needs_full_page=False,
            faithful_evidence=None,
            input_image_path="/tmp/page_0002.png",
        ),
    ]

    context = build_evidence_context(
        question="Which chart?",
        pages=[
            DocumentPage(page_index=0, image_path="/tmp/page_0001.png"),
            DocumentPage(page_index=1, image_path="/tmp/page_0002.png"),
        ],
        retrieved_pages=[RetrievedPage(page_index=0, score=1.0), RetrievedPage(page_index=1, score=0.5)],
        clue_outputs=[],
        page_screening_outputs=[],
        verification_outputs=verification_outputs,
    )

    assert "Verified chart fact." in context.evidence_summary
    assert "page_0002" not in context.evidence_summary
    assert context.retained_image_paths == ["/tmp/page_0001_crop.png"]
    assert context.retained_visual_evidence[0]["is_crop"] is True
