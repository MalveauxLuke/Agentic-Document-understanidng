from sleuth.pipeline.context_builder import build_evidence_context, build_multimodal_evidence_summary
from sleuth.schemas import ClueDiscoveryOutput, DocumentPage, PageScreeningOutput, RetrievedPage


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
