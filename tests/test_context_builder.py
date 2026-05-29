from sleuth.pipeline.context_builder import build_multimodal_evidence_summary
from sleuth.schemas import ClueDiscoveryOutput, PageScreeningOutput


def test_evidence_summary_includes_retained_visual_screening_reasoning():
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

    assert "Retained visual evidence:" in summary
    assert "Page index 1:" in summary
    assert "screening_relevance: Completely Relevant" in summary
    assert "retained chart contains the comparison" in summary
    assert "Page index 2:" not in summary
