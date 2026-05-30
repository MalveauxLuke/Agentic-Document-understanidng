from __future__ import annotations

from pydantic import BaseModel


class DocumentPage(BaseModel):
    page_index: int
    image_path: str
    text: str | None = None


class RetrievedPage(BaseModel):
    page_index: int
    score: float
    reason: str | None = None


class EvidenceItem(BaseModel):
    page_index: int
    evidence_type: str
    content: str
    location: str
    relevance: str
    confidence: str
    crop_region: str = "not_applicable"


class ClueDiscoveryOutput(BaseModel):
    page_index: int
    has_relevant_evidence: bool
    evidence_items: list[EvidenceItem]
    page_summary: str
    key_insights: str
    raw_output: str | None = None
    prompt_used: str | None = None


class PageScreeningOutput(BaseModel):
    page_index: int
    has_visual_element: bool
    relevance: str
    reasoning: str
    keep_page: bool
    raw_output: str | None = None
    prompt_used: str | None = None


class EvidenceVerificationOutput(BaseModel):
    page_index: int
    source_evidence_item_index: int
    verification_stage: str
    crop_hint: str
    crop_location: str
    verification_status: str
    needs_full_page: bool
    crop_problem: str | None = None
    visible_evidence: str = ""
    comparison: str = ""
    faithful_evidence: EvidenceItem | None = None
    notes: list[str] = []
    uncertainties: list[str] = []
    input_image_path: str | None = None
    crop_bbox: list[int] | None = None
    used_full_page_fallback: bool = False
    raw_output: str | None = None
    prompt_used: str | None = None


class DifficultyOutput(BaseModel):
    difficulty_level: int
    instruction_set: str
    raw_output: str | None = None
    prompt_used: str | None = None


class EvidenceContext(BaseModel):
    question: str
    retrieved_pages: list[RetrievedPage]
    clue_outputs: list[ClueDiscoveryOutput]
    page_screening_outputs: list[PageScreeningOutput]
    verification_outputs: list[EvidenceVerificationOutput] = []
    retained_page_indices: list[int]
    retained_image_paths: list[str]
    retained_visual_evidence: list[dict] = []
    evidence_summary: str


class FinalAnswer(BaseModel):
    answer: str
    evidence_references: list[dict]
    raw_output: str | None = None
    prompt_used: str | None = None
    core_decision_model: str | None = None
    core_decision_mode: str | None = None
    difficulty_model_switching_used: bool = False


class PipelineResult(BaseModel):
    question: str
    final_answer: FinalAnswer
    evidence_context: EvidenceContext
    difficulty: DifficultyOutput
    output_dir: str
