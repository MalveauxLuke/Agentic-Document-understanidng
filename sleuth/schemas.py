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
    retained_page_indices: list[int]
    retained_image_paths: list[str]
    evidence_summary: str


class FinalAnswer(BaseModel):
    answer: str
    evidence_references: list[dict]
    raw_output: str | None = None
    prompt_used: str | None = None


class PipelineResult(BaseModel):
    question: str
    final_answer: FinalAnswer
    evidence_context: EvidenceContext
    difficulty: DifficultyOutput
    output_dir: str
