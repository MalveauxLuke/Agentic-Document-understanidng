from __future__ import annotations

from pathlib import Path

from sleuth.agents.clue_discovery import ClueDiscoveryAgent
from sleuth.agents.core_decision import CoreDecisionAgent
from sleuth.agents.difficulty_assessment import DifficultyAssessmentAgent
from sleuth.agents.page_screening import PageScreeningAgent
from sleuth.documents.page_store import build_document_pages
from sleuth.pipeline.context_builder import build_evidence_context
from sleuth.retrieval.base import BaseRetriever
from sleuth.schemas import PipelineResult
from sleuth.utils.file_utils import ensure_dir, write_text
from sleuth.utils.json_utils import save_json


class SleuthPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        clue_agent: ClueDiscoveryAgent,
        page_screening_agent: PageScreeningAgent,
        difficulty_agent: DifficultyAssessmentAgent,
        core_decision_agent: CoreDecisionAgent,
        render_dpi: int = 144,
    ) -> None:
        self.retriever = retriever
        self.clue_agent = clue_agent
        self.page_screening_agent = page_screening_agent
        self.difficulty_agent = difficulty_agent
        self.core_decision_agent = core_decision_agent
        self.render_dpi = render_dpi

    def run(self, pdf_path: str, question: str, top_k: int, out_dir: str) -> PipelineResult:
        output_dir = Path(out_dir)
        pages_dir = ensure_dir(output_dir / "pages")
        agents_dir = ensure_dir(output_dir / "agents")
        final_dir = ensure_dir(output_dir / "final")

        pages = build_document_pages(pdf_path, str(pages_dir), dpi=self.render_dpi)
        page_lookup = {page.page_index: page for page in pages}

        self.retriever.index(pages)
        retrieved_pages = self.retriever.retrieve(question, top_k=top_k)
        save_json(final_dir / "retrieved_pages.json", retrieved_pages)

        clue_outputs = []
        page_screening_outputs = []
        for retrieved_page in retrieved_pages:
            if retrieved_page.page_index not in page_lookup:
                continue
            page = page_lookup[retrieved_page.page_index]
            clue_output = self.clue_agent.run(question, page)
            clue_outputs.append(clue_output)
            save_json(agents_dir / f"clue_page_{page.page_index:04d}.json", clue_output)

            screening_output = self.page_screening_agent.run(question, page)
            page_screening_outputs.append(screening_output)
            save_json(agents_dir / f"screen_page_{page.page_index:04d}.json", screening_output)

        evidence_context = build_evidence_context(
            question=question,
            pages=pages,
            retrieved_pages=retrieved_pages,
            clue_outputs=clue_outputs,
            page_screening_outputs=page_screening_outputs,
        )
        save_json(final_dir / "evidence_context.json", evidence_context)
        write_text(final_dir / "evidence_summary.txt", evidence_context.evidence_summary)

        difficulty = self.difficulty_agent.run(question, evidence_context)
        save_json(final_dir / "difficulty.json", difficulty)

        final_answer = self.core_decision_agent.run(question, evidence_context, difficulty)
        save_json(final_dir / "final_answer.json", final_answer)

        result = PipelineResult(
            question=question,
            final_answer=final_answer,
            evidence_context=evidence_context,
            difficulty=difficulty,
            output_dir=str(output_dir),
        )
        return result
