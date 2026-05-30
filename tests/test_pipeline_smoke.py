from pathlib import Path

import fitz

from sleuth.agents.clue_discovery import ClueDiscoveryAgent
from sleuth.agents.core_decision import CoreDecisionAgent
from sleuth.agents.difficulty_assessment import DifficultyAssessmentAgent
from sleuth.agents.evidence_verification import EvidenceVerificationAgent
from sleuth.instructions.prompt_loader import get_prompt_section, load_agent_prompt_markdown
from sleuth.llm.mock_client import MockClient
from sleuth.pipeline.sleuth_pipeline import SleuthPipeline
from sleuth.retrieval.dummy_retriever import DummyRetriever


ROOT = Path(__file__).resolve().parents[1]


def _make_tiny_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text((36, 72), "The main result is a successful mock pipeline.")
    document.save(path)
    document.close()


def test_mock_pipeline_smoke(tmp_path):
    pdf_path = tmp_path / "tiny.pdf"
    out_dir = tmp_path / "run"
    _make_tiny_pdf(pdf_path)

    prompt_markdown = load_agent_prompt_markdown(ROOT / "agent_prompts.md")
    llm_client = MockClient()
    pipeline = SleuthPipeline(
        retriever=DummyRetriever(),
        clue_agent=ClueDiscoveryAgent(
            llm_client,
            get_prompt_section(prompt_markdown, "clue_discovery"),
        ),
        evidence_verification_agent=EvidenceVerificationAgent(
            llm_client,
            get_prompt_section(prompt_markdown, "evidence_verification_crop"),
            get_prompt_section(prompt_markdown, "evidence_verification_full_page"),
        ),
        difficulty_agent=DifficultyAssessmentAgent(
            llm_client,
            get_prompt_section(prompt_markdown, "difficulty_assessment"),
        ),
        core_decision_agent=CoreDecisionAgent(
            llm_client,
            get_prompt_section(prompt_markdown, "core_decision_text"),
            get_prompt_section(prompt_markdown, "core_decision_visual"),
        ),
        render_dpi=72,
    )

    result = pipeline.run(
        pdf_path=str(pdf_path),
        question="What is the main result?",
        top_k=1,
        out_dir=str(out_dir),
    )

    assert result.final_answer.answer
    assert (out_dir / "pages" / "page_0001.png").exists()
    assert (out_dir / "agents" / "clue_page_0000.json").exists()
    assert (out_dir / "agents" / "verify_page_0000.json").exists()
    assert (out_dir / "final" / "retrieved_pages.json").exists()
    assert (out_dir / "final" / "evidence_context.json").exists()
    assert (out_dir / "final" / "evidence_summary.txt").exists()
    assert (out_dir / "final" / "difficulty.json").exists()
    assert (out_dir / "final" / "final_answer.json").exists()
