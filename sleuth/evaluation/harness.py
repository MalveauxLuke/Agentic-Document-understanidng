from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sleuth.agents.clue_discovery import ClueDiscoveryAgent
from sleuth.agents.core_decision import CoreDecisionAgent
from sleuth.agents.difficulty_assessment import DifficultyAssessmentAgent
from sleuth.agents.page_screening import PageScreeningAgent
from sleuth.documents.page_store import build_document_pages
from sleuth.evaluation.answer_extraction import AnswerExtractor, HeuristicAnswerExtractor
from sleuth.evaluation.dataset import MMLongBenchExample
from sleuth.evaluation.metrics import compute_metrics, compute_metrics_by_category, save_metrics_by_category_csv
from sleuth.evaluation.scoring import eval_score
from sleuth.instructions.instruction_loader import read_markdown_file, save_instruction_copy
from sleuth.instructions.prompt_loader import get_prompt_section, load_agent_prompt_markdown
from sleuth.llm.base import LLMClient
from sleuth.pipeline.context_builder import build_evidence_context
from sleuth.retrieval.base import BaseRetriever
from sleuth.schemas import (
    ClueDiscoveryOutput,
    DifficultyOutput,
    DocumentPage,
    FinalAnswer,
    PageScreeningOutput,
    RetrievedPage,
)
from sleuth.utils.file_utils import ensure_dir, write_text
from sleuth.utils.json_utils import extract_json_from_text, load_json, save_json


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


def _model_validate(model_cls, data: dict[str, Any]):
    validator = getattr(model_cls, "model_validate", None)
    if callable(validator):
        return validator(data)
    return model_cls.parse_obj(data)


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _document_cache_dir(cache_dir: Path, doc_id: str) -> Path:
    return cache_dir / "documents" / _safe_id(doc_id)


def load_or_build_document_pages(
    pdf_path: str,
    doc_id: str,
    cache_dir: Path,
    render_dpi: int,
) -> list[DocumentPage]:
    doc_cache = ensure_dir(_document_cache_dir(cache_dir, doc_id))
    pages_json = doc_cache / "pages.json"
    pages_dir = doc_cache / "pages"

    if pages_json.exists():
        raw_pages = load_json(pages_json)
        pages = [_model_validate(DocumentPage, item) for item in raw_pages]
        if pages and all(Path(page.image_path).exists() for page in pages):
            return pages

    pages = build_document_pages(pdf_path, str(pages_dir), dpi=render_dpi)
    save_json(pages_json, pages)
    return pages


def _retrieval_cache_path(
    cache_dir: Path,
    example: MMLongBenchExample,
    retriever_name: str,
    top_k: int,
) -> Path:
    key = _short_hash(f"{example.doc_id}\n{example.question}\n{retriever_name}\n{top_k}")
    return cache_dir / "retrieval" / f"{_safe_id(example.doc_id)}_{key}.json"


def load_or_run_retrieval(
    retriever: BaseRetriever,
    pages: list[DocumentPage],
    example: MMLongBenchExample,
    retriever_name: str,
    top_k: int,
    cache_dir: Path,
) -> list[RetrievedPage]:
    cache_path = _retrieval_cache_path(cache_dir, example, retriever_name, top_k)
    if cache_path.exists():
        raw_pages = load_json(cache_path)
        return [_model_validate(RetrievedPage, item) for item in raw_pages]

    retriever.index(pages)
    retrieved_pages = retriever.retrieve(example.question, top_k=top_k)
    save_json(cache_path, retrieved_pages)
    return retrieved_pages


def build_base_prompt(example: MMLongBenchExample, pages: list[DocumentPage], retrieved_pages: list[RetrievedPage]) -> str:
    page_lines = "\n".join(
        f"- Page index {retrieved.page_index} (retrieval score {retrieved.score:.4f})"
        for retrieved in retrieved_pages
    )
    return (
        "You are an extractive document question answering model. Answer the question using only the "
        "provided retrieved PDF page images.\n\n"
        f"Question: {example.question}\n\n"
        "Retrieved page images:\n"
        f"{page_lines}\n\n"
        "Give the shortest supported answer. If the answer is not visible in the provided pages, say "
        "No answers found!"
    )


def run_base_example(
    example: MMLongBenchExample,
    pages: list[DocumentPage],
    retrieved_pages: list[RetrievedPage],
    llm_client: LLMClient,
    temperature: float,
    max_new_tokens: int,
) -> FinalAnswer:
    page_lookup = {page.page_index: page for page in pages}
    image_paths = [
        page_lookup[retrieved.page_index].image_path
        for retrieved in retrieved_pages
        if retrieved.page_index in page_lookup
    ]
    prompt = build_base_prompt(example, pages, retrieved_pages)
    raw_output = llm_client.chat(
        [{"role": "user", "content": prompt}],
        images=image_paths,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    data = extract_json_from_text(raw_output) or {}
    answer = str(data.get("answer", raw_output)).strip()
    return FinalAnswer(
        answer=answer or "No answers found!",
        evidence_references=data.get("evidence_references") if isinstance(data.get("evidence_references"), list) else [],
        raw_output=raw_output,
        prompt_used=prompt,
    )


def _agent_cache_path(
    cache_dir: Path,
    example: MMLongBenchExample,
    agent_name: str,
    page_index: int | None = None,
    cache_fingerprint: str = "default",
) -> Path:
    page_suffix = "" if page_index is None else f"_page_{page_index:04d}"
    return cache_dir / "agents" / f"{_safe_id(example.question_id)}_{agent_name}{page_suffix}_{cache_fingerprint}.json"


def _load_agent_output(path: Path, model_cls):
    raw = load_json(path)
    return _model_validate(model_cls, raw)


def run_sleuth_example(
    example: MMLongBenchExample,
    pages: list[DocumentPage],
    retrieved_pages: list[RetrievedPage],
    clue_agent: ClueDiscoveryAgent,
    page_screening_agent: PageScreeningAgent,
    difficulty_agent: DifficultyAssessmentAgent,
    core_decision_agent: CoreDecisionAgent,
    cache_dir: Path,
    cache_fingerprint: str,
) -> tuple[FinalAnswer, list[ClueDiscoveryOutput], list[PageScreeningOutput], DifficultyOutput]:
    page_lookup = {page.page_index: page for page in pages}
    clue_outputs: list[ClueDiscoveryOutput] = []
    page_screening_outputs: list[PageScreeningOutput] = []

    for retrieved_page in retrieved_pages:
        page = page_lookup.get(retrieved_page.page_index)
        if page is None:
            continue

        clue_cache = _agent_cache_path(cache_dir, example, "clue", page.page_index, cache_fingerprint)
        if clue_cache.exists():
            clue_output = _load_agent_output(clue_cache, ClueDiscoveryOutput)
        else:
            clue_output = clue_agent.run(example.question, page)
            save_json(clue_cache, clue_output)
        clue_outputs.append(clue_output)

        screen_cache = _agent_cache_path(cache_dir, example, "screen", page.page_index, cache_fingerprint)
        if screen_cache.exists():
            screening_output = _load_agent_output(screen_cache, PageScreeningOutput)
        else:
            screening_output = page_screening_agent.run(example.question, page)
            save_json(screen_cache, screening_output)
        page_screening_outputs.append(screening_output)

    evidence_context = build_evidence_context(
        question=example.question,
        pages=pages,
        retrieved_pages=retrieved_pages,
        clue_outputs=clue_outputs,
        page_screening_outputs=page_screening_outputs,
    )

    difficulty_cache = _agent_cache_path(cache_dir, example, "difficulty", cache_fingerprint=cache_fingerprint)
    if difficulty_cache.exists():
        difficulty = _load_agent_output(difficulty_cache, DifficultyOutput)
    else:
        difficulty = difficulty_agent.run(example.question, evidence_context)
        save_json(difficulty_cache, difficulty)

    final_answer = core_decision_agent.run(example.question, evidence_context, difficulty)
    return final_answer, clue_outputs, page_screening_outputs, difficulty


def _retrieval_diagnostics(example: MMLongBenchExample, retrieved_pages: list[RetrievedPage]) -> dict[str, Any]:
    gold_pages = sorted(set(example.evidence_pages))
    retrieved_indices = [page.page_index for page in retrieved_pages]
    retrieved_set = set(retrieved_indices)
    if not gold_pages:
        return {
            "gold_evidence_pages": gold_pages,
            "gold_hit_at_k": None,
            "gold_missed_pages": [],
        }
    return {
        "gold_evidence_pages": gold_pages,
        "gold_hit_at_k": bool(set(gold_pages) & retrieved_set),
        "gold_missed_pages": [page for page in gold_pages if page not in retrieved_set],
    }


def _stage_diagnostics(
    method: str,
    example: MMLongBenchExample,
    retrieved_pages: list[RetrievedPage],
    clue_outputs: list[ClueDiscoveryOutput],
    page_screening_outputs: list[PageScreeningOutput],
    score: float,
    raw_score: float,
) -> dict[str, Any]:
    retrieval = _retrieval_diagnostics(example, retrieved_pages)
    gold_pages = set(retrieval["gold_evidence_pages"])
    clue_pages = sorted(
        {
            clue.page_index
            for clue in clue_outputs
            if clue.has_relevant_evidence or clue.evidence_items
        }
    )
    retained_pages = sorted({screen.page_index for screen in page_screening_outputs if screen.keep_page})
    visual_categories = {"Chart", "Table", "Figure", "Layout"}
    has_visual_gold = bool(set(example.categories) & visual_categories)
    clue_hit_gold = bool(gold_pages & set(clue_pages)) if gold_pages and method == "sleuth" else None
    screening_retained_gold = (
        bool(gold_pages & set(retained_pages)) if gold_pages and has_visual_gold and method == "sleuth" else None
    )

    if score > 0.0:
        failure_label = "correct"
    elif retrieval["gold_hit_at_k"] is False:
        failure_label = "retrieval_miss"
    elif method == "sleuth" and clue_hit_gold is False:
        failure_label = "clue_miss"
    elif method == "sleuth" and screening_retained_gold is False:
        failure_label = "screening_drop"
    elif raw_score > score:
        failure_label = "scoring_or_extraction_mismatch"
    else:
        failure_label = "final_wrong"

    return {
        **retrieval,
        "clue_pages_with_evidence": clue_pages,
        "clue_hit_gold": clue_hit_gold,
        "screening_retained_page_indices": retained_pages,
        "screening_retained_gold": screening_retained_gold,
        "failure_label": failure_label,
    }


class MMLongBenchEvaluator:
    def __init__(
        self,
        output_dir: str | Path,
        method: str,
        retriever: BaseRetriever,
        retriever_name: str,
        llm_client: LLMClient,
        top_k: int,
        temperature: float,
        render_dpi: int,
        agent_prompts_md: str,
        sol_instruction_text: str | None,
        max_tokens: dict[str, int],
        answer_extractor: AnswerExtractor | None = None,
    ) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.cache_dir = ensure_dir(self.output_dir / "cache")
        self.method = method
        self.retriever = retriever
        self.retriever_name = retriever_name
        self.llm_client = llm_client
        self.top_k = top_k
        self.temperature = temperature
        self.render_dpi = render_dpi
        self.max_tokens = max_tokens
        self.answer_extractor = answer_extractor or HeuristicAnswerExtractor()

        self.agent_prompt_markdown = load_agent_prompt_markdown(agent_prompts_md)
        save_instruction_copy(self.agent_prompt_markdown, self.output_dir / "agent_prompts_used.md")
        if sol_instruction_text is not None:
            save_instruction_copy(sol_instruction_text, self.output_dir / "sol_instructions_used.md")

        self.clue_agent = ClueDiscoveryAgent(
            llm_client,
            get_prompt_section(self.agent_prompt_markdown, "clue_discovery"),
            sol_instruction_text=sol_instruction_text,
            temperature=temperature,
            max_new_tokens=max_tokens.get("clue_discovery", 1024),
        )
        self.page_screening_agent = PageScreeningAgent(
            llm_client,
            get_prompt_section(self.agent_prompt_markdown, "page_screening"),
            sol_instruction_text=sol_instruction_text,
            temperature=temperature,
            max_new_tokens=max_tokens.get("page_screening", 512),
        )
        self.difficulty_agent = DifficultyAssessmentAgent(
            llm_client,
            get_prompt_section(self.agent_prompt_markdown, "difficulty_assessment"),
            sol_instruction_text=sol_instruction_text,
            temperature=temperature,
            max_new_tokens=max_tokens.get("difficulty_assessment", 512),
        )
        self.core_decision_agent = CoreDecisionAgent(
            llm_client,
            get_prompt_section(self.agent_prompt_markdown, "core_decision_text"),
            get_prompt_section(self.agent_prompt_markdown, "core_decision_visual"),
            sol_instruction_text=sol_instruction_text,
            temperature=temperature,
            max_new_tokens=max_tokens.get("core_decision", 512),
        )
        self.cache_fingerprint = _short_hash(
            json.dumps(
                {
                    "method": method,
                    "retriever": retriever_name,
                    "top_k": top_k,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "llm": getattr(llm_client, "model_name_or_path", llm_client.__class__.__name__),
                    "prompts": self.agent_prompt_markdown,
                },
                sort_keys=True,
            )
        )

    def run_example(self, example: MMLongBenchExample) -> dict[str, Any]:
        pages = load_or_build_document_pages(example.pdf_path, example.doc_id, self.cache_dir, self.render_dpi)
        retrieved_pages = load_or_run_retrieval(
            self.retriever,
            pages,
            example,
            self.retriever_name,
            self.top_k,
            self.cache_dir,
        )

        clue_outputs: list[ClueDiscoveryOutput] = []
        page_screening_outputs: list[PageScreeningOutput] = []
        difficulty_output: DifficultyOutput | None = None

        if self.method == "base":
            final_answer = run_base_example(
                example,
                pages,
                retrieved_pages,
                self.llm_client,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens.get("core_decision", 512),
            )
        elif self.method == "sleuth":
            final_answer, clue_outputs, page_screening_outputs, difficulty_output = run_sleuth_example(
                example,
                pages,
                retrieved_pages,
                self.clue_agent,
                self.page_screening_agent,
                self.difficulty_agent,
                self.core_decision_agent,
                self.cache_dir,
                self.cache_fingerprint,
            )
        else:
            raise ValueError(f"Unsupported evaluation method: {self.method}")

        extraction = self.answer_extractor.extract(example.question, final_answer.answer, example.answer_format)
        raw_score = eval_score(example.answer, final_answer.answer, example.answer_format)
        score = eval_score(example.answer, extraction.extracted_answer, example.answer_format)
        diagnostics = _stage_diagnostics(
            method=self.method,
            example=example,
            retrieved_pages=retrieved_pages,
            clue_outputs=clue_outputs,
            page_screening_outputs=page_screening_outputs,
            score=score,
            raw_score=raw_score,
        )
        return {
            "question_id": example.question_id,
            "row_index": example.row_index,
            "document_id": example.doc_id,
            "doc_type": example.doc_type,
            "question": example.question,
            "ground_truth_answer": example.answer,
            "answer_format": example.answer_format,
            "model_answer": extraction.extracted_answer,
            "raw_model_answer": final_answer.answer,
            "extracted_answer": extraction.extracted_answer,
            "answer_extractor": extraction.extractor,
            "answer_extraction_error": extraction.error,
            "paper_comparable_scoring": extraction.paper_comparable,
            "raw_score": raw_score,
            "score": score,
            "correctness": score > 0.0,
            "category": example.categories[0] if example.categories else "None",
            "categories": example.categories,
            "evidence_pages": example.evidence_pages,
            "gold_evidence_pages": diagnostics["gold_evidence_pages"],
            "evidence_sources": example.evidence_sources,
            "retrieved_page_indices": [page.page_index for page in retrieved_pages],
            "retrieved_pages": _model_dump(retrieved_pages),
            "clue_output": _model_dump(clue_outputs),
            "page_screening_output": _model_dump(page_screening_outputs),
            "difficulty_output": _model_dump(difficulty_output) if difficulty_output is not None else None,
            "final_prompt": final_answer.prompt_used,
            "raw_response": final_answer.raw_output,
            "evidence_references": final_answer.evidence_references,
            **diagnostics,
            "errors": None,
        }

    def _write_metrics(self, predictions: list[dict[str, Any]], failed: list[dict[str, Any]], attempted: int) -> dict[str, Any]:
        metrics = compute_metrics(predictions)
        metrics["failed"] = len(failed)
        metrics["attempted"] = attempted
        metrics["answer_extractor"] = self.answer_extractor.name
        metrics["paper_comparable_scoring"] = (
            all(item.get("paper_comparable_scoring") for item in predictions)
            if predictions
            else self.answer_extractor.paper_comparable
        )
        metrics_by_category = compute_metrics_by_category(predictions)
        save_json(self.output_dir / "metrics.json", metrics)
        save_metrics_by_category_csv(self.output_dir / "metrics_by_category.csv", metrics_by_category)
        write_text(self.output_dir / "summary.txt", json.dumps(metrics, indent=2))
        return metrics

    def run(self, examples: list[MMLongBenchExample], run_config: dict[str, Any]) -> dict[str, Any]:
        predictions_path = self.output_dir / "predictions.jsonl"
        failed_path = self.output_dir / "failed_examples.jsonl"
        predictions: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        save_json(self.output_dir / "run_config.json", run_config)
        with predictions_path.open("w", encoding="utf-8") as pred_f, failed_path.open("w", encoding="utf-8") as fail_f:
            for example in examples:
                try:
                    prediction = self.run_example(example)
                    predictions.append(prediction)
                    pred_f.write(json.dumps(prediction) + "\n")
                    pred_f.flush()
                    self._write_metrics(predictions, failed, attempted=len(examples))
                except Exception as exc:
                    failed_example = {
                        **example.to_dict(),
                        "errors": f"{type(exc).__name__}: {exc}",
                    }
                    failed.append(failed_example)
                    fail_f.write(json.dumps(failed_example) + "\n")
                    fail_f.flush()
                    self._write_metrics(predictions, failed, attempted=len(examples))

        return self._write_metrics(predictions, failed, attempted=len(examples))


def load_sol_instructions_if_needed(mode: str, path: str) -> str | None:
    if mode == "sol":
        return read_markdown_file(path)
    return None
