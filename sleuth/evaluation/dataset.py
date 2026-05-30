from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sleuth.evaluation.categories import normalize_categories


@dataclass(frozen=True)
class MMLongBenchExample:
    question_id: str
    row_index: int
    doc_id: str
    doc_type: str
    question: str
    answer: str
    evidence_pages: list[int]
    source_evidence_pages: list[int]
    evidence_sources: list[str]
    answer_format: str
    pdf_path: str
    categories: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return [stripped]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def _normalize_evidence_pages(raw_pages: list[int], page_base: str, layout: str) -> list[int]:
    normalized_base = page_base.strip().lower().replace("_", "-")
    if normalized_base == "auto":
        normalized_base = "one" if layout in {"official", "bundled_sample"} else "zero"
    if normalized_base in {"one", "1", "one-based", "1-based"}:
        return [page - 1 if page > 0 else page for page in raw_pages]
    if normalized_base in {"zero", "0", "zero-based", "0-based"}:
        return raw_pages
    raise ValueError(f"Unsupported evidence page base: {page_base}")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _official_samples_path(data_dir: Path) -> Path:
    return data_dir / "data" / "samples.json"


def _official_documents_dir(data_dir: Path) -> Path:
    return data_dir / "data" / "documents"


def _bundled_sample_path(data_dir: Path) -> Path:
    return data_dir / "samples" / "mmlongbench_doc" / "sample_question.json"


def resolve_dataset_layout(data_dir: str | Path) -> tuple[list[dict[str, Any]], Path, str]:
    root = Path(data_dir)
    official_samples = _official_samples_path(root)
    if official_samples.exists():
        return _load_json(official_samples), _official_documents_dir(root), "official"

    bundled_sample = _bundled_sample_path(root)
    if bundled_sample.exists():
        sample = _load_json(bundled_sample)
        return [sample], root, "bundled_sample"

    raise FileNotFoundError(
        f"Could not find MMLongBench-Doc data under {root}. Expected "
        f"{official_samples} or {bundled_sample}."
    )


def _resolve_pdf_path(row: dict[str, Any], documents_root: Path, layout: str) -> Path:
    if layout == "bundled_sample" and row.get("pdf_path"):
        return documents_root / str(row["pdf_path"])
    doc_path = documents_root / str(row["doc_id"])
    if doc_path.exists() or doc_path.suffix:
        return doc_path
    return doc_path.with_suffix(".pdf")


def load_mmlongbench_examples(
    data_dir: str | Path,
    limit: int | None = None,
    category: str | None = None,
    evidence_page_base: str = "auto",
) -> list[MMLongBenchExample]:
    rows, documents_root, layout = resolve_dataset_layout(data_dir)
    examples: list[MMLongBenchExample] = []
    category_filter = category.strip() if category else None

    for row_index, row in enumerate(rows):
        effective_row_index = int(row.get("row_index", row_index))
        source_evidence_pages = [int(page) for page in _parse_list(row.get("evidence_pages"))]
        evidence_pages = _normalize_evidence_pages(source_evidence_pages, evidence_page_base, layout)
        evidence_sources = [str(source) for source in _parse_list(row.get("evidence_sources"))]
        categories = normalize_categories(evidence_sources)
        if category_filter and category_filter not in categories:
            continue

        pdf_path = _resolve_pdf_path(row, documents_root, layout)
        question_id = str(row.get("question_id") or row.get("id") or effective_row_index)
        examples.append(
            MMLongBenchExample(
                question_id=question_id,
                row_index=effective_row_index,
                doc_id=str(row["doc_id"]),
                doc_type=str(row.get("doc_type", "")),
                question=str(row["question"]),
                answer=str(row["answer"]),
                evidence_pages=evidence_pages,
                source_evidence_pages=source_evidence_pages,
                evidence_sources=evidence_sources,
                answer_format=str(row.get("answer_format", "Str")),
                pdf_path=str(pdf_path),
                categories=categories,
            )
        )
        if limit is not None and len(examples) >= limit:
            break

    return examples
