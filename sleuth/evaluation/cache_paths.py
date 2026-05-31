from __future__ import annotations

import hashlib
from pathlib import Path

from sleuth.documents.page_store import build_document_pages
from sleuth.evaluation.dataset import MMLongBenchExample
from sleuth.schemas import DocumentPage, RetrievedPage
from sleuth.utils.file_utils import ensure_dir
from sleuth.utils.json_utils import load_json, save_json


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def dpi_key(render_dpi: int) -> str:
    return f"dpi_{int(render_dpi)}"


def document_cache_dir(cache_dir: str | Path, doc_id: str, render_dpi: int) -> Path:
    return Path(cache_dir) / "documents" / dpi_key(render_dpi) / safe_id(doc_id)


def legacy_document_cache_dir(cache_dir: str | Path, doc_id: str) -> Path:
    return Path(cache_dir) / "documents" / safe_id(doc_id)


def colpali_retriever_key(retriever_name: str) -> str:
    return short_hash(retriever_name)


def colpali_embedding_cache_path(
    cache_dir: str | Path,
    retriever_name: str,
    render_dpi: int,
    doc_id: str,
) -> Path:
    return (
        Path(cache_dir)
        / "colpali_embeddings"
        / colpali_retriever_key(retriever_name)
        / dpi_key(render_dpi)
        / f"{safe_id(doc_id)}.pt"
    )


def retrieval_cache_path(
    cache_dir: str | Path,
    example: MMLongBenchExample,
    retriever_name: str,
    top_k: int,
    render_dpi: int,
) -> Path:
    key = short_hash(
        "\n".join(
            [
                str(example.doc_id),
                str(example.question),
                str(retriever_name),
                str(top_k),
                str(render_dpi),
            ]
        )
    )
    return Path(cache_dir) / "retrieval" / dpi_key(render_dpi) / f"{safe_id(example.doc_id)}_{key}.json"


def load_or_build_document_pages(
    pdf_path: str,
    doc_id: str,
    cache_dir: str | Path,
    render_dpi: int,
) -> list[DocumentPage]:
    doc_cache = ensure_dir(document_cache_dir(cache_dir, doc_id, render_dpi))
    pages_json = doc_cache / "pages.json"
    pages_dir = doc_cache / "pages"

    if pages_json.exists():
        raw_pages = load_json(pages_json)
        pages = [DocumentPage.model_validate(item) if hasattr(DocumentPage, "model_validate") else DocumentPage.parse_obj(item) for item in raw_pages]
        if pages and all(Path(page.image_path).exists() for page in pages):
            return pages

    pages = build_document_pages(pdf_path, str(pages_dir), dpi=render_dpi)
    save_json(pages_json, pages)
    return pages


def load_cached_document_pages(
    doc_id: str,
    cache_dir: str | Path,
    render_dpi: int,
    *,
    legacy_cache_dirs: list[str | Path] | None = None,
) -> list[DocumentPage] | None:
    candidate_dirs = [document_cache_dir(cache_dir, doc_id, render_dpi), legacy_document_cache_dir(cache_dir, doc_id)]
    for legacy_cache_dir in legacy_cache_dirs or []:
        candidate_dirs.extend(
            [
                document_cache_dir(legacy_cache_dir, doc_id, render_dpi),
                legacy_document_cache_dir(legacy_cache_dir, doc_id),
            ]
        )

    for candidate_dir in candidate_dirs:
        pages_json = candidate_dir / "pages.json"
        if not pages_json.exists():
            continue
        raw_pages = load_json(pages_json)
        pages = [DocumentPage.model_validate(item) if hasattr(DocumentPage, "model_validate") else DocumentPage.parse_obj(item) for item in raw_pages]
        if pages and all(Path(page.image_path).exists() for page in pages):
            return pages
    return None


def load_cached_retrieval(
    cache_dir: str | Path,
    example: MMLongBenchExample,
    retriever_name: str,
    top_k: int,
    render_dpi: int,
) -> list[RetrievedPage] | None:
    path = retrieval_cache_path(cache_dir, example, retriever_name, top_k, render_dpi)
    if not path.exists():
        return None
    raw_pages = load_json(path)
    return [RetrievedPage.model_validate(item) if hasattr(RetrievedPage, "model_validate") else RetrievedPage.parse_obj(item) for item in raw_pages]
