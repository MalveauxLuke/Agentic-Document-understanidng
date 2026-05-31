from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

from sleuth.evaluation.cache_paths import (
    colpali_embedding_cache_path,
    document_cache_dir,
    retrieval_cache_path,
)
from sleuth.evaluation.dataset import MMLongBenchExample
from sleuth.retrieval.colpali_retriever import ColPaliRetriever
from sleuth.schemas import DocumentPage


def _example(question: str = "Question?") -> MMLongBenchExample:
    return MMLongBenchExample(
        question_id="13",
        row_index=0,
        doc_id="doc.pdf",
        doc_type="Report",
        question=question,
        answer="answer",
        evidence_pages=[],
        source_evidence_pages=[],
        evidence_sources=[],
        answer_format="Str",
        pdf_path="/tmp/doc.pdf",
        categories=["Pure-text"],
    )


def test_dpi_aware_document_and_retrieval_cache_paths(tmp_path):
    assert document_cache_dir(tmp_path, "doc.pdf", 144) == tmp_path / "documents" / "dpi_144" / "doc_pdf"

    path_144 = retrieval_cache_path(tmp_path, _example(), "vidore/colpali-v1.3-hf", 5, 144)
    path_180 = retrieval_cache_path(tmp_path, _example(), "vidore/colpali-v1.3-hf", 5, 180)
    path_top10 = retrieval_cache_path(tmp_path, _example(), "vidore/colpali-v1.3-hf", 10, 144)
    path_other_model = retrieval_cache_path(tmp_path, _example(), "other", 5, 144)

    assert path_144 != path_180
    assert path_144 != path_top10
    assert path_144 != path_other_model
    assert "dpi_144" in str(path_144)


class FakeTensor:
    def __init__(self, value: str) -> None:
        self.value = value
        self.device = "cpu"

    def detach(self):
        return self

    def cpu(self):
        self.device = "cpu"
        return self

    def to(self, device):
        self.device = str(device)
        return self


class FakeTorch:
    @staticmethod
    def save(payload, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str | Path, map_location=None):
        _ = map_location
        with Path(path).open("rb") as f:
            return pickle.load(f)


def _fake_retriever(index_calls: list[int]) -> ColPaliRetriever:
    retriever = object.__new__(ColPaliRetriever)
    retriever.model_name_or_path = "vidore/colpali-v1.3-hf"
    retriever.device = "cuda"
    retriever.top_k_default = 5
    retriever.document_pages = []
    retriever.image_embeddings = None
    retriever.backend = "transformers"
    retriever.torch = FakeTorch
    retriever.model = SimpleNamespace(device="cuda:0")

    def fake_index(document_pages):
        index_calls.append(1)
        retriever.document_pages = list(document_pages)
        retriever.image_embeddings = FakeTensor("encoded-pages")

    retriever.index = fake_index
    return retriever


def test_colpali_index_with_cache_saves_then_loads_without_reencoding(tmp_path):
    pages = [DocumentPage(page_index=0, image_path="/tmp/page.png")]
    cache_path = colpali_embedding_cache_path(tmp_path, "vidore/colpali-v1.3-hf", 144, "doc.pdf")

    first_calls: list[int] = []
    first = _fake_retriever(first_calls)
    first.index_with_cache(pages, cache_path)

    second_calls: list[int] = []
    second = _fake_retriever(second_calls)
    second.index_with_cache(pages, cache_path)

    assert first_calls == [1]
    assert second_calls == []
    assert second.image_embeddings.value == "encoded-pages"
    assert second.image_embeddings.device == "cuda:0"
