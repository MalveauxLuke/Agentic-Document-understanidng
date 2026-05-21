from __future__ import annotations

import numpy as np

from sleuth.retrieval.base import BaseRetriever
from sleuth.schemas import DocumentPage, RetrievedPage


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25TextRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.document_pages: list[DocumentPage] = []
        self.bm25 = None

    def index(self, document_pages: list[DocumentPage]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "BM25 retrieval requires rank-bm25. Install it with: pip install rank-bm25"
            ) from exc

        self.document_pages = list(document_pages)
        corpus = [_tokenize(page.text or "") for page in self.document_pages]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedPage]:
        if self.bm25 is None:
            raise RuntimeError("BM25TextRetriever.index() must be called before retrieve().")
        if not self.document_pages:
            return []

        scores = np.asarray(self.bm25.get_scores(_tokenize(question)), dtype=float)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedPage(
                page_index=self.document_pages[int(index)].page_index,
                score=float(scores[int(index)]),
                reason="bm25 text score",
            )
            for index in ranked_indices
        ]
