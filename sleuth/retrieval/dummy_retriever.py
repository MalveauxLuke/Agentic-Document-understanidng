from __future__ import annotations

from sleuth.retrieval.base import BaseRetriever
from sleuth.schemas import DocumentPage, RetrievedPage


class DummyRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.document_pages: list[DocumentPage] = []

    def index(self, document_pages: list[DocumentPage]) -> None:
        self.document_pages = list(document_pages)

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedPage]:
        selected = self.document_pages[:top_k]
        return [
            RetrievedPage(
                page_index=page.page_index,
                score=float(len(selected) - rank),
                reason="dummy retriever fallback",
            )
            for rank, page in enumerate(selected)
        ]
