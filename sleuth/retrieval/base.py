from __future__ import annotations

from abc import ABC, abstractmethod

from sleuth.schemas import DocumentPage, RetrievedPage


class BaseRetriever(ABC):
    @abstractmethod
    def index(self, document_pages: list[DocumentPage]) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedPage]:
        raise NotImplementedError
