from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError
