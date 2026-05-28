from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sleuth.utils.json_utils import extract_json_from_text


@dataclass(frozen=True)
class AnswerExtractionResult:
    raw_answer: str
    extracted_answer: str
    extractor: str
    paper_comparable: bool
    error: str | None = None


class AnswerExtractor(ABC):
    name = "base"
    paper_comparable = False

    @abstractmethod
    def extract(self, question: str, raw_answer: str, answer_format: str | None = None) -> AnswerExtractionResult:
        raise NotImplementedError


def _answer_from_json(text: str) -> str | None:
    data = extract_json_from_text(text)
    if isinstance(data, dict) and "answer" in data:
        return str(data["answer"]).strip()
    return None


def _clean_candidate(text: str) -> str:
    candidate = text.strip()
    candidate = re.sub(r"^```(?:\w+)?\s*", "", candidate)
    candidate = re.sub(r"\s*```$", "", candidate).strip()
    if candidate.lower().startswith("answer:"):
        candidate = candidate.split(":", 1)[1].strip()
    if candidate.lower().startswith("your answer:"):
        candidate = candidate.split(":", 1)[1].strip()
    return candidate.strip().strip("\"'")


class NoneAnswerExtractor(AnswerExtractor):
    name = "none"
    paper_comparable = False

    def extract(self, question: str, raw_answer: str, answer_format: str | None = None) -> AnswerExtractionResult:
        extracted = _answer_from_json(raw_answer) or _clean_candidate(raw_answer)
        return AnswerExtractionResult(raw_answer, extracted, self.name, self.paper_comparable)


class HeuristicAnswerExtractor(AnswerExtractor):
    name = "heuristic"
    paper_comparable = False

    def extract(self, question: str, raw_answer: str, answer_format: str | None = None) -> AnswerExtractionResult:
        json_answer = _answer_from_json(raw_answer)
        if json_answer is not None:
            return AnswerExtractionResult(raw_answer, _clean_candidate(json_answer), self.name, self.paper_comparable)

        text = raw_answer.strip()
        if not text:
            return AnswerExtractionResult(raw_answer, "No answers found!", self.name, self.paper_comparable)
        if "No answers found!" in text:
            return AnswerExtractionResult(raw_answer, "No answers found!", self.name, self.paper_comparable)

        for pattern in (
            r"(?im)^\s*(?:final\s+answer|answer|your\s+answer)\s*:\s*(.+)$",
            r"(?im)^\s*[-*]\s*(?:answer\s*)?:\s*(.+)$",
        ):
            match = re.search(pattern, text)
            if match:
                return AnswerExtractionResult(raw_answer, _clean_candidate(match.group(1)), self.name, self.paper_comparable)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) == 1:
            return AnswerExtractionResult(raw_answer, _clean_candidate(lines[0]), self.name, self.paper_comparable)

        bullet_lines = [line for line in lines if line.startswith(("-", "*", "•"))]
        if bullet_lines:
            extracted = [re.sub(r"^[-*•]\s*", "", line).strip() for line in bullet_lines]
            return AnswerExtractionResult(raw_answer, str(extracted), self.name, self.paper_comparable)

        return AnswerExtractionResult(raw_answer, _clean_candidate(lines[0]), self.name, self.paper_comparable)


class OpenAICompatibleAnswerExtractor(AnswerExtractor):
    name = "openai_compatible"
    paper_comparable = True

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def extract(self, question: str, raw_answer: str, answer_format: str | None = None) -> AnswerExtractionResult:
        try:
            import requests
        except ImportError as exc:
            return AnswerExtractionResult(
                raw_answer,
                HeuristicAnswerExtractor().extract(question, raw_answer, answer_format).extracted_answer,
                self.name,
                False,
                f"requests unavailable: {exc}",
            )

        prompt = (
            "Extract the shortest final answer from the model output for document QA scoring.\n"
            "Return only the answer string. Do not explain. If the model output says the answer "
            "is not found, return exactly: No answers found!\n\n"
            f"Question: {question}\n"
            f"Answer format: {answer_format or 'unknown'}\n"
            f"Model output:\n{raw_answer}\n\n"
            "Short answer:"
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 128,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            extracted = data["choices"][0]["message"]["content"]
            return AnswerExtractionResult(raw_answer, _clean_candidate(str(extracted)), self.name, self.paper_comparable)
        except Exception as exc:
            heuristic = HeuristicAnswerExtractor().extract(question, raw_answer, answer_format)
            return AnswerExtractionResult(raw_answer, heuristic.extracted_answer, self.name, False, str(exc))


def has_openai_compatible_answer_extractor_env() -> bool:
    return bool(
        os.environ.get("ANSWER_EXTRACTOR_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


def build_answer_extractor(kind: str, mode: str = "sol") -> AnswerExtractor:
    selected = kind
    if kind == "auto":
        selected = "openai_compatible" if mode != "mock" and has_openai_compatible_answer_extractor_env() else "heuristic"
    if selected == "none":
        return NoneAnswerExtractor()
    if selected == "heuristic":
        return HeuristicAnswerExtractor()
    if selected != "openai_compatible":
        raise ValueError(f"Unsupported answer extractor: {kind}")

    api_key = (
        os.environ.get("ANSWER_EXTRACTOR_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        return HeuristicAnswerExtractor()
    base_url = (
        os.environ.get("ANSWER_EXTRACTOR_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.deepseek.com"
    )
    model = (
        os.environ.get("ANSWER_EXTRACTOR_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "deepseek-chat"
    )
    return OpenAICompatibleAnswerExtractor(api_key=api_key, base_url=base_url, model=model)
