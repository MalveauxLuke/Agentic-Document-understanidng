from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests

from sleuth.llm.base import LLMClient


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY is required for OpenAI-compatible mode.")

    def _image_to_data_url(self, image_path: str) -> str:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _messages_with_images(self, messages: list[dict], images: list[str]) -> list[dict[str, Any]]:
        converted = [dict(message) for message in messages]
        if not converted:
            converted = [{"role": "user", "content": ""}]
        last = converted[-1]
        text = last.get("content", "")
        if isinstance(text, list):
            content = list(text)
        else:
            content = [{"type": "text", "text": str(text)}]
        for image_path in images:
            content.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(image_path)}})
        last["content"] = content
        converted[-1] = last
        return converted

    def chat(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = None,
    ) -> str:
        request_messages = self._messages_with_images(messages, images) if images else messages
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "temperature": temperature,
        }
        if max_new_tokens is not None:
            payload["max_tokens"] = max_new_tokens

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"OpenAI-compatible request failed: {response.status_code} {response.text}")
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenAI-compatible response shape: {data}") from exc
