from __future__ import annotations

from typing import Any

from sleuth.llm.base import LLMClient


class QwenVLClient(LLMClient):
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.default_max_new_tokens = max_new_tokens

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen-VL mode requires torch, transformers, accelerate, and qwen-vl-utils. "
                "Install the project environment first; mock mode can still test wiring."
            ) from exc

        self.torch = torch
        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        try:
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            model_kwargs = {"dtype": torch_dtype}
            if device == "cuda":
                model_kwargs["device_map"] = "auto"
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name_or_path,
                    **model_kwargs,
                )
            except TypeError:
                model_kwargs = {"torch_dtype": torch_dtype}
                if device == "cuda":
                    model_kwargs["device_map"] = "auto"
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name_or_path,
                    **model_kwargs,
                )
            if device != "cuda":
                self.model = self.model.to(device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Qwen/Qwen3-VL. GPU memory may be insufficient, "
                "transformers/qwen-vl-utils may need updating, the model identifier "
                "may need correction, or you can use mock mode to test wiring."
            ) from exc

    def _normalize_messages(self, messages: list[dict], images: list[str] | None) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, list):
                normalized_content = content
            else:
                normalized_content = [{"type": "text", "text": str(content)}]
            normalized.append({"role": role, "content": normalized_content})

        if not normalized:
            normalized.append({"role": "user", "content": []})
        if images:
            image_blocks = [{"type": "image", "image": image_path} for image_path in images]
            normalized[-1]["content"] = image_blocks + list(normalized[-1]["content"])
        return normalized

    def chat(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = None,
    ) -> str:
        chat_messages = self._normalize_messages(messages, images)
        try:
            inputs = self.processor.apply_chat_template(
                chat_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to prepare Qwen-VL multimodal messages. Check that transformers "
                "and qwen-vl-utils support this model and message format."
            ) from exc

        target_device = getattr(self.model, "device", self.device)
        inputs = inputs.to(target_device)
        input_len = inputs["input_ids"].shape[-1]
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.default_max_new_tokens,
            "do_sample": temperature > 0.1,
        }
        if temperature > 0.1:
            generation_kwargs["temperature"] = temperature

        with self.torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )
        return generated_text[0].strip()
