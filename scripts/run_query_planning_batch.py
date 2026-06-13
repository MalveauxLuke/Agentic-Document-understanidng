#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
REQUESTED_GENERATION_PARAMETERS = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 200,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def render_prompt(template: str, question: str) -> str:
    if template.count("{question}") != 1:
        raise ValueError("Exact query-planning prompt must contain exactly one {question} placeholder")
    return template.replace("{question}", question)


def extract_json(raw_output: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw_output.strip()
    if text.startswith("```json"):
        text = text[len("```json") :]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, None
    return None, "Model output was not a valid JSON object"


class QwenQueryPlanner:
    def __init__(self, model_name: str, device: str, dtype: str) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError("Query planning requires torch and transformers") from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)
        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        model_kwargs: dict[str, Any] = {"dtype": torch_dtype}
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        except TypeError:
            model_kwargs.pop("dtype", None)
            model_kwargs["torch_dtype"] = torch_dtype
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        target_device = getattr(self.model, "device", "cuda")
        inputs = inputs.to(target_device)
        input_length = inputs["input_ids"].shape[-1]

        # temperature=0 selects greedy decoding. top_p=1 and zero frequency/
        # presence penalties are neutral and therefore require no transformation.
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                top_p=REQUESTED_GENERATION_PARAMETERS["top_p"],
                max_new_tokens=REQUESTED_GENERATION_PARAMETERS["max_tokens"],
            )
        return self.processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
        )[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate query plans for 20 cached MMLongBench questions")
    parser.add_argument("--questions-json", default="colpali_v1_canvas_test_corpus/question_list.json")
    parser.add_argument("--prompt-file", default="prompts/query_planning_exact.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--qid", action="append", default=None, help="Optional QID filter; repeatable/comma-compatible")
    parser.add_argument("--prepare-only", action="store_true", help="Write prompts without loading Qwen")
    return parser.parse_args()


def parse_qids(values: list[str] | None) -> set[str]:
    return {
        piece.strip()
        for value in values or []
        for piece in value.split(",")
        if piece.strip()
    }


def main() -> None:
    args = parse_args()
    questions_path = Path(args.questions_json).resolve()
    prompt_path = Path(args.prompt_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    template = prompt_path.read_text(encoding="utf-8")
    if template.endswith("\n"):
        template = template[:-1]
    questions = read_json(questions_path)
    qids = parse_qids(args.qid)
    if qids:
        questions = [question for question in questions if str(question["question_id"]) in qids]
    questions = questions[: max(0, args.limit)]
    if not questions:
        raise ValueError("No questions matched the requested filters")

    prepared = [
        {
            "question_id": str(question["question_id"]),
            "question": question["question"],
            "modalities": question.get("modalities", []),
            "gold_display_pages": question.get("gold_display_pages", []),
            "prompt": render_prompt(template, str(question["question"])),
        }
        for question in questions
    ]
    write_json(output_dir / "prepared_prompts.json", prepared)

    prompt_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if args.prepare_only:
        manifest = {
            "mode": "prepare_only",
            "question_count": len(prepared),
            "model": args.model,
            "prompt_file": str(prompt_path),
            "prompt_sha256": prompt_sha256,
            "generation_parameters": REQUESTED_GENERATION_PARAMETERS,
        }
        write_json(output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
        return

    planner = QwenQueryPlanner(args.model, args.device, args.dtype)
    results = []
    jsonl_path = output_dir / "query_plans.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for index, item in enumerate(prepared, start=1):
            print(f"[query-planning] {index}/{len(prepared)} qid={item['question_id']}", flush=True)
            raw_output = planner.generate(item["prompt"])
            parsed_output, parse_error = extract_json(raw_output)
            result = {
                **item,
                "model": args.model,
                "generation_parameters": REQUESTED_GENERATION_PARAMETERS,
                "raw_output": raw_output,
                "parsed_output": parsed_output,
                "parse_error": parse_error,
            }
            results.append(result)
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            jsonl.flush()

    write_json(output_dir / "query_plans.json", results)
    manifest = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(results),
        "valid_json_count": sum(result["parsed_output"] is not None for result in results),
        "invalid_json_count": sum(result["parsed_output"] is None for result in results),
        "model": args.model,
        "model_mode": "non-thinking instruct",
        "prompt_file": str(prompt_path),
        "prompt_sha256": prompt_sha256,
        "generation_parameters": REQUESTED_GENERATION_PARAMETERS,
        "effective_transformers_generation": {
            "do_sample": False,
            "top_p": REQUESTED_GENERATION_PARAMETERS["top_p"],
            "max_new_tokens": REQUESTED_GENERATION_PARAMETERS["max_tokens"],
            "frequency_penalty": "neutral at 0",
            "presence_penalty": "neutral at 0",
        },
        "results_json": str(output_dir / "query_plans.json"),
        "results_jsonl": str(jsonl_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
