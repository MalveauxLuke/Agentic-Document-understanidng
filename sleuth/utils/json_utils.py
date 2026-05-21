from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _try_parse_object(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _scan_first_json_object(text: str) -> dict | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    parsed = _try_parse_object(candidate)
                    if parsed is not None:
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


def extract_json_from_text(text: str) -> dict | None:
    if not text:
        return None
    stripped = text.strip()
    parsed = _try_parse_object(stripped)
    if parsed is not None:
        return parsed

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE):
        parsed = _try_parse_object(match.group(1).strip())
        if parsed is not None:
            return parsed

    return _scan_first_json_object(stripped)


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    if isinstance(obj, list):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    return obj


def save_json(path: str | Path, obj: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_to_jsonable(obj), indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
