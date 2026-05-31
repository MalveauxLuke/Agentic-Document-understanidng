from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _split_qid_values(values: Iterable[str] | None) -> list[str]:
    qids: list[str] = []
    for value in values or []:
        for piece in str(value).split(","):
            qid = piece.strip()
            if qid:
                qids.append(qid)
    return qids


def _qids_from_file(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    qid_path = Path(path)
    text = qid_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        for key in ("qids", "question_ids", "question_id"):
            if key in parsed:
                value = parsed[key]
                if isinstance(value, list):
                    return [str(item).strip() for item in value if str(item).strip()]
                return _split_qid_values([str(value)])
        return []
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if parsed is not None:
        return _split_qid_values([str(parsed)])

    qids: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        qids.extend(_split_qid_values([line]))
    return qids


def resolve_qids(values: Iterable[str] | None = None, qid_file: str | Path | None = None) -> set[str] | None:
    qids = [*_split_qid_values(values), *_qids_from_file(qid_file)]
    return set(qids) if qids else None
