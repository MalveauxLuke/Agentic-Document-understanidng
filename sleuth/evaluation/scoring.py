from __future__ import annotations

import ast
import re
from math import isclose
from typing import Any


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth: str, prediction: str, threshold: float = 0.5) -> float:
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


def _precision(value: float) -> int:
    precision = 3
    if "." in str(value):
        precision = len(str(value).split(".")[-1])
    return precision


def is_float_equal(reference: Any, prediction: Any, include_percentage: bool = False, is_close: bool = False) -> bool:
    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except Exception:
        return False

    candidates = [reference / 100, reference, reference * 100] if include_percentage else [reference]
    for item in candidates:
        try:
            if is_close and isclose(item, prediction, rel_tol=0.01):
                return True
            precision = max(min(_precision(prediction), _precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(value: Any) -> str:
    s = str(value).lower().strip()
    for suffix in ("mile", "miles", "million"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    return s


def _is_no_answer(value: Any) -> bool:
    s = str(value).lower().strip()
    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    aliases = {
        "no answer",
        "no answers found",
        "not answerable",
        "unanswerable",
        "insufficient data",
        "insufficient information",
        "answer not found",
        "cannot be answered",
    }
    return s in aliases


def is_exact_match(value: str) -> bool:
    s = str(value)
    if "https://" in s:
        return True
    if s.endswith(".py") or s.endswith("ipynb"):
        return True
    if s.startswith("page"):
        return True
    if re.fullmatch(r"\b\d+(-\d+|\s\d+)?\b", s):
        return True
    if "a.m." in s or "p.m." in s:
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}[-\s]\d{2}\b", s):
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}\b", s):
        return True
    if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", s):
        return True
    return False


def isfloat(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _maybe_parse_list(value: Any) -> list[Any]:
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError):
            parsed = value
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    if isinstance(value, list):
        return value
    return [value]


def _split_list_phrase(value: Any, expected_len: int | None = None) -> list[Any]:
    parsed = _maybe_parse_list(value)
    if len(parsed) != 1 or not isinstance(parsed[0], str):
        return parsed
    text = parsed[0].strip()
    if not text:
        return [text]

    parts: list[str] = []
    if "\n" in text:
        parts = [re.sub(r"^[-*•]\s*", "", line).strip() for line in text.splitlines()]
    elif "," in text or ";" in text:
        parts = re.split(r"\s*(?:,|;)\s*", text)
    elif expected_len and expected_len > 1 and " and " in text.lower():
        parts = re.split(r"\s+and\s+", text, flags=re.IGNORECASE)

    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) > 1:
        return parts
    return parsed


def _clean_list_item(value: Any) -> str:
    item = get_clean_string(value)
    item = re.sub(r"\bpercentage\s+points?\b", "", item)
    item = re.sub(r"\bpercent\b", "", item)
    item = re.sub(r"\bpoints?\b", "", item)
    item = re.sub(r"\s+", " ", item).strip()
    return item


def eval_score(gt: Any, pred: Any, answer_type: str) -> float:
    if answer_type == "Int":
        try:
            gt_int = int(gt)
            pred_int = int(float(pred))
        except Exception:
            pred_int = ""
            gt_int = gt
        score = gt_int == pred_int
    elif answer_type == "Float":
        try:
            gt_float = float(get_clean_string(str(gt)))
            pred_float = float(get_clean_string(str(pred)))
        except Exception:
            pred_float = ""
            gt_float = gt
        score = is_float_equal(gt_float, pred_float, include_percentage=True, is_close=True)
    elif answer_type == "None":
        if _is_no_answer(gt) and _is_no_answer(pred):
            score = True
        elif _is_no_answer(gt) or _is_no_answer(pred):
            score = False
        else:
            gt_clean = get_clean_string(gt)
            pred_clean = get_clean_string(pred)
            score = (gt_clean == pred_clean) if is_exact_match(gt_clean) else anls_compute(gt_clean, pred_clean)
    elif answer_type == "Str":
        gt_clean = get_clean_string(gt)
        pred_clean = get_clean_string(pred)
        score = (gt_clean == pred_clean) if is_exact_match(gt_clean) else anls_compute(gt_clean, pred_clean)
    else:
        gt_list = _split_list_phrase(gt)
        pred_list = _split_list_phrase(pred, expected_len=len(gt_list))
        if len(gt_list) != len(pred_list):
            score = 0.0
        else:
            gt_clean_list = sorted([_clean_list_item(item) for item in gt_list])
            pred_clean_list = sorted([_clean_list_item(item) for item in pred_list])
            if not gt_clean_list:
                score = 0.0
            elif isfloat(gt_clean_list[0]) or is_exact_match(gt_clean_list[0]):
                score = "-".join(gt_clean_list) == "-".join(pred_clean_list)
            else:
                score = min(
                    [anls_compute(gt_value, pred_value) for gt_value, pred_value in zip(gt_clean_list, pred_clean_list)]
                )
    return float(score)
