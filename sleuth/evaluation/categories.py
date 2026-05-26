from __future__ import annotations


CANONICAL_CATEGORIES = ["Chart", "Table", "Pure-text", "Layout", "Figure", "None"]


def normalize_evidence_source(source: str) -> str:
    value = str(source).strip()
    if not value:
        return "None"
    lowered = value.lower()
    if "chart" in lowered:
        return "Chart"
    if "table" in lowered:
        return "Table"
    if "pure-text" in lowered or "plain-text" in lowered or lowered == "text":
        return "Pure-text"
    if "layout" in lowered or "generalized-text" in lowered:
        return "Layout"
    if "figure" in lowered or "image" in lowered:
        return "Figure"
    return value


def normalize_categories(evidence_sources: list[str]) -> list[str]:
    if not evidence_sources:
        return ["None"]
    categories: list[str] = []
    for source in evidence_sources:
        category = normalize_evidence_source(source)
        if category not in categories:
            categories.append(category)
    return categories or ["None"]
