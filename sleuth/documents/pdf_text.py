from __future__ import annotations

from pathlib import Path

import fitz


def extract_pdf_text_by_page(pdf_path: str) -> list[str]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        document = fitz.open(path)
    except Exception as exc:
        raise ValueError(f"Could not open PDF '{path}': {exc}") from exc

    texts: list[str] = []
    try:
        for page_index in range(document.page_count):
            texts.append(document.load_page(page_index).get_text("text"))
    finally:
        document.close()
    return texts
