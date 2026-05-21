from __future__ import annotations

from pathlib import Path


def read_markdown_file(path: str | Path) -> str:
    md_path = Path(path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")
    text = md_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Markdown file is empty: {md_path}")
    return text


def save_instruction_copy(text: str, out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
