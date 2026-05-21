from __future__ import annotations

from pathlib import Path

import fitz


def render_pdf_pages(pdf_path: str, out_dir: str, dpi: int = 144) -> list[str]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        document = fitz.open(path)
    except Exception as exc:
        raise ValueError(f"Could not open PDF '{path}': {exc}") from exc

    image_paths: list[str] = []
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"page_{page_index + 1:04d}.png"
            pixmap.save(image_path)
            image_paths.append(str(image_path))
    finally:
        document.close()
    return image_paths
