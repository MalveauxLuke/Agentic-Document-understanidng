from __future__ import annotations

from sleuth.documents.pdf_renderer import render_pdf_pages
from sleuth.documents.pdf_text import extract_pdf_text_by_page
from sleuth.schemas import DocumentPage


def build_document_pages(pdf_path: str, pages_dir: str, dpi: int = 144) -> list[DocumentPage]:
    image_paths = render_pdf_pages(pdf_path, pages_dir, dpi=dpi)
    page_texts = extract_pdf_text_by_page(pdf_path)
    if len(image_paths) != len(page_texts):
        raise ValueError(
            f"Rendered page count ({len(image_paths)}) does not match text page count ({len(page_texts)})."
        )
    return [
        DocumentPage(page_index=index, image_path=image_path, text=page_texts[index])
        for index, image_path in enumerate(image_paths)
    ]
