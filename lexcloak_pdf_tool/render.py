"""Page rendering -- PNG bytes."""
from __future__ import annotations

import fitz as _fitz

from .redact import open_pdf

Matrix = _fitz.Matrix


def _render_page_doc(doc, page_num: int, dpi: float = 150) -> bytes:
    if page_num < 0 or page_num >= len(doc):
        raise IndexError(
            f"page_num {page_num} out of range for {len(doc)}-page document"
        )
    page = doc[page_num]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=Matrix(zoom, zoom))
    return pix.tobytes("png")


def render_page(pdf_bytes: bytes, page_num: int, dpi: float = 150) -> bytes:
    """Render ``page_num`` of ``pdf_bytes`` to PNG bytes at ``dpi``.

    Default 150 DPI is a reasonable preview size. Use 300 DPI when
    rendering for OCR (Tesseract benefits from higher resolution at page
    margins).
    """
    doc = open_pdf(pdf_bytes)
    try:
        return _render_page_doc(doc, page_num, dpi)
    finally:
        doc.close()
