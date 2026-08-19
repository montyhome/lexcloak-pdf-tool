"""Page rendering -- PNG bytes."""
from __future__ import annotations

import pymupdf as _pymupdf

from .redact import open_pdf

Matrix = _pymupdf.Matrix


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


def render_clip(pdf_bytes: bytes, page_num: int, clip, dpi: float = 150,
                gray: bool = True) -> bytes:
    """Render only ``clip`` of ``page_num`` to PNG bytes at ``dpi``.

    ``clip`` is ``(x0, y0, x1, y1)`` in PDF points. Rendering the clip is
    NOT the same raster as rendering the whole page and cropping: MuPDF
    aligns the output grid to the clip's own (possibly fractional) origin,
    so the two agree only when the clip lands on an integer pixel boundary.
    Callers verifying a region against what a clip render produced must use
    this op rather than composing from :func:`render_page`.

    ``gray`` selects a single-channel grayscale colorspace. It is the
    default because the callers that need a clip are feeding image
    decoders, which want luminance and not colour.

    Raises ``IndexError`` for an out-of-range page and ``ValueError`` for a
    degenerate clip (zero or negative width/height after clamping to the
    page), so a caller cannot silently receive an empty image.
    """
    doc = open_pdf(pdf_bytes)
    try:
        if doc.needs_pass:
            # Renders as blank rather than failing. A blank crop handed to
            # an image decoder reads as "nothing here", which is the wrong
            # answer for every caller that has one.
            raise ValueError(
                "cannot render an encrypted document (needs_pass); "
                "decrypt before calling render_clip")
        if page_num < 0 or page_num >= len(doc):
            raise IndexError(
                f"page_num {page_num} out of range for {len(doc)}-page document"
            )
        page = doc[page_num]
        rect = _pymupdf.Rect(*clip) & page.rect
        if rect.is_empty or rect.width <= 0 or rect.height <= 0:
            raise ValueError(
                f"clip {tuple(clip)} is empty after clamping to page rect "
                f"{tuple(page.rect)}"
            )
        cs = _pymupdf.csGRAY if gray else _pymupdf.csRGB
        pix = page.get_pixmap(clip=rect, dpi=int(dpi), colorspace=cs)
        return pix.tobytes("png")
    finally:
        doc.close()
