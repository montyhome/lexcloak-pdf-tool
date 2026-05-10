"""PDF metadata probes -- page count, page size, encryption state.

Cheap inspection ops for callers that need page count, page size, or
encryption state without touching the document body.
"""
from __future__ import annotations

from .redact import open_pdf


def page_count(pdf_bytes: bytes) -> int:
    """Return the number of pages in ``pdf_bytes``."""
    doc = open_pdf(pdf_bytes)
    try:
        return len(doc)
    finally:
        doc.close()


def page_size(pdf_bytes: bytes, page_num: int) -> tuple[float, float]:
    """Return ``(width, height)`` of ``page_num`` in PDF point-space (72 DPI)."""
    doc = open_pdf(pdf_bytes)
    try:
        if page_num < 0 or page_num >= len(doc):
            raise IndexError(
                f"page_num {page_num} out of range for {len(doc)}-page document"
            )
        rect = doc[page_num].rect
        return float(rect.width), float(rect.height)
    finally:
        doc.close()


def all_page_sizes(pdf_bytes: bytes) -> list[tuple[float, float]]:
    """Return ``[(width, height), ...]`` for every page in PDF point-space (72 DPI).

    Batch counterpart to :func:`page_size`. Opens the document once,
    iterates every page, returns the list. Replaces N sequential
    ``page_size(pdf_bytes, i)`` calls — load-bearing on long docs where
    the IPC re-shoveling cost of repeated full-PDF transfers dominates
    the actual page-size lookup.
    """
    doc = open_pdf(pdf_bytes)
    try:
        return [
            (float(doc[i].rect.width), float(doc[i].rect.height))
            for i in range(len(doc))
        ]
    finally:
        doc.close()


def is_encrypted(pdf_bytes: bytes) -> bool:
    """Return True if the PDF needs a non-empty password to read.

    PyMuPDF auto-authenticates empty-password PDFs on open (silent
    ``authenticate("")`` succeeds), so the common "encrypted-but-empty-pw"
    case reports False here. Used to distinguish "needs unlock UI" from
    "open but flagged as encrypted."
    """
    doc = open_pdf(pdf_bytes)
    try:
        return bool(doc.is_encrypted and doc.needs_pass)
    finally:
        doc.close()


def get_metadata(pdf_bytes: bytes) -> dict:
    """Return ``{"metadata": dict, "has_xmp": bool}``.

    Nested shape so callers can pass through ``metadata`` (a string-coerced
    dict with empty values dropped) without accidentally rendering
    ``has_xmp`` as a metadata field. ``has_xmp`` is a structural signal
    ("XMP exists, will be stripped"), not a metadata field itself; the
    XMP body is intentionally NOT returned.
    """
    doc = open_pdf(pdf_bytes)
    try:
        meta = doc.metadata or {}
        has_xmp = bool(doc.get_xml_metadata())
    finally:
        doc.close()
    fields = {k: str(v) for k, v in meta.items() if v is not None and v != ""}
    return {"metadata": fields, "has_xmp": has_xmp}
