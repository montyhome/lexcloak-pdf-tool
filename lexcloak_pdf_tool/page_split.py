"""Page-range extraction — one contiguous slice of a PDF as a new PDF.

v6 (0.7.0). Built for callers that need to split an over-sized document
into scannable parts while preserving provenance: the slice carries the
source's bookmark outline re-based to the part's local page numbers, so
a citation like "page 437" in the master remains resolvable through the
part's own outline. Page content is copied by ``insert_pdf`` untouched —
no re-rendering, no recompression beyond the save-time ``garbage=3,
deflate=True`` pass.

Page numbers here are 0-based and the range is INCLUSIVE on both ends,
matching every other per-page op in this package. Callers presenting
ranges to humans are expected to convert to 1-based themselves.

v13 (0.15.0) adds an optional ``first_page_label``: the part's page labels
(the numbers a viewer shows, separate from a page's position in the file)
become plain decimal numbers starting there, so the part's pages can read
"93, 94, ..." where they sat in the source. The labels carry no prefix, so
they hold no text. Without it the part carries no labels, as before:
``insert_pdf`` copies pages, not the source catalog's ``/PageLabels``.
"""
from __future__ import annotations

import pymupdf


def _rebase_toc(toc: list, from_page: int, to_page: int) -> list:
    """Slice a ``get_toc(simple=True)`` outline to a 0-based page range.

    Keeps entries whose destination page falls inside the range, re-based
    to the part's local 1-based numbering. Entries without a usable
    destination (``page < 1`` — broken or external links) are dropped:
    they carry no in-document target to preserve.

    ``set_toc`` refuses a hierarchy that starts deeper than level 1 or
    jumps more than one level between consecutive entries — both happen
    naturally when a slice cuts children away from their parents. Levels
    are therefore clamped to the deepest legal value, preserving relative
    nesting where the slice kept it.
    """
    sliced = []
    for entry in toc:
        level, title, page = entry[0], entry[1], entry[2]
        page0 = page - 1
        if page < 1 or not (from_page <= page0 <= to_page):
            continue
        sliced.append([level, title, page0 - from_page + 1])
    prev_level = 0
    for entry in sliced:
        if entry[0] > prev_level + 1:
            entry[0] = prev_level + 1
        prev_level = entry[0]
    return sliced


def validate_first_page_label(value) -> int | None:
    """``None`` (absent) or an integer of at least 1; anything else raises.

    ``bool`` is refused although it is an ``int`` subclass: ``True`` is not a
    page number. A float, even a whole one, is refused rather than truncated.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("first_page_label must be an integer of at least 1")
    if value < 1:
        raise ValueError("first_page_label must be an integer of at least 1")
    return value


def extract_pages_from_doc(src: "pymupdf.Document", from_page: int,
                           to_page: int,
                           first_page_label: int | None = None) -> bytes:
    """The document-object core — shared by the bytes and handle ops.

    Never closes ``src`` (the handle op's cached document must survive).
    Raises ``ValueError`` on an out-of-range or inverted page range, or a
    ``first_page_label`` that is not an integer of at least 1.
    """
    first_page_label = validate_first_page_label(first_page_label)
    n = src.page_count
    if not (0 <= from_page <= to_page < n):
        raise ValueError(
            f"page range [{from_page}, {to_page}] invalid for a "
            f"{n}-page document (0-based, inclusive, from <= to)"
        )
    out = pymupdf.open()
    try:
        out.insert_pdf(src, from_page=from_page, to_page=to_page)
        sliced = _rebase_toc(src.get_toc(simple=True), from_page, to_page)
        if sliced:
            out.set_toc(sliced)
        if first_page_label is not None:
            # One decimal range from the part's first page, no ``/P``.
            out.set_page_labels([{"startpage": 0, "prefix": "", "style": "D",
                                  "firstpagenum": first_page_label}])
        return out.tobytes(garbage=3, deflate=True)
    finally:
        out.close()


def extract_pages(pdf_bytes: bytes, from_page: int, to_page: int,
                  first_page_label: int | None = None) -> bytes:
    """Extract an inclusive 0-based page range as a standalone PDF."""
    first_page_label = validate_first_page_label(first_page_label)
    src = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return extract_pages_from_doc(src, from_page, to_page,
                                      first_page_label)
    finally:
        src.close()
