"""Text extraction + coordinate search."""
from __future__ import annotations

import re

from .redact import Rect, open_pdf


# Maximum rect height -- caps oversized rects to prevent multi-line spans
# from creating giant redaction bars.
_MAX_OCR_LINE_H = 30.0


# ── IPC-clean API ────────────────────────────────────────────────────


def _extract_text_native_doc(doc, page_num: int) -> list[dict]:
    if page_num < 0 or page_num >= len(doc):
        raise IndexError(
            f"page_num {page_num} out of range for {len(doc)}-page document"
        )
    page = doc[page_num]
    words: list[dict] = []
    for x0, y0, x1, y1, text, block_no, line_no, word_no in page.get_text("words"):
        if not text.strip():
            continue
        words.append({
            "text": text,
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "block": int(block_no),
            "line": int(line_no),
            "word": int(word_no),
        })
    return words


def extract_text_native(pdf_bytes: bytes, page_num: int) -> list[dict]:
    """Return native PDF text words + bboxes for ``page_num``.

    Each word: ``{"text": str, "x0": float, "y0": float, "x1": float,
    "y1": float, "block": int, "line": int, "word": int}``.
    Whitespace-only tokens (PyMuPDF emits an empty token at end-of-line
    for some PDFs) are filtered.

    Native text is not always reliable for downstream detection -- span
    boundaries can concatenate without spaces (e.g.,
    "EMAIL ADDRESSmaria@example.com"). Prefer :func:`extract_text_ocr` for
    detection workflows.
    """
    doc = open_pdf(pdf_bytes)
    try:
        return _extract_text_native_doc(doc, page_num)
    finally:
        doc.close()


def _extract_text_ocr_doc(doc, page_num: int,
                          tessdata_path: str | None = None,
                          psm: int = 3) -> dict | None:
    from .ocr import OCR_DPI, _ocr_png_to_dict

    if page_num < 0 or page_num >= len(doc):
        raise IndexError(
            f"page_num {page_num} out of range for {len(doc)}-page document"
        )
    page = doc[page_num]
    try:
        pix = page.get_pixmap(dpi=OCR_DPI)
        png_bytes = pix.tobytes("png")
    except Exception:
        return None
    _ = tessdata_path  # advisory -- _ocr_png_to_dict uses module-level path
    return _ocr_png_to_dict(png_bytes, dpi=OCR_DPI, psm=psm)


def extract_text_ocr(pdf_bytes: bytes, page_num: int,
                     tessdata_path: str | None = None,
                     psm: int = 3) -> dict | None:
    """Run Tesseract OCR on a single page via the subprocess pipeline.

    Returns ``{"text": str, "chars": list[(char, bbox|None)],
    "spans": list[{"text", "bbox", "size"}]}`` in PDF point-space, or
    ``None`` if Tesseract is unavailable or OCR fails.

    ``tessdata_path`` defaults to the auto-resolved path discovered at
    module load -- pass an explicit path to override (e.g. for tests).
    ``psm=3`` (auto page segmentation) is the default and correctly
    handles forms, tables, sidebars, captions, and multi-column layouts.
    """
    doc = open_pdf(pdf_bytes)
    try:
        return _extract_text_ocr_doc(doc, page_num,
                                     tessdata_path=tessdata_path, psm=psm)
    finally:
        doc.close()


def _extract_text_dict_doc(doc, page_num: int) -> list[dict]:
    if page_num < 0 or page_num >= len(doc):
        raise IndexError(
            f"page_num {page_num} out of range for {len(doc)}-page document"
        )
    page = doc[page_num]
    page_dict = page.get_text("dict")
    blocks = page_dict.get("blocks", [])
    for block in blocks:
        if block.get("type") == 1 and "image" in block:
            del block["image"]
    return blocks


def extract_text_dict(pdf_bytes: bytes, page_num: int) -> list[dict]:
    """Return PyMuPDF's ``page.get_text("dict")`` block hierarchy for ``page_num``.

    Used for span-merge work (font-size boundaries) and column-text
    construction. Block / line / span shapes match PyMuPDF's native dict
    output exactly:

    * Block: ``{"type": int, "bbox": [x0,y0,x1,y1], "lines": [...]}``.
      ``type=0`` is text, ``type=1`` is image.
    * Line: ``{"bbox": [x0,y0,x1,y1], "spans": [...], "wmode": int,
      "dir": [dx,dy]}``.
    * Span: ``{"text": str, "bbox": [x0,y0,x1,y1], "size": float,
      "font": str, "color": int, "flags": int}``.

    Defensive: the ``"image": bytes`` field on ``type=1`` blocks is
    stripped before return -- base64-encoding multi-MB images per page
    would wreck the IPC frame budget on image-heavy PDFs.
    """
    doc = open_pdf(pdf_bytes)
    try:
        return _extract_text_dict_doc(doc, page_num)
    finally:
        doc.close()


def _extract_text_plain_doc(doc, page_num: int) -> str:
    if page_num < 0 or page_num >= len(doc):
        raise IndexError(
            f"page_num {page_num} out of range for {len(doc)}-page document"
        )
    return doc[page_num].get_text()


def extract_text_plain(pdf_bytes: bytes, page_num: int) -> str:
    """Return PyMuPDF's ``page.get_text()`` plain-text output for ``page_num``.

    Native text only -- not the OCR path.
    """
    doc = open_pdf(pdf_bytes)
    try:
        return _extract_text_plain_doc(doc, page_num)
    finally:
        doc.close()


def _search_for_doc(doc, page_num: int, needle: str,
                    ocr_chardata: list | None = None,
                    whole_word: bool = False,
                    split: bool = False) -> list:
    from .coords import (search_in_chars,
                         search_whole_word_in_chars,
                         split_search_in_chars)

    if ocr_chardata:
        if split:
            return split_search_in_chars(needle, ocr_chardata)
        if whole_word:
            return search_whole_word_in_chars(needle, ocr_chardata)
        return search_in_chars(needle, ocr_chardata)

    if page_num < 0 or page_num >= len(doc):
        raise IndexError(
            f"page_num {page_num} out of range for {len(doc)}-page document"
        )
    page = doc[page_num]
    if split:
        return _split_search(page, needle, None)
    if whole_word:
        return _page_search_whole_word(page, needle, None)
    rects = page.search_for(needle)
    return _cap_rects(rects) if rects else []


def search_for(pdf_bytes: bytes, page_num: int, needle: str,
               ocr_chardata: list | None = None,
               whole_word: bool = False,
               split: bool = False) -> list:
    """Search for ``needle`` on ``page_num`` and return list of Rects.

    Search modes (mutually exclusive -- ``split`` takes precedence over
    ``whole_word`` if both are set):

    * ``split=True``: head/tail-word union search. Useful for long
      composite strings ("Smith, Jane, MD") where punctuation and
      whitespace breaks may differ between OCR text and the input.
    * ``whole_word=True``: word-boundary-respecting search (rejects
      "El" matching inside "extremely").
    * neither: substring search (PyMuPDF's default ``page.search_for`` /
      :func:`search_in_chars`).

    If ``ocr_chardata`` is provided, the search runs in CharData-space.
    Otherwise the search opens the doc and uses live-page semantics.

    Returns ``list[Rect]`` for in-process callers; the CLI op serializes
    these to flat tuples on the wire.
    """
    if ocr_chardata:
        return _search_for_doc(None, page_num, needle,
                               ocr_chardata=ocr_chardata,
                               whole_word=whole_word, split=split)

    doc = open_pdf(pdf_bytes)
    try:
        return _search_for_doc(doc, page_num, needle,
                               ocr_chardata=None,
                               whole_word=whole_word, split=split)
    finally:
        doc.close()


# ── Live-page helpers ────────────────────────────────────────────────


def _cap_rects(rects: list) -> list:
    """Height-cap rects to ``_MAX_OCR_LINE_H``, anchored to the top edge."""
    capped = []
    for r in rects:
        if r.height > _MAX_OCR_LINE_H:
            capped.append(Rect(r.x0, r.y0, r.x1, r.y0 + _MAX_OCR_LINE_H))
        else:
            capped.append(r)
    return capped


def _page_search_whole_word(page, needle: str, ocr_textpage) -> list:
    """Search for whole-word occurrences on a live page, filtering substrings.

    ``ocr_textpage`` may be ``None`` for native-only fallback paths.
    """
    rects = page.search_for(needle, textpage=ocr_textpage)
    if not rects:
        return []
    rects = _cap_rects(rects)

    boundary_pattern = re.compile(
        r"(?<!\w)" + re.escape(needle) + r"(?!\w)",
        re.IGNORECASE,
    )
    char_pad = 5
    raw = page.get_text("rawdict", textpage=ocr_textpage)
    all_chars = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    all_chars.append(ch)

    verified = []
    for r in rects:
        x0 = r.x0 - char_pad
        x1 = r.x1 + char_pad
        surrounding = "".join(
            ch["c"] for ch in all_chars
            if ch["bbox"][2] > x0 and ch["bbox"][0] < x1
            and ch["bbox"][3] > r.y0 and ch["bbox"][1] < r.y1
        ).strip()
        if not surrounding or boundary_pattern.search(surrounding):
            verified.append(r)
    return verified


def _split_search(page, text: str, ocr_textpage) -> list:
    """Split-search a long string by head + tail union on a live page."""
    parts = re.split(r'[\s@,]+', text)
    parts = [p for p in parts if len(p) >= 3]
    if not parts:
        return []

    head = parts[0]
    tail = parts[-1] if len(parts) > 1 else None

    def _search(needle):
        return page.search_for(needle, textpage=ocr_textpage)

    head_rects = _search(head)
    if not head_rects:
        return []

    if tail and tail != head:
        tail_rects = _search(tail)
        if tail_rects:
            combined = []
            for hr in head_rects:
                for tr in tail_rects:
                    h_mid_y = (hr.y0 + hr.y1) / 2
                    t_mid_y = (tr.y0 + tr.y1) / 2
                    if abs(h_mid_y - t_mid_y) < 5 and tr.x1 > hr.x0:
                        combined.append(Rect(
                            min(hr.x0, tr.x0),
                            min(hr.y0, tr.y0),
                            max(hr.x1, tr.x1),
                            max(hr.y1, tr.y1),
                        ))
                        break
            if combined:
                return combined

    return head_rects
