"""Remove given words' text without touching anything else (v7).

A redaction that only removes text: no fill is drawn, and images and vector
graphics under the boxes are left alone (``images=NONE``, ``graphics=NONE``).
It reaches text that plain extraction does not see -- outside the page, in a
switched-off optional-content group, clipped away, drawn in render mode 3 --
because MuPDF's redaction filter rewrites the content streams themselves.

The hard part is not removing the target text but sparing everything else.
MuPDF chooses the glyphs a redaction box removes by geometry alone, so a box
around one word can also take letters from a tightly-leaded neighbour line,
or from other text drawn in the same place (a second language on another
optional-content layer, say). Three defences:

* **Targets are words, not places.** Each request names a word as the
  ``trace_text`` op reported it: its box plus its text, render mode, opacity
  and layer. Those words are found again in this page's own trace, and every
  character is keyed by position AND drawing properties, so a visible
  character sitting exactly where a target one sits is still "other text".
* **A thin band, not the word box.** Each target is reduced to a band through
  the middle of the word (``BANDS[0]`` of its height, in the page's native
  frame), which crosses the word's glyphs and, on ordinary leading, nothing
  else.
* **Verify, then apply.** The removal is first tried on a one-page copy and
  the characters compared: every target character must be gone and every
  other character must remain. If the page's whole set fails, each word is
  retried alone with progressively wider bands. A word that cannot be removed
  cleanly -- or that is not found at all -- is left in place and reported as
  kept, never forced through.

Boxes arrive in the as-rendered (rotation-applied) frame, like redaction
matches and like ``trace_text``'s own output.
"""
from __future__ import annotations

from collections import Counter

import pymupdf as _pymupdf

from .redact import Rect, _derotate_to_native
from .trace import _restore_layers, _span_words, _switch_layers_on

#: Band heights tried, as shares of a word box's height, narrowest first.
BANDS = (0.1, 0.3, 0.6)
#: Horizontal inset of a band at each end, as a share of the box height.
INSET = 0.02
#: A traced word matches a requested box when every edge is this close (pt).
MATCH_TOL = 0.5


def _page_words(page) -> list[dict]:
    """Every word on ``page`` with its as-rendered box and character keys.

    Keys carry the character, its origin and its span's drawing properties,
    so two characters in one place drawn differently stay distinct.
    """
    rot = page.rotation_matrix
    words = []
    for span in page.get_texttrace():
        props = (int(span.get("type", 0)), round(float(span.get("opacity", 1.0)), 3),
                 span.get("layer") or "",
                 tuple(round(float(c), 3) for c in (span.get("color") or ())))
        for text, box, keys in _span_words(span):
            if box is None:
                continue
            words.append({"text": text, "bbox": list(Rect(box) * rot),
                          "mode": props[0], "opacity": props[1], "layer": props[2],
                          "keys": [k + props for k in keys]})
    return words


def _all_keys(page) -> Counter:
    return Counter(k for w in _page_words(page) for k in w["keys"])


def _matches(word: dict, item: dict) -> bool:
    if any(abs(a - b) > MATCH_TOL for a, b in zip(word["bbox"], item["box"], strict=True)):
        return False
    for field in ("text", "mode", "layer"):
        if field in item and item[field] != word[field]:
            return False
    if "opacity" in item and abs(float(item["opacity"]) - word["opacity"]) > 1e-3:
        return False
    return True


def _band(page, box, share: float):
    """The removal band for an as-rendered ``box``, in add_redact_annot's frame."""
    native = Rect(box) * page.derotation_matrix
    native.normalize()
    h = native.height
    cy = (native.y0 + native.y1) / 2
    band = Rect(native.x0 + INSET * h, cy - share * h / 2,
                native.x1 - INSET * h, cy + share * h / 2)
    if page.rotation:
        return _derotate_to_native(band * page.rotation_matrix, page)
    return band


def _apply(page, bands) -> None:
    for band in bands:
        page.add_redact_annot(band, fill=False)
    page.apply_redactions(images=_pymupdf.PDF_REDACT_IMAGE_NONE,
                          graphics=_pymupdf.PDF_REDACT_LINE_ART_NONE,
                          text=_pymupdf.PDF_REDACT_TEXT_REMOVE)


def _keys_with_layers_on(doc, page_index: int) -> tuple[list[dict], Counter]:
    _, restore = _switch_layers_on(doc)
    try:
        words = _page_words(doc[page_index])
    finally:
        _restore_layers(doc, restore)
    return words, Counter(k for w in words for k in w["keys"])


def _clean(doc, pno: int, targets: list[tuple[dict, float]]) -> bool:
    """Try removing ``targets`` ((word, band share) pairs) on a one-page copy."""
    scratch = _pymupdf.open()
    try:
        scratch.insert_pdf(doc, from_page=pno, to_page=pno)
        page = scratch[0]
        _, before = _keys_with_layers_on(scratch, 0)
        target = Counter(k for w, _ in targets for k in w["keys"])
        others = before - target
        _apply(page, [_band(page, w["bbox"], s) for w, s in targets])
        _, after = _keys_with_layers_on(scratch, 0)
        return not (target & after) and not (others - after)
    finally:
        scratch.close()


def _plan_page(doc, pno: int, items: list[dict]):
    """Pick a band for each requested word, or keep it. Returns (plan, kept)."""
    words, _ = _keys_with_layers_on(doc, pno)
    found: list[tuple[int, dict]] = []
    kept: list[int] = []
    taken: set[int] = set()
    for i, item in enumerate(items):
        match = next((j for j, w in enumerate(words)
                      if j not in taken and _matches(w, item)), None)
        if match is None:
            kept.append(i)          # not found: never report it removed
            continue
        taken.add(match)
        found.append((i, words[match]))
    if not found:
        return [], kept
    if _clean(doc, pno, [(w, BANDS[0]) for _, w in found]):
        return [(i, w, BANDS[0]) for i, w in found], kept
    plan = []
    for i, w in found:
        share = next((s for s in BANDS if _clean(doc, pno, [(w, s)])), None)
        if share is None:
            kept.append(i)
        else:
            plan.append((i, w, share))
    return plan, sorted(kept)


def validate_remove_text(items) -> dict[int, list[dict]]:
    """Coerce the wire field to ``{page: [item, ...]}``; ValueError on bad input."""
    if not isinstance(items, list):
        raise ValueError("remove_text must be a list")
    by_page: dict[int, list[dict]] = {}
    for n, item in enumerate(items):
        try:
            page = int(item["page"])
            box = [float(v) for v in item["box"]]
            clean = {"box": box}
            if "text" in item:
                clean["text"] = str(item["text"])
            if "mode" in item:
                clean["mode"] = int(item["mode"])
            if "opacity" in item:
                clean["opacity"] = float(item["opacity"])
            if "layer" in item:
                clean["layer"] = str(item["layer"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed remove_text entry {n}: {type(exc).__name__}") from None
        if len(box) != 4 or box[0] > box[2] or box[1] > box[3]:
            raise ValueError(f"Malformed remove_text entry {n}: bad box")
        by_page.setdefault(page, []).append(clean)
    return by_page


def remove_text_doc(doc, by_page: dict[int, list[dict]],
                    skip_pages: set[int] = frozenset()) -> dict:
    """Remove each requested word's text from ``doc`` in place.

    Returns ``{"removed": [[page, index], ...], "kept": [[page, index], ...]}``
    where ``index`` is the item's position in that page's list. ``kept`` items
    were not found, or could not be removed without also removing other text
    (or without removing all of their own), and were left untouched. Pages in
    ``skip_pages`` (deleted or blacked out by the caller) and pages out of
    range are in neither list.
    """
    removed, kept = [], []
    for pno in sorted(by_page):
        if pno in skip_pages or not (0 <= pno < len(doc)):
            continue
        plan, left = _plan_page(doc, pno, by_page[pno])
        if plan:
            page = doc[pno]
            _apply(page, [_band(page, w["bbox"], s) for _, w, s in plan])
        removed.extend([pno, i] for i, _, _ in sorted(plan, key=lambda p: p[0]))
        kept.extend([pno, i] for i in left)
    return {"removed": removed, "kept": kept}
