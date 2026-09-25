"""Keep the lines a redaction box does not reach (v0.11.0).

MuPDF's redaction filter removes a glyph when a box reaches more than a tenth
of the way into the glyph's box (measured from above, below and the left, at
7, 12 and 24 pt, on pymupdf 1.28.2). That box is not the ink: it runs across
the glyph's advance and from the font's ascender to its descender. In
Helvetica the ascender sits more than a third of the size above the capitals,
so a box drawn over one line of text reaches into the glyph boxes of the line
below without touching any of its ink, and takes those glyphs out of the
page. Nothing on the page covers them, and a line the reader should still see
is gone from the render and from the text layer.

Two cases, each a box drawn over a value's ink plus a fifth of a point
(``tests/test_keep_uncovered_lines.py``):

* a 12 pt heading 11 pt below a line of 7 pt text lost every letter under
  the box over a name on that line: the box reached 3.6 pt into the
  heading's glyph boxes and stopped 0.4 pt above its capitals;
* single-spaced body text (baseline to baseline equal to the size, 10 to
  12 pt, Helvetica and Times) lost the next line's letters under a box over
  a name with a descender.

``apply_redactions(..., keep_uncovered_lines=True)`` keeps those glyphs:

* **Only a line the box does not reach.** For each box and each line whose
  glyph boxes it touches, the line's ink is measured over the glyphs it
  touches (their outlines, ``TEXT_ACCURATE_BBOXES``). When the box lies
  wholly above that ink or wholly below it, the glyphs are kept. A box that
  reaches the ink by any amount removes them as before, so a box on its own
  line is judged exactly as it always was, including the glyphs beside a
  value that a short box only touches by their side bearing.
* **Only plain visible text.** Fill or stroke render modes, opacity above
  zero, in no optional-content group, horizontal, on a page with no
  ``/Rotate``. A glyph drawn twice at the same origin, or any other text,
  keeps the old rule, and a box touching such a glyph on a line keeps the
  old rule for that line.
* **The fill, images and vector graphics use the box unchanged.** Text is
  removed first, with each box trimmed clear of the glyph boxes of the lines
  it keeps; the boxes are then drawn and applied to images and graphics with
  text left alone.
* **The old rule is the floor.** After the text pass, the old text removal
  is tried on a one-page copy of the result. If it would take any glyph
  other than a kept one, the old text removal is applied to the page itself,
  so a page never keeps a glyph the old rule removes unless its line is one
  the box does not reach.

A page where nothing is kept is burned the old way, in one pass, and its
bytes do not change.
"""
from __future__ import annotations

from collections import Counter

import pymupdf as _pymupdf

from .redact import Rect
from .remove_text import _unpaired
from .trace import _key, _restore_layers, _switch_layers_on

#: Glyph outline boxes, not ascender/descender boxes, for the ink test.
_FLAGS = _pymupdf.TEXTFLAGS_RAWDICT | _pymupdf.TEXT_ACCURATE_BBOXES
#: Render modes that paint the glyph: fill, stroke, fill and stroke.
_VISIBLE_MODES = (0, 1, 2)
#: A line runs left to right when its direction is this close to (1, 0).
_DIR_TOL = 1e-3


def _trace_keys(doc, pno: int) -> tuple[Counter, dict]:
    """Every non-blank character on page ``pno``, layers switched on, keyed
    as ``remove_text`` keys them (character, origin, drawing properties);
    and ``(char, x, y)`` -> key for the characters drawn plainly and drawn
    only once at their origin.

    Only keys: the word boxes ``remove_text`` builds cost a Rect per
    character, and the floor check reads each page twice.
    """
    _, restore = _switch_layers_on(doc)
    try:
        spans = doc[pno].get_texttrace()
    finally:
        _restore_layers(doc, restore)
    keys: Counter = Counter()
    seen: Counter = Counter()
    plain = {}
    for span in spans:
        props = (int(span.get("type", 0)), round(float(span.get("opacity", 1.0)), 3),
                 span.get("layer") or "",
                 tuple(round(float(c), 3) for c in (span.get("color") or ())))
        shown = props[0] in _VISIBLE_MODES and props[1] > 0 and not props[2]
        for ch in span.get("chars", ()):
            c = chr(ch[0]) if ch[0] >= 0 else "\ufffd"
            if c.isspace():
                continue
            cell = _key(c, ch[2])
            key = cell + props
            keys[key] += 1
            seen[cell] += 1
            if shown:
                plain[cell] = key
    return keys, {cell: k for cell, k in plain.items() if seen[cell] == 1}


def _lines(page, plain: dict) -> list[dict]:
    """The page's left-to-right lines: each its glyphs and their extent.

    A glyph carries the box MuPDF's filter tests (``box``: its advance, from
    the span's ascender to its descender), the vertical extent of its ink
    (``ink``, None for a blank), its key when it is plain text (``key``) and
    whether it is a space.
    """
    lines = []
    for block in page.get_text("rawdict", flags=_FLAGS).get("blocks", []):
        for line in block.get("lines", []):
            dx, dy = line.get("dir", (1.0, 0.0))
            if abs(dx - 1.0) > _DIR_TOL or abs(dy) > _DIR_TOL or line.get("wmode"):
                continue
            glyphs = []
            for span in line.get("spans", []):
                asc = span["ascender"] * span["size"]
                desc = span["descender"] * span["size"]
                for ch in span.get("chars", []):
                    x0, y0, x1, y1 = ch["bbox"]
                    oy = ch["origin"][1]
                    space = ch["c"].isspace()
                    glyphs.append({
                        "box": (x0, oy - asc, x1, oy - desc),
                        "ink": None if space or y1 <= y0 else (y0, y1),
                        "key": None if space else plain.get(_key(ch["c"], ch["origin"])),
                        "space": space,
                    })
            if glyphs:
                lines.append({"glyphs": glyphs,
                              "x0": min(g["box"][0] for g in glyphs),
                              "y0": min(g["box"][1] for g in glyphs),
                              "x1": max(g["box"][2] for g in glyphs),
                              "y1": max(g["box"][3] for g in glyphs)})
    return lines


def _meets(box, r) -> bool:
    return (min(box[2], r.x1) > max(box[0], r.x0)
            and min(box[3], r.y1) > max(box[1], r.y0))


def plan(doc, pno: int, rects: list) -> tuple[Counter, list] | None:
    """What a burn of ``rects`` on page ``pno`` keeps, and the boxes that
    remove its text: ``(kept keys, text boxes)``, or None when it keeps
    nothing. ``rects`` are in the unrotated page frame."""
    _, plain_keys = _trace_keys(doc, pno)
    lines = _lines(doc[pno], plain_keys)
    blocked: set[int] = set()
    pairs = []   # (rect index, "above" | "below", touched glyphs)
    for i, r in enumerate(rects):
        for line in lines:
            if not _meets((line["x0"], line["y0"], line["x1"], line["y1"]), r):
                continue
            glyphs = line["glyphs"]
            touched = [g for g in glyphs if _meets(g["box"], r)]
            if not touched:
                continue
            inks = ([g["ink"] for g in touched if g["ink"]]
                    or [g["ink"] for g in glyphs if g["ink"]])
            plain = all(g["space"] or g["key"] for g in touched)
            if inks and plain and r.y1 <= min(a for a, _ in inks):
                pairs.append((i, "below", touched))
            elif inks and plain and r.y0 >= max(b for _, b in inks):
                pairs.append((i, "above", touched))
            else:
                blocked.update(id(g) for g in touched)
    trims = [[r.y0, r.y1] for r in rects]
    kept: Counter = Counter()
    for i, side, touched in pairs:
        free = [g for g in touched if id(g) not in blocked]
        if not free:
            continue
        kept.update(g["key"] for g in free if not g["space"])
        if side == "below":
            trims[i][1] = min(trims[i][1], min(g["box"][1] for g in touched))
        else:
            trims[i][0] = max(trims[i][0], max(g["box"][3] for g in touched))
    if all(t == [r.y0, r.y1] for t, r in zip(trims, rects, strict=True)):
        return None
    text = [Rect(r.x0, t0, r.x1, t1) if t1 > t0 else r
            for r, (t0, t1) in zip(rects, trims, strict=True)]
    return kept, text


def _remove_text(page, rects) -> None:
    for r in rects:
        page.add_redact_annot(r, fill=False)
    page.apply_redactions(images=_pymupdf.PDF_REDACT_IMAGE_NONE,
                          graphics=_pymupdf.PDF_REDACT_LINE_ART_NONE,
                          text=_pymupdf.PDF_REDACT_TEXT_REMOVE)


def floor_holds(doc, pno: int, rects: list, kept: Counter) -> bool:
    """Whether the old text removal of ``rects`` would take nothing from
    page ``pno`` as it now is but ``kept`` characters (tried on a copy)."""
    scratch = _pymupdf.open()
    try:
        scratch.insert_pdf(doc, from_page=pno, to_page=pno)
        before, _ = _trace_keys(scratch, 0)
        _remove_text(scratch[0], rects)
        after, _ = _trace_keys(scratch, 0)
        return not _unpaired(_unpaired(before, after), kept)
    finally:
        scratch.close()


def burn_keeping_lines(doc, pno: int, boxes: list, add_annots) -> bool:
    """Burn ``boxes`` on page ``pno`` keeping the lines they do not reach.

    ``boxes`` are ``(rect, edge strips, label, font size)`` in the frame
    ``add_redact_annot`` takes, and ``add_annots(page, boxes)`` adds the
    burn's own annotations for them. Returns False, having changed nothing,
    when the page is rotated or nothing would be kept: the caller then burns
    it the old way.
    """
    page = doc[pno]
    if page.rotation:
        return False
    rects = [b[0] for b in boxes]
    planned = plan(doc, pno, rects)
    if planned is None:
        return False
    kept, text = planned
    strips = [s for b in boxes for s in b[1]]
    _remove_text(page, strips + text)
    if not floor_holds(doc, pno, strips + rects, kept):
        _remove_text(page, strips + rects)
    add_annots(page, boxes)
    page.apply_redactions(images=_pymupdf.PDF_REDACT_IMAGE_PIXELS,
                          graphics=_pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                          text=_pymupdf.PDF_REDACT_TEXT_NONE)
    return True
