"""Remove given words' text without touching anything else (v7, v9).

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
  kept, never forced through. Characters are compared with a tolerance of
  one rounding step on their origins (``KEY_STEPS``): rewriting a text run
  can move the glyphs left in it by a few millionths of a point, and a
  character that moved that little has not been touched.

**Part of a word (v9).** A request may name some of a word's characters
(``chars``, their positions in ``text``), for a word only partly under
something drawn over it: ``MARCOTTE,`` whose comma a box stops short of.
Each run of named characters gets its own band across just those glyphs,
and the verify is the same, so the comma must still be there afterwards.
A part that fails is halved until each half is removed cleanly or is a
single character no band separates; that character is left, and the word is
reported kept.

Boxes arrive in the as-rendered (rotation-applied) frame, like redaction
matches and like ``trace_text``'s own output.
"""
from __future__ import annotations

from collections import Counter

import pymupdf as _pymupdf

from .redact import Rect, _derotate_to_native
from .render import _render_page_doc
from .trace import _every_layer_drawn, _has_layers, _span_words

#: Band heights tried, as shares of a word box's height, narrowest first.
BANDS = (0.1, 0.3, 0.6)
#: Horizontal inset of a band at each end, as a share of the box height.
INSET = 0.02
#: A traced word matches a requested box when every edge is this close (pt).
MATCH_TOL = 0.5
#: Character keys carry origins rounded to 0.01 pt (``trace._key``). When
#: MuPDF rewrites a text run to drop some of its glyphs, the glyphs it keeps
#: can come back a few millionths of a point from where they were: measured
#: 138.394989 -> 138.395004 on pymupdf 1.28.2. That is nothing on the page,
#: but it tips the rounded key from 138.39 to 138.40, and an exact
#: comparison then calls the kept glyph lost and the removal unclean. Two
#: keys therefore pair when their origins are within this many rounding
#: steps on each axis.
KEY_STEPS = 1


def _page_words(page) -> list[dict]:
    """Every word on ``page`` with its as-rendered box, character keys and
    character boxes.

    Keys carry the character, its origin and its span's drawing properties,
    so two characters in one place drawn differently stay distinct.
    """
    rot = page.rotation_matrix
    words = []
    for span in page.get_texttrace():
        props = (int(span.get("type", 0)), round(float(span.get("opacity", 1.0)), 3),
                 span.get("layer") or "",
                 tuple(round(float(c), 3) for c in (span.get("color") or ())))
        for text, box, keys, boxes in _span_words(span):
            if box is None:
                continue
            words.append({"text": text, "bbox": list(Rect(box) * rot),
                          "mode": props[0], "opacity": props[1], "layer": props[2],
                          "keys": [k + props for k in keys],
                          "chars": [list(Rect(b) * rot) for b in boxes]})
    return words


def _matches(word: dict, item: dict) -> bool:
    if any(abs(a - b) > MATCH_TOL for a, b in zip(word["bbox"], item["box"], strict=True)):
        return False
    for field in ("text", "mode", "layer"):
        if field in item and item[field] != word[field]:
            return False
    return not ("opacity" in item
                and abs(float(item["opacity"]) - word["opacity"]) > 1e-3)


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
    """The page's words and character keys with every optional-content group
    drawn, exactly as ``trace_text`` reads them, so a word it reported is
    found here with the same ``layer``."""
    with _every_layer_drawn(doc, page_index) as page:
        words = _page_words(page)
    return words, Counter(k for w in words for k in w["keys"])


def _cell(key: tuple) -> tuple[tuple, int, int]:
    """A key split into its identity (character and drawing properties) and
    its origin in whole rounding steps."""
    return (key[0],) + tuple(key[3:]), round(key[1] * 100), round(key[2] * 100)


def _unpaired(want: Counter, have: Counter) -> Counter:
    """The keys in ``want`` that find no partner in ``have``, one to one.

    Partners share a character and drawing properties, and their origins are
    at most ``KEY_STEPS`` rounding steps apart on each axis. Exact partners
    are paired first, so the tolerance only ever pairs keys an exact
    comparison would have left over, and each key in ``have`` is used once:
    two identical glyphs drawn in one place still count as two.
    """
    exact = want & have
    want, have = want - exact, have - exact
    pool: Counter = Counter()
    for key, n in have.items():
        pool[_cell(key)] += n
    steps = range(-KEY_STEPS, KEY_STEPS + 1)
    left: Counter = Counter()
    for key, n in want.items():
        ident, x, y = _cell(key)
        for dx in steps:
            for dy in steps:
                cell = (ident, x + dx, y + dy)
                took = min(n, pool[cell])
                pool[cell] -= took
                n -= took
        if n:
            left[key] = n
    return left


# ── Targets: a whole word, or runs of some of its characters ──────────────


def _runs(positions) -> list[list[int]]:
    runs: list[list[int]] = []
    for p in sorted(positions):
        if runs and runs[-1][-1] == p - 1:
            runs[-1].append(p)
        else:
            runs.append([p])
    return runs


def _part(word: dict, positions=None) -> dict:
    """What removing ``positions`` of ``word`` (all of it when None) means:
    the keys that must go, and the boxes the bands cross."""
    if positions is None:
        return {"keys": list(word["keys"]), "boxes": [word["bbox"]],
                "positions": None}
    boxes = []
    for run in _runs(positions):
        cs = [word["chars"][p] for p in run]
        boxes.append([min(c[0] for c in cs), min(c[1] for c in cs),
                      max(c[2] for c in cs), max(c[3] for c in cs)])
    return {"keys": [word["keys"][p] for p in positions], "boxes": boxes,
            "positions": sorted(positions)}


def _bands(page, part: dict, share: float) -> list:
    return [_band(page, box, share) for box in part["boxes"]]


def _clean(doc, pno: int, targets: list[tuple[dict, float]]) -> bool:
    """Try removing ``targets`` ((part, band share) pairs) on a one-page copy."""
    scratch = _pymupdf.open()
    try:
        scratch.insert_pdf(doc, from_page=pno, to_page=pno)
        page = scratch[0]
        _, before = _keys_with_layers_on(scratch, 0)
        target = Counter(k for part, _ in targets for k in part["keys"])
        others = _unpaired(before, target)
        _apply(page, [b for part, s in targets for b in _bands(page, part, s)])
        _, after = _keys_with_layers_on(scratch, 0)
        # Whatever is on the page beyond the other text must not be a target.
        extra = _unpaired(after, others)
        return not _unpaired(others, after) and _unpaired(target, extra) == target
    finally:
        scratch.close()


def _first_share(doc, pno: int, part: dict) -> float | None:
    return next((s for s in BANDS if _clean(doc, pno, [(part, s)])), None)


def _split(doc, pno: int, word: dict, positions: list[int]):
    """Accepted ``(part, share)`` pieces of ``positions``, and those left over.

    Halving keeps the number of trials near-logarithmic where a character at
    a time would try every one.
    """
    part = _part(word, positions)
    share = _first_share(doc, pno, part)
    if share is not None:
        return [(part, share)], []
    if len(positions) == 1:
        return [], list(positions)
    mid = len(positions) // 2
    done_a, left_a = _split(doc, pno, word, positions[:mid])
    done_b, left_b = _split(doc, pno, word, positions[mid:])
    return done_a + done_b, left_a + left_b


def _target(word: dict, item: dict) -> dict:
    return _part(word, item.get("chars"))


def _plan_page(doc, pno: int, items: list[dict]):
    """Pick bands for each requested word or part, or keep it.

    Returns ``(plan, kept)``: ``plan`` is ``(item index, part, share)`` for
    everything that is removed cleanly, ``kept`` the indices of items left
    whole or in part.
    """
    words, _ = _keys_with_layers_on(doc, pno)
    found: list[tuple[int, dict]] = []
    kept: list[int] = []
    taken: set[int] = set()
    for i, item in enumerate(items):
        match = next((j for j, w in enumerate(words)
                      if j not in taken and _matches(w, item)
                      and all(p < len(w["keys"]) for p in item.get("chars") or ())),
                     None)
        if match is None:
            kept.append(i)          # not found: never report it removed
            continue
        taken.add(match)
        found.append((i, words[match]))
    if not found:
        return [], kept
    parts = [(i, _target(w, items[i])) for i, w in found]
    if _clean(doc, pno, [(part, BANDS[0]) for _, part in parts]):
        return [(i, part, BANDS[0]) for i, part in parts], kept
    plan = []
    for (i, w), (_, part) in zip(found, parts, strict=True):
        if part["positions"] is None:
            share = _first_share(doc, pno, part)
            if share is None:
                kept.append(i)
            else:
                plan.append((i, part, share))
            continue
        done, left = _split(doc, pno, w, part["positions"])
        plan += [(i, piece, share) for piece, share in done]
        if left:
            kept.append(i)
    return plan, sorted(set(kept))


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
            if item.get("chars") is not None:
                if not isinstance(item["chars"], list):
                    raise TypeError("chars")
                chars = sorted({int(p) for p in item["chars"]})
                if not chars or chars[0] < 0:
                    raise ValueError("chars")
                clean["chars"] = chars
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
    (or without removing all of their own), and what could not be removed was
    left untouched. An item naming part of a word (``chars``) whose part was
    removed only in some places is ``kept``: whatever could be separated is
    gone, and the rest is still there. Pages in ``skip_pages`` (deleted or
    blacked out by the caller) and pages out of range are in neither list.
    """
    removed, kept = [], []
    for pno in sorted(by_page):
        if pno in skip_pages or not (0 <= pno < len(doc)):
            continue
        plan, left = _plan_page(doc, pno, by_page[pno])
        if plan:
            page = doc[pno]
            _apply(page, [b for _, part, s in plan for b in _bands(page, part, s)])
        done = sorted({i for i, _, _ in plan} - set(left))
        removed.extend([pno, i] for i in done)
        kept.extend([pno, i] for i in left)
    return {"removed": removed, "kept": kept}


# ── A render with some text removed (v9) ─────────────────────────────────


def _copy_for_render(doc, pno: int):
    """A copy to remove text from and render, drawn exactly like ``doc``.

    One page is enough, and cheap, unless the document has optional content:
    a one-page copy loses the document's layer configuration and draws a
    switched-off layer, so then the whole document is copied. Whether it has
    any is read from ``/OCProperties`` itself, never from ``get_ocgs()``,
    which can list no group while the configuration still switches one off.
    """
    if _has_layers(doc) or len(doc) == 1:
        return _pymupdf.open(stream=doc.tobytes(), filetype="pdf")
    copy = _pymupdf.open()
    copy.insert_pdf(doc, from_page=pno, to_page=pno)
    return copy


def _present(keys: list, have: Counter) -> list[int]:
    """Positions in ``keys`` that still find a partner in ``have``."""
    left = _unpaired(Counter(keys), have)
    out = []
    for i, k in enumerate(keys):
        if left.get(k):
            left[k] -= 1                # this one found no partner: gone
        else:
            out.append(i)
    return out


def render_removed_doc(doc, pno: int, items: list[dict], dpi: float) -> dict:
    """Render page ``pno`` as it would look with ``items``' text removed.

    Each item names a word, or some of its characters, exactly as a
    ``remove_text`` entry does. Nothing here is delivered: the caller
    compares this render with the page's own to see which characters change
    nothing when they go. So the removal runs on a copy, with no verify, but
    it must really remove what it is given, or an untouched character would
    read as one whose removal changes nothing. Each run gets the thinnest
    band, and any character still there gets the wider ``remove_text`` bands
    in turn. What even the widest band leaves is reported as ``kept``.

    To let the caller tell a character's own effect from a neighbour's, the
    result also names every OTHER character the copy lost (``lost``).

    Returns ``{"png": bytes, "lost": [[x0, y0, x1, y1], ...], "kept":
    [[x0, y0, x1, y1], ...], "missed": [index, ...]}``: boxes as-rendered,
    ``missed`` the items whose word was not found.
    """
    if not (0 <= pno < len(doc)):
        raise IndexError(f"page_num {pno} out of range for {len(doc)}-page document")
    copy = _copy_for_render(doc, pno)
    if len(copy) == 1:
        pno = 0
    try:
        words, before = _keys_with_layers_on(copy, pno)
        missed: list[int] = []
        targets: list[tuple[dict, list[int]]] = []
        taken: set[int] = set()
        for i, item in enumerate(items):
            match = next((j for j, w in enumerate(words)
                          if j not in taken and _matches(w, item)
                          and all(p < len(w["keys"]) for p in item.get("chars") or ())),
                         None)
            if match is None:
                missed.append(i)
                continue
            taken.add(match)
            word = words[match]
            targets.append((word, list(item.get("chars") or range(len(word["keys"])))))
        page = copy[pno]
        remaining = targets
        for share in BANDS:
            if not remaining:
                break
            _apply(page, [b for w, pos in remaining for b in _bands(page, _part(w, pos), share)])
            _, now = _keys_with_layers_on(copy, pno)
            remaining = [(w, [pos[i] for i in _present([w["keys"][q] for q in pos], now)])
                         for w, pos in remaining]
            remaining = [(w, pos) for w, pos in remaining if pos]
        _, after = _keys_with_layers_on(copy, pno)
        target = Counter(w["keys"][q] for w, pos in targets for q in pos)
        lost = _unpaired(_unpaired(before, target), after)
        boxes = [c for w in words for k, c in zip(w["keys"], w["chars"], strict=True)
                 if lost.get(k)]
        kept = [w["chars"][q] for w, pos in remaining for q in pos]
        png = _render_page_doc(copy, pno, dpi)
        return {"png": png, "lost": boxes, "kept": kept, "missed": missed}
    finally:
        copy.close()
