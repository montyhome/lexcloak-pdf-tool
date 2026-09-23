"""Per-word text trace with drawing properties (protocol v7, v9).

``page.get_text()`` answers "what text is on this page?" and deliberately
leaves out text that is not drawn: it clips to the page, skips content in
optional-content groups that are switched off, and skips glyphs removed by a
clipping path. This op answers the opposite question -- "what text does this
page's content carry, and how is each word drawn?" -- so a caller can compare
it with a render of the page.

Each word carries only facts read from the PDF's own drawing instructions:

* ``mode`` -- the text render mode (0 fill ... 3 invisible ... 7 clip).
* ``opacity`` -- the fill alpha in effect when the word was drawn.
* ``layer`` / ``layer_off`` -- the optional-content group the word belongs
  to, and whether that group is off in the document's current configuration.
* ``clipped`` -- no character of the word survives MuPDF's clip-respecting
  extraction (it lies outside a clipping path, or outside the page).
* ``covered_by`` -- ``"image"`` or ``"path"`` when a fill or image drawn
  LATER covers at least ``COVER_FRACTION`` of the word's span; else ``None``.
* ``bbox`` -- in the ROTATED page frame (``page.rect``), the frame a render
  of the page uses, so a caller can index pixels with it directly.
* ``span`` -- the drawing sequence number of the text span the word came from.
  Words sharing it were drawn by one text-showing run, whatever the page's
  rotation, so a caller can regroup words into runs without guessing a line
  direction from box shapes.
* ``chars`` (protocol 9) -- each character's box, in the order of ``text``,
  rotated like ``bbox``. Present only on a word that a fill, shading or
  image drawn LATER reaches into by any area, so a caller can ask which of
  its characters that draw hides. A cover rarely stops at a word boundary.

The page result also carries ``image_cover``: the share of the page area
covered by image draws (1.0 for a typical scanned page), and ``covers``
(protocol 9): each draw that reaches into a word drawn before it, as its
sequence number, kind (``fill-path``, ``fill-shade``, ``fill-image`` or
``fill-imgmask``) and rotated box. Whether a cover is opaque is not reported:
a render of the page answers that, for blend modes and soft masks too.

Nothing here decides whether a word is "hidden"; that judgement needs the
render and belongs to the caller.
"""
from __future__ import annotations

from .redact import Rect, open_pdf

#: A later fill or image must cover this share of a span's box to count.
COVER_FRACTION = 0.8

_OCCLUDERS = ("fill-path", "fill-image", "fill-shade", "fill-imgmask")
_IMAGE_KINDS = ("fill-image", "fill-imgmask")


def _area(r) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _inter(a, b) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]),
                  min(a[2], b[2]), min(a[3], b[3])))


def _key(c: str, origin) -> tuple:
    # Characters are matched between the trace and ``rawdict`` by character
    # and origin, never by box: the two size glyph boxes differently
    # (ascender/descender versus font bbox), so boxes never compare equal.
    return (c, round(origin[0], 2), round(origin[1], 2))


def _drawn_chars(page) -> set:
    out = set()
    for block in page.get_text("rawdict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    out.add(_key(ch["c"], ch["origin"]))
    return out


def _span_words(span) -> list[tuple[str, tuple, list, list]]:
    """``(text, unrotated bbox, [char keys], [char boxes])`` per
    space-separated word. Keys and boxes run in the order of ``text``."""
    words, buf, box, keys, boxes = [], [], None, [], []
    for ch in span.get("chars", ()):
        c = chr(ch[0])
        x0, y0, x1, y1 = ch[3]
        if c.isspace():
            if buf:
                words.append(("".join(buf), box, keys, boxes))
            buf, box, keys, boxes = [], None, [], []
            continue
        buf.append(c)
        keys.append(_key(c, ch[2]))
        boxes.append((x0, y0, x1, y1))
        box = (x0, y0, x1, y1) if box is None else (
            min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1))
    if buf:
        words.append(("".join(buf), box, keys, boxes))
    return words


def _switch_layers_on(doc) -> tuple[set[str], list[int]]:
    """Turn every switched-off UI layer on; return (off names, configs to restore).

    Uses the live UI configuration, which takes effect immediately.
    ``Document.set_layer`` would need a save and reopen before extraction
    sees it.
    """
    off_names: set[str] = set()
    try:
        ocgs = doc.get_ocgs() or {}
    except Exception:  # noqa: BLE001 -- a broken /OCProperties means no layers
        return off_names, []
    off_names = {v.get("name") or "" for v in ocgs.values() if not v.get("on", True)}
    if not off_names:
        return off_names, []
    restore = []
    for cfg in doc.layer_ui_configs():
        if not cfg.get("on") and not cfg.get("locked"):
            doc.set_layer_ui_config(cfg["number"], action=0)
            restore.append(cfg["number"])
    return off_names, restore


def _restore_layers(doc, restore: list[int]) -> None:
    for number in restore:
        doc.set_layer_ui_config(number, action=2)


def _trace_text_doc(doc, page_num: int) -> dict:
    if page_num < 0 or page_num >= len(doc):
        raise IndexError(
            f"page_num {page_num} out of range for {len(doc)}-page document"
        )
    off_names, restore = _switch_layers_on(doc)
    try:
        page = doc[page_num]
        rot = page.rotation_matrix
        rect = page.rect
        page_area = _area(tuple(rect)) or 1.0
        log = [(kind, tuple(Rect(b) * rot)) for kind, b in page.get_bboxlog()]
        image_area = sum(_inter(tuple(rect), b) for kind, b in log
                         if kind in _IMAGE_KINDS)
        drawn = _drawn_chars(page)
        occluders = [(i, kind, b) for i, (kind, b) in enumerate(log)
                     if kind in _OCCLUDERS]
        words: list[dict] = []
        covers: dict[int, dict] = {}
        for span in page.get_texttrace():
            seq = span.get("seqno", -1)
            sbox = tuple(Rect(span["bbox"]) * rot)
            covered_by = None
            if 0 <= seq < len(log) and _area(sbox) > 0:
                for kind, obox in log[seq + 1:]:
                    if (kind in _OCCLUDERS
                            and _inter(sbox, obox) >= COVER_FRACTION * _area(sbox)):
                        covered_by = "image" if kind in _IMAGE_KINDS else "path"
                        break
            # Every fill, shading or image drawn after this span that reaches
            # into its box (protocol 9). Most spans have none.
            later = [(i, kind, b) for i, kind, b in occluders
                     if i > seq and _inter(sbox, b) > 0] if seq >= 0 else []
            layer = span.get("layer") or ""
            for text, box, keys, boxes in _span_words(span):
                if box is None:
                    continue
                bbox = [float(v) for v in Rect(box) * rot]
                word = {
                    "text": text,
                    "bbox": bbox,
                    "size": float(span.get("size", 0.0)),
                    "mode": int(span.get("type", 0)),
                    "opacity": float(span.get("opacity", 1.0)),
                    "layer": layer,
                    "layer_off": bool(layer) and layer in off_names,
                    "clipped": not any(k in drawn for k in keys),
                    "covered_by": covered_by,
                    "span": int(seq),
                }
                touching = [(i, kind, b) for i, kind, b in later
                            if _inter(bbox, b) > 0]
                if touching:
                    word["chars"] = [[float(v) for v in Rect(c) * rot]
                                     for c in boxes]
                    for i, kind, b in touching:
                        covers[i] = {"seq": i, "kind": kind,
                                     "box": [float(v) for v in b]}
                words.append(word)
        return {
            "rect": [float(v) for v in rect],
            "rotation": int(page.rotation),
            "image_cover": min(1.0, image_area / page_area),
            "words": words,
            "covers": [covers[i] for i in sorted(covers)],
        }
    finally:
        _restore_layers(doc, restore)


def trace_text(pdf_bytes: bytes, page_num: int) -> dict:
    """Trace every word of ``page_num`` with its drawing properties.

    Returns ``{"rect": [x0, y0, x1, y1], "rotation": int,
    "image_cover": float, "words": [...], "covers": [...]}``; see the module
    docstring for the per-word fields and ``covers``. The document's optional-content state is left exactly
    as it was found.
    """
    doc = open_pdf(pdf_bytes)
    try:
        return _trace_text_doc(doc, page_num)
    finally:
        doc.close()
