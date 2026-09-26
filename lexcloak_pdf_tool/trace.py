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
  to (the innermost one marking it; ``""`` for none), and whether the page
  as the document opens leaves the word undrawn because of optional content:
  a group switched off by the default configuration (``/BaseState``,
  ``/ON``, ``/OFF``), an enclosing group that is off around one that is on,
  a form XObject whose own ``/OC`` is off, a group off by its ``/Usage`` or
  ``/Intent``, or a membership dictionary (``/OCMD``) that evaluates off.
  MuPDF itself decides, exactly as it does when it renders the page, so a
  word drawn by the page's default render is never ``layer_off`` (see
  "Switched-off layers" below).
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

**Switched-off layers.** Text MuPDF does not draw is not in its texttrace,
so every word is read from a view of the page with every optional-content
group drawn: a one-page copy of it, which carries no ``/OCProperties``, and
a page without ``/OCProperties`` draws all of its optional content. The
copy is compared, character by character (character, origin and drawing
properties), with the page as the document opens it; a word none of whose
characters the page draws by default is ``layer_off``. Neither MuPDF's
``get_ocgs()`` nor its viewer layer list decides anything: the first reads a
group without ``/Usage`` as on, the second lists only groups in ``/Order``,
and a group needs neither. A document without ``/OCProperties`` has nothing
switched off and is read directly.

Two checks keep that comparison honest, and each fails the page (a
``LayerReadError``, which a caller reports as "could not check") rather than
guess: the document's ``/OCProperties`` must have the shape the PDF
specification gives it, and every character the page draws by default must
also be in the copy, at the same place with the same properties. Neither
check reads or reports any text.
"""
from __future__ import annotations

import re
from collections import Counter
from contextlib import contextmanager

import pymupdf as _pymupdf

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


class LayerReadError(ValueError):
    """The page's optional content cannot be read, so whether its text is
    drawn cannot be told. The message never carries document text."""


#: ``/D`` keys that must be arrays when present.
_CONFIG_ARRAYS = ("ON", "OFF", "Order", "Locked", "AS", "RBGroups")
_REF = re.compile(r"^\s*(\d+)\s+(\d+)\s+R\s*$")


def _has_layers(doc) -> bool:
    """Whether the document carries ``/OCProperties`` at all, well-formed or
    not. Without it MuPDF draws every optional-content mark."""
    try:
        kind, _ = doc.xref_get_key(doc.pdf_catalog(), "OCProperties")
    except Exception:  # noqa: BLE001 -- an unreadable catalog: treat as layered
        return True
    return kind != "null"


def _is_dict(doc, kind: str, value: str) -> bool:
    if kind == "dict":
        return True
    if kind != "xref":
        return False
    m = _REF.match(value)
    if not m:
        return False
    try:
        return doc.xref_object(int(m.group(1)), compressed=True).lstrip().startswith("<<")
    except Exception:  # noqa: BLE001 -- a dangling reference is not a dictionary
        return False


def _check_layers(doc) -> bool:
    """Whether the document has optional content; raise ``LayerReadError``
    when its ``/OCProperties`` does not have the specification's shape (a
    dictionary whose ``/OCGs`` is an array of group dictionaries and whose
    ``/D`` is a dictionary, with arrays where arrays belong)."""
    if not _has_layers(doc):
        return False
    try:
        cat = doc.pdf_catalog()
        if not _is_dict(doc, *doc.xref_get_key(cat, "OCProperties")):
            raise LayerReadError("/OCProperties is not a dictionary")
        kind, value = doc.xref_get_key(cat, "OCProperties/OCGs")
        if kind != "array":
            raise LayerReadError("/OCProperties /OCGs is not an array")
        for ref in value.strip()[1:-1].split(" R"):
            ref = ref.strip()
            if ref and not _is_dict(doc, "xref", ref + " R"):
                raise LayerReadError("an /OCGs entry is not a group dictionary")
        if not _is_dict(doc, *doc.xref_get_key(cat, "OCProperties/D")):
            raise LayerReadError("/OCProperties lacks a default configuration /D")
        for key in _CONFIG_ARRAYS:
            kind, _ = doc.xref_get_key(cat, f"OCProperties/D/{key}")
            if kind not in ("null", "array"):
                raise LayerReadError(f"/D /{key} is not an array")
        kind, _ = doc.xref_get_key(cat, "OCProperties/D/BaseState")
        if kind not in ("null", "name"):
            raise LayerReadError("/D /BaseState is not a name")
    except LayerReadError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any failure to read is a failure
        raise LayerReadError(
            f"optional content cannot be read ({type(exc).__name__})") from None
    return True


@contextmanager
def _every_layer_drawn(doc, pno: int):
    """Page ``pno`` as MuPDF draws it with every optional-content group on.

    A one-page copy of a layered page: the copy carries no ``/OCProperties``
    (removed explicitly, whatever ``insert_pdf`` copies), and MuPDF draws all
    optional content in a document without it -- off groups, groups off by
    ``/Usage`` or ``/Intent``, forms with their own ``/OC`` and membership
    dictionaries alike. The caller's document is only read. A page of a
    document without ``/OCProperties`` already draws everything and is
    yielded itself.
    """
    if not _has_layers(doc):
        yield doc[pno]
        return
    copy = _pymupdf.open()
    try:
        copy.insert_pdf(doc, from_page=pno, to_page=pno)
        copy.xref_set_key(copy.pdf_catalog(), "OCProperties", "null")
        yield copy[0]
    finally:
        copy.close()


def _span_props(span) -> tuple:
    """A span's drawing properties, as ``remove_text`` keys characters."""
    return (int(span.get("type", 0)), round(float(span.get("opacity", 1.0)), 3),
            span.get("layer") or "",
            tuple(round(float(c), 3) for c in (span.get("color") or ())))


def _drawn_by_default(page) -> Counter:
    """Every non-blank character the page draws as the document opens,
    keyed by character, origin and drawing properties."""
    out: Counter = Counter()
    for span in page.get_texttrace():
        props = _span_props(span)
        for _text, _box, keys, _boxes in _span_words(span):
            out.update(k + props for k in keys)
    return out


def _trace_text_doc(doc, page_num: int) -> dict:
    if page_num < 0 or page_num >= len(doc):
        raise IndexError(
            f"page_num {page_num} out of range for {len(doc)}-page document"
        )
    layered = _check_layers(doc)
    page = doc[page_num]
    rot = page.rotation_matrix
    rect = page.rect
    rotation = int(page.rotation)
    shown = _drawn_by_default(page) if layered else None
    with _every_layer_drawn(doc, page_num) as view:
        page_area = _area(tuple(rect)) or 1.0
        log = [(kind, tuple(Rect(b) * rot)) for kind, b in view.get_bboxlog()]
        image_area = sum(_inter(tuple(rect), b) for kind, b in log
                         if kind in _IMAGE_KINDS)
        drawn = _drawn_chars(view)
        occluders = [(i, kind, b) for i, (kind, b) in enumerate(log)
                     if kind in _OCCLUDERS]
        words: list[dict] = []
        covers: dict[int, dict] = {}
        for span in view.get_texttrace():
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
            props = _span_props(span)
            for text, box, keys, boxes in _span_words(span):
                if box is None:
                    continue
                layer_off = False
                if shown is not None:
                    # One to one: a character drawn twice in one place is
                    # drawn by default only as often as the page draws it.
                    hits = 0
                    for k in keys:
                        if shown[k + props] > 0:
                            shown[k + props] -= 1
                            hits += 1
                    layer_off = hits == 0
                bbox = [float(v) for v in Rect(box) * rot]
                word = {
                    "text": text,
                    "bbox": bbox,
                    "size": float(span.get("size", 0.0)),
                    "mode": int(span.get("type", 0)),
                    "opacity": float(span.get("opacity", 1.0)),
                    "layer": layer,
                    "layer_off": layer_off,
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
    if shown is not None and +shown:
        # A character the page draws by default is missing from the view
        # with every layer drawn: the two do not describe the same page, so
        # which words are hidden cannot be told.
        raise LayerReadError("the page with every layer drawn lost text it draws by default")
    return {
        "rect": [float(v) for v in rect],
        "rotation": rotation,
        "image_cover": min(1.0, image_area / page_area),
        "words": words,
        "covers": [covers[i] for i in sorted(covers)],
    }


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
