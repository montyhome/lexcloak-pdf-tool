"""Content a PDF carries that the page, as it opens, does not show (v0.14.0).

Two kinds, each listed for a caller to judge and changed only where the
caller names it. Nothing here decides what content means.

**Images.** :func:`list_unseen_images` reads every image placement and
reports the ones the page as it opens never shows:

* ``outside-crop`` -- drawn wholly outside the crop box, so no viewer shows it;
* ``off-layer`` -- drawn only in optional content the document switches off.

It also counts every placement by state, the shown ones as ``shown`` or
``partly-outside`` (part of the box outside the crop box). Placements are
read from MuPDF's draw log twice: as the document opens, and from a second
copy of the same bytes with ``/OCProperties`` removed, which MuPDF draws
with every optional-content group on. Each unseen placement carries the image
object it draws (``xref``; 0 for an inline image, which has none), that
object's pixel count, and ``xref_shown``: whether any placement of the same
object, anywhere in the document, is shown. Object numbers are resolved only
on the pages that need them, because resolving them hashes every image.

``apply_redactions`` acts on what a caller names. ``blank_images`` replaces
an image object with a one-pixel stencil mask that paints nothing: the
dictionary is rewritten whole, so its soft mask, colour space and filters go
with it, and every placement of it draws nothing, which is why a caller
should name only an object no placement shows. ``remove_images`` removes the
image draws inside a box that lies wholly outside the crop box, text and
drawings untouched; a box inside the crop box, or one another image draw
reaches into, is skipped, because removal takes every image it touches.

**Tagged text.** In marked content tagged ``/ActualText``, text extraction
returns the tag's text in place of the glyphs the content draws. A tag
usually repeats or tidies the drawn text (a ligature, a hyphenation, an
accessibility reading), but nothing requires it to. :func:`list_tag_text`
reports, per page, where extraction returns letters or digits that differ
from the glyphs drawn there (a tag that only drops a hyphen or a space is not
listed):

* ``text`` -- what extraction returns there;
* ``drawn`` -- the glyphs drawn under it (``""`` when there are none);
* ``bbox`` -- in the rotated page frame, as ``trace_text`` reports boxes, or
  ``None`` for a tag that wraps no glyphs.

MuPDF's extraction skips a tag that wraps no glyphs (one around a drawing, an
image or nothing). Other readers may not, so the page's content streams,
the form XObjects they draw and their marked-content property lists are also
read directly, and a tag whose letters and digits extraction never returns
is listed with ``bbox`` ``None``.

A page that cannot be read is listed in ``unreadable_pages``; it is never
reported as having nothing.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher

import pymupdf as _pymupdf

logger = logging.getLogger(__name__)

#: A placement box this far (points) outside the crop box still counts as on it.
EDGE_TOL_PT = 1.0
#: Placement boxes match across the two views when every edge is this close.
BOX_TOL_PT = 0.5

IMAGE_STATES = ("shown", "partly-outside", "outside-crop", "off-layer")
UNSEEN_STATES = ("outside-crop", "off-layer")

_RAW = _pymupdf.TEXTFLAGS_RAWDICT
_DRAWN = _RAW | _pymupdf.TEXT_IGNORE_ACTUALTEXT
_TEXT = _pymupdf.TEXTFLAGS_TEXT

#: Bound on entries per document, so a hostile file cannot make a list unbounded.
MAX_ENTRIES = 200_000
#: Bound on nested form XObjects followed when reading tags directly.
MAX_DEPTH = 32

#: A stencil mask of one sample, 1, which with the default ``/Decode [0 1]``
#: masks the pixel out: drawing it paints nothing.
_BLANK_DICT = ("<</Type/XObject/Subtype/Image/Width 1/Height 1"
               "/ImageMask true/BitsPerComponent 1>>")
_BLANK_DATA = b"\x80"


def open_pdf(pdf_bytes: bytes):
    # Not ``redact.open_pdf``: ``redact`` imports this module.
    return _pymupdf.open(stream=pdf_bytes, filetype="pdf")


# ── images ──────────────────────────────────────────────────────────────

_IMAGE_KINDS = ("fill-image", "fill-imgmask")


def _has_layers(doc) -> bool:
    try:
        return doc.xref_get_key(doc.pdf_catalog(), "OCProperties")[0] != "null"
    except Exception:  # noqa: BLE001 -- an unreadable catalog: treat as layered
        return True


def _every_layer_view(pdf_bytes: bytes):
    """The same bytes opened again with ``/OCProperties`` removed before any
    page loads: MuPDF then draws all optional content. Object numbers are
    those of the source."""
    view = open_pdf(pdf_bytes)
    view.xref_set_key(view.pdf_catalog(), "OCProperties", "null")
    return view


def _image_boxes(page) -> list[tuple]:
    """Every image draw on the page, in drawing order (cheap: no decoding)."""
    return [tuple(float(v) for v in box) for kind, box in page.get_bboxlog()
            if kind in _IMAGE_KINDS]


def _same_box(a, b) -> bool:
    return all(abs(x - y) <= BOX_TOL_PT for x, y in zip(a, b))


def _crop(page) -> tuple:
    # Draw-log boxes are in the unrotated page space, whose origin is the crop
    # box's top-left corner.
    w, h = page.cropbox.width, page.cropbox.height
    return (-EDGE_TOL_PT, -EDGE_TOL_PT, w + EDGE_TOL_PT, h + EDGE_TOL_PT)


def _inside_share(box, crop) -> float:
    w, h = box[2] - box[0], box[3] - box[1]
    if w <= 0 or h <= 0:
        inside = crop[0] <= box[0] <= crop[2] and crop[1] <= box[1] <= crop[3]
        return 1.0 if inside else 0.0
    ix = max(0.0, min(box[2], crop[2]) - max(box[0], crop[0]))
    iy = max(0.0, min(box[3], crop[3]) - max(box[1], crop[1]))
    return (ix * iy) / (w * h)


def _overlaps(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _page_placements(page, view_page) -> list[tuple[str, tuple]]:
    """``(state, box)`` per image draw. ``view_page`` is the page drawn with
    every layer on, or ``None`` when the document has no optional content."""
    crop = _crop(page)
    shown = _image_boxes(page)
    every = _image_boxes(view_page) if view_page is not None else list(shown)
    out = []
    for box in every:
        hit = next((i for i, b in enumerate(shown) if _same_box(b, box)), None)
        if hit is None:
            out.append(("off-layer", box))
            continue
        shown.pop(hit)
        share = _inside_share(box, crop)
        out.append(("outside-crop" if share <= 0.0 else
                     "shown" if share >= 1.0 else "partly-outside", box))
    # A draw the default view makes that the every-layer view does not cannot
    # happen; if it does, it is shown, and saying so is the safe side.
    out.extend(("shown", box) for box in shown)
    return out


def _xref_boxes(page) -> list[tuple[int, tuple]]:
    """``(xref, box)`` per image draw. Slow: MuPDF hashes every image to
    resolve the numbers."""
    return [(int(i.get("xref") or 0), tuple(float(v) for v in i["bbox"]))
            for i in page.get_image_info(xrefs=True) if i.get("bbox") is not None]


def _xref_at(pairs, box) -> int:
    return next((x for x, b in pairs if _same_box(b, box)), 0)


def _pixels(doc, xref: int) -> int:
    try:
        w = doc.xref_get_key(xref, "Width")
        h = doc.xref_get_key(xref, "Height")
        return int(float(w[1])) * int(float(h[1]))
    except Exception:  # noqa: BLE001
        return 0


def _unreadable_document(doc) -> dict:
    """The answer for a document whose structure cannot be read: every page
    unchecked, never "nothing unseen"."""
    return {"unseen": [], "counts": {state: 0 for state in IMAGE_STATES},
            "unreadable_pages": list(range(len(doc)))}


def _list_unseen_images(doc, view) -> dict:
    counts = {state: 0 for state in IMAGE_STATES}
    unseen: list[dict] = []
    unreadable: list[int] = []
    for pno in range(len(doc)):
        try:
            page = doc[pno]
            view_page = view[pno] if view is not None else None
            placements = _page_placements(page, view_page)
            hidden = [(s, b) for s, b in placements
                      if s in ("off-layer", "outside-crop")]
            if hidden:
                default = _xref_boxes(page)
                every = _xref_boxes(view_page) if view_page is not None else default
                for state, box in hidden:
                    xref = _xref_at(every if state == "off-layer" else default, box)
                    unseen.append({"page": pno, "state": state,
                                   "bbox": [float(v) for v in box], "xref": xref})
        except Exception as exc:  # noqa: BLE001 -- reported, never "nothing"
            logger.warning("image listing could not read a page (%s)",
                           type(exc).__name__)
            unreadable.append(pno)
            continue
        for state, _ in placements:
            counts[state] += 1
        if len(unseen) > MAX_ENTRIES:
            raise ValueError("too many image placements to list")
    shown_anywhere = _shown_xrefs(doc, {u["xref"] for u in unseen if u["xref"]},
                                  unreadable)
    for u in unseen:
        u["pixels"] = _pixels(doc, u["xref"]) if u["xref"] else 0
        u["xref_shown"] = u["xref"] in shown_anywhere
    return {"unseen": unseen, "counts": counts, "unreadable_pages": unreadable}


def _shown_xrefs(doc, xrefs: set[int], unreadable: list[int]) -> set[int]:
    """The objects among ``xrefs`` that some page shows, read only on pages
    whose resources (forms included) hold one of them."""
    shown: set[int] = set()
    if not xrefs:
        return shown
    for pno in range(len(doc)):
        page = doc[pno]
        try:
            held = {img[0] for img in page.get_images(full=True)} & (xrefs - shown)
            if not held:
                continue
            crop = _crop(page)
            for xref, box in _xref_boxes(page):
                if xref in held and _inside_share(box, crop) > 0.0:
                    shown.add(xref)
        except Exception:  # noqa: BLE001 -- unknown counts as shown: safe side
            shown |= xrefs
            if pno not in unreadable:
                unreadable.append(pno)
    return shown


def list_unseen_images_pdf(pdf_bytes: bytes) -> dict:
    """Image placements the page as it opens never shows, in ``pdf_bytes``.

    Returns ``{"unseen": [{"page": int, "state": str, "bbox": [x0, y0, x1,
    y1], "xref": int, "pixels": int, "xref_shown": bool}], "counts": {state:
    n}, "unreadable_pages": [int]}``. ``bbox`` is in the unrotated page space
    ``remove_images`` takes; ``counts`` covers every placement, by the states
    in :data:`IMAGE_STATES`.
    """
    doc = open_pdf(pdf_bytes)
    view = None
    try:
        if doc.needs_pass:
            raise ValueError("list_unseen_images needs an unencrypted document")
        if _has_layers(doc):
            try:
                view = _every_layer_view(pdf_bytes)
            except Exception as exc:  # noqa: BLE001 -- e.g. an unreadable catalog
                logger.warning("image listing could not read the layers (%s)",
                               type(exc).__name__)
                return _unreadable_document(doc)
        return _list_unseen_images(doc, view)
    finally:
        if view is not None:
            view.close()
        doc.close()


def list_unseen_images(doc) -> dict:
    """:func:`list_unseen_images_pdf` over an open document. The document is
    only read: the every-layer view is a separate copy of its bytes."""
    if not _has_layers(doc):
        return _list_unseen_images(doc, None)
    try:
        view = _every_layer_view(doc.tobytes())
    except Exception as exc:  # noqa: BLE001 -- e.g. an unreadable catalog
        logger.warning("image listing could not read the layers (%s)",
                       type(exc).__name__)
        return _unreadable_document(doc)
    try:
        return _list_unseen_images(doc, view)
    finally:
        view.close()


def validate_blank_images(xrefs) -> list[int]:
    """The ``blank_images`` field as a sorted list of distinct object numbers."""
    if not isinstance(xrefs, list):
        raise ValueError("blank_images must be a list of object numbers")
    out = set()
    for x in xrefs:
        if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
            raise ValueError("blank_images entries must be positive integers")
        out.add(x)
    return sorted(out)


def validate_remove_images(boxes) -> list[tuple[int, tuple]]:
    """The ``remove_images`` field as ``[(page, (x0, y0, x1, y1)), ...]``."""
    if not isinstance(boxes, list):
        raise ValueError("remove_images must be a list")
    out = []
    for entry in boxes:
        if not isinstance(entry, dict) or set(entry) != {"page", "box"}:
            raise ValueError("remove_images entries are {page, box}")
        page, box = entry["page"], entry["box"]
        if isinstance(page, bool) or not isinstance(page, int) or page < 0:
            raise ValueError("remove_images page must be a page index")
        if not isinstance(box, list) or len(box) != 4 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) for v in box):
            raise ValueError("remove_images box must be four numbers")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("remove_images box must have area")
        out.append((page, tuple(float(v) for v in box)))
    return out


def blank_images(doc, xrefs: list[int]) -> dict:
    """Replace each named image object with a stencil that paints nothing.

    Returns ``{"blanked": n, "skipped": [xref, ...]}``; an object number that
    is out of range or not an image is skipped and left as it was.
    """
    blanked, skipped = 0, []
    for xref in xrefs:
        try:
            if not 0 < xref < doc.xref_length() or not doc.xref_is_stream(xref) \
                    or doc.xref_get_key(xref, "Subtype") != ("name", "/Image"):
                skipped.append(xref)
                continue
            doc.update_object(xref, _BLANK_DICT)
            doc.update_stream(xref, _BLANK_DATA, compress=False)
            blanked += 1
        except Exception as exc:  # noqa: BLE001 -- reported, never half-done
            logger.warning("image blank skipped (%s)", type(exc).__name__)
            skipped.append(xref)
    return {"blanked": blanked, "skipped": skipped}


def remove_images(doc, boxes: list[tuple[int, tuple]]) -> dict:
    """Remove the image draws inside each box, which must lie wholly outside
    its page's crop box with no other image draw reaching into it.

    Returns ``{"removed": n, "skipped": [{"page", "box"}, ...]}``. Text and
    drawings are never touched; a box that fails either test is skipped.
    """
    removed, skipped = 0, []
    by_page: dict[int, list[tuple]] = {}
    for pno, box in boxes:
        by_page.setdefault(pno, []).append(box)
    try:
        view = _every_layer_view(doc.tobytes()) if _has_layers(doc) else None
    except Exception as exc:  # noqa: BLE001 -- nothing removed, all reported
        logger.warning("image removal could not read the layers (%s)",
                       type(exc).__name__)
        return {"removed": 0, "skipped": [{"page": p, "box": list(b)} for p, b in boxes]}
    try:
        for pno, page_boxes in sorted(by_page.items()):
            if not 0 <= pno < len(doc):
                skipped.extend({"page": pno, "box": list(b)} for b in page_boxes)
                continue
            page = doc[pno]
            crop = _crop(page)
            draws = _image_boxes(view[pno] if view is not None else page)
            todo = []
            for box in page_boxes:
                inside = _inside_share(box, crop) > 0.0
                touched = [d for d in draws if _overlaps(d, box)]
                foreign = [d for d in touched if not _contained(d, box)]
                if inside or not touched or foreign:
                    skipped.append({"page": pno, "box": list(box)})
                    continue
                todo.append(box)
            for box in todo:
                page.add_redact_annot(_pymupdf.Rect(box), fill=False)
            if todo:
                page.apply_redactions(images=_pymupdf.PDF_REDACT_IMAGE_REMOVE,
                                      graphics=_pymupdf.PDF_REDACT_LINE_ART_NONE,
                                      text=_pymupdf.PDF_REDACT_TEXT_NONE)
                removed += len(todo)
    finally:
        if view is not None:
            view.close()
    return {"removed": removed, "skipped": skipped}


def _contained(inner, outer) -> bool:
    return (inner[0] >= outer[0] - BOX_TOL_PT and inner[1] >= outer[1] - BOX_TOL_PT
            and inner[2] <= outer[2] + BOX_TOL_PT and inner[3] <= outer[3] + BOX_TOL_PT)


# ── tagged text ─────────────────────────────────────────────────────────


_NOT_WORD = re.compile(r"[\W_]+")
_TAG_KEY = re.compile(rb"/ActualText\s*([(<])")
_REF = re.compile(r"(\d+)\s+\d+\s+R")
_ESCAPES = {0x6E: b"\n", 0x72: b"\r", 0x74: b"\t", 0x62: b"\b", 0x66: b"\f",
            0x0D: b"", 0x0A: b""}


def _compact(text: str) -> str:
    """Letters and digits only, case-folded and NFKC-normalised (which
    expands ligatures), so spacing, punctuation and hyphenation never count
    as a difference."""
    return _NOT_WORD.sub("", unicodedata.normalize("NFKC", text).casefold())


def _literal(data: bytes, i: int) -> bytes:
    """The literal string opening at ``data[i] == b"("``, unescaped."""
    out, depth, i = bytearray(), 1, i + 1
    while i < len(data):
        c = data[i]
        if c == 0x5C:
            i += 1
            if i >= len(data):
                break
            e = data[i]
            if 0x30 <= e <= 0x37:
                j = i
                while j < min(i + 3, len(data)) and 0x30 <= data[j] <= 0x37:
                    j += 1
                out.append(int(data[i:j], 8) & 0xFF)
                i = j
                continue
            out += _ESCAPES.get(e, bytes([e]))
        elif c == 0x28:
            depth += 1
            out.append(c)
        elif c == 0x29:
            depth -= 1
            if depth == 0:
                break
            out.append(c)
        else:
            out.append(c)
        i += 1
    return bytes(out)


def _hex(data: bytes, i: int) -> bytes:
    end = data.find(b">", i)
    digits = re.sub(rb"[^0-9A-Fa-f]", b"", data[i + 1:end if end >= 0 else len(data)])
    if len(digits) % 2:
        digits += b"0"
    return bytes.fromhex(digits.decode("ascii"))


def _text_string(raw: bytes) -> str:
    if raw[:2] in (b"\xfe\xff", b"\xff\xfe"):
        return raw.decode("utf-16", "replace")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8", "replace")
    return raw.decode("latin-1")


def tag_strings(data: bytes) -> list[str]:
    """Every ``/ActualText`` value written in ``data``: a content stream or a
    dictionary's text."""
    out = []
    for m in _TAG_KEY.finditer(data):
        start = m.start(1)
        raw = _literal(data, start) if data[start] == 0x28 else _hex(data, start)
        out.append(_text_string(raw))
    return out


def _page_tag_strings(doc, page) -> list[str]:
    """``/ActualText`` values the page's content reaches: its content streams,
    the form XObjects its resources hold (followed into theirs), and the
    marked-content property lists in those resources."""
    seen: set[int] = set()
    found: list[str] = []

    def walk(owner: int, depth: int) -> None:
        if depth > MAX_DEPTH:
            raise ValueError("form XObjects nest too deeply to read")
        kind, val = doc.xref_get_key(owner, "Resources")
        if kind == "xref":
            m = _REF.match(val)
            if not m:
                return
            owner, prefix = int(m.group(1)), ""
        elif kind == "dict":
            prefix = "Resources/"
        else:
            return
        for sub in ("XObject", "Properties"):
            kind, val = doc.xref_get_key(owner, prefix + sub)
            if kind == "xref":
                m = _REF.match(val)
                if not m:
                    continue
                text = doc.xref_object(int(m.group(1)))
            elif kind == "dict":
                text = val
            else:
                continue
            if sub == "Properties":
                found.extend(tag_strings(text.encode("latin-1", "replace")))
            for ref in _REF.findall(text):
                visit(int(ref), sub, depth)

    def visit(xref: int, sub: str, depth: int) -> None:
        if xref in seen or not 0 < xref < doc.xref_length():
            return
        seen.add(xref)
        if doc.xref_is_stream(xref):
            if sub == "XObject" and doc.xref_get_key(xref, "Subtype")[1] == "/Form":
                found.extend(tag_strings(doc.xref_stream(xref) or b""))
                walk(xref, depth + 1)
        elif sub == "Properties":
            found.extend(tag_strings(
                doc.xref_object(xref).encode("latin-1", "replace")))

    for xref in page.get_contents():
        found.extend(tag_strings(doc.xref_stream(xref) or b""))
    walk(page.xref, 0)
    return found


def _lines(page, flags) -> list[list[tuple[str, tuple]]]:
    """Each text line as ``[(character, box), ...]`` in reading order."""
    out = []
    for block in page.get_text("rawdict", flags=flags).get("blocks", []):
        for line in block.get("lines", ()):
            chars = [(c["c"], tuple(c["bbox"])) for span in line.get("spans", ())
                     for c in span.get("chars", ())]
            if chars:
                out.append(chars)
    return out


def _union(boxes) -> tuple | None:
    boxes = [b for b in boxes if b[2] >= b[0] and b[3] >= b[1]]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _differences(tagged, drawn) -> list[tuple[str, str, tuple | None]]:
    """Where two character sequences differ, as ``(tag side, drawn side,
    box)``. Differences separated by fewer than :data:`_MERGE_GAP` equal
    characters (and no line break) are one region, so a tag that differs
    from its glyphs throughout is one entry, not one per coincidental
    letter in common."""
    a = "".join(c for c, _ in tagged)
    b = "".join(c for c, _ in drawn)
    regions: list[list[int]] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if regions:
            last = regions[-1]
            gap_a, gap_b = a[last[1]:i1], b[last[3]:j1]
            if len(gap_a) < _MERGE_GAP and "\n" not in gap_a and "\n" not in gap_b:
                last[1], last[3] = i2, j2
                continue
        regions.append([i1, i2, j1, j2])
    return [(a[i1:i2], b[j1:j2],
             _union([bx for _, bx in tagged[i1:i2]] + [bx for _, bx in drawn[j1:j2]]))
            for i1, i2, j1, j2 in regions]


#: Equal characters shorter than this between two differences join them.
_MERGE_GAP = 4


def _page_tag_text(doc, pno: int) -> list[dict]:
    page = doc[pno]
    entries: list[dict] = []
    rot = page.rotation_matrix
    with_tag = page.get_text("text", flags=_TEXT)
    drawn_text = page.get_text("text", flags=_TEXT | _pymupdf.TEXT_IGNORE_ACTUALTEXT)
    if with_tag != drawn_text:
        # A tag replaces glyphs within a line, so lines are compared one to
        # one when both readings have the same lines; otherwise the page is
        # compared as one sequence with line breaks kept.
        tagged, drawn = _lines(page, _RAW), _lines(page, _DRAWN)
        if len(tagged) != len(drawn):
            nl = ("\n", (0.0, 0.0, -1.0, -1.0))
            tagged = [[c for line in tagged for c in line + [nl]]]
            drawn = [[c for line in drawn for c in line + [nl]]]
        for t_line, d_line in zip(tagged, drawn):
            if [c for c, _ in t_line] == [c for c, _ in d_line]:
                continue
            for text, under, box in _differences(t_line, d_line):
                text, under = text.strip("\n"), under.strip("\n")
                # Only where extraction returns a letter or digit: a tag that
                # drops a hyphen or a space (most tags in real files) returns
                # nothing the page does not draw.
                if text == under or not _compact(text):
                    continue
                entries.append({"page": pno, "text": text, "drawn": under,
                                "bbox": (None if box is None else
                                         [float(v) for v in _pymupdf.Rect(box) * rot])})
    # Tags extraction never returns: their text is in the content but not in
    # extraction's output.
    returned = _compact(with_tag)
    for s in _page_tag_strings(doc, page):
        c = _compact(s)
        if c and c not in returned:
            entries.append({"page": pno, "text": s, "drawn": "", "bbox": None})
    return entries


def list_tag_text(doc) -> dict:
    """Where text extraction's text differs from what the content draws.

    Returns ``{"entries": [{"page", "text", "drawn", "bbox"}],
    "unreadable_pages": [int]}``; see the module docstring.
    """
    entries: list[dict] = []
    unreadable: list[int] = []
    for pno in range(len(doc)):
        try:
            entries.extend(_page_tag_text(doc, pno))
        except Exception as exc:  # noqa: BLE001 -- reported, never "nothing"
            logger.warning("tag listing could not read a page (%s)",
                           type(exc).__name__)
            unreadable.append(pno)
        if len(entries) > MAX_ENTRIES:
            raise ValueError("too many tag entries to list")
    return {"entries": entries, "unreadable_pages": unreadable}


def list_tag_text_pdf(pdf_bytes: bytes) -> dict:
    """:func:`list_tag_text` over PDF bytes."""
    doc = open_pdf(pdf_bytes)
    try:
        if doc.needs_pass:
            raise ValueError("list_tag_text needs an unencrypted document")
        return list_tag_text(doc)
    finally:
        doc.close()

