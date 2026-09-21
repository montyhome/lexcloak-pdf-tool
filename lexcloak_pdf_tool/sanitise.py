"""Strip what a PDF carries outside its visible page content (v0.9.0).

``apply_redactions`` rewrites page content streams, and ``_scrub_residue``
removes annotations, attachments, document JavaScript and thumbnails. This
module covers the remaining places a delivered file can still hold content
its readers never see. Each function is a separate step so that a caller (and
a test) can tell which one did the work, and each is fail-safe: an object it
cannot parse is left exactly as it was, never half-rewritten, and
:func:`residue_report` says what is left.

* **Active content** -- any ``/OpenAction`` that is not a plain
  destination, additional-actions (``/AA``) dictionaries, and any link or
  annotation action other than go-to, URI and named. JavaScript, launch,
  form-submit and import actions are removed.
* **Extra metadata** -- ``/Info`` keys beyond the eight standard ones, page
  and image ``/Metadata`` streams, ``/PieceInfo``, catalog and page keys
  outside the PDF specification's own, and the trailer ``/ID`` (regenerated
  on save, so a delivered file can no longer be matched to its source by it).
* **Associated files** -- ``/AF`` at the catalog, page and object level, the
  twin of the embedded-file scrub.
* **Image metadata** -- comment and EXIF / XMP segments inside JPEG streams,
  removed without re-encoding: the entropy-coded scan data is copied verbatim.
* **Tagged text** -- ``/ActualText`` and ``/Alt`` on marked-content spans, in
  property lists and on structure elements, on the pages a redaction touched.
  A tag that repeats a sentence keeps the words a burn removed from the
  glyphs, and text extraction returns the tag's text in place of the drawn
  characters, so on a touched page the tag cannot stay.

A tag on a page nothing was redacted from is kept: it carries copy-and-paste
fidelity (ligatures, hyphenation) and accessibility text.
"""
from __future__ import annotations

import logging
import re

import pymupdf as _pymupdf

logger = logging.getLogger(__name__)

#: The eight keys ``strip_metadata`` clears by name, plus ``/Trapped``.
STANDARD_INFO_KEYS = frozenset({
    "Title", "Author", "Subject", "Creator", "Producer",
    "CreationDate", "ModDate", "Keywords", "Trapped",
})

#: Keys the PDF specification defines for the document catalog. ``Info``,
#: ``Root`` and ``ID`` are listed so a library that reports trailer keys
#: through the catalog handle never has them removed.
CATALOG_KEYS = frozenset({
    "Type", "Version", "Extensions", "Pages", "PageLabels", "Names", "Dests",
    "ViewerPreferences", "PageLayout", "PageMode", "Outlines", "Threads",
    "OpenAction", "AA", "URI", "AcroForm", "Metadata", "StructTreeRoot",
    "MarkInfo", "Lang", "SpiderInfo", "OutputIntents", "PieceInfo",
    "OCProperties", "Perms", "Legal", "Requirements", "Collection",
    "NeedsRendering", "DSS", "AF", "DPartRoot",
    "Info", "Root", "ID", "Size", "Prev", "Encrypt",
})

#: Keys the PDF specification defines for a page object, less the three that
#: are metadata-class (``LastModified``, ``Metadata``, ``PieceInfo``) and
#: ``ID`` (a web-capture identifier), which are removed.
PAGE_KEYS = frozenset({
    "Type", "Parent", "Resources", "MediaBox", "CropBox", "BleedBox",
    "TrimBox", "ArtBox", "BoxColorInfo", "Contents", "Rotate", "Group",
    "Thumb", "B", "Dur", "Trans", "Annots", "AA", "StructParents", "PZ",
    "SeparationInfo", "Tabs", "TemplateInstantiated", "PresSteps",
    "UserUnit", "VP", "AF", "OutputIntents", "DPart",
})

#: Action types that stay. Everything else (JavaScript, Launch, SubmitForm,
#: ImportData, GoToR, GoToE, Rendition, ...) is removed. A URI action's
#: destination is a separate question from whether it executes anything.
KEPT_ACTIONS = frozenset({"GoTo", "URI", "Named"})

_S_RE = re.compile(r"/S\s*/(\w+)")


# ── small helpers ───────────────────────────────────────────────────────


def _object_text(doc, xref: int) -> str:
    try:
        return doc.xref_object(xref, compressed=False) or ""
    except Exception:  # noqa: BLE001 -- an unreadable object is skipped
        return ""


def _key_text(doc, xref: int, key: str) -> tuple[str, str]:
    try:
        return doc.xref_get_key(xref, key)
    except Exception:  # noqa: BLE001
        return ("null", "null")


def _null(doc, xref: int, key: str) -> None:
    """Remove ``key`` from ``xref``, or leave it exactly as it was.

    PyMuPDF rejects some legal key spellings (a name containing a space, for
    one). An export must not die over a key it cannot name, so the failure is
    logged and the key stays; ``residue_report`` then counts it, which is how
    the caller finds out.
    """
    try:
        doc.xref_set_key(xref, key, "null")
    except (ValueError, RuntimeError) as exc:
        logger.warning("could not remove a key (%s); left in place",
                       type(exc).__name__)


def _live_keys(doc, xref: int) -> list[str]:
    """Keys of ``xref`` whose value is not null. A removed key is written back
    as ``/Key null``, which the PDF specification treats as absent, so a key
    with a null value is already gone and must not count as left behind."""
    try:
        keys = doc.xref_get_keys(xref)
    except Exception:  # noqa: BLE001
        return []
    return [k for k in keys if _key_text(doc, xref, k)[0] != "null"]


def _dict_text(doc, kind: str, value: str) -> str:
    """The text of a dictionary value, following one level of indirection."""
    if kind == "dict":
        return value
    if kind == "xref":
        try:
            return _object_text(doc, int(value.split()[0]))
        except (ValueError, IndexError):
            return ""
    return ""


def _catalog(doc) -> int:
    return doc.pdf_catalog()


# ── active content ──────────────────────────────────────────────────────


#: What may run by itself when a document opens: only a go-to. A file that
#: opens a web page or changes the view on load is active content.
KEPT_OPEN_ACTIONS = frozenset({"GoTo"})


def _action_is_kept(text: str, allowed: frozenset = KEPT_ACTIONS) -> bool:
    """True for an action of an ``allowed`` type with nothing chained after it.

    ``/Next`` chains another action, so an otherwise harmless go-to could
    lead to a script; an action carrying one is not kept.
    """
    m = _S_RE.search(text)
    if not m or m.group(1) not in allowed:
        return False
    return "/Next" not in text


def _has_action_type(text: str) -> bool:
    return _S_RE.search(text) is not None


def strip_active_content(doc) -> int:
    """Remove executable and navigating actions. Returns how many were removed.

    Walks every object rather than only the catalog, pages and annotations,
    because an action can also hang off an outline entry or a form field.
    Keys named ``/A`` also appear as structure-element attribute dictionaries,
    which carry no ``/S`` action type and are left alone.
    """
    removed = 0
    for xref in range(1, doc.xref_length()):
        text = _object_text(doc, xref)
        if not text or ("/A" not in text and "/OpenAction" not in text):
            continue
        if "/AA" in text:
            if _key_text(doc, xref, "AA")[0] != "null":
                _null(doc, xref, "AA")
                removed += 1
        for key in ("A", "OpenAction", "PA"):
            kind, value = _key_text(doc, xref, key)
            if kind not in ("dict", "xref"):
                continue
            action = _dict_text(doc, kind, value)
            if not _has_action_type(action):
                continue
            allowed = KEPT_OPEN_ACTIONS if key == "OpenAction" else KEPT_ACTIONS
            if not _action_is_kept(action, allowed):
                _null(doc, xref, key)
                removed += 1
    return removed


# ── metadata-class content ──────────────────────────────────────────────


def strip_extra_metadata(doc) -> int:
    """Remove metadata the eight-key clear does not reach. Returns a count.

    Order matters for the caller: this runs after the standard fields are
    emptied, so what remains in ``/Info`` is only keys outside the standard
    set. Keys are removed by name, never by rebuilding the dictionary, so a
    caller that stamps standard fields afterwards is unaffected.
    """
    removed = 0
    info_kind, info_val = _key_text(doc, -1, "Info")
    if info_kind == "xref":
        info = int(info_val.split()[0])
        extra = [k for k in _live_keys(doc, info) if k not in STANDARD_INFO_KEYS]
        if extra:
            # Replace the dictionary rather than null keys one by one: a
            # custom key can carry a name PyMuPDF cannot address (``Form
            # fields``, with a space, is real), and the standard keys are
            # already blank by the time this runs. A caller that stamps
            # standard fields afterwards adds them back to the empty dict.
            doc.update_object(info, "<<>>")
            removed += len(extra)

    cat = _catalog(doc)
    for key in _live_keys(doc, cat):
        if key not in CATALOG_KEYS:
            _null(doc, cat, key)
            removed += 1
    # The catalog's own /PieceInfo is a legal catalog key, and private data.
    if _key_text(doc, cat, "PieceInfo")[0] != "null":
        _null(doc, cat, "PieceInfo")
        removed += 1

    for page in doc:
        for key in _live_keys(doc, page.xref):
            if key not in PAGE_KEYS:
                _null(doc, page.xref, key)
                removed += 1

    # Per-object keys wherever they sit: a page, an image, a form XObject.
    # (The catalog's own /Metadata is removed by ``del_xml_metadata``.)
    for xref in range(1, doc.xref_length()):
        text = _object_text(doc, xref)
        if not text:
            continue
        for key in ("Metadata", "PieceInfo"):
            if f"/{key}" in text and xref != cat:
                if _key_text(doc, xref, key)[0] != "null":
                    _null(doc, xref, key)
                    removed += 1

    # A fresh identifier on save: the permanent half of the source's is
    # otherwise carried into the delivered file.
    if _key_text(doc, -1, "ID")[0] != "null":
        doc.xref_set_key(-1, "ID", "null")
        removed += 1
    return removed


def strip_associated_files(doc) -> int:
    """Remove ``/AF`` (associated files) at every level. Returns a count."""
    removed = 0
    for xref in range(1, doc.xref_length()):
        text = _object_text(doc, xref)
        if "/AF" not in text:
            continue
        if _key_text(doc, xref, "AF")[0] != "null":
            _null(doc, xref, "AF")
            removed += 1
    return removed


# ── JPEG segments ───────────────────────────────────────────────────────

# Segments that carry description rather than pixels. APP0 is kept only as a
# plain JFIF header, APP2 only as an ICC profile, and APP14 (the colour
# transform flag a CMYK image needs) is always kept.
_DROP_MARKERS = frozenset({0xE1, 0xFE, 0xEF} | set(range(0xE3, 0xEE)))


def _keep_segment(marker: int, payload: bytes) -> bool:
    if marker in _DROP_MARKERS:
        return False
    if marker == 0xE0:
        return payload.startswith(b"JFIF\x00")
    if marker == 0xE2:
        return payload.startswith(b"ICC_PROFILE\x00")
    return True


def strip_jpeg_segments(data: bytes) -> bytes:
    """``data`` without comment, EXIF, XMP and other description segments.

    Lossless: every quantisation table, Huffman table, frame header and scan
    is copied byte for byte, so decoded pixels are identical. Anything after
    the final end-of-image marker is dropped, since a decoder never reads it
    and it is a place to hide content. If the structure does not parse, or
    the result lacks a frame or a scan, ``data`` is returned unchanged.
    """
    n = len(data)
    if n < 4 or data[0:2] != b"\xff\xd8":
        return data
    out = bytearray(b"\xff\xd8")
    i = 2
    saw_frame = saw_scan = saw_eoi = False
    while i < n:
        if data[i] != 0xFF:
            return data
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            return data
        marker = data[i]
        i += 1
        if marker == 0xD9:                       # end of image
            out += b"\xff\xd9"
            saw_eoi = True
            break
        if marker == 0x00 or marker == 0x01 or 0xD0 <= marker <= 0xD8:
            out += bytes((0xFF, marker))          # standalone marker
            continue
        if i + 2 > n:
            return data
        length = (data[i] << 8) | data[i + 1]
        if length < 2 or i + length > n:
            return data
        payload = data[i + 2:i + length]
        if marker == 0xDA:                        # start of scan
            saw_scan = True
            out += bytes((0xFF, marker)) + data[i:i + length]
            i += length
            j = i                                 # entropy-coded data
            while j < n - 1:
                if data[j] == 0xFF and data[j + 1] != 0x00 \
                        and not (0xD0 <= data[j + 1] <= 0xD7):
                    break
                j += 1
            else:
                return data
            out += data[i:j]
            i = j
            continue
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            saw_frame = True
        if _keep_segment(marker, payload):
            out += bytes((0xFF, marker)) + data[i:i + length]
        i += length
    if not (saw_frame and saw_scan and saw_eoi):
        return data
    return bytes(out)


def strip_image_metadata(doc) -> int:
    """Strip description segments from every plain-DCT image. Returns a count.

    Only an image whose ``/Filter`` is exactly ``/DCTDecode`` is touched:
    a stacked filter is left alone rather than guessed at.
    """
    changed = 0
    for xref in range(1, doc.xref_length()):
        if _key_text(doc, xref, "Subtype") != ("name", "/Image"):
            continue
        if _key_text(doc, xref, "Filter") != ("name", "/DCTDecode"):
            continue
        try:
            raw = doc.xref_stream_raw(xref)
        except Exception:  # noqa: BLE001
            continue
        if not raw:
            continue
        new = strip_jpeg_segments(raw)
        if new == raw:
            continue
        parms = _key_text(doc, xref, "DecodeParms")
        try:
            doc.update_stream(xref, new, new=False, compress=False)
            # update_stream drops the filter of the stream it replaces.
            doc.xref_set_key(xref, "Filter", "/DCTDecode")
            if parms[0] != "null":
                doc.xref_set_key(xref, "DecodeParms", parms[1])
            changed += 1
        except Exception as exc:  # noqa: BLE001 -- leave the image as it was
            logger.warning("image metadata strip skipped (%s)", type(exc).__name__)
    return changed


def _count_image_segments(doc) -> int:
    """Plain-DCT images that still carry a segment ``strip_jpeg_segments`` drops."""
    left = 0
    for xref in range(1, doc.xref_length()):
        if _key_text(doc, xref, "Subtype") != ("name", "/Image"):
            continue
        if _key_text(doc, xref, "Filter") != ("name", "/DCTDecode"):
            continue
        try:
            raw = doc.xref_stream_raw(xref)
        except Exception:  # noqa: BLE001
            continue
        if raw and strip_jpeg_segments(raw) != raw:
            left += 1
    return left


# ── tagged text ─────────────────────────────────────────────────────────

_TAG_KEYS = (b"ActualText", b"Alt")
_WS = b" \t\r\n\x0c\x00"
_DELIM = b"()<>[]{}/%"


def _skip_literal(data: bytes, i: int) -> int:
    """Index just past the literal string opening at ``data[i] == b"("``."""
    depth, n = 0, len(data)
    while i < n:
        c = data[i]
        if c == 0x5C:                       # backslash escapes the next byte
            i += 2
            continue
        if c == 0x28:
            depth += 1
        elif c == 0x29:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unterminated string")


def strip_tag_keys(data: bytes) -> tuple[bytes, int]:
    """``data`` with every ``/ActualText`` and ``/Alt`` string value removed.

    Scans a content stream token by token, so a key spelled inside a string,
    a comment or inline image data is never mistaken for a real entry.
    Raises ``ValueError`` on a stream it cannot tokenise; the caller then
    leaves the stream untouched.
    """
    out = bytearray()
    i, n, removed = 0, len(data), 0
    while i < n:
        c = data[i]
        if c == 0x28:                                    # literal string
            j = _skip_literal(data, i)
            out += data[i:j]
            i = j
        elif c == 0x25:                                  # comment
            j = i
            while j < n and data[j] not in b"\r\n":
                j += 1
            out += data[i:j]
            i = j
        elif data[i:i + 2] in (b"<<", b">>"):            # dictionary brackets
            out += data[i:i + 2]
            i += 2
        elif c == 0x3C:                                  # hex string
            j = data.index(b">", i) + 1
            out += data[i:j]
            i = j
        elif c == 0x2F:                                  # name
            j = i + 1
            while j < n and data[j] not in _WS and data[j] not in _DELIM:
                j += 1
            name = data[i + 1:j]
            k = j
            while k < n and data[k] in _WS:
                k += 1
            if name in _TAG_KEYS and k < n and data[k] in b"(<" \
                    and data[k:k + 2] != b"<<":          # a string, not a dict
                end = _skip_literal(data, k) if data[k] == 0x28 \
                    else data.index(b">", k) + 1
                i = end                                  # drop key and value
                removed += 1
            else:
                out += data[i:j]
                i = j
        elif data[i:i + 2] in (b"ID", b"BI") and _is_operator(data, i, 2):
            if data[i:i + 2] == b"ID":                   # inline image body
                m = re.compile(rb"[\r\n\t ]EI(?=[\r\n\t ]|$)").search(data, i + 2)
                if m is None:
                    raise ValueError("unterminated inline image")
                out += data[i:m.end()]
                i = m.end()
            else:
                out += data[i:i + 2]
                i += 2
        else:
            out.append(c)
            i += 1
    return bytes(out), removed


def _is_operator(data: bytes, i: int, length: int) -> bool:
    before = data[i - 1:i] if i else b" "
    after = data[i + length:i + length + 1] or b" "
    return before in _WS and after in _WS


def _page_content_xrefs(doc, page) -> list[int]:
    """The page's content streams plus the form XObjects it can draw, since a
    producer may draw page content through a form and the tag travels with it."""
    xrefs: list[int] = []
    try:
        xrefs.extend(page.get_contents())
    except Exception:  # noqa: BLE001
        pass
    try:
        xrefs.extend(x[0] for x in page.get_xobjects())
    except Exception:  # noqa: BLE001
        pass
    return xrefs


def _strip_page_tags(doc, page) -> int:
    removed = 0
    for cx in _page_content_xrefs(doc, page):
        try:
            data = doc.xref_stream(cx)
        except Exception:  # noqa: BLE001
            continue
        if not data or not any(b"/" + k in data for k in _TAG_KEYS):
            continue
        try:
            new, count = strip_tag_keys(data)
        except (ValueError, IndexError):
            logger.warning("tagged-text strip skipped a stream it could not parse")
            continue
        if count:
            doc.update_stream(cx, new)
            removed += count
    return removed


def _is_struct_element(text: str) -> bool:
    return ("/Alt" in text or "/ActualText" in text) and "/S" in text \
        and ("/StructElem" in text or "/K" in text or "/P " in text
             or "/P\n" in text)


def _struct_page(doc, xref: int, page_of: dict[int, int], depth: int = 0):
    """The page index a structure element belongs to, from ``/Pg`` or, when it
    has none, from its nearest ancestor that does. ``None`` when unresolved."""
    seen = set()
    while xref and xref not in seen and depth < 64:
        seen.add(xref)
        kind, val = _key_text(doc, xref, "Pg")
        if kind == "xref":
            return page_of.get(int(val.split()[0]))
        kind, val = _key_text(doc, xref, "P")
        if kind != "xref":
            return None
        xref = int(val.split()[0])
        depth += 1
    return None


def strip_actual_text(doc, pages: set[int] | None) -> int:
    """Remove ``/ActualText`` and ``/Alt`` where a redaction touched the page.

    ``pages`` are the document's page indices the redaction removed or burned
    anything from. Content-stream spans and property lists are stripped on
    those pages; a structure element is stripped when its page is one of them
    or cannot be resolved. With no touched pages nothing is removed.
    """
    if not pages:
        return 0
    removed = 0
    page_of = {doc[i].xref: i for i in range(len(doc))}
    for pno in sorted(pages):
        if 0 <= pno < len(doc):
            removed += _strip_page_tags(doc, doc[pno])
    for xref in range(1, doc.xref_length()):
        text = _tagged_object_text(doc, xref)
        if text is None or not _tag_in_scope(doc, xref, text, pages, page_of):
            continue
        try:
            new, count = strip_tag_keys(text.encode("utf-8"))
        except (ValueError, IndexError):
            logger.warning("tagged-text strip skipped an object it could not parse")
            continue
        if count:
            doc.update_object(xref, new.decode("utf-8"))
            removed += count
    return removed


def _tagged_object_text(doc, xref: int) -> str | None:
    """The dictionary text of a non-stream object that may hold a tag, else
    ``None``. Reading the whole object (not one key) reaches a property list
    nested inside a page or resource dictionary."""
    try:
        if doc.xref_is_stream(xref):
            return None
    except Exception:  # noqa: BLE001
        return None
    text = _object_text(doc, xref)
    if "/ActualText" not in text and "/Alt" not in text:
        return None
    return text


def _tag_in_scope(doc, xref: int, text: str, pages, page_of) -> bool:
    """A structure element is in scope when its page was touched or cannot be
    resolved; any other dictionary carrying a tag (a marked-content property
    list) is in scope whenever the export touched a page at all."""
    if _is_struct_element(text):
        pno = _struct_page(doc, xref, page_of)
        return pno is None or pno in pages
    return True


# ── entry points ────────────────────────────────────────────────────────


def sanitise_document(doc, touched_pages: set[int] | None = None) -> dict:
    """Run every step above on ``doc`` in place; return counts per step."""
    return {
        "active_content": strip_active_content(doc),
        "associated_files": strip_associated_files(doc),
        "image_metadata": strip_image_metadata(doc),
        "tagged_text": strip_actual_text(doc, touched_pages),
    }


def residue_report_pdf(pdf_bytes: bytes,
                       touched_pages: set[int] | None = None) -> dict:
    """``residue_report`` over PDF bytes (the IPC-clean form)."""
    doc = _pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.needs_pass:
            raise ValueError("residue_report needs an unencrypted document")
        return residue_report(doc, touched_pages)
    finally:
        doc.close()


def residue_report(doc, touched_pages: set[int] | None = None) -> dict:
    """Counts, never text, of what each step above would still remove.

    Read-only. A delivered file is clean when every count is zero. Tagged text
    is counted only on ``touched_pages``, matching what the export removes.
    """
    report = {
        "active_content": 0, "extra_info_keys": 0, "page_metadata": 0,
        "piece_info": 0, "unknown_keys": 0, "associated_files": 0,
        "image_metadata": _count_image_segments(doc), "tagged_text": 0,
    }
    info_kind, info_val = _key_text(doc, -1, "Info")
    if info_kind == "xref":
        info = int(info_val.split()[0])
        report["extra_info_keys"] = sum(
            1 for k in doc.xref_get_keys(info) if k not in STANDARD_INFO_KEYS
            and _key_text(doc, info, k)[0] != "null")
    cat = _catalog(doc)
    report["unknown_keys"] += sum(
        1 for k in _live_keys(doc, cat) if k not in CATALOG_KEYS)
    if _key_text(doc, cat, "PieceInfo")[0] != "null":
        report["piece_info"] += 1
    for page in doc:
        report["unknown_keys"] += sum(
            1 for k in _live_keys(doc, page.xref) if k not in PAGE_KEYS)
    page_of = {doc[i].xref: i for i in range(len(doc))} if touched_pages else {}
    for xref in range(1, doc.xref_length()):
        text = _object_text(doc, xref)
        if not text:
            continue
        if "/AA" in text and _key_text(doc, xref, "AA")[0] != "null":
            report["active_content"] += 1
        for key in ("A", "OpenAction", "PA"):
            if f"/{key}" not in text:
                continue
            kind, value = _key_text(doc, xref, key)
            if kind in ("dict", "xref"):
                action = _dict_text(doc, kind, value)
                allowed = KEPT_OPEN_ACTIONS if key == "OpenAction" else KEPT_ACTIONS
                if _has_action_type(action) and not _action_is_kept(action, allowed):
                    report["active_content"] += 1
        if "/Metadata" in text and xref != cat \
                and _key_text(doc, xref, "Metadata")[0] != "null":
            report["page_metadata"] += 1
        if "/PieceInfo" in text and xref != cat \
                and _key_text(doc, xref, "PieceInfo")[0] != "null":
            report["piece_info"] += 1
        if "/AF" in text and _key_text(doc, xref, "AF")[0] != "null":
            report["associated_files"] += 1
        if touched_pages:
            tagged = _tagged_object_text(doc, xref)
            if tagged is not None and _tag_in_scope(
                    doc, xref, tagged, touched_pages, page_of):
                try:
                    report["tagged_text"] += strip_tag_keys(tagged.encode("utf-8"))[1]
                except (ValueError, IndexError):
                    report["tagged_text"] += 1       # unreadable counts as left
    if touched_pages:
        for pno in sorted(touched_pages):
            if not (0 <= pno < len(doc)):
                continue
            for cx in _page_content_xrefs(doc, doc[pno]):
                try:
                    data = doc.xref_stream(cx)
                except Exception:  # noqa: BLE001
                    continue
                if data and any(b"/" + k in data for k in _TAG_KEYS):
                    try:
                        report["tagged_text"] += strip_tag_keys(data)[1]
                    except (ValueError, IndexError):
                        report["tagged_text"] += 1   # unreadable counts as left
    return report
