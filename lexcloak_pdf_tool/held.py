"""A held copy: a document burned one page at a time (v0.16.0, protocol 14).

``apply_redactions`` burns a whole document in one call. A held copy runs the
same steps, through the same functions, a page at a time: each page's text
removal, burn and finishing whenever its boxes are settled, in any order, and
the whole-document steps once, on a copy, when the file is wanted. That file
is the one ``apply_redactions`` gives for the same payload: every page's
pixels, words, images, annotations and links, every document structure, and
the objects with their numbers blanked. It is not byte-identical, because
objects are numbered in the order they are made.
``tests/test_page_burn.py`` holds the comparison.

Three rules make it true, and each is kept here:

1. **Image edits and form flattening run once, when the copy is opened**
   (``blank_images``, ``remove_images``, then the flattening ``bake``).
   Neither reads a match. A burn that reaches an image writes a redacted copy
   and the page then draws the copy, so an edit made to the image after a
   burn lands on an object no page draws any more.
2. **Putting a page back covers what the page step changed.** MuPDF writes a
   burned page's content, its instanced forms and its redacted images as new
   objects and points the page at them, so nothing another page draws is
   changed. In place it changes the page dictionary, the objects only that
   page reaches (its annotations, an indirect ``/Annots`` array), and the
   structure elements the page's ``/StructParents``, and those of the forms it
   draws, reach through the ``/ParentTree``: MuPDF edits their alternate text
   for the characters it removes. A page whose tags are stripped without a
   burn changes its content streams and forms where they are. A page put back
   (:meth:`HeldDocument.restore_page`) gets all of that back from the
   prepared copy. When such an object is shared with another page and has
   changed, that page goes back too, so no page keeps a change made for
   inputs it no longer has.
3. **A held page is finished.** Its share of the assembly runs in the page
   step: a blacked-out page is covered, its annotations other than links are
   deleted, and its content tags are stripped when the page is touched. So a
   held page renders and extracts as it will in the delivered file, and a
   page's checks can read the held copy.

The assembly (:meth:`HeldDocument.assemble`) takes the whole payload and
checks every page against it. A page burned with other inputs, or not
burned, is put back and burned again on the fork, so a stale page never
reaches the file, whatever the caller believed. The held copy itself is left
as it was.

**Size.** A burn leaves the streams it creates uncompressed, which would make
a held copy of a scanned document several times its delivered size. Each
page step compresses them the way the clean save would
(``_compress_like_save``), so the save copies them as they are and writes the
objects a single burn writes. The save's search for duplicate objects
compares streams as stored, so on the fork a stream whose compressed form
could be matched differently is put back uncompressed first
(``_merge_differs``). A step also deletes the redaction annotations it made
and no longer reaches (``HeldDocument._prune``). Nothing else a burn makes is
ever deleted: MuPDF hands one font, colour space or image to every page that
asks for the same one, so a later page's burn may draw what an earlier one
made, and what no page draws is left to the save's garbage collection.
"""
from __future__ import annotations

import hashlib
import json
import re

import pymupdf as _pymupdf

from .metadata import _set_metadata_doc, validate_metadata_fields
from .redact import (
    _KEEP_ANNOT_TYPES,
    _blackout_page,
    _burn_page,
    _burn_plan,
    _flatten_form_fields,
    _remove_pages,
    _save_clean,
    _scrub_residue,
    _strip_metadata_doc,
    _touched_pages,
    _validate_pages,
    _validate_redaction_payload,
    open_pdf,
)
from .remove_text import remove_text_doc, validate_remove_text
from .sanitise import _page_content_xrefs, _strip_page_tags
from .side_text import REMOVE_KINDS, TEXT_KINDS, _walk as _side_text_walk
from .side_text import rewrite_side_text, validate_side_text_edits
from .unseen import (
    blank_images as _blank_images,
    remove_images as _remove_images,
    validate_blank_images,
    validate_remove_images,
)

_m = _pymupdf.mupdf

#: An object reference in an object's printed source.
_REF = re.compile(rb"(\d+) (\d+) R")
#: Side-text kinds a page step can take out of the file before the rewrite
#: runs (see :meth:`HeldDocument.assemble`).
_BURN_REMOVES = frozenset({"tag", "link"})


class HeldPageError(ValueError):
    """A page operation the held copy's state does not allow."""


def _pdf(doc):
    return _m.pdf_specifics(doc.this)


def _serialise(doc) -> bytes:
    """``doc`` written as it is: every object under its own number, streams
    as stored, and the trailer's ``/ID`` kept rather than regenerated in
    ``doc``. A save with no garbage collection changes no object of ``doc``."""
    return doc.tobytes(no_new_id=True)


def _raw(doc, xref: int) -> bytes | None:
    return doc.xref_stream_raw(xref) if doc.xref_is_stream(xref) else None


def _differs(a, b, xref: int) -> bool:
    """Whether object ``xref`` is not the same in ``a`` and ``b``."""
    return (a.xref_object(xref, compressed=True) != b.xref_object(xref, compressed=True)
            or _raw(a, xref) != _raw(b, xref))


def _put_back(doc, prepared, xref: int) -> None:
    """Object ``xref`` as the prepared copy has it, its stream bytes as stored."""
    doc.update_object(xref, prepared.xref_object(xref, compressed=True))
    raw = _raw(prepared, xref)
    if raw is not None:
        pdf = _pdf(doc)
        _m.pdf_update_stream(pdf, _m.pdf_new_indirect(pdf, xref, 0),
                             _m.fz_new_buffer_from_copied_data(raw), 1)


# ── compressing what a burn created, as the save would ──────────────────────


def _bitmap(obj, length: int) -> tuple[int, int] | None:
    """``(width, height)`` when the save would encode this stream as a
    CCITT G4 bitmap: MuPDF's ``is_bitmap_stream`` (pdf-write.c), read with
    the same calls."""
    if not _m.pdf_name_eq(_m.pdf_dict_get(obj, _m.PDF_ENUM_NAME_Subtype),
                          _m.PDF_ENUM_NAME_Image):
        return None
    w = _m.pdf_dict_get_int(obj, _m.PDF_ENUM_NAME_Width)
    h = _m.pdf_dict_get_int(obj, _m.PDF_ENUM_NAME_Height)
    if (w + 7) // 8 * h != length:
        return None
    if _m.pdf_dict_get_bool(obj, _m.PDF_ENUM_NAME_ImageMask):
        return w, h
    bpc = _m.pdf_dict_get(obj, _m.PDF_ENUM_NAME_BitsPerComponent)
    if not _m.pdf_is_int(bpc) or _m.pdf_to_int(bpc) != 1:
        return None
    if not _m.pdf_name_eq(_m.pdf_dict_get(obj, _m.PDF_ENUM_NAME_ColorSpace),
                          _m.PDF_ENUM_NAME_DeviceGray):
        return None
    return w, h


def _xml_metadata(obj) -> bool:
    return (_m.pdf_name_eq(_m.pdf_dict_get(obj, _m.PDF_ENUM_NAME_Type),
                           _m.PDF_ENUM_NAME_Metadata)
            and _m.pdf_name_eq(_m.pdf_dict_get(obj, _m.PDF_ENUM_NAME_Subtype),
                               _m.PDF_ENUM_NAME_XML))


def _canon(obj):
    """A value as ``pdf_objcmp`` compares it, with every reference blanked:
    dictionaries by sorted key, so two values it calls equal are equal here."""
    if _m.pdf_is_indirect(obj):
        return ("R",)
    if _m.pdf_is_dict(obj):
        return ("D", tuple(sorted(
            (_m.pdf_to_name(_m.pdf_dict_get_key(obj, i)),
             _canon(_m.pdf_dict_get_val(obj, i)))
            for i in range(_m.pdf_dict_len(obj)))))
    if _m.pdf_is_array(obj):
        return ("A", tuple(_canon(_m.pdf_array_get(obj, i))
                           for i in range(_m.pdf_array_len(obj))))
    if _m.pdf_is_name(obj):
        return ("N", _m.pdf_to_name(obj))
    if _m.pdf_is_int(obj):
        return ("I", _m.pdf_to_int64(obj))
    if _m.pdf_is_real(obj):
        return ("F", _m.pdf_to_real(obj))
    if _m.pdf_is_string(obj):
        return ("S", _m.pdf_to_text_string(obj))
    if _m.pdf_is_bool(obj):
        return ("B", bool(_m.pdf_to_bool(obj)))
    return ("Z",)


def _stream_key(doc, pdf, xref: int) -> tuple:
    """What the save's duplicate search compares for a stream (its dictionary
    and its bytes as stored), with references blanked."""
    return (_canon(_m.pdf_load_object(pdf, xref)),
            hashlib.sha256(doc.xref_stream_raw(xref) or b"").digest())


class _Packed:
    """A stream a page step compressed: its keys before and after (see
    :func:`_stream_key`), and its dictionary's source before."""

    __slots__ = ("xref", "plain", "packed", "source")

    def __init__(self, xref: int, plain: tuple, packed: tuple, source: str):
        self.xref, self.plain, self.packed, self.source = xref, plain, packed, source


def _compress_like_save(doc, xrefs) -> list[_Packed]:
    """Encode each unfiltered stream among ``xrefs`` exactly as the clean save
    would, and mark it so the save copies it as it is. Returns what it did.

    The save (``deflate=True``, pymupdf 1.28.2, MuPDF's ``copystream``) gives
    an unfiltered stream ``/FlateDecode`` at zlib's default level and drops
    its ``/DecodeParms``, or, for a one-bit grey image or a stencil mask of
    exactly its declared size, CCITT G4 with ``/DecodeParms << /K -1 /Columns
    w >>``; it leaves XMP metadata uncompressed. This does the same through
    MuPDF's own encoders, not Python's ``zlib``, which can be another version
    and write other bytes. ``Document.update_stream(compress=True)`` does not
    do the same: it compresses at the best level and adds an empty
    ``/DecodeParms``.

    One thing still differs: the save looks for duplicate objects before it
    compresses, comparing streams as stored, so a stream compressed here can
    be found equal to a stream the source holds compressed, or no longer
    equal to one it holds uncompressed. :meth:`HeldDocument.assemble` puts
    such a stream back uncompressed on the fork (:func:`_unpack`), from what
    this returns.
    """
    pdf = _pdf(doc)
    done: list[_Packed] = []
    for xref in xrefs:
        if not doc.xref_is_stream(xref):
            continue
        obj = _m.pdf_new_indirect(pdf, xref, 0)
        if _m.pdf_dict_get(obj, _m.PDF_ENUM_NAME_Filter).m_internal \
                or _xml_metadata(obj):
            continue
        source = doc.xref_object(xref, compressed=True)
        plain = _stream_key(doc, pdf, xref)
        data = _m.fz_new_buffer_from_copied_data(doc.xref_stream_raw(xref))
        size, ptr = _m.ll_fz_buffer_storage(data.m_internal)
        bitmap = _bitmap(obj, size)
        if bitmap is not None:
            w, h = bitmap
            packed = _m.FzBuffer(_m.ll_fz_compress_ccitt_fax_g4(ptr, w, h, (w + 7) // 8))
            _m.pdf_update_stream(pdf, obj, packed, 1)
            _m.pdf_dict_put(obj, _m.PDF_ENUM_NAME_Filter,
                            _m.PDF_ENUM_NAME_CCITTFaxDecode)
            parms = _m.pdf_dict_put_dict(obj, _m.PDF_ENUM_NAME_DecodeParms, 1)
            _m.pdf_dict_put_int(parms, _m.PDF_ENUM_NAME_K, -1)
            _m.pdf_dict_put_int(parms, _m.PDF_ENUM_NAME_Columns, w)
        else:
            out, n = _m.fz_new_deflated_data_from_buffer(data, _m.FZ_DEFLATE_DEFAULT)
            packed = _m.FzBuffer(_m.fz_new_buffer_from_data(out, n))
            _m.pdf_update_stream(pdf, obj, packed, 1)
            _m.pdf_dict_put(obj, _m.PDF_ENUM_NAME_Filter, _m.PDF_ENUM_NAME_FlateDecode)
            _m.pdf_dict_del(obj, _m.PDF_ENUM_NAME_DecodeParms)
        done.append(_Packed(xref, plain, _stream_key(doc, pdf, xref), source))
    return done


def _unpack(doc, packed: _Packed) -> None:
    """A stream :func:`_compress_like_save` compressed, as it was before."""
    data = doc.xref_stream(packed.xref)
    doc.update_object(packed.xref, packed.source)
    pdf = _pdf(doc)
    _m.pdf_update_stream(pdf, _m.pdf_new_indirect(pdf, packed.xref, 0),
                         _m.fz_new_buffer_from_copied_data(data), 1)


def _merge_differs(packed: list[_Packed], others: set[tuple]) -> list[_Packed]:
    """The compressed streams the save's duplicate search could treat other
    than it treats them uncompressed, the form a single burn leaves.

    ``others`` are the keys of every other stream the fork holds. The search
    compares a stream's dictionary and stored bytes, so a compressed stream
    can match what its uncompressed form never would: a stream the source
    holds compressed, or another stream whose uncompressed form differs only
    in what compression rewrites (``/DecodeParms``). And it can miss what
    its uncompressed form matches: a stream held uncompressed. Each is
    returned. References are blanked in the keys, so a match through
    references only the search's later rounds could resolve is returned too;
    putting a stream back uncompressed is always the single burn's form.
    """
    plain_of: dict[tuple, set[tuple]] = {}
    for p in packed:
        plain_of.setdefault(p.packed, set()).add(p.plain)
    return [p for p in packed
            if p.plain in others or p.packed in others or len(plain_of[p.packed]) > 1]


# ── what each page reaches, and what names it ───────────────────────────────


def _refs(doc, xref: int) -> set[int]:
    try:
        text = doc.xref_object(xref, compressed=True).encode("latin-1", "replace")
    except Exception:  # noqa: BLE001 -- an object that cannot be read leads nowhere
        return set()
    return {int(m.group(1)) for m in _REF.finditer(text)}


class _Index:
    """For each page of the prepared copy: the objects only it reaches, and
    the objects a page step may change in place that other pages share.

    A page reaches what its dictionary leads to, never another page or the
    page tree, so a destination naming another page leads nowhere. Built
    once, from the prepared copy, which never changes.
    """

    def __init__(self, doc):
        n = len(doc)
        self.page_xref = [doc[i].xref for i in range(n)]
        #: The page dictionaries and page-tree nodes, where a walk stops.
        self.stop = stop = set(self.page_xref)
        for xref in range(1, doc.xref_length()):
            try:
                if doc.xref_get_key(xref, "Type") == ("name", "/Pages"):
                    stop.add(xref)
            except Exception:  # noqa: BLE001
                continue
        refs: dict[int, set[int]] = {}
        reach: list[set[int]] = []
        count: dict[int, int] = {}
        for pno in range(n):
            seen: set[int] = set()
            todo = [self.page_xref[pno]]
            while todo:
                x = todo.pop()
                if x in seen or not 0 < x < doc.xref_length():
                    continue
                seen.add(x)
                if x not in refs:
                    refs[x] = _refs(doc, x)
                todo.extend(y for y in refs[x] if y not in stop)
            seen.discard(self.page_xref[pno])
            reach.append(seen)
            for x in seen:
                count[x] = count.get(x, 0) + 1
        named = self._named(doc, reach)
        named_count: dict[int, int] = {}
        for elems in named:
            for x in elems:
                named_count[x] = named_count.get(x, 0) + 1
        #: page -> objects only it reaches or names
        self.own: list[list[int]] = []
        #: page -> objects it may change in place that another page shares
        self.shared: list[list[int]] = []
        #: object -> the pages that may change it in place
        self.sharers: dict[int, set[int]] = {}
        for pno in range(n):
            own = {x for x in reach[pno] if count[x] == 1}
            own |= {x for x in named[pno] if named_count[x] == 1 and x not in stop}
            shared = {x for x in named[pno] if named_count[x] > 1 and x not in stop}
            page = doc[pno]
            shared |= {x for x in _page_content_xrefs(doc, page)
                       if count.get(x, 0) > 1}
            kind, value = doc.xref_get_key(self.page_xref[pno], "Annots")
            if kind == "xref":
                annots = int(value.split()[0])
                if count.get(annots, 0) > 1:
                    shared.add(annots)
            del page
            self.own.append(sorted(own))
            self.shared.append(sorted(shared))
            for x in shared:
                self.sharers.setdefault(x, set()).add(pno)

    def _named(self, doc, reach: list[set[int]]) -> list[set[int]]:
        """page -> the structure elements its step may edit: those named in
        the ``/ParentTree`` by the page's ``/StructParents`` and by those of
        the forms it reaches (what MuPDF's filter edits, pdf-op-filter.c), and
        those whose ``/Pg`` is the page."""
        n = len(self.page_xref)
        named: list[set[int]] = [set() for _ in range(n)]
        page_of = {x: i for i, x in enumerate(self.page_xref)}
        for xref in range(1, doc.xref_length()):
            try:
                kind, value = doc.xref_get_key(xref, "Pg")
            except Exception:  # noqa: BLE001
                continue
            if kind == "xref":
                pno = page_of.get(int(value.split()[0]))
                if pno is not None:
                    named[pno].add(xref)
        pdf = _pdf(doc)
        tree = _m.pdf_dict_getp(_m.pdf_trailer(pdf), "Root/StructTreeRoot/ParentTree")
        if not _m.pdf_is_dict(tree):
            return named
        keyed: dict[int, bool] = {}

        def has_key(x: int) -> bool:
            if x not in keyed:
                keyed[x] = doc.xref_get_key(x, "StructParents")[0] in ("int", "float")
            return keyed[x]

        for pno in range(n):
            owners = [self.page_xref[pno]] + [x for x in sorted(reach[pno]) if has_key(x)]
            for owner in owners:
                obj = _m.pdf_load_object(pdf, owner)
                key = _m.pdf_dict_get_int_default(obj, _m.PDF_ENUM_NAME_StructParents, -1)
                if key < 0:
                    continue
                found = _m.pdf_lookup_number(tree, key)
                if not _m.pdf_is_array(found):
                    continue
                if not _m.pdf_is_indirect(found):
                    parent = _m.pdf_obj_parent_num(found)
                    if parent > 0:
                        named[pno].add(parent)
                for i in range(_m.pdf_array_len(found)):
                    item = _m.pdf_array_get(found, i)
                    if _m.pdf_is_indirect(item):
                        named[pno].add(_m.pdf_to_num(item))
                    elif _m.pdf_is_dict(item):
                        parent = _m.pdf_obj_parent_num(item)
                        if parent > 0:
                            named[pno].add(parent)
        return named


# ── a page's inputs ─────────────────────────────────────────────────────────


def _inputs_key(pg_matches: list[dict], redact_label: str, keep_uncovered_lines: bool,
                blackout: bool, removal: list[dict] | None, strip_tags: bool) -> str:
    """Everything a page step reads, as one string: two steps with the same
    key leave the page the same. A blacked-out page reads only that it is."""
    if blackout:
        return json.dumps(["blackout"])
    boxes = [[m["rect"]["x0"], m["rect"]["y0"], m["rect"]["x1"], m["rect"]["y1"],
              m.get("redact_label") or redact_label] for m in pg_matches]
    key: dict = {"boxes": boxes, "removal": removal or [],
                 "strip_tags": bool(strip_tags)}
    if boxes:
        key["keep_uncovered_lines"] = bool(keep_uncovered_lines)
    return json.dumps(key, sort_keys=True)


#: The key of a page step with nothing to do: an unburned page reads the same.
_IDLE = _inputs_key([], "", False, False, None, False)


class _State:
    """Which pages a document holds burned, and what each step left."""

    def __init__(self):
        self.burned: dict[int, str] = {}         # page -> inputs key
        self.removal: dict[int, dict] = {}       # page -> its text-removal outcome
        #: object -> how a page step compressed it. Kept when the page goes
        #: back: a later step may draw the same object (see ``_restore``).
        self.packed: dict[int, _Packed] = {}

    def copy(self) -> "_State":
        out = _State()
        out.burned = dict(self.burned)
        out.removal = dict(self.removal)
        out.packed = dict(self.packed)
        return out


def _finish_page(doc, pno: int, touched: bool) -> None:
    """What the assembly does to one page, done now: every annotation but
    links deleted (``_scrub_residue``) and, on a touched page, the content
    tags stripped (``strip_actual_text``).

    A function of its own so that no ``Page`` or ``Annot`` outlives it: MuPDF
    keeps a page loaded while anything refers to it, and a page loaded
    across a restore would still hold the dictionary the restore replaced.
    """
    page = doc[pno]
    doomed = [a for a in page.annots() if a.type[0] not in _KEEP_ANNOT_TYPES]
    for annot in doomed:
        page.delete_annot(annot)
    if touched:
        _strip_page_tags(doc, page)


def _bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {type(value).__name__}")
    return value


def _categories(active_categories) -> set[str] | None:
    if active_categories is None:
        return None
    if not isinstance(active_categories, (list, tuple, set, frozenset)):
        raise ValueError("active_categories must be a list of category names")
    return set(active_categories)


# ── the held copy ───────────────────────────────────────────────────────────


class HeldDocument:
    """One document held open and burned a page at a time.

    ``doc`` is the held copy, which page steps change and every read sees;
    ``prepared`` is the copy as it was opened, after the image edits and
    form flattening, which nothing writes and restores read from.
    """

    def __init__(self, pdf_bytes: bytes, *, blank_images=None, remove_images=None):
        blank = (validate_blank_images(blank_images)
                 if blank_images is not None else None)
        remove = (validate_remove_images(remove_images)
                  if remove_images is not None else None)
        doc = open_pdf(pdf_bytes)
        prepared = None
        try:
            if doc.needs_pass:
                raise ValueError("a held copy needs an unencrypted document")
            # The side-text ids a caller plans against are the source's
            # (``list_side_text`` on these bytes); kept to report the
            # rewrite as ``apply_redactions`` would (see ``assemble``).
            walk = _side_text_walk(doc)
            self._side_kinds = {e["id"]: e["kind"] for e in walk.entries}
            #: The image edits' outcome, as ``apply_redactions`` reports it;
            #: ``None`` when neither field was given.
            self.unseen: dict | None = None
            if blank is not None or remove is not None:
                b = _blank_images(doc, blank or [])
                r = _remove_images(doc, remove or [])
                self.unseen = {"blanked": b["blanked"], "removed": r["removed"],
                               "skipped_blank": b["skipped"],
                               "skipped_remove": r["skipped"]}
            _flatten_form_fields(doc)
            prepared = open_pdf(_serialise(doc))
            self._index = _Index(prepared)
        except BaseException:
            doc.close()
            if prepared is not None:
                prepared.close()
            raise
        self.doc = doc
        self.prepared = prepared
        self._state = _State()
        self._prepared_keys: set[tuple] | None = None

    # ── state ──

    @property
    def burned_pages(self) -> list[int]:
        """The pages held burned, in page order."""
        return sorted(self._state.burned)

    def close(self) -> None:
        for doc in (self.doc, self.prepared):
            try:
                doc.close()
            except Exception:  # noqa: BLE001 -- closing must never raise
                pass

    def _page(self, pno) -> int:
        if isinstance(pno, bool) or not isinstance(pno, int):
            raise ValueError(f"page must be an integer, got {type(pno).__name__}")
        if not 0 <= pno < len(self.doc):
            raise IndexError(f"page {pno} out of range for {len(self.doc)}-page document")
        return pno

    # ── one page ──

    def burn_page(self, pno, matches, *, redact_label: str = "",
                  active_categories=None, keep_uncovered_lines: bool = False,
                  blackout: bool = False, remove_text=None,
                  strip_tags: bool = False) -> dict | None:
        """Burn page ``pno`` into the held copy and finish it.

        ``matches`` are this page's, in the shape ``apply_redactions`` takes,
        filtered as it filters them (disabled, inactive types). ``blackout``
        covers the whole page instead. ``remove_text`` lists this page's
        ``remove_text`` entries; ``strip_tags`` says the page is in
        ``strip_tags_pages``. Returns the text-removal outcome
        (``{"removed": [[page, i], ...], "kept": [...]}``) when ``remove_text``
        was given, else ``None``. A page already burned is refused: put it
        back first (:meth:`restore_page`).
        """
        pno = self._page(pno)
        if pno in self._state.burned:
            raise HeldPageError(f"page {pno} is already burned; put it back first")
        if not isinstance(redact_label, str):
            raise ValueError("redact_label must be a string")
        keep = _bool(keep_uncovered_lines, "keep_uncovered_lines")
        blackout = _bool(blackout, "blackout")
        strip_tags = _bool(strip_tags, "strip_tags")
        active = _categories(active_categories)
        if not isinstance(matches, list):
            raise ValueError("matches must be a list")
        _validate_redaction_payload(matches, None, None)
        if any(m["page"] != pno for m in matches):
            raise ValueError("every match must be on the page being burned")
        items = None
        if remove_text is not None:
            plan = validate_remove_text(remove_text)
            if set(plan) - {pno}:
                raise ValueError("every remove_text entry must be on the page being burned")
            items = plan.get(pno, [])
        pg_matches = _burn_plan(matches, set(), {pno} if blackout else set(),
                                active).get(pno, [])
        self._step(self.doc, self._state, pno, pg_matches, redact_label, keep,
                   blackout, items, strip_tags, compress=True)
        return None if remove_text is None else self._state.removal.get(
            pno, {"removed": [], "kept": []})

    def _step(self, doc, state: _State, pno: int, pg_matches: list[dict],
              redact_label: str, keep: bool, blackout: bool,
              items: list[dict] | None, strip_tags: bool, *, compress: bool) -> None:
        """One page's text removal, burn and finishing, into ``doc``.

        The order is ``apply_redactions``': text removal, then the burn (or
        the blackout), then what its assembly does page by page (annotations
        deleted, tags stripped on a touched page). A step that raises leaves
        the page put back.
        """
        first = doc.xref_length()
        key = _inputs_key(pg_matches, redact_label, keep, blackout, items, strip_tags)
        state.burned[pno] = key
        try:
            if items and not blackout:
                state.removal[pno] = remove_text_doc(doc, {pno: items})
            if blackout:
                _blackout_page(doc, pno)
            elif pg_matches:
                _burn_page(doc, pno, pg_matches, redact_label, keep)
            _finish_page(doc, pno, bool(pg_matches or blackout or items or strip_tags))
            if compress:
                self._prune(doc, pno, first)
                for pk in _compress_like_save(doc, range(first, doc.xref_length())):
                    state.packed[pk.xref] = pk
        except BaseException:
            self._restore(doc, state, pno)
            raise

    def _prune(self, doc, pno: int, first: int) -> None:
        """Delete the annotations and forms this step made that the page no
        longer reaches: the burn's own redaction annotations and their
        appearances, which MuPDF takes off the page and leaves in the file
        for the save's garbage collection. Held, they would make the copy
        grow with every box. Nothing else is deleted: MuPDF keeps one font,
        colour space and image per document for every page that asks for
        it, so an unreached one may be drawn by a later step (see
        ``_restore``), and annotations and forms are never kept that way.
        """
        ix = self._index
        walk_into = set(ix.own[pno])
        reached, todo = set(), [ix.page_xref[pno]]
        while todo:
            x = todo.pop()
            if x in reached:
                continue
            reached.add(x)
            todo.extend(y for y in _refs(doc, x)
                        if y not in ix.stop and (y >= first or y in walk_into))
        pdf = _pdf(doc)
        for x in range(first, doc.xref_length()):
            if x in reached:
                continue
            if (doc.xref_get_key(x, "Type") == ("name", "/Annot")
                    or doc.xref_get_key(x, "Subtype") == ("name", "/Form")):
                _m.pdf_delete_object(pdf, x)

    def restore_page(self, pno) -> list[int]:
        """Put page ``pno`` back as it was before its burn; return the pages
        put back, in page order.

        That is ``pno``, and any burned page sharing with it an object a
        page step changes in place (a structure element two pages' marked
        content names, say) when that object has changed: one page cannot
        take its share back alone. Each page returned must be burned again.
        """
        pno = self._page(pno)
        if pno not in self._state.burned:
            raise HeldPageError(f"page {pno} is not burned")
        return self._restore(self.doc, self._state, pno)

    def _restore(self, doc, state: _State, pno: int) -> list[int]:
        ix, prepared = self._index, self.prepared
        group, todo = set(), [pno]
        while todo:
            p = todo.pop()
            if p in group or p not in state.burned:
                continue
            group.add(p)
            for x in ix.shared[p]:
                if _differs(prepared, doc, x):
                    todo.extend(q for q in ix.sharers[x] if q not in group)
        # The objects a page's burn made are not deleted: MuPDF keeps one
        # font, colour space and image per document for every page that
        # needs it, so a later page's burn can draw an object an earlier one
        # made. One no page draws any more is left for the save's garbage
        # collection, as a single burn leaves the objects it replaced.
        for p in sorted(group):
            for x in [ix.page_xref[p]] + ix.own[p]:
                if _differs(prepared, doc, x):
                    _put_back(doc, prepared, x)
            state.burned.pop(p, None)
            state.removal.pop(p, None)
        for x in sorted({x for p in group for x in ix.shared[p]}):
            if _differs(prepared, doc, x):
                _put_back(doc, prepared, x)
        return sorted(group)

    # ── the whole document ──

    def assemble(self, matches, *, redact_label: str = "", active_categories=None,
                 removed_pages=None, blackout_pages=None, remove_text=None,
                 keep_uncovered_lines: bool = False, side_text=None,
                 strip_tags_pages=None, metadata: dict | None = None
                 ) -> tuple[bytes, dict]:
        """The delivered file for this payload, made on a fork of the held copy.

        Takes ``apply_redactions``' payload (the image plan is the one the
        copy was opened with) and gives its file. On the fork, every page
        whose held burn does not match the payload is put back and burned
        again, removed pages are put back before they go, and then come the
        side-text rewrite, the page removal, the residue scrub, the metadata
        strip, ``metadata`` (merged as ``set_metadata`` merges it) and one
        clean save. The held copy is left as it was.

        Returns ``(bytes, outcome)``. ``outcome`` has ``burned_at_assembly``
        (the pages burned on the fork), and ``removal`` and ``side`` as
        ``apply_redactions`` reports them when ``remove_text`` and
        ``side_text`` are given.

        The side-text outcome counts as applied an edit to a structure
        element's alternate text or to a link that a page step had already
        taken out of the file, as ``apply_redactions``, which rewrites before
        it burns, would. The file is the same either way.
        """
        _validate_redaction_payload(matches, removed_pages, blackout_pages)
        if not isinstance(redact_label, str):
            raise ValueError("redact_label must be a string")
        keep = _bool(keep_uncovered_lines, "keep_uncovered_lines")
        active = _categories(active_categories)
        removal_plan = (validate_remove_text(remove_text)
                        if remove_text is not None else None)
        side_edits = (validate_side_text_edits(side_text)
                      if side_text is not None else None)
        tag_pages = (_validate_pages(strip_tags_pages, "strip_tags_pages")
                     if strip_tags_pages is not None else set())
        if metadata is not None:
            validate_metadata_fields(metadata)
        n = len(self.doc)
        # Entries outside the document are ignored, as apply_redactions
        # ignores them, except that a match on such a page is refused: it
        # burns the page Python's negative indexing names there.
        removed = set(removed_pages) if removed_pages else set()
        blackout = (set(blackout_pages) - removed) if blackout_pages else set()
        valid_removed = {p for p in removed if 0 <= p < n}
        if valid_removed and len(valid_removed) >= n:
            raise ValueError("Cannot export: all pages have been removed")
        by_page = _burn_plan(matches, removed, blackout, active)
        for p in by_page:
            if not 0 <= p < n:
                raise IndexError(f"page {p} not in document")
        want = {p: _inputs_key(by_page.get(p, []), redact_label, keep, p in blackout,
                               (removal_plan or {}).get(p), p in tag_pages)
                for p in range(n) if p not in removed}

        fork = open_pdf(_serialise(self.doc))
        try:
            state = self._state.copy()
            for p in range(n):
                if p in state.burned and want.get(p) != state.burned[p]:
                    self._restore(fork, state, p)
            redo = [p for p in sorted(want)
                    if p not in state.burned and want[p] != _IDLE]
            for p in redo:
                self._step(fork, state, p, by_page.get(p, []), redact_label, keep,
                           p in blackout, (removal_plan or {}).get(p),
                           p in tag_pages, compress=False)
            self._unpack_where_merges_differ(fork, state)
            outcome: dict = {"burned_at_assembly": redo}
            if removal_plan is not None:
                done: dict = {"removed": [], "kept": []}
                for p in sorted(removal_plan):
                    page_outcome = state.removal.get(p)
                    if page_outcome is not None:
                        done["removed"] += page_outcome["removed"]
                        done["kept"] += page_outcome["kept"]
                outcome["removal"] = done
            if side_edits is not None:
                outcome["side"] = self._rewrite(fork, side_edits)
            touched = _touched_pages(n, by_page, blackout, removal_plan, tag_pages,
                                     removed)
            _remove_pages(fork, removed)
            _scrub_residue(fork, touched_pages=touched)
            _strip_metadata_doc(fork)
            if metadata:
                _set_metadata_doc(fork, metadata)
            return _save_clean(fork), outcome
        finally:
            fork.close()

    def _unpack_where_merges_differ(self, fork, state: _State) -> None:
        """Put back uncompressed, on the fork, every stream whose compression
        could change what the save finds duplicated (:func:`_merge_differs`)."""
        packed = list(state.packed.values())
        if not packed:
            return
        if self._prepared_keys is None:
            ppdf = _pdf(self.prepared)
            self._prepared_keys = {
                _stream_key(self.prepared, ppdf, x)
                for x in range(1, self.prepared.xref_length())
                if self.prepared.xref_is_stream(x)}
        others = set(self._prepared_keys)
        pdf = _pdf(fork)
        for x in range(self.prepared.xref_length(), fork.xref_length()):
            if x not in state.packed and fork.xref_is_stream(x):
                others.add(_stream_key(fork, pdf, x))
        for pk in _merge_differs(packed, others):
            _unpack(fork, pk)

    def _rewrite(self, doc, edits: list[dict]) -> dict:
        """``rewrite_side_text``, reported as ``apply_redactions`` reports it.

        ``apply_redactions`` rewrites before it burns. Here the pages are
        burned first, and a burn takes some strings out of the file itself:
        MuPDF deletes a structure element's alternate text for characters it
        removes, and the links a box touches. An edit whose string the burn
        already took is applied there and skipped here, so it is counted as
        applied when the source listed it with a kind that takes the edit.
        """
        outcome = rewrite_side_text(doc, edits)
        by_id = {e["id"]: e for e in edits}
        applied, skipped = outcome["applied"], []
        for entry_id in outcome["skipped"]:
            kind = self._side_kinds.get(entry_id)
            edit = by_id.get(entry_id)
            if kind in _BURN_REMOVES and edit is not None and (
                    kind in REMOVE_KINDS if edit["remove"] else kind in TEXT_KINDS):
                applied += 1
            else:
                skipped.append(entry_id)
        return {"applied": applied, "skipped": skipped}
