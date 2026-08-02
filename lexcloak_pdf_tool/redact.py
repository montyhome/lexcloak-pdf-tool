"""Redaction: black-box matches + strip metadata + optional re-encryption."""
from __future__ import annotations

import io
import logging

import fitz as _fitz

PDF_ENCRYPT_AES_256 = _fitz.PDF_ENCRYPT_AES_256
PDF_PERM_ACCESSIBILITY = _fitz.PDF_PERM_ACCESSIBILITY
Rect = _fitz.Rect
TEXT_ALIGN_CENTER = _fitz.TEXT_ALIGN_CENTER


def open_pdf(pdf_bytes: bytes) -> _fitz.Document:
    """Open a PDF from bytes -- module-local helper."""
    return _fitz.open(stream=pdf_bytes, filetype="pdf")


def _validate_redaction_payload(matches: list[dict],
                                removed_pages: list[int] | None,
                                blackout_pages: list[int] | None = None) -> None:
    """Coerce + sanity-check the wire payload before opening the PDF.

    Mutates each match in place to coerce ``page`` to int and rect coords
    to float so downstream code can rely on the typed shape. Coerces
    ``removed_pages`` and ``blackout_pages`` entries to int in place too.
    Raises ``ValueError`` with a named-field message on the first malformed
    entry; the caller surfaces this as a 400-class error.

    Without this guard, a bad payload (e.g., ``{"page": "abc"}``,
    ``{"rect": {"x0": "??"}}``, swapped x0/x1) would surface as an opaque
    PyMuPDF error deep inside ``page.add_redact_annot``.

    The optional per-match ``redact_label`` (v0.6.4) is type-checked here for
    the same reason: a non-string label reaches ``add_redact_annot``'s
    ``text=`` and fails inside PyMuPDF instead of at the wire boundary.
    Unknown keys are NOT rejected -- this validator has always ignored them,
    which is what lets a v0.6.4-aware app send per-match labels to an older
    bundled CLI and get the document-level label rather than a hard failure.
    """
    for field_name, page_list in (("removed_pages", removed_pages),
                                  ("blackout_pages", blackout_pages)):
        if page_list is not None:
            try:
                page_list[:] = [int(p) for p in page_list]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed redaction payload: {field_name} "
                    f"contains non-integer entry -- {type(exc).__name__}"
                ) from None

    for i, m in enumerate(matches):
        if not isinstance(m, dict):
            raise ValueError(
                f"Malformed redaction payload: matches[{i}] is not a dict"
            )
        try:
            m["page"] = int(m["page"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"Malformed redaction payload: matches[{i}].page "
                f"missing or non-integer"
            )

        rect = m.get("rect")
        if not isinstance(rect, dict):
            raise ValueError(
                f"Malformed redaction payload: matches[{i}].rect "
                f"missing or not a dict"
            )
        try:
            x0 = float(rect["x0"])
            y0 = float(rect["y0"])
            x1 = float(rect["x1"])
            y1 = float(rect["y1"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"Malformed redaction payload: matches[{i}].rect "
                f"has missing or non-numeric x0/y0/x1/y1"
            )
        if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1:
            raise ValueError(
                f"Malformed redaction payload: matches[{i}].rect "
                f"has invalid bounds x0={x0} y0={y0} x1={x1} y1={y1}"
            )
        rect["x0"] = x0
        rect["y0"] = y0
        rect["x1"] = x1
        rect["y1"] = y1

        label = m.get("redact_label")
        if label is not None and not isinstance(label, str):
            raise ValueError(
                f"Malformed redaction payload: matches[{i}].redact_label "
                f"is not a string (got {type(label).__name__})"
            )


def _strip_metadata_doc(doc) -> None:
    """Remove all document metadata + XMP from a live ``fitz.Document``.

    Strips author, title, subject, creator, producer, creation/mod dates,
    keywords, and the full XMP metadata stream.
    """
    doc.set_metadata({
        "author": "",
        "title": "",
        "subject": "",
        "creator": "",
        "producer": "",
        "creationDate": "",
        "modDate": "",
        "keywords": "",
    })
    doc.del_xml_metadata()


def strip_metadata(pdf_bytes_or_doc):
    """Strip document metadata + XMP. Accepts bytes (IPC form) or Document.

    * ``bytes`` -> returns ``bytes`` with metadata stripped (IPC-clean form).
    * ``fitz.Document`` -> mutates in place, returns ``None``.
    """
    if isinstance(pdf_bytes_or_doc, (bytes, bytearray)):
        doc = open_pdf(bytes(pdf_bytes_or_doc))
        try:
            _strip_metadata_doc(doc)
            buf = io.BytesIO()
            doc.save(buf, garbage=4, deflate=True, clean=True)
        finally:
            doc.close()
        return buf.getvalue()
    _strip_metadata_doc(pdf_bytes_or_doc)
    return None


def _has_any_widget(doc) -> bool:
    """True if any page carries a ``/Widget`` annotation.

    ``doc.is_form_pdf`` alone false-negatives on CATALOG-ORPHAN widgets --
    page-level /Widget annotations never registered in a document /AcroForm
    dictionary (form-filler / generator output; Session 660 found a whole
    synthetic tax corpus orphan-shaped, and a live packaged-app export
    shipped all 46 widget values because the flatten guard skipped). Real
    authority-published forms register their fields and ARE seen by
    ``is_form_pdf``; the orphan shape is the residual exposure class.
    ``doc.bake`` flattens orphan widgets correctly once actually called
    (verified empirically, both pymupdf lines) -- detection was the only
    gap. The scan reads each page's annotation-xref list (no widget objects
    are loaded), so the common no-widget document pays one cheap /Annots
    array read per page.
    """
    for page in doc:
        for entry in page.annot_xrefs():
            if entry[1] == _fitz.PDF_ANNOT_WIDGET:
                return True
    return False


def _flatten_form_fields(doc) -> None:
    """Flatten AcroForm widget fields into static page content.

    Form-field values live in the widget ``/V`` and the document AcroForm
    dictionary -- OUTSIDE the page content stream that
    ``page.apply_redactions()`` scrubs. Without this step a redaction box drawn
    over a form field blacks out the *visual* but leaves the underlying value
    fully extractable via ``get_text()`` and present in the raw output bytes --
    a real PII leak for a redaction tool. The leak bites even when the app
    "redacts" the field: the value is visible to the detector through
    ``get_text`` (PyMuPDF surfaces widget text), so a match IS produced at the
    widget rect, yet the box only covers the rendering and the ``/V`` survives.

    ``doc.bake(widgets=True)`` converts every interactive widget to static page
    content BEFORE the redaction loop, so the now-static value under a redaction
    rect is removed by ``apply_redactions`` and no interactive field (and no
    residual ``/V``) survives in the exported, safe-to-share PDF. Form fields
    the user did NOT redact remain as static, non-interactive content -- the
    correct shape for a final redacted artifact (an interactive form can carry
    hidden values, scripts, and reset actions a redacted output must not ship).

    Guarded by ``is_form_pdf`` OR a per-page widget scan so the
    overwhelmingly-common no-widget PDF takes the identical,
    byte-for-byte-unchanged path. The scan half exists because
    ``is_form_pdf`` false-negatives on catalog-orphan widgets (v0.6.2,
    Sessions 659/660 -- see ``_has_any_widget``); ``is_form_pdf`` runs first
    as the cheap short-circuit for well-formed forms. ``annots=False``
    leaves non-widget annotations untouched -- annotation-borne text is a
    distinct surface, out of scope for this AcroForm fix.
    """
    if not (doc.is_form_pdf or _has_any_widget(doc)):
        return
    # bake(annots=False, widgets=True): flatten only interactive form widgets.
    # Non-widget annotations are handled by _scrub_residue (deleted, not baked).
    doc.bake(annots=False, widgets=True)


# Annotation types the residue scrub KEEPS. Links carry no free text of their
# own -- a /Link is a rect plus a destination -- and non-overlapping links are
# a deliberate out-of-scope decision for the residue work (they are a distinct
# vector wanting their own ruling, not a text leak). Everything else goes.
#
# Belt-and-braces, not the primary mechanism: ``page.annots()`` does not yield
# Link annotations at all (pymupdf 1.27.2 -- they are reachable only through
# ``annot_xrefs`` / ``get_links``), so the walk below never sees a link even
# before this set is consulted. The explicit entry is what keeps links
# surviving if that upstream behaviour ever changes.
_KEEP_ANNOT_TYPES = frozenset({_fitz.PDF_ANNOT_LINK})


def null_page_thumbnails(doc) -> int:
    """Drop every page's ``/Thumb``; return how many were dropped.

    **``scrub(thumbnails=True)`` cannot be relied on.** PyMuPDF's page loop
    early-``continue``s on ``if not (clean_pages or hidden_text)`` *before*
    reaching its thumbnail branch (pymupdf 1.27.2, ``Document.scrub``), so
    the flag is silently a no-op on exactly the flag-set a redaction tool
    must use -- ``clean_pages=False`` (never re-run content machinery over
    finalized bytes) and ``hidden_text=False`` (keep the invisible OCR
    layer). Measured 2026-08-02: a ``/Thumb`` raster survived a full burn
    with ``thumbnails=True`` set, recoverable as 820 bytes of PNG.

    That matters because a thumbnail is a cached raster of the page as it
    looked *before* the redaction -- a picture of the content just burned
    away. Nulling the key orphans the stream, which the callers' saves
    (``garbage=4``) then collect.
    """
    dropped = 0
    for page in doc:
        if doc.xref_get_key(page.xref, "Thumb")[0] != "null":
            doc.xref_set_key(page.xref, "Thumb", "null")
            dropped += 1
    return dropped


def _scrub_residue(doc) -> None:
    """Strip non-page-content residue that survives ``apply_redactions``.

    ``apply_redactions`` only rewrites *page content streams*. Four classes of
    payload live outside them, are never seen by detection, and shipped intact
    in every export before v0.6.6:

    * **Annotation text** -- sticky notes (``/Text``), ``/FreeText``,
      comments. Two distinct shapes, both verified surviving a default burn
      against v0.6.5: a ``/Text`` note's content is NOT returned by
      ``page.get_text()``, so no detector can match it and no user can
      redact it; a ``/FreeText``'s text IS returned, so a match is produced
      and a box drawn -- but the box burns the page CONTENT stream while the
      text lives in the annot's own appearance stream, so it survives while
      the user is told it was redacted.
    * **Embedded / attached files** -- a whole second document riding along.
      Verified: the payload came back byte-for-byte from ``embfile_get`` on a
      v0.6.5 export.
    * **Document-level JavaScript** -- executable content in a "safe to share"
      artifact.
    * **Page thumbnails** -- a cached raster of the page as it looked
      *before* the burn, i.e. a picture of the unredacted content.

    **Delete, do not bake.** ``doc.bake(annots=True)`` would paint annotation
    appearance streams into the page content -- converting text that detection
    never saw into permanent, extractable page content that nothing redacted.
    That is strictly worse than the status quo: it would launder undetected
    residue into the very layer the tool promises is clean. Deletion is the
    only direction that shrinks the leak.

    Ordering: callers must run this AFTER ``apply_redactions`` (which consumes
    the redaction annots it created) and after any blackout/removal passes, so
    the scrub sees the final annot set. It runs before ``_strip_metadata_doc``,
    which owns metadata and XMP -- hence ``metadata``/``xml_metadata`` are
    False here rather than duplicated.

    ``hidden_text=False`` is load-bearing and mirrors ``reduce_size``: a
    scanned redacted PDF carries its searchable layer as invisible
    (render-mode-3) text over the page image, and ``scrub``'s default would
    delete it -- destroying text selection on exactly the documents this tool
    targets. ``clean_pages``/``redactions`` stay off for the same reason
    ``reduce_size`` keeps them off: this op must never re-run redaction
    machinery over finalized bytes.
    """
    for page in doc:
        # Snapshot first: deleting while iterating page.annots() invalidates
        # the generator mid-walk and silently skips entries.
        doomed = [a for a in page.annots() if a.type[0] not in _KEEP_ANNOT_TYPES]
        for annot in doomed:
            page.delete_annot(annot)

    doc.scrub(
        attached_files=True,
        embedded_files=True,
        javascript=True,
        # Set for intent, but NOT trusted -- see null_page_thumbnails below.
        thumbnails=True,
        # Owned by _strip_metadata_doc, which runs immediately after.
        metadata=False,
        xml_metadata=False,
        # See docstring -- each of these would damage a redacted artifact.
        hidden_text=False,
        clean_pages=False,
        redactions=False,
        remove_links=False,
        reset_fields=False,
        reset_responses=False,
    )

    # scrub's own thumbnail branch is unreachable on this flag-set.
    null_page_thumbnails(doc)

    # scrub(javascript=True) empties the action body to `/JS ()` but leaves
    # the /Names/JavaScript name tree in place -- and its NAMES are author
    # chosen, so a script named for a matter or custodian would ride out in
    # a document that is supposed to carry nothing along. Drop the tree.
    doc.xref_set_key(doc.pdf_catalog(), "Names/JavaScript", "null")


# A rect within this many points of a page edge is treated as flush against
# it for off-page overscan purposes (OCR bboxes clamp at 0, so "flush" in
# practice means exactly-at-the-edge plus float fuzz).
_PAGE_EDGE_EPSILON = 1.0
# How far past the page bounds an overscan region extends. Glyphs drawn
# outside the page box (print headers/footers clipped by the scan, cropped
# margins) still extract via ``get_text`` when any sliver of the glyph pokes
# into the page -- yet ``apply_redactions`` KEEPS a glyph whose overlap with
# the redaction region is a sliver (empirically <~2pt of the glyph box, both
# pymupdf 1.27.x and 1.28.x). Session 659 found live blackout exports with
# descender glyphs (p/y/g) + form furniture extractable from header lines
# hanging ~1pt into the page. Extending the redaction region well past the
# edge makes those glyphs fully contained, so the text filter removes them.
# 10k pt is far beyond any real content offset while staying comfortably
# inside PDF implementation coordinate limits (+/-32767).
_EDGE_OVERSCAN = 10000.0


def _edge_overscan_strips(rect, page_rect) -> list:
    """Off-page scrub strips for a match rect flush against a page edge.

    A redaction rect can only cover the in-page part of a clipped glyph
    (payload validation rejects negative coords), so a header line hanging
    off the page survives the text filter as extractable slivers. For every
    side of ``rect`` flush against ``page_rect`` (within
    ``_PAGE_EDGE_EPSILON``), emit an off-page strip extending
    ``_EDGE_OVERSCAN`` past that edge, sharing the rect's cross-axis span.
    The strips are separate redaction annots so the USER's rect -- its fill
    geometry and label placement -- stays byte-identical; interior rects
    (the overwhelmingly common case) get no strips at all.

    Both rects are in the as-rendered (rotation-applied) frame; the caller
    derotates strips exactly like the main rect before burning.
    """
    strips = []
    if rect.y0 <= page_rect.y0 + _PAGE_EDGE_EPSILON:  # flush top
        strips.append(Rect(rect.x0, page_rect.y0 - _EDGE_OVERSCAN,
                           rect.x1, page_rect.y0 + _PAGE_EDGE_EPSILON))
    if rect.y1 >= page_rect.y1 - _PAGE_EDGE_EPSILON:  # flush bottom
        strips.append(Rect(rect.x0, page_rect.y1 - _PAGE_EDGE_EPSILON,
                           rect.x1, page_rect.y1 + _EDGE_OVERSCAN))
    if rect.x0 <= page_rect.x0 + _PAGE_EDGE_EPSILON:  # flush left
        strips.append(Rect(page_rect.x0 - _EDGE_OVERSCAN, rect.y0,
                           page_rect.x0 + _PAGE_EDGE_EPSILON, rect.y1))
    if rect.x1 >= page_rect.x1 - _PAGE_EDGE_EPSILON:  # flush right
        strips.append(Rect(page_rect.x1 - _PAGE_EDGE_EPSILON, rect.y0,
                           page_rect.x1 + _EDGE_OVERSCAN, rect.y1))
    return strips


def _derotate_to_native(rect, page):
    """Map an as-rendered (rotation-applied) rect into the native frame
    ``add_redact_annot`` expects -- the S590 two-step transform (derotation
    + MediaBox-origin shift). No-op frame-wise on an unrotated page; the
    caller guards on ``page.rotation``.
    """
    rect = rect * page.derotation_matrix
    rect.normalize()
    mb = page.mediabox
    return Rect(rect.x0 + mb.x0, rect.y0 - mb.y0,
                rect.x1 + mb.x0, rect.y1 - mb.y0)


def _save_clean(doc) -> bytes:
    """Serialize ``doc`` to unencrypted bytes with the standard save params."""
    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True, clean=True)
    return buf.getvalue()


def _save_encrypted(doc, password: str) -> tuple[bytes, bool]:
    """Save ``doc`` AES-256 encrypted under ``password``; return ``(bytes, applied)``.

    Assumes a non-empty ``password`` and cleartext input -- callers guard both.
    On any PyMuPDF save failure the encryption is dropped and the document is
    re-saved unprotected, returning ``protection_applied=False``: a failed
    encryption must never block the redacted download. The warning logs the
    exception *type* only -- never the password, never ``str(exc)``.

    Shared by ``_apply_redactions_doc`` (the library-back-compat re-encrypt
    path) and :func:`lexcloak_pdf_tool.encryption.encrypt` (the standalone
    ``encrypt`` op) so the two cannot drift on save params or fallback
    behaviour (Session 342, Risk 2).
    """
    buf = io.BytesIO()
    try:
        doc.save(
            buf,
            garbage=4,
            deflate=True,
            clean=True,
            encryption=PDF_ENCRYPT_AES_256,
            user_pw=password,
            owner_pw=password,
            permissions=int(PDF_PERM_ACCESSIBILITY),
        )
        return buf.getvalue(), True
    except Exception as exc:  # noqa: BLE001 -- degrade to unprotected, never block
        logging.getLogger(__name__).warning(
            "PDF encryption failed; saving unprotected. exception_type=%s",
            type(exc).__name__,
        )
        return _save_clean(doc), False


def _apply_redactions_doc(doc, matches: list[dict],
                          redact_label: str = "",
                          active_categories: list[str] | set[str] | None = None,
                          removed_pages: list[int] | None = None,
                          blackout_pages: list[int] | None = None,
                          *,
                          output_protection: dict | None = None
                          ) -> tuple[bytes, bool]:
    """Apply redactions to an open ``fitz.Document`` and return (bytes, protected).

    Mutates ``doc`` in place: AcroForm widgets are flattened to static content,
    redaction annotations are applied, ``blackout_pages`` are fully blacked out,
    ``removed_pages`` are deleted, out-of-content residue is scrubbed (see
    ``_scrub_residue``), metadata is stripped. Caller owns ``doc`` lifecycle --
    this helper does NOT close it. Used by both the bytes-IPC entry point and
    the v0.4.0 stateful handle protocol.

    ``removed_pages`` vs ``blackout_pages`` are the two "drop" outcomes a page
    can have (Session 592 triage redesign): a removed page is deleted from the
    output; a blackout page stays in the output but is covered edge-to-edge in
    solid black with its underlying text/images SCRUBBED (a true redaction, not
    a cosmetic draw -- nothing extractable survives). A blackout never depends
    on per-match completeness, so a dropped page can never ship partially
    redacted. ``removed_pages`` wins any overlap: a deleted page needs no
    blackout, so a page listed in both is simply deleted.

    ``redact_label`` is the document-level default; since v0.6.4 a match may
    carry its own ``redact_label`` key that wins for that box alone. Both
    labelled and unlabelled matches burn in the SAME ``apply_redactions``
    pass -- the label is a per-annotation property, so per-match labels cost
    nothing beyond the dict lookup (the rejected alternative was one burn
    pass per distinct label, which would multiply the dominant export cost).
    """
    _validate_redaction_payload(matches, removed_pages, blackout_pages)
    # Flatten form fields BEFORE redacting: a widget /V survives a redaction
    # box otherwise (see _flatten_form_fields). No-op for non-form PDFs.
    _flatten_form_fields(doc)

    removed_set = set(removed_pages) if removed_pages else set()
    # A removed page is deleted outright, so blacking it out would be wasted
    # work -- remove takes precedence over blackout on any overlap.
    blackout_set = (set(blackout_pages) - removed_set) if blackout_pages else set()
    # `[]` means caller turned every category off — must NOT collapse to None.
    active_set = set(active_categories) if active_categories is not None else None
    by_page: dict[int, list[dict]] = {}
    for m in matches:
        if not m.get("enabled", True):
            continue
        if m["page"] in removed_set:
            continue
        if m["page"] in blackout_set:
            # The whole-page cover below supersedes any individual box on a
            # blackout page; skip per-match work so the page's safety never
            # rides on per-match completeness.
            continue
        if active_set is not None and m.get("type") not in active_set:
            if m.get("type") not in ("Manual Region", "Custom"):
                continue
        by_page.setdefault(m["page"], []).append(m)

    for pg_num, pg_matches in by_page.items():
        page = doc[pg_num]
        page_rect = Rect(page.rect)  # as-rendered (rotation-applied) box
        for m in pg_matches:
            r = m["rect"]
            rect = Rect(r["x0"], r["y0"], r["x1"], r["y1"])
            box_h = rect.height
            font_size = min(11, max(5, box_h * 0.7))
            # Off-page scrub strips for rects flush against a page edge --
            # computed in the as-rendered frame BEFORE derotation so the
            # flush test runs against the same box the app's rects live in
            # (Session 659; see _edge_overscan_strips).
            strips = _edge_overscan_strips(rect, page_rect)
            # Match rects arrive in the app's as-rendered (rotation-applied)
            # space -- the frame render_page / page_size / OCR geometry all
            # share. ``add_redact_annot`` instead interprets its rect in the
            # page's *native* (unrotated, MediaBox-origin) space, so on a
            # ``/Rotate`` page an untransformed burn lands displaced
            # (point-mirrored at 180, transposed at 90/270) -- privacy-grade
            # on the landscape legal / medical pages that carry rotation
            # flags. ``_derotate_to_native`` is the S590 two-step transform
            # (derotation + MediaBox-origin shift), pinned by affine-fit /
            # invert ground truth against the strict goldens -- see its
            # docstring. ``font_size`` keeps the as-rendered ``box_h`` so the
            # label tracks the visible box. Root-caused S514; fixed S590;
            # extraction-verified (fill AND text-scrub, both pymupdf lines)
            # S659.
            if page.rotation:
                rect = _derotate_to_native(rect, page)
                strips = [_derotate_to_native(s, page) for s in strips]
            for s in strips:
                page.add_redact_annot(s, fill=(0, 0, 0))
            # Per-match label (v0.6.4) overrides the document-level one for
            # THIS box only -- what lets one person's boxes carry a pseudonym
            # ("Patient A") while the rest of the document keeps its default.
            # Absent and empty-string both mean "no per-match label, use the
            # document one": ``redact_label=""`` already means "plain black
            # box" document-wide, so giving the same value a second, inverted
            # meaning per-match ("suppress the document label here") would be
            # a trap. A payload carrying no per-match labels therefore takes
            # byte-identical decisions to v0.6.3.
            label = m.get("redact_label") or redact_label
            if label:
                page.add_redact_annot(
                    rect,
                    text=label,
                    fontname="helv",
                    fontsize=font_size,
                    text_color=(1, 1, 1),
                    fill=(0, 0, 0),
                    align=TEXT_ALIGN_CENTER,
                )
            else:
                page.add_redact_annot(rect, fill=(0, 0, 0))
        page.apply_redactions()

    # Full-page blackout (Session 592): cover each blackout page edge-to-edge
    # in solid black and SCRUB the content beneath it. Runs before the
    # ``removed_set`` delete so both operate on original page indices, and on
    # the disjoint page set the per-match loop skipped (blackout pages are
    # excluded from ``by_page``), so no page gets ``apply_redactions`` twice.
    # The cover rect is the full *displayed* page mapped into native
    # (add_redact_annot) space via the same S590-pinned derotation as the
    # per-match path, then inflated ``_EDGE_OVERSCAN`` past every edge
    # (Session 659): content drawn OUTSIDE the page box -- clipped print
    # headers/footers -- pokes sliver glyphs into the page that extract via
    # ``get_text`` yet survive a page-bounds redaction region (the text
    # filter keeps sliver-overlap glyphs). The inflated region fully contains
    # them; the rendered page is identical (fills clip to the page), so the
    # blackout invariant is solid black AND zero extractable chars.
    for pg_num in sorted(blackout_set):
        if not (0 <= pg_num < len(doc)):
            continue
        page = doc[pg_num]
        rect = Rect(page.rect)
        if page.rotation:
            rect = _derotate_to_native(rect, page)
        rect = Rect(rect.x0 - _EDGE_OVERSCAN, rect.y0 - _EDGE_OVERSCAN,
                    rect.x1 + _EDGE_OVERSCAN, rect.y1 + _EDGE_OVERSCAN)
        page.add_redact_annot(rect, fill=(0, 0, 0))
        page.apply_redactions()

    if removed_set:
        valid_removed = {p for p in removed_set if 0 <= p < len(doc)}
        if valid_removed and len(valid_removed) >= len(doc):
            raise ValueError("Cannot export: all pages have been removed")
        for pg_num in sorted(valid_removed, reverse=True):
            doc.delete_page(pg_num)

    # Strip residue that lives OUTSIDE page content streams (annotation text,
    # attachments, document JavaScript, pre-burn page thumbnails) -- none of it
    # is touched by apply_redactions. Runs after every content pass so it sees
    # the final annot set, and before _strip_metadata_doc, which owns metadata.
    _scrub_residue(doc)

    _strip_metadata_doc(doc)

    mode = (output_protection or {}).get("mode") if output_protection else None

    if output_protection is None or mode == "none":
        out_bytes, protection_applied = _save_clean(doc), True
    elif mode in ("same", "new"):
        password = output_protection.get("password") or ""
        if not password:
            out_bytes, protection_applied = _save_clean(doc), False
        else:
            out_bytes, protection_applied = _save_encrypted(doc, password)
    else:
        logging.getLogger(__name__).warning(
            "apply_redactions: unknown output_protection mode; saving unprotected."
        )
        out_bytes, protection_applied = _save_clean(doc), False

    return out_bytes, protection_applied


def apply_redactions(pdf_bytes: bytes, matches: list[dict],
                     redact_label: str = "",
                     active_categories: list[str] | set[str] | None = None,
                     removed_pages: list[int] | None = None,
                     blackout_pages: list[int] | None = None,
                     *,
                     output_protection: dict | None = None
                     ) -> tuple[bytes, bool]:
    """Black-box redact enabled matches, return (new PDF bytes, protection_applied).

    Parameters
    ----------
    pdf_bytes
        Source PDF as bytes.
    matches
        List of ``{"page": int, "rect": {"x0": ..., "y0": ..., "x1": ...,
        "y1": ...}, "enabled": bool, "type": str}``. Disabled matches and
        matches whose ``type`` is not in ``active_categories`` are skipped
        (manual / custom regions are always included). A match may also
        carry an optional ``"redact_label": str`` (v0.6.4) labelling that
        box alone -- see ``redact_label`` below.
    redact_label
        Optional text to overlay on each black box ("REDACTED",
        "[b](6)", custom). Empty = plain black box. This is the
        DOCUMENT-level default; since v0.6.4 an individual match may carry
        its own ``redact_label`` key, which wins for that box only (absent
        or empty falls back here). Mixing labelled and unlabelled matches
        in one payload is supported and costs no extra burn pass.
    active_categories
        Optional set/list of category names. If provided, only matches
        whose ``type`` is in this set are redacted.
    removed_pages
        Optional list of page numbers to delete from the PDF.
    blackout_pages
        Optional list of page numbers to fully black out (the whole page is
        covered in solid black with its content scrubbed) while keeping the
        page in the output. ``removed_pages`` wins any overlap.
    output_protection
        Optional ``{"mode": "same"|"new"|"none", "password": str?}``.
        Modes "same"/"new" require a non-empty password. Re-encryption
        failures fall back to unprotected output rather than blocking the
        download (``protection_applied=False`` signals this).

    Returns
    -------
    tuple[bytes, bool]
        ``(new_pdf_bytes, protection_applied)`` where ``protection_applied``
        means "the requested protection shipped":

        * ``output_protection=None``         -> always ``True``
        * ``{"mode": "none"}``               -> always ``True``
        * ``{"mode": "same"|"new", ...}``    -> ``True`` iff the encrypted
                                                save succeeded; ``False``
                                                on fallback.

    Raises
    ------
    ValueError
        On malformed wire payload (bad page index, non-numeric rect coords,
        inverted bounds).
    """
    doc = open_pdf(pdf_bytes)
    try:
        return _apply_redactions_doc(
            doc, matches,
            redact_label=redact_label,
            active_categories=active_categories,
            removed_pages=removed_pages,
            blackout_pages=blackout_pages,
            output_protection=output_protection,
        )
    finally:
        doc.close()
