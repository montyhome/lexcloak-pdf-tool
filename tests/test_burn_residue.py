"""Burn-residue vectors: payload that survives ``apply_redactions`` because
it does not live in a page content stream (v0.6.6).

``apply_redactions`` rewrites page content streams. Four classes of payload
sit outside them and shipped intact in every export through v0.6.5 --
each verified surviving a real burn before the fix:

* **Annotation text** -- ``/Text`` sticky notes, ``/FreeText``, comments.
  ``page.get_text()`` does not return annot ``/Contents``, so detection never
  sees this text and no user can redact it.
* **Embedded / attached files** -- recovered byte-for-byte from a v0.6.5
  export via ``embfile_get``.
* **Document-level JavaScript**.
* **Page thumbnails** -- a cached raster of the page *before* the burn.

The fix deletes rather than bakes: ``bake(annots=True)`` would paint
annotation appearance streams into page content, converting text detection
never saw into permanent extractable content nothing redacted.

The 6-criteria framework:

* **Happy path** -- each of the four vectors is absent from the export, by
  structural API *and* by raw-byte sentinel scan.
* **Boundary / type** -- residue on a page with no matches (that page never
  calls ``apply_redactions``); on a blackout page; on a removed page; multi
  page; a document with no residue at all; the encrypted-output path.
* **Sad paths** (>=2 per happy) -- Link annots must SURVIVE (deliberate scope
  fence); the invisible OCR text layer must survive; visible unredacted text
  must survive; page count must not change.
* **No logic mirroring** -- sentinels are literal constants asserted against
  literal expectations; never recomputed through the code under test.
* **Side-effect verification** -- assertions read the exported BYTES (raw
  scan + re-opened document), not in-memory state.
* **Mock integrity** -- the scrub is exercised through the real public
  ``apply_redactions`` entry point, never by calling ``_scrub_residue``
  directly.

Synthetic fixtures only (per ``feedback_synthetic_fixtures_only``).
"""
from __future__ import annotations

import fitz
import pytest

from lexcloak_pdf_tool import apply_redactions


# ── Sentinels ────────────────────────────────────────────────────────
# Distinctive literals so a raw-byte scan cannot false-positive on ordinary
# PDF structure. Synthetic names, never real PII.

NOTE_TEXT = "STICKYSENTINEL-Zephyr-Quillon-8814"
FREETEXT_TEXT = "FREETEXTSENTINEL-Marisol-Vantalow"
ATTACH_NAME = "case-notes.txt"
ATTACH_PAYLOAD = b"ATTACHSENTINEL-embedded-payload-bytes"
JS_SENTINEL = "DOCJSSENTINEL"
BODY_TEXT = "Reference 553-01-8842 filed 2026-03-04."
OCR_TOKEN = "OCR-INVISIBLE-LAYER-42"
LINK_URI = "https://example.invalid/scope-fence"

# The body rect, in the coordinates the fixture writes BODY_TEXT at.
BODY_MATCH = {
    "page": 0,
    "rect": {"x0": 40.0, "y0": 88.0, "x1": 400.0, "y1": 106.0},
    "type": "Manual Region",
    "enabled": True,
}


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_residue_pdf(
    *,
    n_pages: int = 1,
    annot_page: int = 0,
    with_attachment: bool = True,
    with_javascript: bool = True,
    with_link: bool = False,
    ocr_token: str | None = None,
) -> bytes:
    """Synthetic PDF carrying the residue vectors under test."""
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text(fitz.Point(50, 100), BODY_TEXT, fontsize=12)
        if ocr_token is not None:
            page.insert_text(fitz.Point(50, 140), ocr_token,
                             fontsize=12, render_mode=3)

    page = doc[annot_page]
    page.add_text_annot(fitz.Point(300, 200), NOTE_TEXT)
    ft = page.add_freetext_annot(fitz.Rect(300, 300, 560, 340), FREETEXT_TEXT)
    ft.update()
    if with_link:
        page.insert_link({
            "kind": fitz.LINK_URI,
            "from": fitz.Rect(50, 400, 200, 420),
            "uri": LINK_URI,
        })

    if with_attachment:
        doc.embfile_add(ATTACH_NAME, ATTACH_PAYLOAD)
    if with_javascript:
        xref = doc.get_new_xref()
        doc.update_object(
            xref, "<</S/JavaScript/JS(var s = '%s';)>>" % JS_SENTINEL)
        doc.xref_set_key(doc.pdf_catalog(), "Names/JavaScript/Names",
                         "[(sentinel) %d 0 R]" % xref)

    out = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return out


def _annot_inventory(pdf_bytes: bytes) -> list[tuple[str, str]]:
    """(subtype, contents) for every annotation in the exported bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [(a.type[1], a.info.get("content", ""))
                for page in doc for a in page.annots()]
    finally:
        doc.close()


def _embfile_names(pdf_bytes: bytes) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return list(doc.embfile_names())
    finally:
        doc.close()


def _page_text(pdf_bytes: bytes, pno: int = 0) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc.load_page(pno).get_text()
    finally:
        doc.close()


def _page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()


# ── Happy path: each vector is gone from the export ──────────────────


def test_baseline_fixture_actually_carries_every_vector():
    """Guard the guards: a fixture that stopped carrying residue would make
    every absence assertion below vacuously true."""
    src = _make_residue_pdf(with_link=True)
    subtypes = {t for t, _ in _annot_inventory(src)}
    assert {"Text", "FreeText"} <= subtypes
    assert _embfile_names(src) == [ATTACH_NAME]
    assert NOTE_TEXT.encode() in src
    assert FREETEXT_TEXT.encode() in src
    assert JS_SENTINEL.encode() in src


def test_annotation_text_absent_from_export():
    out, _ = apply_redactions(_make_residue_pdf(), [BODY_MATCH])
    assert _annot_inventory(out) == []
    assert NOTE_TEXT.encode() not in out
    assert FREETEXT_TEXT.encode() not in out


def test_embedded_attachment_absent_from_export():
    out, _ = apply_redactions(_make_residue_pdf(), [BODY_MATCH])
    assert _embfile_names(out) == []
    assert ATTACH_PAYLOAD not in out
    assert ATTACH_NAME.encode() not in out


def test_document_javascript_body_is_emptied():
    """scrub(javascript=True) rewrites the action body to ``/JS ()``."""
    out, _ = apply_redactions(_make_residue_pdf(), [BODY_MATCH])
    assert JS_SENTINEL.encode() not in out


def test_document_javascript_name_tree_is_dropped():
    """Emptying the action body is not enough: the /Names/JavaScript tree
    survives it, and the entry NAMES are author-chosen -- a script named for
    a matter or custodian would ride out in a "carries nothing along"
    export. The catalog key must go, not just the script body."""
    out, _ = apply_redactions(_make_residue_pdf(), [BODY_MATCH])
    doc = fitz.open(stream=out, filetype="pdf")
    try:
        names = doc.xref_get_key(doc.pdf_catalog(), "Names/JavaScript")
    finally:
        doc.close()
    assert names == ("null", "null")
    assert b"sentinel" not in out


def _attach_thumbnail(pdf_bytes: bytes, pno: int = 0) -> bytes:
    """Give page ``pno`` a /Thumb raster of its own pre-burn appearance."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[pno]
    pix = page.get_pixmap(dpi=12)
    thumb_xref = doc.get_new_xref()
    doc.update_object(thumb_xref, "<<>>")
    doc.update_stream(thumb_xref, pix.tobytes("png"))
    doc.xref_set_key(page.xref, "Thumb", "%d 0 R" % thumb_xref)
    out = doc.tobytes(garbage=0, deflate=True)
    doc.close()
    return out


def _thumb_key(pdf_bytes: bytes, pno: int = 0):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc.xref_get_key(doc[pno].xref, "Thumb")
    finally:
        doc.close()


def test_page_thumbnail_absent_from_export():
    """A /Thumb is a cached raster of the page as it looked BEFORE the burn
    -- a picture of the content just redacted away.

    Regression guard for a live PyMuPDF footgun: ``scrub(thumbnails=True)``
    is a NO-OP on the flag-set this tool must use, because scrub's page loop
    early-continues on ``not (clean_pages or hidden_text)`` before reaching
    its thumbnail branch. Measured against pymupdf 1.27.2 on 2026-08-02: an
    820-byte PNG survived a full burn with the flag set. If this test ever
    starts passing without ``null_page_thumbnails``, upstream changed.
    """
    src = _attach_thumbnail(_make_residue_pdf())
    assert _thumb_key(src) != ("null", "null")     # fixture guard

    out, _ = apply_redactions(src, [BODY_MATCH])
    assert _thumb_key(out) == ("null", "null")


def test_thumbnail_raster_bytes_do_not_survive_the_burn():
    """Nulling the key must actually orphan the stream, not just unlink a
    reference that garbage collection then keeps alive."""
    src = _attach_thumbnail(_make_residue_pdf())
    pre = fitz.open(stream=src, filetype="pdf")
    try:
        pre_streams = sum(1 for x in range(1, pre.xref_length())
                          if pre.xref_is_stream(x))
    finally:
        pre.close()

    out, _ = apply_redactions(src, [BODY_MATCH])
    post = fitz.open(stream=out, filetype="pdf")
    try:
        post_streams = sum(1 for x in range(1, post.xref_length())
                           if post.xref_is_stream(x))
    finally:
        post.close()
    assert post_streams < pre_streams


def test_thumbnail_scrubbed_on_a_page_with_no_matches():
    """Same unvisited-page boundary as the annotation case -- the thumbnail
    of an unredacted page still pictures unredacted content."""
    src = _attach_thumbnail(_make_residue_pdf(n_pages=2), pno=1)
    assert _thumb_key(src, 1) != ("null", "null")   # fixture guard

    out, _ = apply_redactions(src, [BODY_MATCH])    # match on page 0 only
    assert _thumb_key(out, 1) == ("null", "null")


# ── Scope fence: what must SURVIVE ───────────────────────────────────


def test_link_annotations_survive_the_scrub():
    """Non-overlapping links are a deliberate out-of-scope decision -- a
    /Link is a rect plus a destination and carries no free text of its own.
    If this ever starts failing, it is a scope change, not a bug fix."""
    out, _ = apply_redactions(_make_residue_pdf(with_link=True), [BODY_MATCH])
    doc = fitz.open(stream=out, filetype="pdf")
    try:
        uris = [lk.get("uri") for lk in doc[0].get_links()]
    finally:
        doc.close()
    assert LINK_URI in uris
    # ...and the text-bearing annots next to it still went.
    assert NOTE_TEXT.encode() not in out


def test_invisible_ocr_layer_survives_the_scrub():
    """hidden_text=False in the scrub flag-set is load-bearing: a scanned
    redacted PDF carries its searchable layer as render-mode-3 text."""
    src = _make_residue_pdf(ocr_token=OCR_TOKEN)
    out, _ = apply_redactions(src, [BODY_MATCH])
    assert OCR_TOKEN in _page_text(out)


def test_unredacted_visible_text_survives_the_scrub():
    src = _make_residue_pdf(n_pages=2)
    out, _ = apply_redactions(src, [BODY_MATCH])
    # Page 0's body was the match; page 1 was never targeted.
    assert "553-01-8842" not in _page_text(out, 0)
    assert "553-01-8842" in _page_text(out, 1)


def test_page_count_unchanged_by_the_scrub():
    src = _make_residue_pdf(n_pages=3)
    out, _ = apply_redactions(src, [BODY_MATCH])
    assert _page_count(out) == 3


def test_document_with_no_residue_is_unaffected():
    src = _make_residue_pdf(with_attachment=False, with_javascript=False)
    doc = fitz.open(stream=src, filetype="pdf")
    for page in doc:
        for annot in list(page.annots()):
            page.delete_annot(annot)
    clean_src = doc.tobytes(garbage=4, deflate=True)
    doc.close()

    out, _ = apply_redactions(clean_src, [BODY_MATCH])
    assert _annot_inventory(out) == []
    assert _page_count(out) == 1
    assert "553-01-8842" not in _page_text(out)


# ── Boundary: pages the redaction loop never visits ──────────────────


def test_residue_on_page_with_no_matches_is_still_scrubbed():
    """The per-match loop only calls apply_redactions on pages carrying a
    match. A page with residue but no match is never visited by it -- the
    scrub must walk every page, not just the redacted ones."""
    src = _make_residue_pdf(n_pages=2, annot_page=1)
    out, _ = apply_redactions(src, [BODY_MATCH])   # match is on page 0 only
    assert _annot_inventory(out) == []
    assert NOTE_TEXT.encode() not in out
    assert FREETEXT_TEXT.encode() not in out


def test_residue_on_blackout_page_is_scrubbed():
    src = _make_residue_pdf(n_pages=2, annot_page=1)
    out, _ = apply_redactions(src, [BODY_MATCH], blackout_pages=[1])
    assert _annot_inventory(out) == []
    assert NOTE_TEXT.encode() not in out


def test_residue_on_removed_page_goes_with_the_page():
    src = _make_residue_pdf(n_pages=2, annot_page=1)
    out, _ = apply_redactions(src, [BODY_MATCH], removed_pages=[1])
    assert _page_count(out) == 1
    assert _annot_inventory(out) == []
    assert NOTE_TEXT.encode() not in out


def test_residue_scrubbed_on_every_page_of_a_multipage_doc():
    doc = fitz.open(stream=_make_residue_pdf(n_pages=3), filetype="pdf")
    for pno in (1, 2):
        doc[pno].add_text_annot(fitz.Point(300, 200), NOTE_TEXT)
    src = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    assert len(_annot_inventory(src)) == 4      # 2 on page 0, 1 each on 1-2

    out, _ = apply_redactions(src, [BODY_MATCH])
    assert _annot_inventory(out) == []
    assert NOTE_TEXT.encode() not in out


# ── Boundary: the encrypted-output path ──────────────────────────────


def test_residue_scrubbed_under_encrypted_output():
    """The scrub runs before the save, so it must hold on the encrypt arm
    too -- otherwise encryption merely hides un-scrubbed residue."""
    src = _make_residue_pdf()
    out, protected = apply_redactions(
        src, [BODY_MATCH],
        output_protection={"mode": "new", "password": "pw-sentinel"},
    )
    assert protected is True

    doc = fitz.open(stream=out, filetype="pdf")
    try:
        assert doc.authenticate("pw-sentinel")
        inventory = [(a.type[1], a.info.get("content", ""))
                     for page in doc for a in page.annots()]
        names = list(doc.embfile_names())
    finally:
        doc.close()
    assert inventory == []
    assert names == []


# ── Sad path: a source document carrying stale redaction annots ──────


def test_unapplied_source_redaction_annots_do_not_survive():
    """A source PDF can arrive with someone else's un-applied /Redact
    annots. They must not ride into a Lex Cloak export as live annotations."""
    doc = fitz.open(stream=_make_residue_pdf(), filetype="pdf")
    doc[0].add_redact_annot(fitz.Rect(400, 500, 500, 520), fill=(0, 0, 0))
    src = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    assert any(t == "Redact" for t, _ in _annot_inventory(src))

    out, _ = apply_redactions(src, [BODY_MATCH])
    assert _annot_inventory(out) == []


@pytest.mark.parametrize("annot_page", [0, 1])
def test_scrub_is_independent_of_which_page_carries_residue(annot_page):
    src = _make_residue_pdf(n_pages=2, annot_page=annot_page)
    out, _ = apply_redactions(src, [BODY_MATCH])
    assert _annot_inventory(out) == []
    assert NOTE_TEXT.encode() not in out
