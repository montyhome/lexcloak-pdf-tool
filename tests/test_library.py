"""Library-API tests for ``lexcloak_pdf_tool``.

Exercises every public IPC-clean function plus CharData serialize /
deserialize round-trips. The 6-criteria framework:

* **Happy path** -- render, page_count, page_size, is_encrypted,
  extract_native, search_for, apply_redactions, strip_metadata,
  extract_text_ocr (when Tesseract is available) all return correct
  results on a fixture PDF.
* **Boundary** -- page_num at 0 (first page) and len(doc)-1 (last page).
* **Sad path** -- out-of-range page_num raises IndexError; malformed match
  payload raises ValueError; corrupt PDF bytes raise on open_pdf.
* **Idempotence** -- serialize/deserialize/serialize round-trips and
  matches.
* **Encrypted-PDF gate** -- ``is_encrypted`` correctly distinguishes
  empty-pw (False -- PyMuPDF auto-authenticates) from real-pw (True).
"""
from __future__ import annotations

import fitz
import pytest

from lexcloak_pdf_tool import (
    all_page_sizes,
    apply_redactions,
    extract_text_native,
    insert_cover_page,
    is_encrypted,
    page_count,
    page_size,
    pymupdf_version,
    render_page,
    search_for,
    set_metadata,
    strip_metadata,
)
from lexcloak_pdf_tool.cover_page import (
    _COVER_PAGE_FOOTER,
    _COVER_PAGE_TITLE,
    _insert_cover_page_doc,
)
from lexcloak_pdf_tool.metadata import _ALLOWED_METADATA_KEYS, _set_metadata_doc
from lexcloak_pdf_tool.coords import (
    deserialize_chardata,
    search_in_chars,
    search_whole_word_in_chars,
    serialize_chardata,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _make_pdf(text: str = "Patient SSN 123-45-6789",
              x: float = 50, y: float = 100,
              fontsize: float = 12, n_pages: int = 1) -> bytes:
    """Build an n-page PDF with text on each page."""
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text(fitz.Point(x, y), text, fontsize=fontsize)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_encrypted_pdf(password: str) -> bytes:
    """Build a single-page password-protected PDF."""
    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text(
        fitz.Point(50, 100), "Confidential", fontsize=12)
    out = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw=password,
        owner_pw=password,
    )
    doc.close()
    return out


# ── page_count / page_size / is_encrypted ────────────────────────────


def test_page_count_single_page():
    assert page_count(_make_pdf(n_pages=1)) == 1


def test_page_count_multi_page():
    assert page_count(_make_pdf(n_pages=7)) == 7


def test_page_size_default_letter():
    w, h = page_size(_make_pdf(), 0)
    assert w == pytest.approx(612.0)
    assert h == pytest.approx(792.0)


def test_page_size_first_page_boundary():
    w, h = page_size(_make_pdf(n_pages=3), 0)
    assert w > 0 and h > 0


def test_page_size_last_page_boundary():
    pdf = _make_pdf(n_pages=3)
    page_size(pdf, 2)


def test_page_size_out_of_range_raises():
    with pytest.raises(IndexError):
        page_size(_make_pdf(n_pages=2), 5)


def test_page_size_negative_page_raises():
    with pytest.raises(IndexError):
        page_size(_make_pdf(), -1)


# ── all_page_sizes ───────────────────────────────────────────────────


def test_all_page_sizes_single_page():
    sizes = all_page_sizes(_make_pdf(n_pages=1))
    assert len(sizes) == 1
    w, h = sizes[0]
    assert w == pytest.approx(612.0)
    assert h == pytest.approx(792.0)


def test_all_page_sizes_multi_page_uniform_dims():
    sizes = all_page_sizes(_make_pdf(n_pages=5))
    assert len(sizes) == 5
    for w, h in sizes:
        assert w == pytest.approx(612.0)
        assert h == pytest.approx(792.0)


def test_all_page_sizes_mixed_dimensions():
    """Pages with different dimensions surface accurately."""
    doc = fitz.open()
    doc.new_page(width=612, height=792)   # US Letter
    doc.new_page(width=595, height=842)   # A4
    doc.new_page(width=792, height=612)   # Letter landscape
    pdf = doc.tobytes()
    doc.close()
    sizes = all_page_sizes(pdf)
    assert len(sizes) == 3
    assert sizes[0][0] == pytest.approx(612.0)
    assert sizes[0][1] == pytest.approx(792.0)
    assert sizes[1][0] == pytest.approx(595.0)
    assert sizes[1][1] == pytest.approx(842.0)
    assert sizes[2][0] == pytest.approx(792.0)
    assert sizes[2][1] == pytest.approx(612.0)


def test_all_page_sizes_matches_per_page_page_size():
    """Behavioral equivalence with N sequential page_size calls — load-bearing
    for the engine-side fallback path when an old subprocess lacks the op."""
    pdf = _make_pdf(n_pages=4)
    batch = all_page_sizes(pdf)
    sequential = [page_size(pdf, i) for i in range(4)]
    assert batch == sequential


def test_all_page_sizes_corrupt_pdf_raises():
    """Fail-fast on bad input, matching the rest of the library API."""
    with pytest.raises(Exception):  # noqa: B017 -- PyMuPDF raises various
        all_page_sizes(b"not a pdf")


def test_all_page_sizes_empty_pw_encrypted_pdf_passthrough():
    """PyMuPDF auto-authenticates empty-password PDFs, so all_page_sizes
    succeeds with the same dims a plaintext doc would return."""
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    pdf = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="", user_pw="")
    doc.close()
    sizes = all_page_sizes(pdf)
    assert len(sizes) == 1
    assert sizes[0][0] == pytest.approx(612.0)
    assert sizes[0][1] == pytest.approx(792.0)


def test_is_encrypted_plaintext_pdf():
    assert is_encrypted(_make_pdf()) is False


def test_is_encrypted_real_password():
    # PyMuPDF auto-authenticates empty-password PDFs silently, so only a
    # real-password PDF surfaces as is_encrypted=True here.
    assert is_encrypted(_make_encrypted_pdf("secret123")) is True


def test_open_corrupt_pdf_raises():
    # The library API is fail-fast on bad input -- bad bytes should raise
    # rather than return a sentinel.
    with pytest.raises(Exception):  # noqa: B017 -- PyMuPDF raises various types
        page_count(b"not a pdf")


# ── render_page ──────────────────────────────────────────────────────


def test_render_page_returns_png_bytes():
    png = render_page(_make_pdf(), 0, dpi=72)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 100


def test_render_page_dpi_affects_size():
    pdf = _make_pdf()
    small = render_page(pdf, 0, dpi=50)
    large = render_page(pdf, 0, dpi=200)
    assert len(large) > len(small)


def test_render_page_out_of_range_raises():
    with pytest.raises(IndexError):
        render_page(_make_pdf(), 99, dpi=72)


# ── extract_text_native ──────────────────────────────────────────────


def test_extract_text_native_finds_words():
    pdf = _make_pdf("Patient SSN 123-45-6789")
    words = extract_text_native(pdf, 0)
    texts = [w["text"] for w in words]
    assert "Patient" in texts
    assert "SSN" in texts


def test_extract_text_native_word_shape():
    pdf = _make_pdf()
    words = extract_text_native(pdf, 0)
    for w in words:
        assert set(w.keys()) >= {"text", "x0", "y0", "x1", "y1",
                                  "block", "line", "word"}
        assert isinstance(w["x0"], float)
        assert w["x1"] > w["x0"]
        assert w["y1"] > w["y0"]


def test_extract_text_native_empty_page_returns_empty():
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    pdf = doc.tobytes()
    doc.close()
    assert extract_text_native(pdf, 0) == []


# ── search_for ───────────────────────────────────────────────────────


def test_search_for_native_path_finds_text():
    pdf = _make_pdf("Patient John Smith here")
    rects = search_for(pdf, 0, "John")
    assert len(rects) >= 1
    r = rects[0]
    assert r.x1 > r.x0
    assert r.y1 > r.y0


def test_search_for_no_match_returns_empty():
    pdf = _make_pdf("Patient John Smith")
    assert search_for(pdf, 0, "unicornicide") == []


def test_search_for_chardata_path():
    chardata = [
        ("J", (10.0, 10.0, 15.0, 20.0)),
        ("o", (15.0, 10.0, 20.0, 20.0)),
        ("h", (20.0, 10.0, 25.0, 20.0)),
        ("n", (25.0, 10.0, 30.0, 20.0)),
    ]
    rects = search_for(b"unused", 0, "John", ocr_chardata=chardata)
    assert len(rects) == 1


# ── apply_redactions ─────────────────────────────────────────────────


def test_apply_redactions_returns_bytes_and_protection_flag():
    pdf = _make_pdf("Patient SSN 123-45-6789")
    out, protection_applied = apply_redactions(pdf, [], output_protection=None)
    assert isinstance(out, bytes)
    assert out[:4] == b"%PDF"
    assert protection_applied is True


def test_apply_redactions_actually_redacts():
    pdf = _make_pdf("Patient SSN 123-45-6789")
    matches = [{
        "id": "x", "type": "SSN", "page": 0,
        "rect": {"x0": 30, "y0": 80, "x1": 300, "y1": 120},
        "enabled": True, "text": "123-45-6789",
    }]
    out, _ = apply_redactions(pdf, matches)
    out_doc = fitz.open(stream=out, filetype="pdf")
    redacted_text = out_doc[0].get_text()
    out_doc.close()
    assert "123-45-6789" not in redacted_text


def test_apply_redactions_invalid_match_raises_valueerror():
    pdf = _make_pdf()
    matches = [{
        "id": "bad", "type": "SSN", "page": "abc",  # not int
        "rect": {"x0": 10, "y0": 10, "x1": 50, "y1": 30},
        "enabled": True,
    }]
    with pytest.raises(ValueError, match="non-integer"):
        apply_redactions(pdf, matches)


def test_apply_redactions_inverted_rect_raises():
    pdf = _make_pdf()
    matches = [{
        "id": "bad", "type": "SSN", "page": 0,
        "rect": {"x0": 100, "y0": 100, "x1": 50, "y1": 50},  # x1 < x0
        "enabled": True,
    }]
    with pytest.raises(ValueError, match="invalid bounds"):
        apply_redactions(pdf, matches)


# ── apply_redactions: /Rotate geometry (Session 590) ───────────────────
#
# The app supplies match rects in as-rendered (rotation-applied) space, but
# add_redact_annot interprets them in the page's native (unrotated) space.
# Pre-S590, _apply_redactions_doc burned with no rotation transform, so on a
# /Rotate page the box landed point-mirrored (180) / transposed (90, 270) and
# the covered content stayed readable in the export. Privacy-grade. The fix
# derotates each rect (+ a MediaBox-origin shift for cropped pages) before the
# burn. The redact app's strict-xfail goldens
# (tests/test_session_514_apply_rotation.py) are the integration oracle; this
# pins the same invariant directly on the library function.


def _is_dark_at(pdf_bytes: bytes, page: int, x_pt: float, y_pt: float,
                dpi: float = 144.0) -> bool:
    """True if the as-rendered pixel at (x_pt, y_pt) point-space is near-black.

    Renders the page as displayed (get_pixmap honors /Rotate), so the point
    coordinates are in the same as-rendered frame the app supplies rects in.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = dpi / 72.0
        pix = doc[page].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        r, g, b = pix.pixel(int(x_pt * zoom), int(y_pt * zoom))[:3]
        return r < 40 and g < 40 and b < 40
    finally:
        doc.close()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_apply_redactions_rotated_page_lands_in_as_rendered_space(rotation):
    """A redaction supplied in as-rendered space lands on that exact region in
    the export for every /Rotate value. The supplied center (150, 120) must
    render black; pre-fix it stayed white because the box was displaced into
    unrotated space."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    if rotation:
        page.set_rotation(rotation)
    pdf = doc.tobytes()
    doc.close()
    matches = [{
        "id": "m", "type": "Manual Region", "page": 0,
        "rect": {"x0": 100, "y0": 100, "x1": 200, "y1": 140},
        "enabled": True, "text": "",
    }]
    out, _ = apply_redactions(pdf, matches)
    assert _is_dark_at(out, 0, 150, 120), (
        f"/Rotate {rotation}: redaction did not land on the supplied "
        f"as-rendered region (pre-S590 it burned in unrotated space)"
    )
    # A point well outside the supplied box stays clear — the box is localized,
    # not a whole-page smear.
    assert not _is_dark_at(out, 0, 40, 40), (
        f"/Rotate {rotation}: unexpected dark pixel far from the supplied box"
    )


# ── apply_redactions: active_categories filter ─────────────────────────
#
# Regression guards for the empty-list-vs-None distinction. The earlier
# `if active_categories else None` collapsed `[]` ("user turned every
# category off") into `None` ("no filter"), causing every match to be
# redacted regardless of UI state. Each test below pins one of the four
# code paths through the filter.


def _ssn_match(extra: dict | None = None) -> dict:
    """SSN match positioned over the fixture text 'Patient SSN 123-45-6789'."""
    m = {
        "id": "ssn", "type": "SSN", "page": 0,
        "rect": {"x0": 30, "y0": 80, "x1": 300, "y1": 120},
        "enabled": True, "text": "123-45-6789",
    }
    if extra:
        m.update(extra)
    return m


def _read_page_text(pdf_bytes: bytes, page: int = 0) -> str:
    out_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return out_doc[page].get_text()
    finally:
        out_doc.close()


def test_apply_redactions_active_categories_none_redacts_all():
    """`active_categories=None` (no filter) still redacts every enabled match."""
    pdf = _make_pdf("Patient SSN 123-45-6789")
    out, _ = apply_redactions(pdf, [_ssn_match()], active_categories=None)
    assert "123-45-6789" not in _read_page_text(out)


def test_apply_redactions_active_categories_empty_skips_detector_matches():
    """`active_categories=[]` (every UI category off) must NOT redact
    detector-typed matches. Privacy regression guard."""
    pdf = _make_pdf("Patient SSN 123-45-6789")
    out, _ = apply_redactions(pdf, [_ssn_match()], active_categories=[])
    assert "123-45-6789" in _read_page_text(out)


def test_apply_redactions_active_categories_empty_still_redacts_manual_region():
    """`active_categories=[]` keeps the Manual Region / Custom bypass —
    user-drawn regions are not category-gated."""
    pdf = _make_pdf("Patient SSN 123-45-6789")
    manual = _ssn_match({"type": "Manual Region", "id": "manual"})
    out, _ = apply_redactions(pdf, [manual], active_categories=[])
    assert "123-45-6789" not in _read_page_text(out)


def test_apply_redactions_active_categories_empty_still_redacts_custom():
    """`Custom` matches also bypass the category filter."""
    pdf = _make_pdf("Patient SSN 123-45-6789")
    custom = _ssn_match({"type": "Custom", "id": "custom"})
    out, _ = apply_redactions(pdf, [custom], active_categories=[])
    assert "123-45-6789" not in _read_page_text(out)


def test_apply_redactions_active_categories_populated_filters_by_type():
    """Populated list redacts only matches whose type is in the set."""
    pdf = _make_pdf("Patient SSN 123-45-6789")
    ssn = _ssn_match()
    person = _ssn_match({"type": "Person Name", "id": "person"})
    out, _ = apply_redactions(pdf, [ssn, person], active_categories=["SSN"])
    # Only one rect covers the text region, so both matches target the same
    # text; the SSN match drives the redaction. Verify the type filter does
    # not block it.
    assert "123-45-6789" not in _read_page_text(out)


def test_apply_redactions_active_categories_populated_skips_other_types():
    """Match whose type is NOT in active_categories is skipped."""
    pdf = _make_pdf("Patient SSN 123-45-6789")
    person = _ssn_match({"type": "Person Name", "id": "person"})
    out, _ = apply_redactions(pdf, [person], active_categories=["SSN"])
    assert "123-45-6789" in _read_page_text(out)


def test_apply_redactions_active_categories_mixed_obeys_filter_and_bypass():
    """Mixed match list with empty active_categories: only Manual Region
    survives the filter; detector matches are skipped."""
    pdf = _make_pdf("Patient SSN 123-45-6789", n_pages=2)
    p1_detector = _ssn_match()
    p2_manual = _ssn_match({"type": "Manual Region", "id": "manual", "page": 1})
    out, _ = apply_redactions(pdf, [p1_detector, p2_manual],
                              active_categories=[])
    # Detector match on page 0 must NOT redact.
    assert "123-45-6789" in _read_page_text(out, 0)
    # Manual region on page 1 must redact.
    assert "123-45-6789" not in _read_page_text(out, 1)


# ── strip_metadata ───────────────────────────────────────────────────


def test_strip_metadata_bytes_form_strips_author():
    doc = fitz.open()
    doc.set_metadata({"author": "Dr. Jane Doe", "title": "Confidential"})
    doc.new_page(width=612, height=792)
    pdf = doc.tobytes()
    doc.close()

    out = strip_metadata(pdf)
    out_doc = fitz.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert (meta.get("author") or "") == ""
    assert (meta.get("title") or "") == ""


def test_strip_metadata_doc_form_returns_none_and_mutates():
    """Live ``Document`` form preserves the legacy in-place semantics."""
    doc = fitz.open()
    doc.set_metadata({"author": "Dr. Jane Doe"})
    doc.new_page()
    result = strip_metadata(doc)
    assert result is None
    assert (doc.metadata or {}).get("author") in (None, "")
    doc.close()


# ── set_metadata ─────────────────────────────────────────────────────


def test_set_metadata_round_trip_subject_producer_keywords():
    """The Spec-13 happy path: Subject/Producer/Keywords land verbatim."""
    pdf = _make_pdf()
    fields = {
        "subject": "Auto-redacted by Lex Cloak. Review before distribution.",
        "producer": "Lex Cloak 1.7.8",
        "keywords": "auto-redacted, review-required",
    }
    out = set_metadata(pdf, fields)

    out_doc = fitz.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert meta["subject"] == fields["subject"]
    assert meta["producer"] == fields["producer"]
    assert meta["keywords"] == fields["keywords"]


def test_set_metadata_preserves_existing_fields_not_in_payload():
    """Merge semantics: fields not in ``payload`` survive untouched."""
    doc = fitz.open()
    doc.set_metadata({"author": "Original Author", "title": "Original Title"})
    doc.new_page()
    pdf = doc.tobytes()
    doc.close()

    out = set_metadata(pdf, {"subject": "added subject"})
    out_doc = fitz.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert meta["author"] == "Original Author"
    assert meta["title"] == "Original Title"
    assert meta["subject"] == "added subject"


def test_set_metadata_empty_dict_is_noop():
    """Empty fields round-trips the PDF without metadata changes."""
    doc = fitz.open()
    doc.set_metadata({"author": "A", "title": "T"})
    doc.new_page()
    pdf = doc.tobytes()
    doc.close()

    out = set_metadata(pdf, {})
    out_doc = fitz.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert meta["author"] == "A"
    assert meta["title"] == "T"


def test_set_metadata_unknown_key_raises_value_error():
    """Caller-supplied unknown keys fail fast with a named-key message."""
    pdf = _make_pdf()
    with pytest.raises(ValueError, match=r"unknown metadata key"):
        set_metadata(pdf, {"subject": "ok", "bogus_field": "nope"})


def test_set_metadata_non_dict_payload_raises_value_error():
    """``fields`` must be a dict; anything else fails fast."""
    pdf = _make_pdf()
    with pytest.raises(ValueError, match=r"fields must be a dict"):
        set_metadata(pdf, "not a dict")  # type: ignore[arg-type]


def test_set_metadata_unicode_keywords_round_trip():
    """Non-ASCII keywords (e.g., accented language codes) survive."""
    pdf = _make_pdf()
    out = set_metadata(pdf, {"keywords": "auto-redactée, revue-requise"})
    out_doc = fitz.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert meta["keywords"] == "auto-redactée, revue-requise"


def test_set_metadata_long_value_round_trip():
    """Producer-style strings well past typical lengths round-trip cleanly."""
    pdf = _make_pdf()
    long_value = "Lex Cloak " + ("x" * 4096)
    out = set_metadata(pdf, {"producer": long_value})
    out_doc = fitz.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert meta["producer"] == long_value


def test_set_metadata_doc_form_mutates_in_place():
    """``_set_metadata_doc`` mutates the live Document, returns None."""
    doc = fitz.open()
    doc.set_metadata({"author": "A"})
    doc.new_page()
    result = _set_metadata_doc(doc, {"subject": "S"})
    assert result is None
    meta = doc.metadata or {}
    assert meta["author"] == "A"
    assert meta["subject"] == "S"
    doc.close()


def test_set_metadata_doc_form_empty_dict_is_noop():
    """No-op short-circuits before PyMuPDF call (no metadata mutation)."""
    doc = fitz.open()
    doc.set_metadata({"author": "A"})
    doc.new_page()
    _set_metadata_doc(doc, {})
    assert (doc.metadata or {})["author"] == "A"
    doc.close()


def test_set_metadata_allowed_keys_matches_pymupdf_contract():
    """The validator's whitelist must be a subset of what PyMuPDF accepts.

    Regression guard: if PyMuPDF drops a key we list, our pre-check would
    pass an unsupported field through and surface a deeper error.
    """
    doc = fitz.open()
    doc.new_page()
    # PyMuPDF accepts the entire allowed-set; verify by setting all at once.
    valid_payload = {k: "" for k in _ALLOWED_METADATA_KEYS}
    # 'encryption' read-back is None for plaintext; setting "" leaves it None
    # but PyMuPDF doesn't reject the key name. Same for 'format'.
    doc.set_metadata(valid_payload)
    doc.close()


def test_set_metadata_redacted_pdf_then_strip_round_trip():
    """Lex Cloak pipeline: redact -> set Spec-13 metadata -> strip == idempotent
    on metadata wipe. Demonstrates set + strip work in the same flow."""
    pdf = _make_pdf("Patient SSN 123-45-6789")
    # Apply Spec-13 fields
    intermediate = set_metadata(pdf, {
        "subject": "Auto-redacted by Lex Cloak. Review before distribution.",
        "producer": "Lex Cloak 1.7.8",
        "keywords": "auto-redacted, review-required",
    })
    # Verify they landed
    doc = fitz.open(stream=intermediate, filetype="pdf")
    assert (doc.metadata or {})["subject"].startswith("Auto-redacted")
    doc.close()
    # Then strip should wipe them
    stripped = strip_metadata(intermediate)
    doc = fitz.open(stream=stripped, filetype="pdf")
    meta = doc.metadata or {}
    doc.close()
    assert (meta.get("subject") or "") == ""
    assert (meta.get("keywords") or "") == ""


# ── insert_cover_page ───────────────────────────────────────────────


def _default_context(date="2026-05-17", n=12, p=5, version="1.7.8") -> dict:
    return {
        "date": date,
        "redacted_count": n,
        "page_count": p,
        "product_version": version,
    }


def test_insert_cover_page_adds_one_page_at_index_zero():
    """Page count grows by 1; original page-0 content shifts to page 1."""
    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text(
        fitz.Point(50, 100), "ORIGINAL_PAGE_ONE_MARKER", fontsize=12)
    doc.new_page(width=612, height=792).insert_text(
        fitz.Point(50, 100), "ORIGINAL_PAGE_TWO_MARKER", fontsize=12)
    pdf = doc.tobytes()
    doc.close()

    out = insert_cover_page(pdf, _default_context(p=2))
    out_doc = fitz.open(stream=out, filetype="pdf")
    try:
        assert len(out_doc) == 3
        assert "ORIGINAL_PAGE_ONE_MARKER" in out_doc[1].get_text()
        assert "ORIGINAL_PAGE_TWO_MARKER" in out_doc[2].get_text()
    finally:
        out_doc.close()


def test_insert_cover_page_renders_title_and_footer_with_em_dash():
    """Verbatim title + footer including em-dash (U+2014) render correctly.

    The title is searched whole (no ligature glyphs). The footer is
    split because PyMuPDF's renderer emits an ``ﬁ`` ligature (U+FB01) for
    the ``fi`` in ``local-first``, and ``search_for`` does not normalize
    the literal-string query against the rendered ligature.
    """
    pdf = _make_pdf()
    out = insert_cover_page(pdf, _default_context())
    out_doc = fitz.open(stream=out, filetype="pdf")
    try:
        cover = out_doc[0]
        # Title (em-dash is U+2014, must be preserved verbatim)
        assert cover.search_for(_COVER_PAGE_TITLE), (
            "cover page title with em-dash not found via search"
        )
        # Footer split around the ``fi`` ligature in ``first``.
        assert cover.search_for("Lex Cloak —"), "footer em-dash missing"
        assert cover.search_for("PDF redaction."), "footer body missing"
        assert cover.search_for("lexcloak.com"), "footer URL missing"
    finally:
        out_doc.close()


def test_insert_cover_page_renders_template_substitutions():
    """Date / N / P substitute into the body sentence."""
    pdf = _make_pdf()
    out = insert_cover_page(pdf, _default_context(date="2026-05-17", n=42, p=7))
    out_doc = fitz.open(stream=out, filetype="pdf")
    try:
        cover = out_doc[0]
        assert cover.search_for("2026-05-17"), "date not rendered"
        # N + P render as standalone integers
        assert cover.search_for("42 items"), "redacted count not rendered"
        assert cover.search_for("7 pages"), "page count not rendered"
    finally:
        out_doc.close()


def test_insert_cover_page_renders_body_anchor_phrase():
    """Body text includes the recipient-review-required clause verbatim."""
    pdf = _make_pdf()
    out = insert_cover_page(pdf, _default_context())
    out_doc = fitz.open(stream=out, filetype="pdf")
    try:
        cover = out_doc[0]
        # No ligatures in this phrase; safe for direct text-extraction check.
        text = cover.get_text()
        assert "downstream recipients should apply their own review" in text
        assert "before further distribution" in text
    finally:
        out_doc.close()


def test_insert_cover_page_matches_a4_page_size():
    """A4 input -> A4 cover. Otherwise the cover looks wrong in viewers."""
    a4_width, a4_height = 595.0, 842.0
    doc = fitz.open()
    doc.new_page(width=a4_width, height=a4_height)
    pdf = doc.tobytes()
    doc.close()

    out = insert_cover_page(pdf, _default_context(p=1))
    out_doc = fitz.open(stream=out, filetype="pdf")
    try:
        cover_rect = out_doc[0].rect
        assert float(cover_rect.width) == pytest.approx(a4_width)
        assert float(cover_rect.height) == pytest.approx(a4_height)
    finally:
        out_doc.close()


def test_insert_cover_page_large_redaction_count_renders_cleanly():
    """4-digit counts don't break the layout (boundary stress)."""
    pdf = _make_pdf()
    out = insert_cover_page(pdf, _default_context(n=9999, p=500))
    out_doc = fitz.open(stream=out, filetype="pdf")
    try:
        cover = out_doc[0]
        assert cover.search_for("9999 items")
        assert cover.search_for("500 pages")
    finally:
        out_doc.close()


def test_insert_cover_page_zero_redaction_count_still_renders():
    """N=0 ships verbatim per the no-pluralization rule."""
    pdf = _make_pdf()
    out = insert_cover_page(pdf, _default_context(n=0, p=1))
    out_doc = fitz.open(stream=out, filetype="pdf")
    try:
        assert out_doc[0].search_for("0 items")
    finally:
        out_doc.close()


def test_insert_cover_page_missing_required_key_raises():
    pdf = _make_pdf()
    with pytest.raises(ValueError, match=r"missing required context key"):
        insert_cover_page(pdf, {"date": "2026-05-17", "redacted_count": 1})


def test_insert_cover_page_unknown_key_raises():
    pdf = _make_pdf()
    with pytest.raises(ValueError, match=r"unknown context key"):
        insert_cover_page(pdf, {
            **_default_context(),
            "bogus_key": "x",
        })


def test_insert_cover_page_non_dict_context_raises():
    pdf = _make_pdf()
    with pytest.raises(ValueError, match=r"context must be a dict"):
        insert_cover_page(pdf, "not a dict")  # type: ignore[arg-type]


def test_insert_cover_page_product_version_optional():
    """product_version is accepted-but-not-required (forward-compat)."""
    pdf = _make_pdf()
    ctx = {"date": "2026-05-17", "redacted_count": 1, "page_count": 1}
    out = insert_cover_page(pdf, ctx)
    out_doc = fitz.open(stream=out, filetype="pdf")
    try:
        assert len(out_doc) == 2
    finally:
        out_doc.close()


def test_insert_cover_page_doc_form_mutates_in_place():
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    pre_len = len(doc)
    _insert_cover_page_doc(doc, _default_context(p=1))
    assert len(doc) == pre_len + 1
    doc.close()


# ── CharData serialize / deserialize ─────────────────────────────────


def test_serialize_deserialize_round_trip():
    """Round-trip preserves every char + bbox + line break marker."""
    chardata = [
        ("H", (1.0, 2.0, 3.0, 4.0)),
        ("i", (3.0, 2.0, 5.0, 4.0)),
        ("\n", None),
        ("X", (1.0, 5.0, 3.0, 7.0)),
    ]
    serialized = serialize_chardata(chardata)
    assert serialized == [
        ("H", 1.0, 2.0, 3.0, 4.0),
        ("i", 3.0, 2.0, 5.0, 4.0),
        ("\n", None, None, None, None),
        ("X", 1.0, 5.0, 3.0, 7.0),
    ]
    deserialized = deserialize_chardata(serialized)
    assert deserialized == chardata


def test_serialize_idempotent_on_flat_form():
    flat = [("a", 1.0, 2.0, 3.0, 4.0), ("b", None, None, None, None)]
    assert serialize_chardata(flat) == [tuple(e) for e in flat]


def test_deserialize_idempotent_on_tuple_form():
    tup = [("a", (1.0, 2.0, 3.0, 4.0)), ("b", None)]
    assert deserialize_chardata(tup) == tup


def test_serialize_empty_returns_empty():
    assert serialize_chardata([]) == []
    assert deserialize_chardata([]) == []


def test_search_in_chars_after_round_trip_matches_in_process():
    """search_in_chars must work on chardata that's been through
    serialize / json / deserialize round-trip -- the IPC use case."""
    chardata = [
        ("J", (10.0, 10.0, 15.0, 20.0)),
        ("o", (15.0, 10.0, 20.0, 20.0)),
        ("h", (20.0, 10.0, 25.0, 20.0)),
        ("n", (25.0, 10.0, 30.0, 20.0)),
    ]
    in_process_rects = search_in_chars("John", chardata)
    import json as _json
    serialized = serialize_chardata(chardata)
    via_json = _json.loads(_json.dumps(serialized))
    rehydrated = deserialize_chardata(via_json)
    round_trip_rects = search_in_chars("John", rehydrated)
    assert len(in_process_rects) == len(round_trip_rects)
    for a, b in zip(in_process_rects, round_trip_rects):
        assert a.x0 == b.x0 and a.y0 == b.y0
        assert a.x1 == b.x1 and a.y1 == b.y1


# ── search_whole_word_in_chars boundary semantics ────────────────────


def _ocr_line_chars(text: str, y0: float = 10.0, y1: float = 20.0,
                    char_w: float = 5.0, x_start: float = 10.0) -> list:
    """Build chardata mimicking Tesseract OCR output for a single line.

    Spaces in ``text`` are emitted as ``(' ', None)`` to match the
    line-break / inter-word marker convention used by the OCR pipeline.
    Non-space chars get sequential bboxes on a single baseline.
    """
    chars: list = []
    x = x_start
    for c in text:
        if c == " ":
            chars.append((" ", None))
        else:
            chars.append((c, (x, y0, x + char_w, y1)))
            x += char_w
    return chars


def test_whole_word_matches_when_preceded_by_punctuation_no_space_bbox():
    """Regression: a needle preceded by ``label:`` (no bbox-bearing space
    between) used to be rejected because bbox-proximity surrounding-char
    extraction concatenated adjacent chars without a separator, so the
    word-boundary regex saw the needle glued to neighbours. The fix
    operates on chardata-reconstructed text where None-bbox markers
    render as spaces, so the boundary check sees real word boundaries.

    Mirrors real-world OCR shape: "Name: Susan R." rendered as a
    sequence of letter bboxes interleaved with None-bbox space markers.
    """
    chars = _ocr_line_chars("Name: Susan R. Smith")
    rects = search_whole_word_in_chars("Susan", chars)
    assert len(rects) == 1


def test_whole_word_rejects_substring_inside_longer_word():
    """``Susan`` inside ``Susannah`` is a true substring, not a whole word."""
    chars = _ocr_line_chars("The Susannah file")
    rects = search_whole_word_in_chars("Susan", chars)
    assert rects == []


def test_whole_word_matches_at_text_start():
    """No char before idx 0 — boundary check must treat start-of-text
    as a word boundary."""
    chars = _ocr_line_chars("Susan called")
    rects = search_whole_word_in_chars("Susan", chars)
    assert len(rects) == 1


def test_whole_word_matches_at_text_end():
    """No char after idx + len(needle) — boundary check must treat
    end-of-text as a word boundary."""
    chars = _ocr_line_chars("called Susan")
    rects = search_whole_word_in_chars("Susan", chars)
    assert len(rects) == 1


def test_whole_word_matches_multiple_occurrences():
    """Same needle appearing twice in the same line should yield two rects."""
    chars = _ocr_line_chars("Susan met Susan today")
    rects = search_whole_word_in_chars("Susan", chars)
    assert len(rects) == 2


def test_whole_word_punctuation_boundary():
    """Adjacent punctuation (``.``, ``,``, ``:``) is not ``\\w`` — must
    not block a whole-word match."""
    chars = _ocr_line_chars("Susan,Bob;Carol.Dave")
    for needle in ("Susan", "Bob", "Carol", "Dave"):
        rects = search_whole_word_in_chars(needle, chars)
        assert len(rects) == 1, f"{needle!r} should match exactly once"


def test_whole_word_empty_inputs():
    """Defensive: empty needle and empty chars both return [] cleanly."""
    chars = _ocr_line_chars("Susan called")
    assert search_whole_word_in_chars("", chars) == []
    assert search_whole_word_in_chars("Susan", []) == []


# ── pymupdf_version probe ────────────────────────────────────────────


def test_pymupdf_version_returns_string():
    v = pymupdf_version()
    assert isinstance(v, str)
    assert any(c.isdigit() for c in v)
