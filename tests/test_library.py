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

import pymupdf
import pytest

from lexcloak_pdf_tool import (
    all_page_sizes,
    apply_redactions,
    encrypt,
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
    doc = pymupdf.open()
    for _ in range(n_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text(pymupdf.Point(x, y), text, fontsize=fontsize)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_encrypted_pdf(password: str) -> bytes:
    """Build a single-page password-protected PDF."""
    doc = pymupdf.open()
    doc.new_page(width=612, height=792).insert_text(
        pymupdf.Point(50, 100), "Confidential", fontsize=12)
    out = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
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
    doc = pymupdf.open()
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
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    pdf = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="", user_pw="")
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


# ── encrypt (Session 342) ────────────────────────────────────────────
# The encrypt-on-exit half of the decrypt/encrypt pipeline bracket. Mirrors
# reduce_size's cleartext-only invariant + apply_redactions's save-failure
# fallback. Shares the AES-256 save block with apply_redactions via
# redact._save_encrypted, so a params drift here would fail both paths.


def test_encrypt_valid_password_round_trips():
    """A valid password produces AES-256 output that authenticates with the
    right password and rejects a wrong one; content survives."""
    clear = _make_pdf("Patient SSN 123-45-6789")
    out, applied = encrypt(clear, "pw-123")
    assert applied is True
    assert out[:4] == b"%PDF"
    # Right password unlocks.
    d = pymupdf.open(stream=out, filetype="pdf")
    assert d.is_encrypted
    assert d.authenticate("pw-123") > 0
    assert "123-45-6789" in d[0].get_text()
    d.close()
    # Wrong password fails on a fresh open (authenticate is one-shot).
    d2 = pymupdf.open(stream=out, filetype="pdf")
    assert d2.authenticate("WRONG") == 0
    d2.close()


def test_encrypt_empty_password_is_noop_returns_input_unchanged():
    """Empty password is a no-op: the exact input bytes come back, byte-for-
    byte, with protection_applied=False (never a re-saved copy)."""
    clear = _make_pdf("Unprotected")
    out, applied = encrypt(clear, "")
    assert applied is False
    assert out is clear or out == clear


def test_encrypt_non_ascii_password_round_trips():
    """A non-ASCII (Unicode) password authenticates correctly."""
    clear = _make_pdf("Dossier")
    pw = "clé-secrète-Ünïcödé-🔒"
    out, applied = encrypt(clear, pw)
    assert applied is True
    d = pymupdf.open(stream=out, filetype="pdf")
    assert d.authenticate(pw) > 0
    d.close()


def test_encrypt_large_multipage_pdf_round_trips():
    """A large multi-page document encrypts + authenticates without loss."""
    clear = _make_pdf("Page body text", n_pages=50)
    out, applied = encrypt(clear, "big-doc-pw")
    assert applied is True
    d = pymupdf.open(stream=out, filetype="pdf")
    assert d.authenticate("big-doc-pw") > 0
    assert d.page_count == 50
    d.close()


def test_encrypt_save_failure_falls_back_to_unprotected(monkeypatch):
    """A PyMuPDF failure on the *encrypted* save degrades to unprotected
    output (protection_applied=False) rather than raising -- a failed
    encryption must never block the download. The fallback clean-save
    still succeeds, so content survives."""
    clear = _make_pdf("Fallback body")
    real_save = pymupdf.Document.save

    def flaky_save(self, *args, **kwargs):
        # Only the encrypted save (carries an ``encryption`` kwarg) fails;
        # the unprotected fallback save passes through.
        if kwargs.get("encryption"):
            raise RuntimeError("simulated encrypted-save failure")
        return real_save(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Document, "save", flaky_save)
    out, applied = encrypt(clear, "pw-that-cannot-apply")
    assert applied is False
    d = pymupdf.open(stream=out, filetype="pdf")
    assert not (d.is_encrypted and d.needs_pass)  # unprotected fallback
    assert "Fallback body" in d[0].get_text()
    d.close()


def test_encrypt_rejects_already_encrypted_input():
    """Encrypted input is a programming error -- the pipeline only ever feeds
    cleartext into encrypt. Raises ValueError naming the cleartext invariant."""
    with pytest.raises(ValueError, match="cleartext"):
        encrypt(_make_encrypted_pdf("secret123"), "new-pw")


def test_encrypt_non_string_password_raises():
    """Password must be a string; a non-string raises a named ValueError."""
    with pytest.raises(ValueError, match="password"):
        encrypt(_make_pdf(), 12345)


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
    doc = pymupdf.open()
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
    out_doc = pymupdf.open(stream=out, filetype="pdf")
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


def test_apply_redactions_non_string_per_match_label_raises():
    """A non-string ``redact_label`` is caught at the wire boundary rather
    than failing opaquely inside ``add_redact_annot``'s ``text=``."""
    pdf = _make_pdf()
    matches = [{
        "id": "bad", "type": "SSN", "page": 0,
        "rect": {"x0": 10, "y0": 10, "x1": 50, "y1": 30},
        "enabled": True, "redact_label": 123,
    }]
    with pytest.raises(ValueError, match="redact_label"):
        apply_redactions(pdf, matches)


def test_apply_redactions_tolerates_unknown_per_match_keys():
    """LOAD-BEARING CROSS-VERSION CONTRACT -- do not "tighten" this into
    unknown-key rejection without a protocol bump.

    The closed app pairs a v0.6.4-aware build with whatever bundled CLI was
    frozen into it, which may be OLDER. Because this validator ignores keys
    it does not know, a v0.6.3 CLI handed a per-match ``redact_label`` drops
    it and stamps the document-level label -- a graceful degradation, not a
    hard failure. Verified out-of-band against real v0.6.3 code at the time
    this test was written. Adding strict key rejection here (as
    ``cover_page`` and ``metadata`` do for THEIR payloads) would turn every
    future additive per-match field into a breaking change for already-frozen
    builds.
    """
    pdf = _make_pdf("Patient SSN 123-45-6789")
    matches = [{
        "id": "x", "type": "SSN", "page": 0,
        "rect": {"x0": 30, "y0": 80, "x1": 300, "y1": 120},
        "enabled": True, "text": "123-45-6789",
        "some_future_field": {"nested": True},
    }]
    out, _ = apply_redactions(pdf, matches, redact_label="REDACTED")
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    try:
        text = out_doc[0].get_text()
    finally:
        out_doc.close()
    assert "123-45-6789" not in text
    assert "REDACTED" in text


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
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = dpi / 72.0
        pix = doc[page].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
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
    doc = pymupdf.open()
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
    out_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
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


# ── apply_redactions: blackout_pages (Session 592 triage redesign) ─────
#
# A "dropped" page can have one of two outcomes: removed_pages deletes it
# from the output; blackout_pages keeps it but covers it edge-to-edge in
# solid black with the underlying content SCRUBBED. The load-bearing
# property is that a blackout never depends on per-match completeness — a
# dropped page can never ship partially redacted or readable. These pin
# that contract directly on the library function.


def test_apply_redactions_blackout_scrubs_page_text():
    """A blacked-out page's text is removed from the output, not just covered."""
    pdf = _make_pdf("Patient SSN 123-45-6789")
    out, _ = apply_redactions(pdf, [], blackout_pages=[0])
    assert "123-45-6789" not in _read_page_text(out, 0)
    assert _read_page_text(out, 0).strip() == ""


def test_apply_redactions_blackout_no_per_match_dependency():
    """Blackout fully blacks the page with NO matches supplied — the page's
    safety does not ride on per-match completeness."""
    pdf = _make_pdf("Patient SSN 123-45-6789")
    out, _ = apply_redactions(pdf, [], blackout_pages=[0])
    # Solid black edge-to-edge: text region, center, and a far corner are all
    # dark — the cover is the whole page, not just where content happened to be.
    assert _is_dark_at(out, 0, 50, 100)   # over the (now scrubbed) text
    assert _is_dark_at(out, 0, 306, 396)  # page center
    assert _is_dark_at(out, 0, 560, 740)  # far corner, no content there


def test_apply_redactions_blackout_keeps_page_in_output():
    """Blackout keeps the page (unlike remove, which deletes it)."""
    pdf = _make_pdf(n_pages=3)
    out, _ = apply_redactions(pdf, [], blackout_pages=[1])
    assert page_count(out) == 3


def test_apply_redactions_blackout_leaves_other_pages_untouched():
    """Only the blackout page is covered; siblings keep their content + stay clear."""
    pdf = _make_pdf("Patient SSN 123-45-6789", n_pages=3)
    out, _ = apply_redactions(pdf, [], blackout_pages=[1])
    assert "123-45-6789" in _read_page_text(out, 0)
    assert "123-45-6789" in _read_page_text(out, 2)
    assert not _is_dark_at(out, 0, 560, 740)  # sibling far corner stays clear
    assert _read_page_text(out, 1).strip() == ""


def test_apply_redactions_remove_precedence_over_blackout():
    """A page listed in BOTH removed_pages and blackout_pages is deleted —
    remove wins, blackout on it is a no-op (the page is gone)."""
    pdf = _make_pdf(n_pages=3)
    out, _ = apply_redactions(pdf, [], removed_pages=[1], blackout_pages=[1])
    assert page_count(out) == 2


def test_apply_redactions_blackout_and_remove_combo():
    """Blackout one page + remove another in the same call: removed page gone,
    blackout page stays-but-black, untouched page intact."""
    pdf = _make_pdf("Patient SSN 123-45-6789", n_pages=3)
    out, _ = apply_redactions(pdf, [], removed_pages=[2], blackout_pages=[0])
    assert page_count(out) == 2          # page 2 deleted
    assert _read_page_text(out, 0).strip() == ""   # page 0 blacked + scrubbed
    assert _is_dark_at(out, 0, 306, 396)
    assert "123-45-6789" in _read_page_text(out, 1)  # page 1 untouched


def test_apply_redactions_blackout_all_pages_allowed():
    """Blacking out EVERY page is valid (the doc stays, all black) — unlike
    removing every page, which raises. A fully-redacted doc is a real output."""
    pdf = _make_pdf("Patient SSN 123-45-6789", n_pages=2)
    out, _ = apply_redactions(pdf, [], blackout_pages=[0, 1])
    assert page_count(out) == 2
    assert _read_page_text(out, 0).strip() == ""
    assert _read_page_text(out, 1).strip() == ""
    assert _is_dark_at(out, 0, 306, 396)
    assert _is_dark_at(out, 1, 306, 396)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_apply_redactions_blackout_rotated_page(rotation):
    """Blackout covers the whole page for every /Rotate value — center and all
    four corners render black in as-displayed space, and text is scrubbed."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(pymupdf.Point(50, 100), "Patient SSN 123-45-6789", fontsize=12)
    if rotation:
        page.set_rotation(rotation)
    pdf = doc.tobytes()
    doc.close()
    out, _ = apply_redactions(pdf, [], blackout_pages=[0])
    assert _read_page_text(out, 0).strip() == "", (
        f"/Rotate {rotation}: blackout left extractable text")
    # As-displayed dimensions swap at 90/270; sample within whichever frame.
    w, h = (792, 612) if rotation in (90, 270) else (612, 792)
    for x, y in ((w * 0.5, h * 0.5), (15, 15), (w - 15, 15),
                 (15, h - 15), (w - 15, h - 15)):
        assert _is_dark_at(out, 0, x, y), (
            f"/Rotate {rotation}: blackout missed ({x:.0f},{y:.0f})")


def test_apply_redactions_blackout_non_integer_raises():
    """Malformed blackout_pages (non-int) raises a named ValueError."""
    pdf = _make_pdf()
    with pytest.raises(ValueError, match="blackout_pages.*non-integer"):
        apply_redactions(pdf, [], blackout_pages=["nope"])


def test_apply_redactions_blackout_out_of_range_ignored():
    """An out-of-range blackout index is silently ignored, not a crash —
    a stale frontend index can't abort the export."""
    pdf = _make_pdf("Patient SSN 123-45-6789", n_pages=1)
    out, _ = apply_redactions(pdf, [], blackout_pages=[99])
    assert page_count(out) == 1
    assert "123-45-6789" in _read_page_text(out, 0)  # the real page untouched


# ── strip_metadata ───────────────────────────────────────────────────


def test_strip_metadata_bytes_form_strips_author():
    doc = pymupdf.open()
    doc.set_metadata({"author": "Dr. Jane Doe", "title": "Confidential"})
    doc.new_page(width=612, height=792)
    pdf = doc.tobytes()
    doc.close()

    out = strip_metadata(pdf)
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert (meta.get("author") or "") == ""
    assert (meta.get("title") or "") == ""


def test_strip_metadata_doc_form_returns_none_and_mutates():
    """Live ``Document`` form preserves the legacy in-place semantics."""
    doc = pymupdf.open()
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

    out_doc = pymupdf.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert meta["subject"] == fields["subject"]
    assert meta["producer"] == fields["producer"]
    assert meta["keywords"] == fields["keywords"]


def test_set_metadata_preserves_existing_fields_not_in_payload():
    """Merge semantics: fields not in ``payload`` survive untouched."""
    doc = pymupdf.open()
    doc.set_metadata({"author": "Original Author", "title": "Original Title"})
    doc.new_page()
    pdf = doc.tobytes()
    doc.close()

    out = set_metadata(pdf, {"subject": "added subject"})
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert meta["author"] == "Original Author"
    assert meta["title"] == "Original Title"
    assert meta["subject"] == "added subject"


def test_set_metadata_empty_dict_is_noop():
    """Empty fields round-trips the PDF without metadata changes."""
    doc = pymupdf.open()
    doc.set_metadata({"author": "A", "title": "T"})
    doc.new_page()
    pdf = doc.tobytes()
    doc.close()

    out = set_metadata(pdf, {})
    out_doc = pymupdf.open(stream=out, filetype="pdf")
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
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert meta["keywords"] == "auto-redactée, revue-requise"


def test_set_metadata_long_value_round_trip():
    """Producer-style strings well past typical lengths round-trip cleanly."""
    pdf = _make_pdf()
    long_value = "Lex Cloak " + ("x" * 4096)
    out = set_metadata(pdf, {"producer": long_value})
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert meta["producer"] == long_value


def test_set_metadata_doc_form_mutates_in_place():
    """``_set_metadata_doc`` mutates the live Document, returns None."""
    doc = pymupdf.open()
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
    doc = pymupdf.open()
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
    doc = pymupdf.open()
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
    doc = pymupdf.open(stream=intermediate, filetype="pdf")
    assert (doc.metadata or {})["subject"].startswith("Auto-redacted")
    doc.close()
    # Then strip should wipe them
    stripped = strip_metadata(intermediate)
    doc = pymupdf.open(stream=stripped, filetype="pdf")
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
    doc = pymupdf.open()
    doc.new_page(width=612, height=792).insert_text(
        pymupdf.Point(50, 100), "ORIGINAL_PAGE_ONE_MARKER", fontsize=12)
    doc.new_page(width=612, height=792).insert_text(
        pymupdf.Point(50, 100), "ORIGINAL_PAGE_TWO_MARKER", fontsize=12)
    pdf = doc.tobytes()
    doc.close()

    out = insert_cover_page(pdf, _default_context(p=2))
    out_doc = pymupdf.open(stream=out, filetype="pdf")
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
    out_doc = pymupdf.open(stream=out, filetype="pdf")
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
    out_doc = pymupdf.open(stream=out, filetype="pdf")
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
    out_doc = pymupdf.open(stream=out, filetype="pdf")
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
    doc = pymupdf.open()
    doc.new_page(width=a4_width, height=a4_height)
    pdf = doc.tobytes()
    doc.close()

    out = insert_cover_page(pdf, _default_context(p=1))
    out_doc = pymupdf.open(stream=out, filetype="pdf")
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
    out_doc = pymupdf.open(stream=out, filetype="pdf")
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
    out_doc = pymupdf.open(stream=out, filetype="pdf")
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
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    try:
        assert len(out_doc) == 2
    finally:
        out_doc.close()


def test_insert_cover_page_doc_form_mutates_in_place():
    doc = pymupdf.open()
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


# ── numeric token boundary (opt-in, v0.6.5) ──────────────────────────
#
# `.`, `-` and `/` are word boundaries in prose but intra-number separators
# inside a number, and one \w-class rule served both -- so a bare "12"
# matched INSIDE the statute citation "18-12-107.5" (diagnosed on a real
# Colorado court document, where the consumer then amplified that 2-char
# rect to the whole enclosing token and boxed text no detector had matched).
#
# The rule is OPT-IN via numeric_token_boundary=True. The tests below pass
# it explicitly; the DEFAULT-behavior tests in the next block pin that
# callers who don't ask get exactly the pre-v0.6.4 semantics. Both halves
# matter: the default is what user-typed needles rely on.


def test_numeric_needle_rejected_inside_longer_number():
    """The headline case: `12` is a fragment of `18-12-107.5`, not a token."""
    chars = _ocr_line_chars("See 18-12-107.5 for details")
    assert search_whole_word_in_chars("12", chars, numeric_token_boundary=True) == []


def test_numeric_needle_rejected_inside_slash_bounded_number():
    """Same rule for `/` -- `15` inside the date `07/15/1973`."""
    chars = _ocr_line_chars("DOB 07/15/1973 on file")
    assert search_whole_word_in_chars("15", chars, numeric_token_boundary=True) == []


def test_numeric_needle_rejected_inside_dot_bounded_number():
    """Same rule for `.` -- `5` inside the version-like `107.5.2`."""
    chars = _ocr_line_chars("Section 107.5.2 applies")
    assert search_whole_word_in_chars("5", chars, numeric_token_boundary=True) == []


def test_numeric_needle_still_matches_as_its_own_token():
    """Guard against over-tightening: a standalone number still matches."""
    chars = _ocr_line_chars("Age 12 years")
    assert len(search_whole_word_in_chars("12", chars, numeric_token_boundary=True)) == 1


def test_numeric_needle_matches_before_sentence_final_period():
    """A trailing separator with NO digit after it does not bind, so a
    sentence-final number still matches ("Age: 12." -> `12` is a token)."""
    chars = _ocr_line_chars("Patient age is 12. Next line")
    assert len(search_whole_word_in_chars("12", chars, numeric_token_boundary=True)) == 1


def test_numeric_needle_matches_after_leading_hyphen():
    """A leading separator with no digit BEHIND it does not bind either --
    `12` in `-12` is still its own token (the `-` reads as a minus sign or
    a dash, not an intra-number separator)."""
    chars = _ocr_line_chars("Delta -12 units")
    assert len(search_whole_word_in_chars("12", chars, numeric_token_boundary=True)) == 1


def test_full_numeric_needle_matches_the_whole_citation():
    """Searching for the WHOLE number still places -- the rule rejects
    fragments, not the number itself."""
    chars = _ocr_line_chars("See 18-12-107.5 for details")
    assert len(search_whole_word_in_chars("18-12-107.5", chars, numeric_token_boundary=True)) == 1


def test_alpha_needle_still_matches_inside_hyphenated_name():
    """UNCHANGED BY DESIGN, and the reason the rule is numeric-only: a
    hyphenated surname is a real occurrence of the name, so `Smith` must
    still match inside `Smith-Jones`. If this ever flips, the rule has
    leaked out of its numeric scope."""
    chars = _ocr_line_chars("Contact Smith-Jones today")
    assert len(search_whole_word_in_chars("Smith", chars, numeric_token_boundary=True)) == 1


def test_alpha_needle_still_matches_around_slash():
    """Same guard for `/` on an alpha needle -- `and/or` splits into tokens."""
    chars = _ocr_line_chars("terms and/or conditions")
    assert len(search_whole_word_in_chars("and", chars, numeric_token_boundary=True)) == 1


def test_mixed_alphanumeric_needle_keeps_the_looser_rule():
    """A needle that is not bare-numeric (`A12`) is not numeric-shaped, so
    the separator clause does not apply to it."""
    chars = _ocr_line_chars("Case A12-99 filed")
    assert len(search_whole_word_in_chars("A12", chars, numeric_token_boundary=True)) == 1


def test_numeric_boundary_rejects_only_the_glued_occurrence():
    """A needle appearing BOTH glued and standalone yields exactly the
    standalone rect -- the rule filters per-occurrence, not per-needle."""
    chars = _ocr_line_chars("Under 18-12-107.5 the age is 12 exactly")
    assert len(search_whole_word_in_chars("12", chars, numeric_token_boundary=True)) == 1


# ── numeric boundary: the DEFAULT is the historical looser rule ───────
#
# LOAD-BEARING. Whether a numeric fragment is noise depends on where the
# needle came from, and only the caller knows that. A detector-inferred
# needle is worth tightening; a human-typed one is not -- someone who asks
# to redact "12" may mean every 12, and in a redaction tool an unwanted box
# costs one click while a missed one leaks. v0.6.4 briefly made the
# tightened rule unconditional and silently changed both kinds of caller.
# These tests are what stop that from happening again.


@pytest.mark.parametrize("needle,text", [
    ("12", "See 18-12-107.5 for details"),
    ("15", "DOB 07/15/1973 on file"),
    ("5", "Section 107.5.2 applies"),
    ("4021", "Acct 12-4021-99 ref"),
])
def test_numeric_needle_still_matches_when_glued_by_default(needle, text):
    """Without the flag, a glued numeric fragment STILL matches -- exactly
    the pre-v0.6.4 behavior. A caller relying on this (user-typed search
    terms) must keep getting it."""
    chars = _ocr_line_chars(text)
    assert len(search_whole_word_in_chars(needle, chars)) == 1


def test_default_and_opt_in_agree_on_every_non_glued_case():
    """The flag changes ONLY glued numeric fragments. Everything else --
    alpha, mixed, standalone, whole-number, sentence-final -- must return
    identical results with and without it."""
    cases = [
        ("12", "Age 12 years"),
        ("12", "Patient age is 12. Next line"),
        ("12", "Delta -12 units"),
        ("18-12-107.5", "See 18-12-107.5 for details"),
        ("Smith", "Contact Smith-Jones today"),
        ("and", "terms and/or conditions"),
        ("A12", "Case A12-99 filed"),
    ]
    for needle, text in cases:
        chars = _ocr_line_chars(text)
        loose = search_whole_word_in_chars(needle, chars)
        tight = search_whole_word_in_chars(needle, chars,
                                           numeric_token_boundary=True)
        assert len(loose) == len(tight), (
            f"{needle!r} in {text!r}: flag changed a case it must not "
            f"({len(loose)} -> {len(tight)})")


def test_numeric_token_boundary_is_keyword_only():
    """Positional callers can't trip the flag by accident -- the third
    positional arg was never a parameter, so this must raise."""
    chars = _ocr_line_chars("See 18-12-107.5 for details")
    with pytest.raises(TypeError):
        search_whole_word_in_chars("12", chars, True)


# ── pymupdf_version probe ────────────────────────────────────────────


def test_pymupdf_version_returns_string():
    v = pymupdf_version()
    assert isinstance(v, str)
    assert any(c.isdigit() for c in v)
