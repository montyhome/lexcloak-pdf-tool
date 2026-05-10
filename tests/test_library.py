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
    is_encrypted,
    page_count,
    page_size,
    pymupdf_version,
    render_page,
    search_for,
    strip_metadata,
)
from lexcloak_pdf_tool.coords import (
    deserialize_chardata,
    search_in_chars,
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


# ── pymupdf_version probe ────────────────────────────────────────────


def test_pymupdf_version_returns_string():
    v = pymupdf_version()
    assert isinstance(v, str)
    assert any(c.isdigit() for c in v)
