"""Tests for the ``reduce_size`` op (library + wire dispatch).

The 6-criteria framework:

* **Happy path** -- lossless default shrinks-or-equals and preserves text +
  the invisible OCR layer; a DPI preset downsamples an image-heavy PDF;
  grayscale path; truthful ``info`` dict.
* **Boundary / type** -- single + multi page, absurdly-high DPI (no image
  exceeds it -> lossless), quality at 1 and 100, image just below vs above
  the threshold.
* **Sad paths** (>=2 per happy) -- encrypted input, dpi 0 / negative /
  non-int / bool, quality out of range / non-int all raise ``ValueError``.
* **No logic mirroring** -- expectations are hard-coded shapes + measured
  sizes, never recomputed through the same call.
* **Side-effect verification** -- output re-opens + re-scans clean; page
  count + text + OCR layer survive; the no-grow guard returns the input
  *bytes-identical*.
* **Mock integrity** -- ``rewrite_images`` raising falls back to lossless;
  ``subset_fonts`` raising is non-fatal; ``save`` raising propagates;
  a save that grows the file trips the no-grow guard.

Synthetic fixtures only (per ``feedback_synthetic_fixtures_only``).
"""
from __future__ import annotations

import base64
import os

import fitz
import pytest

from lexcloak_pdf_tool import reduce_size
from lexcloak_pdf_tool.reduce_size import _validate_reduce_params


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_text_pdf(text: str = "Patient SSN 123-45-6789",
                   n_pages: int = 1, ocr_token: str | None = None) -> bytes:
    """Born-digital text PDF; optional invisible (render-mode-3) OCR layer."""
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text(fitz.Point(50, 100), text, fontsize=12)
        if ocr_token is not None:
            page.insert_text(fitz.Point(50, 140), ocr_token,
                             fontsize=12, render_mode=3)
    out = doc.tobytes()
    doc.close()
    return out


def _make_image_pdf(px: int = 1500, rect_pt: float = 200.0) -> bytes:
    """High-DPI raster: ``px``-square random RGB image in a small rect.

    px=1500 in a 200pt (≈2.78in) box is ≈540 DPI -- well above any preset,
    so ``rewrite_images`` always has something to downsample.
    """
    pm = fitz.Pixmap(fitz.csRGB, px, px, os.urandom(px * px * 3), False)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(50, 50, 50 + rect_pt, 50 + rect_pt), pixmap=pm)
    out = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return out


def _make_encrypted_pdf(password: str = "pw") -> bytes:
    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text(
        fitz.Point(50, 100), "Confidential", fontsize=12)
    out = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256,
                      user_pw=password, owner_pw=password)
    doc.close()
    return out


def _open_clean(pdf_bytes: bytes) -> fitz.Document:
    """Open output bytes, asserting they parse as a valid PDF."""
    return fitz.open(stream=pdf_bytes, filetype="pdf")


# ── Happy path ───────────────────────────────────────────────────────


def test_lossless_default_info_shape_and_no_grow():
    src = _make_text_pdf()
    out, info = reduce_size(src)
    assert set(info) == {"orig_size", "new_size", "applied_dpi"}
    assert info["orig_size"] == len(src)
    assert info["new_size"] == len(out)
    assert info["applied_dpi"] is None          # lossless -> no dpi
    assert len(out) <= len(src)                  # never larger


def test_lossless_preserves_text_and_ocr_layer():
    src = _make_text_pdf(text="Patient SSN 123-45-6789",
                         ocr_token="OCR-INVISIBLE-LAYER-42")
    out, _ = reduce_size(src)
    doc = _open_clean(out)
    try:
        text = doc.load_page(0).get_text()
    finally:
        doc.close()
    # Visible text AND the invisible OCR/selectable layer both survive
    # (hidden_text=False in the scrub flag-set is load-bearing here).
    assert "123-45-6789" in text
    assert "OCR-INVISIBLE-LAYER-42" in text


def test_dpi_downsample_shrinks_image_pdf():
    src = _make_image_pdf()
    out, info = reduce_size(src, dpi=150, quality=50)
    assert info["applied_dpi"] == 150
    assert info["new_size"] < info["orig_size"]   # real reduction
    assert info["new_size"] == len(out)
    _open_clean(out).close()                      # still a valid PDF


def test_grayscale_downsample_shrinks_and_opens():
    src = _make_image_pdf()
    out, info = reduce_size(src, dpi=150, quality=60, grayscale=True)
    assert info["applied_dpi"] == 150
    assert info["new_size"] < info["orig_size"]
    _open_clean(out).close()


def test_lower_dpi_yields_smaller_output_than_higher_dpi():
    src = _make_image_pdf()
    _, info_hi = reduce_size(src, dpi=200, quality=75)
    _, info_lo = reduce_size(src, dpi=72, quality=75)
    # Monotonic: a more aggressive (lower) DPI target produces fewer bytes.
    assert info_lo["new_size"] < info_hi["new_size"] < info_hi["orig_size"]


# ── Boundary / type stress ───────────────────────────────────────────


def test_multipage_page_count_preserved():
    src = _make_text_pdf(n_pages=5)
    out, _ = reduce_size(src)
    doc = _open_clean(out)
    try:
        assert doc.page_count == 5
    finally:
        doc.close()


def test_absurdly_high_dpi_is_effectively_lossless():
    # No image exceeds 100000 DPI, so nothing is downsampled. rewrite_images
    # still runs without error; output stays valid and not larger.
    src = _make_image_pdf()
    out, info = reduce_size(src, dpi=100000, quality=75)
    assert info["applied_dpi"] == 100000
    assert info["new_size"] <= info["orig_size"]
    _open_clean(out).close()


def test_zero_image_text_pdf_with_dpi_is_safe_noop():
    # DPI requested on a born-digital text PDF: no images to rewrite, must
    # not raise, must not grow, text preserved.
    src = _make_text_pdf(text="No images here 555-0100")
    out, info = reduce_size(src, dpi=150)
    assert info["new_size"] <= info["orig_size"]
    doc = _open_clean(out)
    try:
        assert "555-0100" in doc.load_page(0).get_text()
    finally:
        doc.close()


@pytest.mark.parametrize("quality", [1, 100])
def test_quality_boundary_values_accepted(quality):
    src = _make_image_pdf()
    out, info = reduce_size(src, dpi=150, quality=quality)
    assert info["applied_dpi"] == 150
    _open_clean(out).close()


# ── preserve_metadata: the Spec-13 stamp across the scrub (v0.6.6) ───
#
# The lossless scrub sets metadata=True, which wipes the whole dict --
# including a marking the CALLER applied after redacting. Lex Cloak's
# Spec-13 "Auto-redacted" notice lives in `subject` and was erased by every
# successful compression before this flag existed (B3 M1, confirmed
# empirically 2026-07-30 and re-confirmed against v0.6.5 on 2026-08-02).

SPEC_13_SUBJECT = "Auto-redacted by Lex Cloak. Review before distribution."


def _make_stamped_pdf(**meta) -> bytes:
    """Text PDF carrying caller-applied metadata, sized so the no-grow guard
    does not short-circuit the scrub (a returned original would preserve the
    stamp trivially and prove nothing)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(40):
        page.insert_text(fitz.Point(50, 80 + i * 16),
                         "Reference 553-01-8842 filed 2026-03-04.", fontsize=11)
    doc.set_metadata(meta)
    out = doc.tobytes(garbage=0, deflate=False)
    doc.close()
    return out


def _meta(pdf_bytes: bytes) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return {k: v for k, v in doc.metadata.items() if v}
    finally:
        doc.close()


def test_default_still_strips_all_metadata():
    """Historical behavior is the default -- preserve_metadata is opt-in."""
    src = _make_stamped_pdf(subject=SPEC_13_SUBJECT, producer="SomeTool 9")
    out, _ = reduce_size(src)
    assert out != src                       # scrub actually ran
    assert "subject" not in _meta(out)
    assert "producer" not in _meta(out)


def test_preserve_subject_keeps_the_spec_13_stamp():
    src = _make_stamped_pdf(subject=SPEC_13_SUBJECT)
    out, _ = reduce_size(src, preserve_metadata=("subject",))
    assert out != src
    assert _meta(out)["subject"] == SPEC_13_SUBJECT


def test_preserve_subject_does_not_leak_other_metadata():
    """Preserving one key must not resurrect the rest -- the narrow request
    is honoured narrowly, so an unnamed key stays stripped even though it
    IS a legal key to name."""
    src = _make_stamped_pdf(
        subject=SPEC_13_SUBJECT, producer="SomeTool 9",
        creator="Scanner X", title="Case notes", author="A. Person",
    )
    out, _ = reduce_size(src, preserve_metadata=("subject",))
    surviving = _meta(out)
    assert surviving["subject"] == SPEC_13_SUBJECT
    for gone in ("producer", "creator", "title", "author"):
        assert gone not in surviving
    assert b"Scanner X" not in out
    assert b"Case notes" not in out


def test_preserve_all_three_spec_13_fields():
    """The Spec-13 marking is subject + producer + keywords, all three a
    stated launch invariant -- the scrub wipes all three, so all three must
    be recoverable in one call."""
    src = _make_stamped_pdf(
        subject=SPEC_13_SUBJECT, producer="Lex Cloak 1.8.19",
        keywords="redacted, auto-redacted", creator="Scanner X",
    )
    out, _ = reduce_size(
        src, preserve_metadata=("subject", "producer", "keywords"))
    surviving = _meta(out)
    assert surviving["subject"] == SPEC_13_SUBJECT
    assert surviving["producer"] == "Lex Cloak 1.8.19"
    assert surviving["keywords"] == "redacted, auto-redacted"
    # The key NOT named is still stripped -- opting in one does not opt in all.
    assert "creator" not in surviving


def test_preserve_metadata_accepts_multiple_keys():
    src = _make_stamped_pdf(subject=SPEC_13_SUBJECT, title="Exhibit B",
                            producer="SomeTool 9")
    out, _ = reduce_size(src, preserve_metadata=["subject", "title"])
    surviving = _meta(out)
    assert surviving["subject"] == SPEC_13_SUBJECT
    assert surviving["title"] == "Exhibit B"
    assert "producer" not in surviving


def test_preserve_metadata_of_absent_key_is_not_an_error():
    src = _make_stamped_pdf(subject=SPEC_13_SUBJECT)
    out, _ = reduce_size(src, preserve_metadata=("title",))
    assert _meta(out).get("title") is None
    assert "subject" not in _meta(out)


def test_preserve_metadata_survives_the_dpi_downsample_path():
    """The stamp must outlive the lossy arm too, not just the lossless one."""
    pm = fitz.Pixmap(fitz.csRGB, 1500, 1500, os.urandom(1500 * 1500 * 3), False)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(50, 50, 250, 250), pixmap=pm)
    doc.set_metadata({"subject": SPEC_13_SUBJECT, "producer": "SomeTool 9"})
    src = doc.tobytes(garbage=4, deflate=True)
    doc.close()

    out, info = reduce_size(src, dpi=72, quality=40,
                            preserve_metadata=("subject",))
    assert info["applied_dpi"] == 72          # the lossy arm really ran
    assert _meta(out)["subject"] == SPEC_13_SUBJECT
    assert "producer" not in _meta(out)


@pytest.mark.parametrize("key", ["producer", "creator", "creationDate",
                                 "modDate", "title", "author", "keywords"])
def test_every_writable_metadata_key_is_nameable(key):
    """The check is known-key, not a safety allowlist: whether a key is safe
    to keep is the caller's call, so every writable key must be nameable."""
    src = _make_stamped_pdf(subject=SPEC_13_SUBJECT, **{key: "sentinel-value"})
    out, _ = reduce_size(src, preserve_metadata=(key,))
    assert _meta(out).get(key) == "sentinel-value"


@pytest.mark.parametrize("bad", [("nope",), ("Subject",), ("subj",), ("",),
                                 ("format",), ("encryption",)])
def test_preserve_metadata_refuses_unknown_keys(bad):
    """Typo rejection is the point: a misspelled key is otherwise a silent
    no-op, and a silently-dropped marking is the failure this prevents.
    `format`/`encryption` are derived, not settable, so they are unknown."""
    with pytest.raises(ValueError, match="unknown metadata key"):
        reduce_size(_make_stamped_pdf(subject=SPEC_13_SUBJECT),
                    preserve_metadata=bad)


@pytest.mark.parametrize("bad", ["subject", 42, {"subject": 1}, object()])
def test_preserve_metadata_refuses_non_sequence(bad):
    """A bare str must be rejected, not silently iterated into characters."""
    with pytest.raises(ValueError, match="list or tuple"):
        reduce_size(_make_stamped_pdf(subject=SPEC_13_SUBJECT),
                    preserve_metadata=bad)


def test_preserve_metadata_empty_tuple_is_historical_behavior():
    src = _make_stamped_pdf(subject=SPEC_13_SUBJECT)
    out, _ = reduce_size(src, preserve_metadata=())
    assert "subject" not in _meta(out)


# ── Thumbnails: scrub(thumbnails=True) is a no-op on this flag-set ───


def test_thumbnails_are_actually_stripped():
    """The module docstring has always claimed this op scrubs thumbnails.
    It did not: scrub's page loop early-continues on
    ``not (clean_pages or hidden_text)`` before its thumbnail branch, and
    this op passes both False deliberately. Measured against pymupdf 1.27.2
    on 2026-08-02; ``null_page_thumbnails`` closes it."""
    doc = fitz.open(stream=_make_stamped_pdf(), filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(dpi=12)
    tx = doc.get_new_xref()
    doc.update_object(tx, "<<>>")
    doc.update_stream(tx, pix.tobytes("png"))
    doc.xref_set_key(page.xref, "Thumb", "%d 0 R" % tx)
    src = doc.tobytes(garbage=0, deflate=True)
    doc.close()

    pre = fitz.open(stream=src, filetype="pdf")
    try:
        assert pre.xref_get_key(pre[0].xref, "Thumb") != ("null", "null")
    finally:
        pre.close()

    out, _ = reduce_size(src)
    post = fitz.open(stream=out, filetype="pdf")
    try:
        assert post.xref_get_key(post[0].xref, "Thumb") == ("null", "null")
    finally:
        post.close()


# ── Sad paths ────────────────────────────────────────────────────────


def test_encrypted_input_raises():
    with pytest.raises(ValueError, match="cleartext"):
        reduce_size(_make_encrypted_pdf())


@pytest.mark.parametrize("dpi", [0, -1, -300])
def test_non_positive_dpi_raises(dpi):
    with pytest.raises(ValueError, match="positive"):
        reduce_size(_make_text_pdf(), dpi=dpi)


@pytest.mark.parametrize("dpi", ["150", 150.0, True])
def test_non_int_dpi_raises(dpi):
    with pytest.raises(ValueError, match="dpi"):
        reduce_size(_make_text_pdf(), dpi=dpi)


@pytest.mark.parametrize("quality", [0, 101, -5])
def test_quality_out_of_range_raises(quality):
    with pytest.raises(ValueError, match="1..100"):
        reduce_size(_make_text_pdf(), quality=quality)


@pytest.mark.parametrize("quality", ["75", 75.0, False])
def test_non_int_quality_raises(quality):
    with pytest.raises(ValueError, match="quality"):
        reduce_size(_make_text_pdf(), quality=quality)


def test_corrupt_input_raises():
    with pytest.raises(Exception):
        reduce_size(b"%PDF-1.7 not actually a pdf")


def test_validate_helper_accepts_none_dpi():
    # None dpi is the lossless contract -- must not raise.
    _validate_reduce_params(None, 75)


# ── Side-effect verification + no-grow guard ─────────────────────────


def test_no_grow_guard_returns_input_bytes_identical(monkeypatch):
    # Force the re-saved candidate to be LARGER than the input; the guard
    # must hand back the original bytes byte-for-byte, untouched.
    src = _make_text_pdf()

    def fat_save(self, buf, *a, **k):
        buf.write(b"%PDF-1.7\n" + b"0" * (len(src) + 4096))

    monkeypatch.setattr(fitz.Document, "save", fat_save)
    out, info = reduce_size(src)
    assert out == src                            # bytes-identical original
    assert info["orig_size"] == info["new_size"] == len(src)
    assert info["applied_dpi"] is None


# ── Mock integrity ───────────────────────────────────────────────────


def test_rewrite_images_failure_falls_back_to_lossless(monkeypatch):
    src = _make_image_pdf()

    def boom(self, *a, **k):
        raise RuntimeError("synthetic image-rewrite failure")

    monkeypatch.setattr(fitz.Document, "rewrite_images", boom)
    out, info = reduce_size(src, dpi=150)
    # Graceful: no exception, downsample skipped (applied_dpi None), still
    # a valid PDF (lossless result or no-grow original).
    assert info["applied_dpi"] is None
    _open_clean(out).close()


def test_subset_fonts_failure_is_non_fatal(monkeypatch):
    src = _make_text_pdf()

    def boom(self, *a, **k):
        raise RuntimeError("synthetic font-subset failure")

    monkeypatch.setattr(fitz.Document, "subset_fonts", boom)
    out, info = reduce_size(src)               # lossless path still completes
    assert info["new_size"] == len(out)
    _open_clean(out).close()


def test_save_failure_propagates(monkeypatch):
    # The library is a faithful primitive: a genuine save failure is NOT
    # swallowed (route-level graceful degradation handles it upstream).
    src = _make_text_pdf()

    def boom(self, *a, **k):
        raise RuntimeError("synthetic disk-full")

    monkeypatch.setattr(fitz.Document, "save", boom)
    with pytest.raises(RuntimeError, match="disk-full"):
        reduce_size(src)


# ── Wire dispatch ────────────────────────────────────────────────────


def test_wire_dispatch_reduce_size():
    from lexcloak_pdf_tool.__main__ import _handle
    src = _make_image_pdf()
    resp = _handle({"op": "reduce_size",
                    "pdf_b64": base64.b64encode(src).decode("ascii"),
                    "dpi": 150, "quality": 60})
    assert resp["ok"] is True
    result = resp["result"]
    assert result["info"]["applied_dpi"] == 150
    out = base64.b64decode(result["pdf_b64"])
    assert len(out) == result["info"]["new_size"]
    _open_clean(out).close()


def test_wire_dispatch_encrypted_returns_error():
    from lexcloak_pdf_tool.__main__ import _handle
    resp = _handle({"op": "reduce_size",
                    "pdf_b64": base64.b64encode(_make_encrypted_pdf()).decode("ascii")})
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"
    assert "cleartext" in resp["error"]


def test_wire_dispatch_handle_variant():
    from lexcloak_pdf_tool.__main__ import _handle
    src = _make_image_pdf()
    h = _handle({"op": "open_doc",
                 "pdf_b64": base64.b64encode(src).decode("ascii")})["result"]["handle"]
    try:
        resp = _handle({"op": "reduce_size_h", "handle": h,
                        "dpi": 150, "quality": 60})
        assert resp["ok"] is True
        assert resp["result"]["info"]["applied_dpi"] == 150
    finally:
        _handle({"op": "close_doc", "handle": h})
