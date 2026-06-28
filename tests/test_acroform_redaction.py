"""AcroForm widget-field redaction regression tests.

Incident (Session 625, 2026-06-28 test-suite audit §7): form-field values live
in the widget ``/V`` and the document AcroForm dictionary -- OUTSIDE the page
content stream that ``page.apply_redactions()`` scrubs. Pre-fix,
``_apply_redactions_doc`` drew the redaction box but never flattened widgets, so
the field value:

  * survived in the redacted output's ``get_text()``,
  * survived in the raw output bytes,
  * kept an interactive widget carrying the original ``/V``,

even though the app's detector DOES see the value (PyMuPDF surfaces widget text
in ``get_text``) and therefore DOES produce a match + draw a box over it. A real
PII leak for a legal/medical redaction tool, empirically confirmed against
installed v0.5.1 before the fix.

Fix (v0.5.2): ``_flatten_form_fields`` bakes every interactive widget to static
content (``doc.bake(widgets=True)``) BEFORE the redaction loop, so the now-static
value under a redaction rect is removed by ``apply_redactions`` and no widget /V
survives. Un-redacted form fields remain as static content -- correct retention,
not data loss.

6-criteria: exact PII-absence in BYTES (not return code); hard-coded golden PII
(not re-derived); ≥2 unhappy paths (un-redacted field retained as static;
no-form PDF unaffected); boundary/type stress (checkbox + text widget,
multi-page, empty doc); side-effect verification on output bytes.
"""
from __future__ import annotations

import fitz

from lexcloak_pdf_tool import apply_redactions

# Hard-coded golden PII — never re-derived from a detector.
SSN = "123-45-6789"
NAME = "John Q Public"
FIELD_PII = f"{NAME} {SSN}"
WIDGET_RECT = (100.0, 100.0, 400.0, 130.0)


# ── Fixtures ────────────────────────────────────────────────────────


def _text_widget(name: str, value: str, rect: tuple) -> fitz.Widget:
    w = fitz.Widget()
    w.field_name = name
    w.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    w.field_value = value
    w.rect = fitz.Rect(*rect)
    w.text_fontsize = 11
    return w


def _make_acroform_pdf(field_value: str = FIELD_PII,
                       rect: tuple = WIDGET_RECT,
                       extra_text_fields: list[tuple] | None = None,
                       add_checkbox: bool = False,
                       n_pages: int = 1) -> bytes:
    """Build a synthetic AcroForm PDF.

    ``extra_text_fields`` is a list of ``(name, value, rect)`` placed on page 0
    and never redacted by the tests (they assert retention-as-static).
    """
    doc = fitz.open()
    for _ in range(n_pages):
        doc.new_page(width=612, height=792)
    for i in range(n_pages):
        doc[i].add_widget(_text_widget(f"pii_field_{i}", field_value, rect))
    if extra_text_fields:
        for name, value, r in extra_text_fields:
            doc[0].add_widget(_text_widget(name, value, r))
    if add_checkbox:
        cb = fitz.Widget()
        cb.field_name = "consent"
        cb.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
        cb.field_value = True
        cb.rect = fitz.Rect(100, 200, 120, 220)
        doc[0].add_widget(cb)
    out = doc.tobytes()
    doc.close()
    return out


def _match_for(rect: tuple, page: int = 0, type_: str = "Manual Region") -> dict:
    x0, y0, x1, y1 = rect
    return {
        "id": "m", "type": type_, "page": page,
        "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "enabled": True, "text": "",
    }


def _is_dark_at(pdf_bytes: bytes, page: int, x_pt: float, y_pt: float,
                dpi: float = 144.0) -> bool:
    """True if the as-rendered pixel at (x_pt, y_pt) point-space is near-black."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = dpi / 72.0
        pix = doc[page].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        r, g, b = pix.pixel(int(x_pt * zoom), int(y_pt * zoom))[:3]
        return r < 40 and g < 40 and b < 40
    finally:
        doc.close()


# ── Golden: the leak is closed end-to-end ────────────────────────────


def test_redacted_form_field_value_absent_from_output_text():
    """The widget /V must NOT survive in the redacted output's get_text()."""
    pdf = _make_acroform_pdf()
    out, _ = apply_redactions(pdf, [_match_for(WIDGET_RECT)])
    doc = fitz.open(stream=out, filetype="pdf")
    text = doc[0].get_text()
    doc.close()
    assert SSN not in text
    assert NAME not in text


def test_redacted_form_field_value_absent_from_raw_bytes():
    """Strongest assertion: the PII must not appear anywhere in output bytes."""
    pdf = _make_acroform_pdf()
    out, _ = apply_redactions(pdf, [_match_for(WIDGET_RECT)])
    assert SSN.encode() not in out
    assert NAME.encode() not in out


def test_redacted_form_leaves_no_interactive_widget():
    """No interactive widget (which would carry /V) survives in the output."""
    pdf = _make_acroform_pdf(add_checkbox=True)
    out, _ = apply_redactions(pdf, [_match_for(WIDGET_RECT)])
    doc = fitz.open(stream=out, filetype="pdf")
    surviving = list(doc[0].widgets())
    is_form = doc.is_form_pdf
    doc.close()
    assert surviving == []
    assert not is_form


def test_redacted_form_field_box_renders_dark():
    """The redaction mark is present: the widget center renders near-black."""
    pdf = _make_acroform_pdf()
    out, _ = apply_redactions(pdf, [_match_for(WIDGET_RECT)])
    cx = (WIDGET_RECT[0] + WIDGET_RECT[2]) / 2
    cy = (WIDGET_RECT[1] + WIDGET_RECT[3]) / 2
    assert _is_dark_at(out, 0, cx, cy), "redaction box did not render over the field"


# ── Boundary / type stress ───────────────────────────────────────────


def test_multipage_form_fields_all_redacted():
    """PII in a form field on each of several pages is gone from every page."""
    pdf = _make_acroform_pdf(n_pages=3)
    matches = [_match_for(WIDGET_RECT, page=p) for p in range(3)]
    out, _ = apply_redactions(pdf, matches)
    assert SSN.encode() not in out
    doc = fitz.open(stream=out, filetype="pdf")
    try:
        for p in range(3):
            assert SSN not in doc[p].get_text(), f"page {p} leaked the field value"
            assert list(doc[p].widgets()) == [], f"page {p} kept a widget"
    finally:
        doc.close()


def test_empty_value_form_field_redaction_is_clean():
    """A form field with an empty value flattens + redacts without error."""
    pdf = _make_acroform_pdf(field_value="")
    out, _ = apply_redactions(pdf, [_match_for(WIDGET_RECT)])
    assert out[:4] == b"%PDF"
    doc = fitz.open(stream=out, filetype="pdf")
    try:
        assert list(doc[0].widgets()) == []
    finally:
        doc.close()


def test_checkbox_widget_flattened_too():
    """A non-text widget (checkbox) is also flattened — no widget survives."""
    pdf = _make_acroform_pdf(add_checkbox=True)
    # Redact only the text field; the checkbox is elsewhere but still flattens.
    out, _ = apply_redactions(pdf, [_match_for(WIDGET_RECT)])
    doc = fitz.open(stream=out, filetype="pdf")
    try:
        assert list(doc[0].widgets()) == []
    finally:
        doc.close()


# ── Encrypted output + form field (end-to-end compliance) ────────────


def test_encrypted_output_form_field_value_absent():
    """mode=new re-encryption must not hide an un-scrubbed field /V: decrypt
    the output and confirm the field PII is gone from the text stream."""
    pdf = _make_acroform_pdf()
    password = "s3cret-pw"
    out, protected = apply_redactions(
        pdf, [_match_for(WIDGET_RECT)],
        output_protection={"mode": "new", "password": password},
    )
    assert protected is True
    doc = fitz.open(stream=out, filetype="pdf")
    assert doc.needs_pass, "output should be encrypted"
    assert doc.authenticate(password) > 0
    try:
        assert SSN not in doc[0].get_text()
        assert NAME not in doc[0].get_text()
    finally:
        doc.close()


# ── Unhappy paths (PII that SHOULD survive / no spurious change) ──────


def test_unredacted_form_field_value_retained_as_static_content():
    """A form field the user did NOT redact is retained (flattened to static
    content), NOT deleted — flatten preserves data, it does not destroy it."""
    keep_value = "Retain Me 555-0100"
    pdf = _make_acroform_pdf(
        extra_text_fields=[("keep", keep_value, (100, 400, 400, 430))],
    )
    # Redact ONLY the PII field; the "keep" field is left alone.
    out, _ = apply_redactions(pdf, [_match_for(WIDGET_RECT)])
    doc = fitz.open(stream=out, filetype="pdf")
    text = doc[0].get_text()
    doc.close()
    assert SSN not in text                  # redacted field gone
    assert "Retain Me 555-0100" in text     # un-redacted field retained as static


def test_no_form_pdf_redaction_unaffected():
    """A non-form PDF redacts exactly as before — the flatten step is a no-op
    (is_form_pdf guard), and output carries no widgets."""
    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text(
        fitz.Point(50, 100), f"Patient SSN {SSN}", fontsize=12)
    pdf = doc.tobytes()
    doc.close()
    matches = [_match_for((30, 80, 300, 120), type_="SSN")]
    out, _ = apply_redactions(pdf, matches)
    assert SSN.encode() not in out
    odoc = fitz.open(stream=out, filetype="pdf")
    try:
        assert SSN not in odoc[0].get_text()
        assert list(odoc[0].widgets()) == []
    finally:
        odoc.close()
