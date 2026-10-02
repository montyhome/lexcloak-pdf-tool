"""A form XObject keeps its text state through the redaction rewrite (v0.14.1).

MuPDF's content filter drops a text-state operator set outside ``BT`` before
a form's ``Do``, and rewrites each form as if it started from the default
text state, so a form that resets ``0 Tr`` after the page drew invisible
text (``3 Tr``) loses that reset and draws nothing. ``form_state`` wraps
each rewritten form's ``Do`` in the default state the rewrite assumed.

Every fixture is built in memory with invented text. "Kept" is judged by
the render (ink in the stamp's box at 144 DPI) and by the trace's render
mode; "removed" by the raw content streams, not by ``get_text``.
"""
from __future__ import annotations

import pymupdf
import pytest

from lexcloak_pdf_tool import form_state
from lexcloak_pdf_tool.form_state import (ContentError, apply_page_redactions,
                                          operators, save_document)
from lexcloak_pdf_tool.redact import apply_redactions

W, H = 300, 200
STAMP = "STAMP-0001"
#: Where the stamp draws, in page coordinates (baseline 20 pt from the top).
STAMP_BOX = pymupdf.Rect(195, 8, 275, 26)
HIDDEN = "Qwzv hidden words"
#: Over the first hidden word only (baseline 100 pt from the top).
HIT = pymupdf.Rect(18, 90, 52, 103)
REMOVE_ONLY = dict(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                   graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                   text=pymupdf.PDF_REDACT_TEXT_REMOVE)


def _page(content: bytes, form: bytes | None = None, *, oc: bool = False,
          form_name: str = "Fm0", nested: bytes | None = None):
    """A one-page document: ``content`` on the page, ``form`` as an XObject
    named ``form_name`` (optionally on a layer), ``nested`` as a form the
    first one invokes as ``/Fm9``. Font ``/F1`` is Helvetica throughout."""
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    font = doc.get_new_xref()
    doc.update_object(font, "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>")
    xobjects = ""
    if form is not None:
        inner = ""
        if nested is not None:
            nx = doc.get_new_xref()
            doc.update_object(nx, f"<</Type/XObject/Subtype/Form/BBox[0 0 {W} {H}]"
                                  f"/Resources<</Font<</F1 {font} 0 R>>>>>>")
            doc.update_stream(nx, nested)
            inner = f"/XObject<</Fm9 {nx} 0 R>>"
        fx = doc.get_new_xref()
        oc_entry = f"/OC {doc.add_ocg('Stamp', on=True)} 0 R" if oc else ""
        doc.update_object(fx, f"<</Type/XObject/Subtype/Form/BBox[0 0 {W} {H}]"
                              f"/Resources<</Font<</F1 {font} 0 R>>{inner}>>{oc_entry}>>")
        doc.update_stream(fx, form)
        xobjects = f"/XObject<</{form_name} {fx} 0 R>>"
    doc.xref_set_key(page.xref, "Resources", f"<</Font<</F1 {font} 0 R>>{xobjects}>>")
    cx = doc.get_new_xref()
    doc.update_object(cx, "<<>>")
    doc.update_stream(cx, content)
    doc.xref_set_key(page.xref, "Contents", f"{cx} 0 R")
    return doc


#: The page draws an invisible text layer, then shows the stamp form.
LAYER = b"BT /F1 10 Tf 3 Tr 20 100 Td (" + HIDDEN.encode() + b") Tj ET "
STAMP_FORM = b"BT 0 Tr /F1 10 Tf 200 180 Td (" + STAMP.encode() + b") Tj ET"
STAMP_FORM_NO_TR = b"BT /F1 10 Tf 200 180 Td (" + STAMP.encode() + b") Tj ET"


def _ink(page, rect=STAMP_BOX) -> int:
    pix = page.get_pixmap(dpi=144, clip=rect, colorspace=pymupdf.csGRAY)
    return sum(1 for b in pix.samples if b < 128)


def _modes(page, needle: str) -> list[int]:
    return [s["type"] for s in page.get_texttrace()
            if needle in "".join(chr(c[0]) for c in s["chars"])]


def _streams(doc) -> bytes:
    """Every live stream: the document as it would be written."""
    live = pymupdf.open(stream=doc.tobytes(garbage=4))
    return b"\n".join(live.xref_stream(x) or b"" for x in range(1, live.xref_length())
                      if live.xref_is_stream(x))


def _rewrite(doc, how: str, fix: bool):
    page = doc[0]
    if how == "remove":
        page.add_redact_annot(HIT, fill=False)
        kw = REMOVE_ONLY
    else:
        page.add_redact_annot(HIT, fill=(0, 0, 0))
        kw = {}
    return apply_page_redactions(page, **kw) if fix else page.apply_redactions(**kw)


# ── The defect, and the repair ─────────────────────────────────────────────


@pytest.mark.parametrize("how", ["remove", "burn"])
@pytest.mark.parametrize("oc", [False, True], ids=["plain", "on-a-layer"])
@pytest.mark.parametrize("content, form", [
    (LAYER + b"q /Fm0 Do Q", STAMP_FORM),                 # the form resets 0 Tr
    (LAYER + b"q 0 Tr /Fm0 Do Q", STAMP_FORM_NO_TR),      # the page resets it
    (LAYER + b"q 0 Tr /Fm0 Do Q", STAMP_FORM),            # both do
], ids=["form-resets", "page-resets", "both-reset"])
def test_the_stamp_still_draws_after_the_rewrite(content, form, oc, how):
    doc = _page(content, form, oc=oc)
    before = _ink(doc[0])
    assert before > 300 and _modes(doc[0], STAMP) == [0]
    assert _rewrite(doc, how, fix=True) is True
    assert _ink(doc[0]) == before
    assert _modes(doc[0], STAMP) == [0]
    # the redaction itself happened: the first hidden word is gone
    assert b"Qwzv" not in _streams(doc)


@pytest.mark.parametrize("how", ["remove", "burn"])
def test_without_the_repair_mupdf_hides_the_stamp(how):
    """The defect this module exists for, pinned on the unrepaired path so
    a MuPDF that fixes it upstream shows up here."""
    doc = _page(LAYER + b"q /Fm0 Do Q", STAMP_FORM)
    assert _ink(doc[0]) > 300
    _rewrite(doc, how, fix=False)
    assert _ink(doc[0]) == 0
    assert _modes(doc[0], STAMP) == [3]
    assert STAMP.encode() in _streams(doc)        # still in the file


@pytest.mark.parametrize("op, value, default", [
    ("Tc", 3, 0), ("Tw", 9, 0), ("Tz", 150, 100), ("Ts", 4, 0), ("Tr", 3, 0)])
def test_every_text_state_parameter_survives(op, value, default):
    """The page sends a non-default value; the form sets the default. Each
    parameter is dropped the same way, and each is restored."""
    content = (f"BT {value} {op} /F1 10 Tf 20 60 Td (page text here) Tj ET "
               "q /Fm0 Do Q").encode()
    form = f"BT {default} {op} /F1 10 Tf 150 150 Td (FORM TEXT) Tj ET".encode()

    def shape(page):
        return [(s["type"], [round(v, 2) for v in s["bbox"]])
                for s in page.get_texttrace()
                if "FORM" in "".join(chr(c[0]) for c in s["chars"])]

    plain = _page(content, form)
    expected = shape(plain[0])
    plain[0].add_redact_annot(pymupdf.Rect(18, 130, 50, 142), fill=False)
    plain[0].apply_redactions(**REMOVE_ONLY)
    assert shape(plain[0]) != expected                  # MuPDF changes it
    fixed = _page(content, form)
    fixed[0].add_redact_annot(pymupdf.Rect(18, 130, 50, 142), fill=False)
    assert apply_page_redactions(fixed[0], **REMOVE_ONLY) is True
    assert shape(fixed[0]) == expected


def test_a_nested_form_keeps_its_state_too():
    nested = b"BT 0 Tr /F1 10 Tf 200 150 Td (NESTED-0002) Tj ET"
    outer = b"BT 3 Tr /F1 10 Tf 200 180 Td (outer hidden) Tj ET q /Fm9 Do Q"
    doc = _page(LAYER + b"q /Fm0 Do Q", outer, nested=nested)
    box = pymupdf.Rect(195, 38, 290, 56)
    before = _ink(doc[0], box)
    assert before > 300
    doc[0].add_redact_annot(HIT, fill=False)
    assert apply_page_redactions(doc[0], **REMOVE_ONLY) is True
    assert _ink(doc[0], box) == before
    assert _modes(doc[0], "NESTED") == [0]


def test_an_escaped_resource_name_is_resolved():
    doc = _page(LAYER + b"q /F#6d0 Do Q", STAMP_FORM, form_name="Fm0")
    before = _ink(doc[0])
    assert before > 300
    doc[0].add_redact_annot(HIT, fill=False)
    assert apply_page_redactions(doc[0], **REMOVE_ONLY) is True
    assert _ink(doc[0]) == before


def test_the_page_state_after_the_form_is_the_pages_own():
    """The wrap's Q restores the page: invisible text drawn after the form
    stays invisible, visible text stays visible."""
    content = (LAYER + b"q /Fm0 Do Q BT 3 Tr /F1 10 Tf 20 40 Td (after hidden) Tj ET "
               b"BT 0 Tr /F1 10 Tf 20 20 Td (after shown) Tj ET")
    doc = _page(content, STAMP_FORM)
    doc[0].add_redact_annot(HIT, fill=False)
    assert apply_page_redactions(doc[0], **REMOVE_ONLY) is True
    assert _modes(doc[0], "after hidden") == [3]
    assert _modes(doc[0], "after shown") == [0]
    assert _modes(doc[0], STAMP) == [0]


# ── Pages left exactly as MuPDF writes them ───────────────────────────────


def test_a_form_that_relies_on_inherited_state_is_left_alone():
    """The form draws with the caller's 3 Tr and sets nothing: invisible by
    design. Wrapping it in the default state would show it, so the page is
    left as MuPDF writes it -- and the stamp stays invisible."""
    content = LAYER + b"q /Fm0 Do Q"
    plain = _page(content, STAMP_FORM_NO_TR)
    assert _ink(plain[0]) == 0
    plain[0].add_redact_annot(HIT, fill=False)
    plain[0].apply_redactions(**REMOVE_ONLY)
    fixed = _page(content, STAMP_FORM_NO_TR)
    fixed[0].add_redact_annot(HIT, fill=False)
    assert apply_page_redactions(fixed[0], **REMOVE_ONLY) is False
    assert _ink(fixed[0]) == 0
    assert _modes(fixed[0], STAMP) == [3]
    assert _streams(fixed) == _streams(plain)


def test_a_page_without_forms_is_written_exactly_as_mupdf_writes_it():
    plain = _page(LAYER + b"BT 0 Tr /F1 10 Tf 20 60 Td (shown) Tj ET")
    fixed = _page(LAYER + b"BT 0 Tr /F1 10 Tf 20 60 Td (shown) Tj ET")
    for doc in (plain, fixed):
        doc[0].add_redact_annot(HIT, fill=(0, 0, 0))
    plain[0].apply_redactions()
    assert apply_page_redactions(fixed[0]) is False
    assert _streams(fixed) == _streams(plain)


def test_unreadable_content_is_left_as_mupdf_writes_it(monkeypatch):
    """A content stream the reader refuses: the redaction is still applied,
    nothing is wrapped, and nothing is raised."""
    def refuse(_data):
        raise ContentError("unreadable token")
        yield  # pragma: no cover
    monkeypatch.setattr(form_state, "operators", refuse)
    doc = _page(LAYER + b"q /Fm0 Do Q", STAMP_FORM)
    doc[0].add_redact_annot(HIT, fill=False)
    assert apply_page_redactions(doc[0], **REMOVE_ONLY) is False
    assert b"Qwzv" not in _streams(doc)
    assert _modes(doc[0], STAMP) == [3]            # MuPDF's own result


def test_a_non_numeric_operand_is_left_as_mupdf_writes_it():
    doc = _page(LAYER + b"q /X Tr /Fm0 Do Q", STAMP_FORM)
    doc[0].add_redact_annot(HIT, fill=False)
    assert apply_page_redactions(doc[0], **REMOVE_ONLY) is False
    assert b"Qwzv" not in _streams(doc)


def test_a_form_nested_past_the_depth_limit_is_left_alone(monkeypatch):
    monkeypatch.setattr(form_state, "MAX_DEPTH", 0)
    doc = _page(LAYER + b"q /Fm0 Do Q", STAMP_FORM)
    doc[0].add_redact_annot(HIT, fill=False)
    assert apply_page_redactions(doc[0], **REMOVE_ONLY) is False


def test_mupdf_errors_still_raise(monkeypatch):
    doc = _page(LAYER + b"q /Fm0 Do Q", STAMP_FORM)

    def boom(self, **kw):
        raise RuntimeError("redaction failed")
    monkeypatch.setattr(pymupdf.Page, "apply_redactions", boom)
    with pytest.raises(RuntimeError, match="redaction failed"):
        apply_page_redactions(doc[0], **REMOVE_ONLY)


# ── The content reader ────────────────────────────────────────────────────


def test_operators_reads_strings_hex_dicts_comments_and_inline_images():
    data = (b"% a comment with /Fm0 Do in it\n"
            b"/P <</MCID 3 /Alt (a (nested) Do \\) string)>> BDC "
            b"BT (x\\(Do) Tj <446f> Tj [(a) -20 (b)] TJ ET EMC "
            b"BI /W 2 /H 1 /BPC 8 /CS /G ID \x00 Do \xff EI "
            b"q 1 0 0 1 0 0 cm /Fm0 Do Q")
    ops = [op for op, _, _, _ in operators(data)]
    assert ops == [b"BDC", b"BT", b"Tj", b"Tj", b"TJ", b"ET", b"EMC", b"BI",
                   b"q", b"cm", b"Do", b"Q"]
    tj = [args for op, args, _, _ in operators(data) if op == b"Tj"]
    assert tj == [[("str", b"(x\\(Do)")], [("hex", b"<446f>")]]
    *_, (op, args, start, end) = [o for o in operators(data) if o[0] == b"Do"]
    assert data[start:end] == b"/Fm0 Do"


@pytest.mark.parametrize("data", [b"BT (never closed Tj ET",
                                  b"BI /W 1 /H 1 ID \x00\x00",
                                  b"q <</A 1>> > Do"])
def test_operators_refuses_what_is_not_a_content_stream(data):
    with pytest.raises(ContentError):
        list(operators(data))


def test_operators_on_an_empty_stream():
    assert list(operators(b"")) == []


# ── Through the op ─────────────────────────────────────────────────────────


def test_apply_redactions_burn_keeps_the_stamp_with_and_without_kept_lines():
    for keep in (False, True):
        doc = _page(LAYER + b"BT 0 Tr /F1 10 Tf 20 60 Td (Shownvalue rest) Tj ET "
                            b"q /Fm0 Do Q", STAMP_FORM)
        src = doc.tobytes()
        match = {"page": 0, "type": "Person Name", "text": "Shownvalue",
                 "rect": {"x0": 19, "y0": 131, "x1": 72, "y1": 143}}
        out, _ = apply_redactions(src, [match], active_categories=["Person Name"],
                                  keep_uncovered_lines=keep)
        result = pymupdf.open(stream=out)
        assert _ink(result[0]) == _ink(pymupdf.open(stream=src)[0])
        assert _modes(result[0], STAMP) == [0]
        assert b"Shownvalue" not in _streams(result)


# ── The clean save: the same filter, on every page ────────────────────────


SAVE = dict(garbage=4, deflate=True, clean=True)


@pytest.mark.parametrize("content, form", [
    (LAYER + b"q /Fm0 Do Q", STAMP_FORM),
    (LAYER + b"q 0 Tr /Fm0 Do Q", STAMP_FORM_NO_TR),
], ids=["form-resets", "page-resets"])
def test_a_clean_save_alone_hides_the_stamp_and_save_document_keeps_it(content, form):
    plain = _page(content, form)
    before = _ink(plain[0])
    assert before > 300
    import io
    buf = io.BytesIO()
    plain.save(buf, **SAVE)
    assert _ink(pymupdf.open(stream=buf.getvalue())[0]) == 0      # the defect
    kept = pymupdf.open(stream=save_document(_page(content, form), **SAVE))
    assert _ink(kept[0]) == before
    assert _modes(kept[0], STAMP) == [0]
    assert _modes(kept[0], "Qwzv") == [3]          # the layer stays invisible


def test_a_document_that_sets_no_non_default_state_saves_as_before():
    content = b"BT /F1 10 Tf 20 100 Td (plain text) Tj ET q /Fm0 Do Q"
    import io
    buf = io.BytesIO()
    _page(content, STAMP_FORM).save(buf, **SAVE)
    ours = save_document(_page(content, STAMP_FORM), **SAVE)
    assert _streams(pymupdf.open(stream=ours)) == _streams(pymupdf.open(stream=buf.getvalue()))
    assert b"0 Tc 0 Tw 100 Tz" not in _streams(pymupdf.open(stream=ours))


def test_a_save_without_clean_is_left_to_mupdf():
    doc = _page(LAYER + b"q /Fm0 Do Q", STAMP_FORM)
    out = pymupdf.open(stream=save_document(doc, garbage=4, deflate=True))
    assert _ink(out[0]) > 300
    assert b"0 Tc 0 Tw 100 Tz" not in _streams(out)


def test_an_encrypted_clean_save_keeps_the_stamp():
    data = save_document(_page(LAYER + b"q /Fm0 Do Q", STAMP_FORM), **SAVE,
                         encryption=pymupdf.PDF_ENCRYPT_AES_256,
                         user_pw="pw-1", owner_pw="pw-1")
    out = pymupdf.open(stream=data)
    assert out.needs_pass
    assert out.authenticate("pw-1")
    assert _modes(out[0], STAMP) == [0]
    assert _ink(out[0]) > 300


def test_a_clean_save_leaves_a_form_relying_on_inherited_state_alone():
    plain = _page(LAYER + b"q /Fm0 Do Q", STAMP_FORM_NO_TR)
    import io
    buf = io.BytesIO()
    plain.save(buf, **SAVE)
    ours = save_document(_page(LAYER + b"q /Fm0 Do Q", STAMP_FORM_NO_TR), **SAVE)
    out = pymupdf.open(stream=ours)
    assert _modes(out[0], STAMP) == [3]
    assert _ink(out[0]) == 0
    assert _streams(out) == _streams(pymupdf.open(stream=buf.getvalue()))


def test_a_form_shown_on_two_pages_is_wrapped_once():
    nested = b"BT 0 Tr /F1 10 Tf 200 150 Td (NESTED-0002) Tj ET"
    outer = b"BT 3 Tr /F1 10 Tf 200 180 Td (outer hidden) Tj ET q /Fm9 Do Q"
    doc = _page(LAYER + b"q /Fm0 Do Q", outer, nested=nested)
    doc.fullcopy_page(0)
    box = pymupdf.Rect(195, 38, 290, 56)
    before = _ink(doc[0], box)
    out = pymupdf.open(stream=save_document(doc, **SAVE))
    assert [_ink(out[p], box) for p in (0, 1)] == [before, before]
    forms = [out.xref_stream(x) for x in range(1, out.xref_length())
             if out.xref_is_stream(x) and out.xref_get_key(x, "Subtype")[1] == "/Form"]
    assert all(f.count(b"0 Tc 0 Tw 100 Tz") <= 1 for f in forms)


def test_the_ops_that_save_keep_the_stamp():
    from lexcloak_pdf_tool.metadata import set_metadata
    from lexcloak_pdf_tool.redact import strip_metadata
    from lexcloak_pdf_tool.reduce_size import reduce_size
    # An uncompressed comment the clean save drops, so reduce_size's
    # no-grow guard keeps its re-saved bytes.
    padding = b"% " + b"x" * 20000 + b"\n"
    src = _page(padding + LAYER + b"q /Fm0 Do Q", STAMP_FORM).tobytes()
    before = _ink(pymupdf.open(stream=src)[0])
    outs = {"strip_metadata": strip_metadata(src),
            "set_metadata": set_metadata(src, {"title": "Invented"}),
            "reduce_size": reduce_size(src)[0]}
    for name, out in outs.items():
        assert out != src, name                        # really re-saved
        assert _ink(pymupdf.open(stream=out)[0]) == before, name


# ── The narrower rules (each pinned by a mutation check) ───────────────────


def test_a_page_mupdf_did_not_rewrite_is_not_wrapped(monkeypatch):
    """Only form instances the rewrite produced were filtered from the
    default state; a page still showing an original form is left alone."""
    doc = _page(LAYER + b"q /Fm0 Do Q", STAMP_FORM)
    before = doc[0].read_contents()
    monkeypatch.setattr(pymupdf.Page, "apply_redactions", lambda self, **kw: None)
    assert apply_page_redactions(doc[0], **REMOVE_ONLY) is False
    assert doc[0].read_contents() == before


def test_a_form_shown_inside_bt_is_never_wrapped():
    """``q`` is not allowed inside ``BT``: such a stream is left alone."""
    doc = _page(LAYER + b"q /Fm0 Do Q", STAMP_FORM)
    form = form_state._page_xobjects(doc[0])["Fm0"]
    names = {"Fm0": form}
    assert form_state._wrap(doc, b"BT /Fm0 Do ET", names, lambda x: True) is None
    assert form_state._wrap(doc, b"q /Fm0 Do Q", names, lambda x: True) == (
        b"q " + form_state.PREFIX + b"/Fm0 Do Q Q")
    assert form_state._wrap(doc, b"q /Fm0 Do Q", names, lambda x: False) is None


def test_the_reader_restores_state_at_Q():
    """``3 Tr`` set inside ``q ... Q`` is gone at the Do: the form that
    inherits the default is safe, and the page is restored."""
    content = (b"BT 0 Tr /F1 10 Tf 20 60 Td (shown) Tj ET "
               b"q BT 3 Tr /F1 10 Tf 20 100 Td (" + HIDDEN.encode() + b") Tj ET Q "
               b"q /Fm0 Do Q")
    doc = _page(content, STAMP_FORM_NO_TR)
    reading = form_state._read_page(doc, doc[0])
    assert (reading.safe, reading.at_risk) == (True, True)
    unsafe = _page(content.replace(b" Q q /Fm0", b" q /Fm0"), STAMP_FORM_NO_TR)
    assert form_state._read_page(unsafe, unsafe[0]).safe is False


def test_a_form_an_unsafe_page_also_shows_keeps_mupdfs_content():
    """Page 1 can be restored; page 2 also shows a form that relies on
    inherited 3 Tr, so it cannot. The form both pages show keeps the content
    MuPDF wrote; page 1's own Do is still wrapped."""
    nested = b"BT 0 Tr /F1 10 Tf 200 150 Td (NESTED-0002) Tj ET"
    outer = b"BT 0 Tr /F1 10 Tf 200 180 Td (OUTER-0003) Tj ET q /Fm9 Do Q"

    def build():
        doc = _page(LAYER + b"q /Fm0 Do Q", outer, nested=nested)
        relying = doc.get_new_xref()
        font = doc.xref_get_key(doc[0].xref, "Resources/Font/F1")[1]
        doc.update_object(relying, f"<</Type/XObject/Subtype/Form/BBox[0 0 {W} {H}]"
                                   f"/Resources<</Font<</F1 {font}>>>>>>")
        doc.update_stream(relying, b"BT /F1 10 Tf 20 20 Td (relies) Tj ET")
        res = doc.xref_get_key(doc[0].xref, "Resources")[1]
        page2 = doc.new_page(width=W, height=H)
        doc.xref_set_key(page2.xref, "Resources",
                         res.replace("/XObject<<", f"/XObject<</FmR {relying} 0 R"))
        cx = doc.get_new_xref()
        doc.update_object(cx, "<<>>")
        doc.update_stream(cx, LAYER + b"q /FmR Do Q q /Fm0 Do Q")
        doc.xref_set_key(page2.xref, "Contents", f"{cx} 0 R")
        return doc

    import io
    buf = io.BytesIO()
    build().save(buf, **SAVE)
    plain = pymupdf.open(stream=buf.getvalue())
    ours = pymupdf.open(stream=save_document(build(), **SAVE))

    def outer_form(d):
        return next(d.xref_stream(x) for x in range(1, d.xref_length())
                    if d.xref_is_stream(x) and b"OUTER" in (d.xref_stream(x) or b""))

    assert outer_form(ours) == outer_form(plain)
    assert form_state.PREFIX in ours[0].read_contents()
    assert form_state.PREFIX not in ours[1].read_contents()
