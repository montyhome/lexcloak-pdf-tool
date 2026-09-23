"""A PDF whose xref leaves an object number undefined (v0.9.1).

A file's ``/Size`` can exceed what its xref sections cover, which is common in
linearized files carrying an incremental update. The numbers no section
mentions have no body; a reference to one is a dangling reference, which a
conforming reader resolves as null. PyMuPDF's ``Document.scrub`` walks every
number below ``xref_length()`` for ``javascript=True`` and
``xml_metadata=True``, and loading an undefined one raises ``cannot find
object in xref``. Through v0.9.0 that failed every ``apply_redactions`` and
``reduce_size`` call on such a file, with or without redaction boxes.

``scrub_objects`` now does that walk and passes over exactly the numbers the
xref never defines. These tests pin that:

* **Happy path** -- ``apply_redactions`` (with and without boxes) and
  ``reduce_size`` succeed, on the library entry points and on the handle ops
  over the real subprocess.
* **The guarantee holds** -- JavaScript and XMP are still removed, asserted on
  the delivered bytes (raw sentinel scan and a re-opened document), and the
  page and its form field survive.
* **No vacuous pass** -- the fixture is checked to carry the undefined number
  and the dangling reference that make it fail.
* **Fail closed** -- a *defined* object that will not load still raises; it is
  never skipped.

Synthetic fixture, hand-built so the xref shape is exact.
"""
from __future__ import annotations

import base64
import json
import struct
import subprocess
import sys

import pymupdf
import pytest

from lexcloak_pdf_tool import apply_redactions, reduce_size
from lexcloak_pdf_tool.redact import scrub_objects


JS_SENTINEL = b"JSSENTINEL-Tamsin-Orlov-5521"
XMP_SENTINEL = b"XMPSENTINEL-Quarry-Lane-0917"
PAGE_TEXT = "Line 7 total"
FIELD_NAME = "f1_07"
FIELD_VALUE = "FIELDVALUE"
UNDEFINED = 7


def _pdf_with_undefined_entry() -> bytes:
    """One page, one text field, a JavaScript open action and catalog XMP.

    The trailer says ``/Size 10`` but the xref sections cover 0-6 and 8-9, so
    object 7 is defined nowhere. The catalog's ``/PieceInfo`` points at it,
    making the reference dangling.
    """
    content = f"BT /F1 12 Tf 20 150 Td ({PAGE_TEXT}) Tj ET".encode()
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/">' + XMP_SENTINEL + b"</x:xmpmeta>"
    objs = {
        1: b"<< /Type /Catalog /Pages 2 0 R /OpenAction 5 0 R /Metadata 6 0 R"
           b" /AcroForm << /Fields [9 0 R] >> /PieceInfo 7 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R"
           b" /Resources << /Font << /F1 8 0 R >> >> /Annots [9 0 R] >>",
        4: b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        5: b"<< /S /JavaScript /JS (app.alert\\('" + JS_SENTINEL + b"'\\);) >>",
        6: b"<< /Type /Metadata /Subtype /XML /Length %d >>\nstream\n" % len(xmp)
           + xmp + b"\nendstream",
        8: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        9: b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (" + FIELD_NAME.encode()
           + b") /V (" + FIELD_VALUE.encode() + b") /Rect [20 40 180 60] /P 3 0 R"
           b" /DA (/Helv 10 Tf 0 g) >>",
    }
    out = bytearray(b"%PDF-1.7\n")
    offsets = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += b"%d 0 obj\n" % num + objs[num] + b"\nendobj\n"
    startxref = len(out)
    out += b"xref\n0 7\n0000000000 65535 f \n"
    for num in range(1, 7):
        out += b"%010d 00000 n \n" % offsets[num]
    out += b"8 2\n"
    for num in (8, 9):
        out += b"%010d 00000 n \n" % offsets[num]
    out += b"trailer\n<< /Size 10 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % startxref
    return bytes(out)


def _assert_clean(out: bytes) -> pymupdf.Document:
    """JavaScript and XMP gone from the delivered bytes; every object loads."""
    assert JS_SENTINEL not in out
    assert XMP_SENTINEL not in out
    doc = pymupdf.open(stream=out, filetype="pdf")
    assert doc.get_xml_metadata() == ""
    for xref in range(1, doc.xref_length()):
        text = doc.xref_object(xref)  # raises if the delivered file is broken
        assert "/Type /Metadata" not in text
        if doc.xref_get_key(xref, "S")[1] == "/JavaScript":
            assert doc.xref_get_key(xref, "JS")[1] in ("", "()")
    return doc


# ── the fixture is what it claims to be ─────────────────────────────


def test_fixture_leaves_an_object_number_undefined():
    doc = pymupdf.open(stream=_pdf_with_undefined_entry(), filetype="pdf")
    pdf = pymupdf.mupdf.pdf_specifics(doc.this)
    assert doc.xref_length() == 10
    assert not pymupdf.mupdf.pdf_object_exists(pdf, UNDEFINED)
    assert pymupdf.mupdf.pdf_object_exists(pdf, UNDEFINED + 1)
    with pytest.raises(RuntimeError, match="cannot find object in xref"):
        doc.xref_object(UNDEFINED)
    assert doc.xref_get_key(doc.pdf_catalog(), "PieceInfo") == ("xref", "7 0 R")
    # And it carries what the scrub has to remove.
    assert doc.get_xml_metadata() != ""
    assert doc.xref_get_key(5, "S")[1] == "/JavaScript"


# ── library entry points ───────────────────────────────────────────


def test_apply_redactions_with_no_boxes_succeeds_and_scrubs():
    out, protected = apply_redactions(_pdf_with_undefined_entry(), [])
    assert protected is True
    doc = _assert_clean(out)
    assert len(doc) == 1
    assert PAGE_TEXT in doc[0].get_text()


def test_apply_redactions_with_a_box_succeeds_and_burns():
    src = pymupdf.open(stream=_pdf_with_undefined_entry(), filetype="pdf")
    rect = src[0].search_for(PAGE_TEXT)[0]
    matches = [{"page": 0, "enabled": True, "type": "custom",
                "rect": {"x0": rect.x0, "y0": rect.y0,
                         "x1": rect.x1, "y1": rect.y1}}]
    out, _ = apply_redactions(_pdf_with_undefined_entry(), matches)
    doc = _assert_clean(out)
    assert PAGE_TEXT not in doc[0].get_text()


def test_reduce_size_succeeds_scrubs_and_keeps_the_field():
    src = _pdf_with_undefined_entry()
    out, info = reduce_size(src)
    # The no-grow guard returns the input unchanged when the result is not
    # smaller, which would make the checks below read the unscrubbed file.
    assert info["new_size"] < info["orig_size"]
    doc = _assert_clean(out)
    assert len(doc) == 1
    assert PAGE_TEXT in doc[0].get_text()
    widgets = list(doc[0].widgets())
    assert [(w.field_name, w.field_value) for w in widgets] == [
        (FIELD_NAME, FIELD_VALUE)]


# ── handle ops, over the real subprocess ───────────────────────────


def _call(ops: list[dict]) -> list[dict]:
    frame = struct.Struct(">I")
    proc = subprocess.Popen([sys.executable, "-m", "lexcloak_pdf_tool"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    replies = []
    try:
        for op in ops:
            if "handle" in op and op["handle"] is None:
                op = {**op, "handle": replies[0]["result"]["handle"]}
            body = json.dumps({"protocol_version": 4, **op}).encode()
            proc.stdin.write(frame.pack(len(body)) + body)
            proc.stdin.flush()
            (length,) = frame.unpack(proc.stdout.read(4))
            replies.append(json.loads(proc.stdout.read(length)))
    finally:
        proc.kill()
        proc.wait(timeout=5)
    return replies


@pytest.mark.parametrize("op", ["apply_redactions_h", "reduce_size_h"])
def test_handle_ops_succeed_and_scrub(op):
    b64 = base64.b64encode(_pdf_with_undefined_entry()).decode("ascii")
    extra = {"matches": []} if op == "apply_redactions_h" else {}
    opened, reply = _call([{"op": "open_doc", "pdf_b64": b64},
                           {"op": op, "handle": None, **extra}])
    assert opened["ok"] is True, opened
    assert reply["ok"] is True, reply
    _assert_clean(base64.b64decode(reply["result"]["pdf_b64"]))


# ── fail closed ─────────────────────────────────────────────────────


def test_a_defined_object_that_will_not_load_still_raises(monkeypatch):
    """Only numbers the xref never defines are passed over. A defined object
    that fails to load might carry script or XMP, so the walk must raise.

    Called directly: a real file with a defined-but-unloadable object makes
    MuPDF repair the xref on load, which leaves nothing to test.
    """
    doc = pymupdf.open(stream=_pdf_with_undefined_entry(), filetype="pdf")
    real = pymupdf.Document.xref_object

    def broken(self, xref, *args, **kwargs):
        if xref == 5:  # the JavaScript action -- defined
            raise RuntimeError("simulated load failure")
        return real(self, xref, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Document, "xref_object", broken)
    with pytest.raises(RuntimeError, match="simulated load failure"):
        scrub_objects(doc, javascript=True, xml_metadata=True)
