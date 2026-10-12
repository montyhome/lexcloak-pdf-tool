"""A save leaves the catalog's /Metadata as the document says (v0.14.2).

PyMuPDF saves with ``preserve_metadata=1`` by default. After writing, MuPDF
then puts ``Root/Metadata`` back into the open document from the newest xref
section that has one, and the source file's own section still does. So once
``del_xml_metadata`` has run, a save brings the old reference back into the
open document, pointing at the object number the XMP stream had. The bytes
of that save are right (the catalog was already written); the next save of
the same document is not. ``save_document`` saves twice when a form's text
state needs restoring, and after the first save's garbage collection has
renumbered the objects, the second wrote ``/Metadata`` as a reference to an
unrelated object, a page dictionary here. A handle-held document saved by
two ops in turn met the same thing with no form at all.

Every fixture is built in memory with invented text.
"""
from __future__ import annotations

import re

import pymupdf
import pytest

from lexcloak_pdf_tool import form_state
from lexcloak_pdf_tool.form_state import save_document
from lexcloak_pdf_tool.redact import (_strip_metadata_doc, apply_redactions,
                                      strip_metadata)
from test_form_text_state import HIT, LAYER, STAMP_FORM, H, W

XMP = ('<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
       '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
       '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
       '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
       '<dc:creator>Quillon Vasterby</dc:creator>'
       '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')
SAVE = dict(garbage=4, deflate=True, clean=True)
_REF = re.compile(r"(\d+) 0 R")


def _document(*, xmp: bool = True, pages: int = 1) -> bytes:
    """``pages`` pages that draw an invisible text layer and then a stamp
    form resetting ``0 Tr``, so a clean save takes ``save_document``'s
    second save. An unreferenced object sits ahead of the XMP stream, so
    garbage collection renumbers everything after it: the XMP's old number
    then lands on the first page's dictionary."""
    doc = pymupdf.open()
    orphan = doc.get_new_xref()
    doc.update_object(orphan, "<</Unused true>>")
    if xmp:
        doc.set_xml_metadata(XMP)
    doc.set_metadata({"author": "Quillon Vasterby", "title": "Invented"})
    for _ in range(pages):
        page = doc.new_page(width=W, height=H)
        font = doc.get_new_xref()
        doc.update_object(font, "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>")
        fx = doc.get_new_xref()
        doc.update_object(fx, f"<</Type/XObject/Subtype/Form/BBox[0 0 {W} {H}]"
                              f"/Resources<</Font<</F1 {font} 0 R>>>>>>")
        doc.update_stream(fx, STAMP_FORM)
        doc.xref_set_key(page.xref, "Resources",
                         f"<</Font<</F1 {font} 0 R>>/XObject<</Fm0 {fx} 0 R>>>>")
        cx = doc.get_new_xref()
        doc.update_object(cx, "<<>>")
        doc.update_stream(cx, LAYER + b"q /Fm0 Do Q")
        doc.xref_set_key(page.xref, "Contents", f"{cx} 0 R")
    return doc.tobytes()


@pytest.fixture
def restores(monkeypatch):
    """How many times ``save_document`` wrapped a page, i.e. took its second
    save. A test that never reaches the second save proves nothing here."""
    calls = []
    real = form_state._restore

    def spy(*args, **kwargs):
        wrote = real(*args, **kwargs)
        calls.append(wrote)
        return wrote

    monkeypatch.setattr(form_state, "_restore", spy)
    return calls


def _references(text: str) -> list[int]:
    return [int(n) for n in _REF.findall(text)]


def _assert_references_hold(pdf: bytes, *, xmp: bool) -> None:
    """Every reference in the catalog and trailer names an object that
    exists, and the ones with a known type have it: ``/Metadata`` an XML
    metadata stream, ``/Pages`` the page tree, ``/Info`` a dictionary that
    is not a page. ``xmp`` says whether the XMP stream should be there."""
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    cat = doc.pdf_catalog()
    for where, text in (("catalog", doc.xref_object(cat, compressed=True)),
                        ("trailer", doc.pdf_trailer(compressed=True))):
        for xref in _references(text):
            assert 0 < xref < doc.xref_length(), f"{where} names missing {xref}"
            assert doc.xref_object(xref, compressed=True) != "null", (
                f"{where} names free object {xref}")

    kind, value = doc.xref_get_key(cat, "Metadata")
    if xmp:
        assert kind == "xref"
        target = int(value.split()[0])
        assert doc.xref_is_stream(target)
        assert doc.xref_get_key(target, "Type") == ("name", "/Metadata")
        assert doc.get_xml_metadata() == XMP
    else:
        assert kind == "null", f"/Metadata is {kind} {value}"
        assert doc.get_xml_metadata() == ""

    kind, value = doc.xref_get_key(cat, "Pages")
    assert kind == "xref"
    assert doc.xref_get_key(int(value.split()[0]), "Type") == ("name", "/Pages")
    kind, value = doc.xref_get_key(-1, "Info")
    if kind == "xref":
        info = int(value.split()[0])
        assert doc.xref_get_key(info, "Type")[0] == "null"
        assert "/Contents" not in doc.xref_object(info, compressed=True)


def test_the_fixture_takes_the_second_save(restores):
    """Guards the fixture: without the second save nothing below is tested."""
    strip_metadata(_document())
    assert any(restores)


def test_stripped_xmp_stays_gone_through_the_second_save(restores):
    out = strip_metadata(_document())
    assert any(restores)
    _assert_references_hold(out, xmp=False)


@pytest.mark.parametrize("matches", [[], [{"page": 0, "rect": {
    "x0": HIT.x0, "y0": HIT.y0, "x1": HIT.x1, "y1": HIT.y1}}]],
    ids=["no-matches", "one-match"])
def test_a_redaction_export_leaves_no_metadata_reference(matches, restores):
    out, _ = apply_redactions(_document(pages=2), matches)
    assert any(restores)
    _assert_references_hold(out, xmp=False)


def test_kept_xmp_still_resolves_to_itself_after_the_second_save(restores):
    """The same renumbering with the XMP kept: the reference written must
    be the stream's new number, not its old one."""
    doc = pymupdf.open(stream=_document(), filetype="pdf")
    out = save_document(doc, **SAVE)
    assert any(restores)
    _assert_references_hold(out, xmp=True)


def test_a_document_saved_again_after_a_strip_keeps_no_reference():
    """A handle-held document: one op strips and saves, a later op saves the
    same open document again. No form is needed for this one."""
    doc = pymupdf.open(stream=_document(), filetype="pdf")
    _strip_metadata_doc(doc)
    _assert_references_hold(save_document(doc, **SAVE), xmp=False)
    _assert_references_hold(save_document(doc, **SAVE), xmp=False)


def test_the_open_document_is_not_given_the_reference_back():
    doc = pymupdf.open(stream=_document(), filetype="pdf")
    _strip_metadata_doc(doc)
    save_document(doc, **SAVE)
    assert doc.xref_get_key(doc.pdf_catalog(), "Metadata")[0] == "null"
