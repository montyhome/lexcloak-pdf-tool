"""The image listing hashes only what a page draws (v0.16.0).

``list_unseen_images`` finds each image draw's object by hashing images
until one matches the draw's pixel digest. Until 0.15.2 it hashed every image
the page's resources name (PyMuPDF's ``get_image_info(xrefs=True)``), so
scans sharing one resource dictionary, which names every scan, made every
page hash every scan in the file: the listing grew with pages times images.
Now the images the page's content draws by name are hashed first, and every
named image only when some draw matches none of them.

The 6-criteria framework:

* **Happy path** -- on scans sharing one dictionary, the listing is the one
  hashing every image gives, and each page decodes only the images it draws.
* **Sad paths** -- a draw no named image matches (an inline image) falls back
  to hashing every image, with the same answer; so does a page whose
  resources are inherited from the page tree, and one whose content does not
  parse.
* **Boundary** -- an image drawn through a form; one drawn twice; a page that
  draws no image.
* **No logic mirroring** -- every answer is compared with PyMuPDF's own
  ``get_image_info(xrefs=True)``, the method the listing used before.
* **Side-effect verification** -- listing leaves the document's bytes as they
  were.
* **Mock integrity** -- documents are built in memory and read through the
  real functions; only ``pymupdf.Pixmap`` is wrapped, to count decodes, and
  it still decodes.

Synthetic fixtures only; every value is invented.
"""
from __future__ import annotations

import pymupdf
import pytest

from lexcloak_pdf_tool import unseen
from lexcloak_pdf_tool.unseen import _xref_boxes, list_unseen_images, list_unseen_images_pdf


def _image(doc, rgb, size=(24, 24)) -> int:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, *size), False)
    pix.set_rect(pix.irect, rgb)
    page = doc.new_page()
    xref = page.insert_image(pymupdf.Rect(0, 0, 10, 10), pixmap=pix)
    doc.delete_page(len(doc) - 1)
    return xref


def _shared_dictionary_scans(n: int, *, inline_on: int | None = None) -> bytes:
    """``n`` pages each drawing its own image from one resource dictionary
    that names all ``n``, plus one image drawn only below the last page's crop
    box. ``inline_on`` adds an inline image to that page's content."""
    doc = pymupdf.open()
    xrefs = [_image(doc, (40 + 7 * i % 200, 90, 160)) for i in range(n)]
    hidden = _image(doc, (200, 30, 30))
    names = " ".join(f"/Im{i} {x} 0 R" for i, x in enumerate(xrefs))
    res = doc.get_new_xref()
    doc.update_object(res, f"<< /XObject << {names} /Hid {hidden} 0 R >> >>")
    for i in range(n):
        page = doc.new_page(width=200, height=200)
        content = f"q 180 0 0 180 10 10 cm /Im{i} Do Q\n"
        if i == n - 1:
            content += "q 40 0 0 20 20 -60 cm /Hid Do Q\n"
        if i == inline_on:
            content += "q 20 0 0 20 150 150 cm BI /W 2 /H 2 /CS /G /BPC 8 ID \x10\x80\xc0\xff EI Q\n"
        cx = doc.get_new_xref()
        doc.update_object(cx, "<<>>")
        doc.update_stream(cx, content.encode("latin-1"))
        doc.xref_set_key(page.xref, "Resources", f"{res} 0 R")
        doc.xref_set_key(page.xref, "Contents", f"{cx} 0 R")
    data = doc.tobytes(garbage=1)
    doc.close()
    return data


def _every_named(page) -> list:
    return [(int(i.get("xref") or 0), tuple(float(v) for v in i["bbox"]))
            for i in page.get_image_info(xrefs=True) if i.get("bbox") is not None]


@pytest.fixture
def decodes(monkeypatch):
    """Count the images ``unseen`` decodes to hash them."""
    count = {"n": 0}
    real = pymupdf.Pixmap

    class Counting(real):
        def __init__(self, *args):
            if len(args) == 2 and isinstance(args[1], int):
                count["n"] += 1
            super().__init__(*args)

    monkeypatch.setattr(unseen._pymupdf, "Pixmap", Counting)
    return count


def test_shared_dictionary_scans_list_what_hashing_every_image_lists(decodes):
    data = _shared_dictionary_scans(30)
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        for pno in range(len(doc)):
            fresh = pymupdf.open(stream=data, filetype="pdf")
            want = _every_named(fresh[pno])
            fresh.close()
            decodes["n"] = 0
            got = _xref_boxes(doc[pno])
            assert got == want, pno
            drawn = 2 if pno == len(doc) - 1 else 1
            assert decodes["n"] == drawn, (pno, decodes["n"])
    finally:
        doc.close()


def test_the_listing_decodes_each_page_once_not_once_per_named_image(decodes):
    data = _shared_dictionary_scans(30)
    before = bytes(data)
    out = list_unseen_images_pdf(data)
    assert data == before
    assert out["unseen"] == [{"page": 29, "state": "outside-crop",
                              "bbox": [20.0, 240.0, 60.0, 260.0], "xref": out["unseen"][0]["xref"],
                              "pixels": 576, "xref_shown": False}]
    assert out["counts"] == {"shown": 30, "partly-outside": 0, "outside-crop": 1,
                             "off-layer": 0}
    # The hidden image's page, and every page whose resources name it: one
    # decode for what each draws, not 31.
    assert decodes["n"] <= 2 * 30 + 2


def test_an_inline_image_falls_back_to_hashing_every_image(decodes):
    data = _shared_dictionary_scans(6, inline_on=2)
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        want = _every_named(pymupdf.open(stream=data, filetype="pdf")[2])
        assert (0, (150.0, 30.0, 170.0, 50.0)) in want
        decodes["n"] = 0
        assert _xref_boxes(doc[2]) == want
        assert decodes["n"] == 1 + 7          # the drawn one, then every named one
    finally:
        doc.close()


def test_inherited_resources_and_unreadable_content_fall_back():
    data = _shared_dictionary_scans(4)
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        res = doc.xref_get_key(doc[1].xref, "Resources")[1]
        pages = doc.xref_get_key(doc[1].xref, "Parent")[1]
        doc.xref_set_key(int(pages.split()[0]), "Resources", res)
        doc.xref_set_key(doc[1].xref, "Resources", "null")
        cx = doc[2].get_contents()[0]
        doc.update_stream(cx, b"q 180 0 0 180 10 10 cm /Im2 Do Q (unterminated")
        data = doc.tobytes()
    finally:
        doc.close()
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        assert unseen._drawn_images(doc, doc[1]) is None
        assert unseen._drawn_images(doc, doc[2]) is None
        for pno in (1, 2):
            want = _every_named(pymupdf.open(stream=data, filetype="pdf")[pno])
            assert _xref_boxes(doc[pno]) == want
    finally:
        doc.close()


def test_an_image_drawn_through_a_form_and_twice():
    doc = pymupdf.open()
    a, b = _image(doc, (10, 120, 10)), _image(doc, (120, 10, 10))
    form = doc.get_new_xref()
    doc.update_object(form, f"<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] "
                            f"/Resources << /XObject << /B {b} 0 R >> >> >>")
    doc.update_stream(form, b"/B Do")
    page = doc.new_page(width=200, height=200)
    cx = doc.get_new_xref()
    doc.update_object(cx, "<<>>")
    doc.update_stream(cx, b"q 50 0 0 50 10 10 cm /A Do Q q 50 0 0 50 80 10 cm /A Do Q "
                          b"q 50 0 0 50 10 100 cm /F Do Q")
    doc.xref_set_key(page.xref, "Resources",
                     f"<< /XObject << /A {a} 0 R /F {form} 0 R >> >>")
    doc.xref_set_key(page.xref, "Contents", f"{cx} 0 R")
    empty = doc.new_page()
    data = doc.tobytes()
    doc.close()
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        assert unseen._drawn_images(doc, doc[0]) == [a, b]
        assert _xref_boxes(doc[0]) == _every_named(doc[0])
        assert sorted(x for x, _ in _xref_boxes(doc[0])) == [a, a, b]
        assert _xref_boxes(doc[1]) == []
        assert empty is not None
    finally:
        doc.close()


def test_the_handle_listing_is_the_bytes_listing():
    data = _shared_dictionary_scans(8)
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        assert list_unseen_images(doc) == list_unseen_images_pdf(data)
    finally:
        doc.close()
