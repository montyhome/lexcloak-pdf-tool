"""Content a PDF carries that the page, as it opens, does not show (v0.14.0).

An image drawn only on a switched-off optional-content layer, or wholly
outside the crop box, shipped intact in a burned file; and a span tagged
``/ActualText`` made text extraction return the tag's text in place of the
glyphs drawn, whatever the tag said. ``list_unseen_images`` and
``list_tag_text`` report both; ``apply_redactions``'s ``blank_images`` and
``strip_tags_pages`` act on what a caller names.

The 6-criteria framework:

* **Happy path** -- each unseen image is reported with its state and, once
  named, its pixels are absent from the delivered bytes; each tag that says
  something the page does not draw is listed with what is drawn under it,
  and once its page is named the tag is gone from the delivered bytes.
* **Sad paths** -- an image that is shown, on a layer that is on, or shown
  elsewhere in the file is reported shown; a tag that matches its glyphs, or
  differs only in case or hyphenation, is not listed; a tag on a page nobody
  named survives byte for byte; a malformed field fails before anything
  changes; an object that is not an image is skipped; an encrypted source is
  refused by both listers.
* **Boundary** -- a rotated page, an image drawn through a form XObject, an
  image placed twice (one placement unseen), a page with no optional
  content, a tag in a property list, in a form XObject, around nothing and
  around a drawing, the handle ops, protocol 12.
* **No logic mirroring** -- every expected state and string is typed out.
* **Side-effect verification** -- assertions read the delivered BYTES (every
  decoded and undecoded stream) and render pages.
* **Mock integrity** -- every PDF is built in memory with PyMuPDF and driven
  through the real entry points and the CLI op table.

Synthetic fixtures only; every value is invented.
"""
from __future__ import annotations

import base64

import pymupdf
import pytest

from lexcloak_pdf_tool import (
    apply_redactions,
    extract_text_plain,
    list_tag_text_pdf,
    list_unseen_images_pdf,
)
from lexcloak_pdf_tool.__main__ import _OPS, PROTOCOL_VERSION, _handle
from lexcloak_pdf_tool.trace import trace_text
from lexcloak_pdf_tool.unseen import validate_blank_images

MARK = "wxmarkerqz"
VISIBLE = "an ordinary paragraph"


# ── builders and readers ─────────────────────────────────────────────────


def _image(rgb=(196, 60, 60)) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 24, 24), False)
    pix.set_rect(pix.irect, rgb)
    return pix.tobytes("png")


def _page(doc):
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 90), VISIBLE, fontsize=12)
    return page


def _bytes(doc) -> bytes:
    data = doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    doc.close()
    return data


def _image_on_layer(on: bool) -> bytes:
    doc = pymupdf.open()
    page = _page(doc)
    oc = doc.add_ocg("scratch", on=on)
    page.insert_image(pymupdf.Rect(60, 200, 108, 248), stream=_image(), oc=oc)
    return _bytes(doc)


def _image_at(rect, crop=None, rotation=0) -> bytes:
    doc = pymupdf.open()
    page = _page(doc)
    page.insert_image(pymupdf.Rect(*rect), stream=_image())
    if crop is not None:
        page.set_cropbox(pymupdf.Rect(*crop))
    if rotation:
        page.set_rotation(rotation)
    return _bytes(doc)


def _states(pdf: bytes) -> dict[str, int]:
    """Placement counts by state, the zero ones left out."""
    return {k: v for k, v in list_unseen_images_pdf(pdf)["counts"].items() if v}


def _unseen(pdf: bytes) -> list[tuple[int, str, bool]]:
    """``(page, state, has an object)`` per unseen placement."""
    return [(u["page"], u["state"], u["xref"] > 0)
            for u in list_unseen_images_pdf(pdf)["unseen"]]


def _pixels_present(pdf: bytes, sample: bytes) -> bool:
    """True if an image stream in ``pdf`` still holds ``sample``'s pixels."""
    want = pymupdf.Pixmap(sample).samples
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        for x in range(1, doc.xref_length()):
            if doc.xref_is_stream(x) and \
                    doc.xref_get_key(x, "Subtype") == ("name", "/Image"):
                try:
                    pix = pymupdf.Pixmap(doc, x)
                except Exception:  # noqa: BLE001
                    continue
                if pix.samples == want:
                    return True
    finally:
        doc.close()
    return False


def _tagged(body: bytes, resources=None, pages: int = 1) -> bytes:
    """A document whose LAST page draws ``body`` after a visible line."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = _page(doc)
    page.insert_font(fontname="helv")
    page.clean_contents()
    if resources is not None:
        resources(doc, page)
    xref = page.get_contents()[0]
    doc.update_stream(xref, doc.xref_stream(xref) + b"\n" + body)
    return _bytes(doc)


def _span(tag: str, drawn: str = "visible words") -> bytes:
    return (b"BT /helv 12 Tf 60 150 Td /Span <</ActualText (" + tag.encode()
            + b")>> BDC (" + drawn.encode() + b") Tj EMC ET")


def _resources(doc, page) -> tuple[int, str]:
    kind, val = doc.xref_get_key(page.xref, "Resources")
    if kind == "xref":
        return int(val.split()[0]), ""
    return page.xref, "Resources/"


def _property_list(doc, page):
    target, prefix = _resources(doc, page)
    doc.xref_set_key(target, prefix + "Properties",
                     f"<</MC0 <</ActualText ({MARK})>>>>")


def _form(doc, page):
    fx = doc.get_new_xref()
    doc.update_object(fx, "<</Type/XObject/Subtype/Form/BBox[0 0 300 50]"
                          "/Resources<</Font<</F1<</Type/Font/Subtype/Type1"
                          "/BaseFont/Helvetica>>>>>>>>")
    doc.update_stream(fx, f"BT /F1 12 Tf 10 20 Td /Span <</ActualText ({MARK})>> "
                          f"BDC (visible words) Tj EMC ET".encode(), new=True)
    target, prefix = _resources(doc, page)
    doc.xref_set_key(target, prefix + "XObject", f"<</Fm1 {fx} 0 R>>")


def _entries(pdf: bytes) -> list[tuple[int, str, str, bool]]:
    return [(e["page"], e["text"], e["drawn"], e["bbox"] is None)
            for e in list_tag_text_pdf(pdf)["entries"]]


def _in_streams(pdf: bytes, needle: str) -> bool:
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        for x in range(1, doc.xref_length()):
            if doc.xref_is_stream(x):
                for blob in (doc.xref_stream(x), doc.xref_stream_raw(x)):
                    if blob and needle.encode() in blob:
                        return True
            elif needle in doc.xref_object(x, compressed=False):
                return True
    finally:
        doc.close()
    return False


# ── images: listing ──────────────────────────────────────────────────────


def test_an_image_on_a_switched_off_layer_is_off_layer():
    pdf = _image_on_layer(on=False)
    assert _states(pdf) == {"off-layer": 1}
    (u,) = list_unseen_images_pdf(pdf)["unseen"]
    assert (u["page"], u["state"], u["xref_shown"], u["pixels"]) == (0, "off-layer", False, 576)
    assert u["bbox"] == [60.0, 200.0, 108.0, 248.0]


def test_an_image_on_a_layer_that_is_on_is_shown():
    pdf = _image_on_layer(on=True)
    assert _states(pdf) == {"shown": 1} and _unseen(pdf) == []


def test_an_image_wholly_outside_the_crop_box_is_outside_crop():
    pdf = _image_at((60, 600, 108, 648), crop=(0, 0, 612, 400))
    assert _states(pdf) == {"outside-crop": 1}
    assert _unseen(pdf) == [(0, "outside-crop", True)]


def test_an_image_straddling_the_crop_edge_is_partly_outside_and_not_unseen():
    pdf = _image_at((60, 380, 108, 428), crop=(0, 0, 612, 400))
    assert _states(pdf) == {"partly-outside": 1} and _unseen(pdf) == []


def test_an_image_inside_the_crop_box_is_shown():
    assert _states(_image_at((60, 200, 108, 248))) == {"shown": 1}


def test_a_rotated_page_reads_the_crop_box_unrotated():
    """Draw-log boxes come unrotated; a rotated page's crop box must be read
    the same way, or an image in view reads as outside it."""
    shown = _image_at((60, 700, 108, 748), rotation=90)
    outside = _image_at((60, 600, 108, 648), crop=(0, 0, 612, 400), rotation=90)
    assert _states(shown) == {"shown": 1}
    assert _states(outside) == {"outside-crop": 1}


def _twice(second: str) -> bytes:
    """One image object drawn in view and again, either inside a
    switched-off optional-content block or outside the crop box."""
    doc = pymupdf.open()
    page = _page(doc)
    xref = page.insert_image(pymupdf.Rect(60, 200, 108, 248), stream=_image())
    name = next(i[7] for i in page.get_images(full=True) if i[0] == xref)
    page.clean_contents()
    content = page.get_contents()[0]
    if second == "off-layer":
        ocg = doc.add_ocg("scratch", on=False)
        target, prefix = _resources(doc, page)
        doc.xref_set_key(target, prefix + "Properties", f"<</MC1 {ocg} 0 R>>")
        draw = f"/OC /MC1 BDC q 48 0 0 48 300 544 cm /{name} Do Q EMC"
    else:
        draw = f"q 48 0 0 48 300 150 cm /{name} Do Q"
        page.set_cropbox(pymupdf.Rect(0, 0, 612, 400))
    doc.update_stream(content, doc.xref_stream(content) + f"\n{draw}\n".encode())
    return _bytes(doc)


@pytest.mark.parametrize("second", ["off-layer", "outside-crop"])
def test_an_image_also_shown_elsewhere_says_so(second):
    pdf = _twice(second)
    assert _states(pdf) == {"shown": 1, second: 1}
    (u,) = list_unseen_images_pdf(pdf)["unseen"]
    assert u["state"] == second and u["xref"] > 0 and u["xref_shown"] is True


def test_an_image_shown_on_another_page_says_so():
    doc = pymupdf.open()
    first = _page(doc)
    xref = first.insert_image(pymupdf.Rect(60, 200, 108, 248), stream=_image())
    second = _page(doc)
    oc = doc.add_ocg("scratch", on=False)
    second.insert_image(pymupdf.Rect(60, 200, 108, 248), xref=xref)
    name = next(i[7] for i in second.get_images(full=True) if i[0] == xref)
    second.clean_contents()
    target, prefix = _resources(doc, second)
    doc.xref_set_key(target, prefix + "Properties", f"<</MC1 {oc} 0 R>>")
    content = second.get_contents()[0]
    doc.update_stream(content, f"/OC /MC1 BDC q 48 0 0 48 300 544 cm /{name} Do Q EMC\n".encode())
    (u,) = list_unseen_images_pdf(_bytes(doc))["unseen"]
    assert (u["page"], u["state"], u["xref_shown"]) == (1, "off-layer", True)


def test_an_image_drawn_through_a_form_on_an_off_layer_is_off_layer():
    src = pymupdf.open()
    sp = src.new_page(width=100, height=100)
    sp.insert_image(sp.rect, stream=_image())
    doc = pymupdf.open()
    page = _page(doc)
    oc = doc.add_ocg("scratch", on=False)
    page.show_pdf_page(pymupdf.Rect(60, 200, 160, 300), src, 0, oc=oc)
    pdf = _bytes(doc)
    assert _states(pdf) == {"off-layer": 1}
    assert _unseen(pdf) == [(0, "off-layer", True)]


def _inline_outside_crop(extra: bytes = b"") -> bytes:
    doc = pymupdf.open()
    page = _page(doc)
    page.clean_contents()
    content = page.get_contents()[0]
    doc.update_stream(content, doc.xref_stream(content) + extra + (
        b"\nq 48 0 0 48 60 150 cm BI /W 2 /H 2 /CS /RGB /BPC 8 ID "
        + INLINE_PIXELS + b" EI Q\n"))
    page.set_cropbox(pymupdf.Rect(0, 0, 612, 400))
    return _bytes(doc)


INLINE_PIXELS = bytes([196, 60, 61, 196, 60, 62, 196, 60, 63, 196, 60, 64])


def test_an_inline_image_outside_the_crop_box_has_no_object():
    pdf = _inline_outside_crop()
    assert _states(pdf) == {"outside-crop": 1}
    assert _unseen(pdf) == [(0, "outside-crop", False)]
    (u,) = list_unseen_images_pdf(pdf)["unseen"]
    assert u["pixels"] == 0 and u["xref_shown"] is False


def test_a_document_without_layers_lists_every_placement_shown():
    doc = pymupdf.open()
    page = _page(doc)
    page.insert_image(pymupdf.Rect(60, 200, 108, 248), stream=_image())
    page.insert_image(pymupdf.Rect(200, 200, 248, 248), stream=_image((10, 10, 200)))
    pdf = _bytes(doc)
    probe = pymupdf.open(stream=pdf)
    assert probe.xref_get_key(probe.pdf_catalog(), "OCProperties")[0] == "null"
    assert list_unseen_images_pdf(pdf) == {
        "unseen": [], "unreadable_pages": [],
        "counts": {"shown": 2, "partly-outside": 0, "outside-crop": 0, "off-layer": 0}}


def test_the_listing_leaves_the_layer_state_alone():
    pdf = _image_on_layer(on=False)
    list_unseen_images_pdf(pdf)
    assert _states(pdf) == {"off-layer": 1}


def test_an_encrypted_source_is_refused():
    doc = pymupdf.open()
    _page(doc)
    pdf = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    with pytest.raises(ValueError):
        list_unseen_images_pdf(pdf)
    with pytest.raises(ValueError):
        list_tag_text_pdf(pdf)


# ── images: blanking and removal ─────────────────────────────────────────


def _xref(pdf: bytes) -> int:
    return list_unseen_images_pdf(pdf)["unseen"][0]["xref"]


def test_a_blanked_image_leaves_no_pixels_in_the_delivered_file():
    pdf = _image_on_layer(on=False)
    assert _pixels_present(pdf, _image())
    out, _ = apply_redactions(pdf, [], blank_images=[_xref(pdf)])
    assert not _pixels_present(out, _image())
    # The stencil paints nothing, so no image with pixels is left unseen.
    assert all(u["pixels"] <= 1 for u in list_unseen_images_pdf(out)["unseen"])


def test_a_blanked_image_paints_nothing_even_with_its_layer_on():
    pdf = _image_on_layer(on=False)
    out, _ = apply_redactions(pdf, [], blank_images=[_xref(pdf)])
    doc = pymupdf.open(stream=out, filetype="pdf")
    doc.xref_set_key(doc.pdf_catalog(), "OCProperties", "null")
    pix = doc[0].get_pixmap(clip=pymupdf.Rect(60, 200, 108, 248))
    assert set(pix.samples) == {255}


def test_an_image_nobody_named_survives():
    pdf = _image_on_layer(on=False)
    out, _ = apply_redactions(pdf, [])
    assert _pixels_present(out, _image())


def test_blanking_reports_and_skips_what_is_not_an_image():
    pdf = _image_on_layer(on=False)
    sink: dict = {}
    page_xref = pymupdf.open(stream=pdf)[0].xref
    apply_redactions(pdf, [], blank_images=[_xref(pdf), page_xref, 99999],
                     unseen_sink=sink)
    assert sink == {"blanked": 1, "removed": 0,
                    "skipped_blank": [page_xref, 99999], "skipped_remove": []}


@pytest.mark.parametrize("bad", [None, "7", [0], [-3], [True], [1.5]])
def test_a_malformed_blank_images_field_fails_before_anything_changes(bad):
    with pytest.raises(ValueError):
        validate_blank_images(bad)
    if bad is not None:
        with pytest.raises(ValueError):
            apply_redactions(_image_on_layer(on=False), [], blank_images=bad)


def test_validate_blank_images_sorts_and_dedupes():
    assert validate_blank_images([9, 4, 9]) == [4, 9]


def _box(pdf: bytes) -> list[float]:
    return list_unseen_images_pdf(pdf)["unseen"][0]["bbox"]


def test_an_inline_image_outside_the_crop_box_is_removed():
    pdf = _inline_outside_crop()
    sink: dict = {}
    out, _ = apply_redactions(pdf, [], remove_images=[{"page": 0, "box": _box(pdf)}],
                              unseen_sink=sink)
    assert sink["removed"] == 1 and sink["skipped_remove"] == []
    assert INLINE_PIXELS not in b"".join(
        pymupdf.open(stream=out).xref_stream(x) for x in pymupdf.open(stream=out)[0].get_contents())
    assert _unseen(out) == []
    assert VISIBLE in pymupdf.open(stream=out)[0].get_text()


def test_an_image_object_outside_the_crop_box_is_removed():
    pdf = _image_at((60, 600, 108, 648), crop=(0, 0, 612, 400))
    out, _ = apply_redactions(pdf, [], remove_images=[{"page": 0, "box": _box(pdf)}])
    assert not _pixels_present(out, _image())
    assert _states(out) == {}


def test_removal_skips_a_box_inside_the_crop_box():
    pdf = _image_at((60, 200, 108, 248))
    sink: dict = {}
    out, _ = apply_redactions(pdf, [], unseen_sink=sink,
                              remove_images=[{"page": 0, "box": [60, 200, 108, 248]}])
    assert sink["removed"] == 0 and len(sink["skipped_remove"]) == 1
    assert _pixels_present(out, _image())


def test_removal_skips_a_box_another_image_reaches_into():
    """Removal takes every image a box touches. A shown image that extends
    past the crop box into the same area must not be taken with it."""
    doc = pymupdf.open()
    page = _page(doc)
    page.insert_image(pymupdf.Rect(40, 300, 140, 660), stream=_image((10, 10, 200)))
    page.insert_image(pymupdf.Rect(60, 600, 108, 648), stream=_image())
    page.set_cropbox(pymupdf.Rect(0, 0, 612, 400))
    pdf = _bytes(doc)
    sink: dict = {}
    out, _ = apply_redactions(pdf, [], unseen_sink=sink,
                              remove_images=[{"page": 0, "box": [60, 600, 108, 648]}])
    assert sink["removed"] == 0
    assert _pixels_present(out, _image((10, 10, 200)))


def test_removal_reads_the_box_unrotated_on_a_rotated_page():
    pdf = _image_at((60, 600, 108, 648), crop=(0, 0, 612, 400), rotation=90)
    out, _ = apply_redactions(pdf, [], remove_images=[{"page": 0, "box": _box(pdf)}])
    assert not _pixels_present(out, _image())


@pytest.mark.parametrize("bad", [
    "x", [{"page": 0}], [{"page": -1, "box": [0, 0, 1, 1]}],
    [{"page": 0, "box": [0, 0, 1]}], [{"page": 0, "box": [5, 5, 1, 1]}],
    [{"page": 0, "box": [0, 0, 1, 1], "extra": 1}], [{"page": True, "box": [0, 0, 1, 1]}],
])
def test_a_malformed_remove_images_field_fails(bad):
    with pytest.raises(ValueError):
        apply_redactions(_inline_outside_crop(), [], remove_images=bad)


# ── tagged text: listing ─────────────────────────────────────────────────


def test_a_tag_that_differs_from_its_glyphs_is_listed_with_them():
    assert _entries(_tagged(_span(MARK))) == [(0, MARK, "visible words", False)]


def test_a_tag_in_a_property_list_is_listed():
    pdf = _tagged(b"BT /helv 12 Tf 60 150 Td /Span /MC0 BDC (visible words) Tj EMC ET",
                  _property_list)
    assert _entries(pdf) == [(0, MARK, "visible words", False)]


def test_a_tag_inside_a_form_xobject_is_listed():
    pdf = _tagged(b"q 1 0 0 1 60 600 cm /Fm1 Do Q", _form)
    assert _entries(pdf) == [(0, MARK, "visible words", False)]


@pytest.mark.parametrize("body", [
    b"BT /helv 12 Tf 60 150 Td /Span <</ActualText (" + MARK.encode() + b")>> BDC EMC ET",
    b"/Span <</ActualText (" + MARK.encode() + b")>> BDC EMC",
    b"/Figure <</ActualText (" + MARK.encode() + b")>> BDC 0 0 1 rg 60 600 100 40 re f EMC",
], ids=["empty-span", "outside-text-object", "around-a-drawing"])
def test_a_tag_wrapping_no_glyphs_is_listed_without_a_box(body):
    assert _entries(_tagged(body)) == [(0, MARK, "", True)]


def test_a_hex_tag_wrapping_no_glyphs_is_decoded():
    utf16 = ("﻿" + MARK).encode("utf-16-be").hex().encode()
    pdf = _tagged(b"/Span <</ActualText <" + utf16 + b">>> BDC EMC")
    assert _entries(pdf) == [(0, MARK, "", True)]


@pytest.mark.parametrize("tag, drawn", [
    ("visible words", "visible words"),
    ("", "-"),
    (" ", "visible words"),
], ids=["matching", "drops-a-hyphen", "says-nothing-readable"])
def test_a_tag_that_says_what_is_drawn_is_not_listed(tag, drawn):
    assert _entries(_tagged(_span(tag, drawn))) == []


def test_a_case_only_tag_is_listed_as_it_is_for_the_caller_to_judge():
    assert _entries(_tagged(_span("VISIBLE WORDS"))) == \
        [(0, "VISIBLE WORDS", "visible words", False)]


def test_a_listed_box_is_in_the_rotated_page_frame():
    """Boxes are in the frame ``trace_text`` uses, so a caller can index a
    render with either: the entry's box holds the traced glyphs' box."""
    doc = pymupdf.open(stream=_tagged(_span(MARK)))
    doc[0].set_rotation(90)
    rotated = doc.tobytes()
    (entry,) = list_tag_text_pdf(rotated)["entries"]
    word = next(w for w in trace_text(rotated, 0)["words"] if w["text"] == "visible")
    cx, cy = (word["bbox"][0] + word["bbox"][2]) / 2, (word["bbox"][1] + word["bbox"][3]) / 2
    x0, y0, x1, y1 = entry["bbox"]
    assert x0 <= cx <= x1 and y0 <= cy <= y1
    assert y1 - y0 > x1 - x0      # turned a quarter: the run now reads down


def test_a_page_without_tags_has_no_entries():
    doc = pymupdf.open()
    _page(doc)
    assert list_tag_text_pdf(_bytes(doc)) == {"entries": [], "unreadable_pages": []}


# ── tagged text: removal and the drawn reading ───────────────────────────


def test_strip_tags_pages_removes_the_named_page_tags():
    pdf = _tagged(_span(MARK), pages=2)
    out, _ = apply_redactions(pdf, [], strip_tags_pages=[1])
    assert not _in_streams(out, MARK)
    assert list_tag_text_pdf(out)["entries"] == []
    assert "visible words" in pymupdf.open(stream=out)[1].get_text()


def test_a_benign_tag_on_a_page_nobody_named_survives():
    """Removing every tag would cost copy and paste fidelity everywhere; a
    tag on a page the caller did not name stays."""
    doc = pymupdf.open()
    for body in (_span("visible words"), _span(MARK)):
        page = _page(doc)
        page.insert_font(fontname="helv")
        page.clean_contents()
        xref = page.get_contents()[0]
        doc.update_stream(xref, doc.xref_stream(xref) + b"\n" + body)
    pdf = _bytes(doc)
    out, _ = apply_redactions(pdf, [], strip_tags_pages=[1])
    kept = pymupdf.open(stream=out)
    first = b"".join(kept.xref_stream(x) for x in kept[0].get_contents())
    assert b"/ActualText" in first
    assert not _in_streams(out, MARK)


@pytest.mark.parametrize("bad", ["1", [-1], [True], [0.5]])
def test_a_malformed_strip_tags_pages_field_fails(bad):
    with pytest.raises(ValueError):
        apply_redactions(_tagged(_span(MARK)), [], strip_tags_pages=bad)


def test_drawn_only_reads_the_glyphs_not_the_tag():
    pdf = _tagged(_span(MARK))
    assert MARK in extract_text_plain(pdf, 0)
    drawn = extract_text_plain(pdf, 0, drawn_only=True)
    assert MARK not in drawn and "visible words" in drawn


# ── CLI ──────────────────────────────────────────────────────────────────


def _cmd(op: str, **kw) -> dict:
    return _handle({"protocol_version": PROTOCOL_VERSION, "op": op, **kw})


def test_protocol_12_carries_the_new_ops():
    assert PROTOCOL_VERSION == 12
    for op in ("list_unseen_images", "list_unseen_images_h",
               "list_tag_text", "list_tag_text_h"):
        assert op in _OPS


def test_the_cli_ops_list_blank_and_strip():
    images = _image_on_layer(on=False)
    b64 = base64.b64encode(images).decode()
    listed = _cmd("list_unseen_images", pdf_b64=b64)
    assert listed["ok"]
    xref = listed["result"]["unseen"][0]["xref"]
    resp = _cmd("apply_redactions", pdf_b64=b64, matches=[], blank_images=[xref])
    assert resp["ok"] and resp["result"]["unseen_images"] == {
        "blanked": 1, "removed": 0, "skipped_blank": [], "skipped_remove": []}
    assert not _pixels_present(base64.b64decode(resp["result"]["pdf_b64"]), _image())

    tagged = base64.b64encode(_tagged(_span(MARK))).decode()
    listed = _cmd("list_tag_text", pdf_b64=tagged)
    assert listed["ok"] and listed["result"]["entries"][0]["text"] == MARK
    resp = _cmd("apply_redactions", pdf_b64=tagged, matches=[], strip_tags_pages=[0])
    assert resp["ok"] and not _in_streams(base64.b64decode(resp["result"]["pdf_b64"]), MARK)
    text = _cmd("extract_text_plain", pdf_b64=tagged, page=0, drawn_only=True)
    assert text["ok"] and MARK not in text["result"]["text"]


def test_the_cli_answers_blank_images_only_when_the_request_carried_it():
    b64 = base64.b64encode(_image_on_layer(on=False)).decode()
    resp = _cmd("apply_redactions", pdf_b64=b64, matches=[])
    assert resp["ok"] and "unseen_images" not in resp["result"]


def test_drawn_only_must_be_a_boolean():
    b64 = base64.b64encode(_tagged(_span(MARK))).decode()
    assert not _cmd("extract_text_plain", pdf_b64=b64, page=0, drawn_only="yes")["ok"]


def test_the_handle_ops_list_blank_and_strip():
    pdf = _image_on_layer(on=False)
    handle = _OPS["open_doc"]({"pdf_b64": base64.b64encode(pdf).decode()})["handle"]
    try:
        listed = _OPS["list_unseen_images_h"]({"handle": handle})
        assert [u["state"] for u in listed["unseen"]] == ["off-layer"]
        result = _OPS["apply_redactions_h"]({
            "handle": handle, "matches": [],
            "blank_images": [listed["unseen"][0]["xref"]]})
        assert result["unseen_images"]["blanked"] == 1
    finally:
        _OPS["close_doc"]({"handle": handle})
    tagged = _tagged(_span(MARK))
    handle = _OPS["open_doc"]({"pdf_b64": base64.b64encode(tagged).decode()})["handle"]
    try:
        assert _OPS["list_tag_text_h"]({"handle": handle})["entries"][0]["text"] == MARK
        drawn = _OPS["extract_text_plain_h"]({"handle": handle, "page": 0,
                                              "drawn_only": True})
        assert MARK not in drawn["text"]
    finally:
        _OPS["close_doc"]({"handle": handle})


def test_a_document_whose_layers_cannot_be_read_is_unchecked_not_clear(monkeypatch):
    """A catalog MuPDF cannot read (measured on a real file: an assertion
    inside ``xref_set_key``) must leave every page unchecked."""
    from lexcloak_pdf_tool import unseen

    def boom(_bytes):
        raise AssertionError("unreadable catalog")

    monkeypatch.setattr(unseen, "_every_layer_view", boom)
    pdf = _image_on_layer(on=False)
    assert list_unseen_images_pdf(pdf) == {
        "unseen": [], "unreadable_pages": [0],
        "counts": {"shown": 0, "partly-outside": 0, "outside-crop": 0, "off-layer": 0}}
    sink: dict = {}
    apply_redactions(pdf, [], unseen_sink=sink,
                     remove_images=[{"page": 0, "box": [60, 600, 108, 648]}])
    assert sink["removed"] == 0 and len(sink["skipped_remove"]) == 1


# ── gaps the mutation check found ────────────────────────────────────────


def test_an_image_mostly_inside_the_crop_box_is_partly_outside():
    pdf = _image_at((60, 360, 108, 405), crop=(0, 0, 612, 400))
    assert _states(pdf) == {"partly-outside": 1}


def test_a_placement_outside_the_crop_box_elsewhere_does_not_count_as_shown():
    """``xref_shown`` means a reader sees the object somewhere. A second
    placement that is itself outside the crop box is not that."""
    doc = pymupdf.open()
    first = _page(doc)
    xref = first.insert_image(pymupdf.Rect(60, 600, 108, 648), stream=_image())
    first.set_cropbox(pymupdf.Rect(0, 0, 612, 400))
    second = _page(doc)
    second.insert_image(pymupdf.Rect(60, 600, 108, 648), xref=xref)
    second.set_cropbox(pymupdf.Rect(0, 0, 612, 400))
    unseen = list_unseen_images_pdf(_bytes(doc))["unseen"]
    assert [(u["page"], u["xref_shown"]) for u in unseen] == [(0, False), (1, False)]


def test_removal_skips_a_box_with_no_image_in_it():
    pdf = _inline_outside_crop()
    sink: dict = {}
    apply_redactions(pdf, [], unseen_sink=sink,
                     remove_images=[{"page": 0, "box": [300, 600, 340, 640]}])
    assert sink["removed"] == 0 and len(sink["skipped_remove"]) == 1


def _glyphless_props(doc, page):
    target, prefix = _resources(doc, page)
    doc.xref_set_key(target, prefix + "Properties",
                     f"<</MC0 <</ActualText ({MARK})>>>>")


def _glyphless_form(doc, page):
    fx = doc.get_new_xref()
    doc.update_object(fx, "<</Type/XObject/Subtype/Form/BBox[0 0 300 50]>>")
    doc.update_stream(fx, f"/Span <</ActualText ({MARK})>> BDC EMC".encode(), new=True)
    target, prefix = _resources(doc, page)
    doc.xref_set_key(target, prefix + "XObject", f"<</Fm1 {fx} 0 R>>")


def test_a_glyphless_tag_in_a_property_list_is_listed():
    pdf = _tagged(b"/Span /MC0 BDC EMC", _glyphless_props)
    assert _entries(pdf) == [(0, MARK, "", True)]


def test_a_glyphless_tag_inside_a_form_xobject_is_listed():
    pdf = _tagged(b"q 1 0 0 1 60 600 cm /Fm1 Do Q", _glyphless_form)
    assert _entries(pdf) == [(0, MARK, "", True)]
