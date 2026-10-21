"""Both listings read what the page shows (v0.15.2).

Two gaps in 0.14.0 to 0.15.1, each letting content no reader sees leave in
a delivered file:

* **A crop box that reaches past the media box.** A page shows its crop box
  clipped to its media box, and the draw log's origin is that area's corner,
  but placements were measured against the crop box's own width and height.
  An image beside or below the shown area read as ``shown``, so it was never
  blanked or removed.
* **A tag around an image.** MuPDF's extraction returns the text of an
  ``/ActualText`` tag that wraps an image and no glyphs. The line comparison
  reads characters and the tag has none, and the direct read skipped any tag
  whose letters extraction returns, so the tag was never listed.

The 6-criteria framework:

* **Happy path** -- an image in the area a crop box past the media box does
  not show is ``outside-crop`` on each of the four sides and under
  ``/Rotate`` 90 and 270, and is removed or blanked; a tag around an image
  is listed with ``drawn`` ``""`` and no box.
* **Sad paths** -- an image inside the shown area is ``shown``; a tag around
  an image whose word the page also draws is still listed; one with no letter
  or digit is not; a tag around glyphs and an image, a ligature tag and a
  hyphen-only tag add nothing; a tag the lines list is not listed twice.
* **Boundary** -- no crop box; a crop box with no media box; a crop box that
  misses the media box; a shown area of one point and of less; ``/UserUnit``;
  a crop box inherited from the page tree; an image partly outside, inline,
  and in a form; a tag around a drawing, around nothing, in a property list,
  in a form, and around nested forms.
* **No logic mirroring** -- every state, box, count and string is typed out
  from the construction.
* **Side-effect verification** -- listing leaves an open document's bytes as
  they were; blanking one image leaves every other image's stream byte for
  byte; stripping one page's tags leaves the other page's.
* **Mock integrity** -- every PDF is built in memory with PyMuPDF and read
  through the real entry points; a page whose content cannot be parsed is
  unreadable, never empty.

Synthetic fixtures only; every value is invented.
"""
from __future__ import annotations

import base64

import pymupdf
import pytest

from lexcloak_pdf_tool import apply_redactions, list_tag_text_pdf, list_unseen_images_pdf
from lexcloak_pdf_tool.__main__ import _OPS
from lexcloak_pdf_tool.unseen import list_tag_text, list_unseen_images

MARK = "wxshownqz"
LINE = "an ordinary line"
PIXELS = 576          # the 24 x 24 test image


# ── builders ─────────────────────────────────────────────────────────────


def _image(rgb=(196, 60, 60)) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 24, 24), False)
    pix.set_rect(pix.irect, rgb)
    return pix.tobytes("png")


def _draw(rect, name: str) -> str:
    x0, y0, x1, y1 = rect
    return f"q {x1 - x0:g} 0 0 {y1 - y0:g} {x0:g} {y0:g} cm /{name} Do Q"


def _boxed(draws, crop=None, media="[0 0 612 792]", rotate=0, unit=None,
           rgbs=((196, 60, 60),)) -> tuple[bytes, list[int]]:
    """One page drawing each image in ``rgbs`` at the PDF-space rects
    ``draws[i]`` lists for it, with its boxes written as given (``None``
    leaves a box out). Returns the bytes and each image's object number."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    xrefs, names = [], []
    for rgb in rgbs:
        xref = page.insert_image(pymupdf.Rect(0, 0, 10, 10), stream=_image(rgb))
        xrefs.append(xref)
        names.append(next(i[7] for i in page.get_images(full=True) if i[0] == xref))
    page.clean_contents()
    content = "\n".join(_draw(rect, names[i]) for i, rects in enumerate(draws)
                        for rect in rects)
    doc.update_stream(page.get_contents()[0], content.encode())
    doc.xref_set_key(page.xref, "MediaBox", "null" if media is None else media)
    if crop is not None:
        doc.xref_set_key(page.xref, "CropBox", crop)
    if rotate:
        doc.xref_set_key(page.xref, "Rotate", str(rotate))
    if unit is not None:
        doc.xref_set_key(page.xref, "UserUnit", str(unit))
    data = doc.tobytes(no_new_id=True)
    doc.close()
    return data, xrefs


def _listing(*args, **kwargs) -> tuple[dict, list[int]]:
    pdf, xrefs = _boxed(*args, **kwargs)
    return list_unseen_images_pdf(pdf), xrefs


def _counts(**kw) -> dict:
    out = {"shown": 0, "partly-outside": 0, "outside-crop": 0, "off-layer": 0}
    out.update({k.replace("_", "-"): v for k, v in kw.items()})
    return out


def _outside(bbox, xref, shown=False) -> dict:
    return {"page": 0, "state": "outside-crop", "bbox": bbox, "xref": xref,
            "pixels": PIXELS, "xref_shown": shown}


def _pixels_present(pdf: bytes, rgb) -> bool:
    want = pymupdf.Pixmap(_image(rgb)).samples
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        for x in range(1, doc.xref_length()):
            if doc.xref_is_stream(x) and \
                    doc.xref_get_key(x, "Subtype") == ("name", "/Image"):
                try:
                    if pymupdf.Pixmap(doc, x).samples == want:
                        return True
                except Exception:  # noqa: BLE001
                    continue
    finally:
        doc.close()
    return False


# ── a crop box past the media box ────────────────────────────────────────

#: Each side: the crop box, a PDF-space rect the page does not show (inside
#: the crop box where the crop box passes the media box, or beyond the shown
#: area by less than the crop box overshoots it), and where the draw log puts
#: it: unrotated, y down, origin at the shown area's top-left corner.
SIDES = {
    # shown (50, 100)-(400, 792): the crop box's 108 points above the paper
    # moved the origin down, so its full height reached 108 points too low.
    "top": ("[50 100 400 900]", (100, 30, 140, 60), [50.0, 732.0, 90.0, 762.0]),
    # shown (50, 0)-(400, 700): the image is in the crop box, below the paper.
    "bottom": ("[50 -100 400 700]", (100, -60, 140, -30), [50.0, 730.0, 90.0, 760.0]),
    # shown (0, 100)-(400, 700): 100 points of crop box left of the paper.
    "left": ("[-100 100 400 700]", (450, 300, 490, 330), [450.0, 370.0, 490.0, 400.0]),
    # shown (50, 100)-(612, 700): the image is in the crop box, right of the paper.
    "right": ("[50 100 800 700]", (650, 300, 690, 330), [600.0, 370.0, 640.0, 400.0]),
}


@pytest.mark.parametrize("side", sorted(SIDES))
def test_an_image_where_a_crop_box_passes_the_media_box_is_outside_crop(side):
    crop, rect, bbox = SIDES[side]
    listing, (xref,) = _listing([[rect]], crop=crop)
    assert listing == {"unseen": [_outside(bbox, xref)],
                       "counts": _counts(outside_crop=1), "unreadable_pages": []}


@pytest.mark.parametrize("side", sorted(SIDES))
def test_an_image_inside_the_clipped_area_is_shown(side):
    crop, _, _ = SIDES[side]
    listing, _ = _listing([[(100, 400, 140, 430)]], crop=crop)
    assert listing == {"unseen": [], "counts": _counts(shown=1), "unreadable_pages": []}


@pytest.mark.parametrize("rotate", [90, 270])
@pytest.mark.parametrize("side", sorted(SIDES))
def test_a_turned_page_reads_the_clipped_area_unrotated(side, rotate):
    crop, rect, bbox = SIDES[side]
    listing, (xref,) = _listing([[rect, (100, 400, 140, 430)]], crop=crop, rotate=rotate)
    assert listing == {"unseen": [_outside(bbox, xref, shown=True)],
                       "counts": _counts(shown=1, outside_crop=1), "unreadable_pages": []}


def test_an_image_across_the_clipped_edge_is_partly_outside():
    """Shown area (50, 100)-(400, 792): the image straddles its bottom edge,
    which the unclipped height put 108 points lower."""
    listing, _ = _listing([[(100, 80, 140, 120)]], crop="[50 100 400 900]")
    assert listing == {"unseen": [], "counts": _counts(partly_outside=1),
                       "unreadable_pages": []}


def test_the_same_object_shown_and_hidden_says_it_is_shown():
    listing, (xref,) = _listing([[(100, 30, 140, 60), (100, 400, 140, 430)]],
                                crop="[50 100 400 900]")
    assert listing["unseen"] == [_outside([50.0, 732.0, 90.0, 762.0], xref, shown=True)]
    assert listing["counts"] == _counts(shown=1, outside_crop=1)


def test_user_unit_scales_the_shown_area_with_the_draw_log():
    listing, (xref,) = _listing([[(100, 30, 140, 60), (100, 400, 140, 430)]],
                                crop="[50 100 400 900]", unit=2)
    assert listing["unseen"] == [_outside([100.0, 1464.0, 180.0, 1524.0], xref, shown=True)]
    assert listing["counts"] == _counts(shown=1, outside_crop=1)


def test_a_crop_box_inherited_from_the_page_tree_is_clipped_too():
    doc = pymupdf.open(stream=_boxed([[(100, 30, 140, 60)]])[0])
    pages = int(doc.xref_get_key(doc.pdf_catalog(), "Pages")[1].split()[0])
    doc.xref_set_key(pages, "CropBox", "[50 100 400 900]")
    assert doc.xref_get_key(doc[0].xref, "CropBox")[0] == "null"
    listing = list_unseen_images(doc)
    assert [u["state"] for u in listing["unseen"]] == ["outside-crop"]
    assert listing["unseen"][0]["bbox"] == [50.0, 732.0, 90.0, 762.0]


def _inline_and_form(crop: str) -> bytes:
    """An inline image and an image drawn through a form, both at PDF
    (100, 30)-(140, 60), below the area a crop box past the paper shows."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    src = pymupdf.open()
    sp = src.new_page(width=40, height=30)
    sp.insert_image(sp.rect, stream=_image())
    page.show_pdf_page(pymupdf.Rect(300, 732, 340, 762), src, 0)
    page.clean_contents()
    content = page.get_contents()[0]
    doc.update_stream(content, doc.xref_stream(content) + (
        b"\nq 40 0 0 30 100 30 cm BI /W 2 /H 2 /CS /RGB /BPC 8 ID "
        + bytes([196, 60, 61] * 4) + b" EI Q\n"))
    doc.xref_set_key(page.xref, "CropBox", crop)
    return doc.tobytes(no_new_id=True)


def test_an_inline_image_and_a_form_image_below_the_clipped_area_are_outside_crop():
    listing = list_unseen_images_pdf(_inline_and_form("[50 100 400 900]"))
    assert listing["counts"] == _counts(outside_crop=2)
    assert sorted((u["bbox"], u["xref"] > 0) for u in listing["unseen"]) == [
        ([50.0, 732.0, 90.0, 762.0], False), ([250.0, 732.0, 290.0, 762.0], True)]


# ── what a page shows, at the edges ──────────────────────────────────────


def test_no_crop_box_shows_the_media_box():
    listing, (xref,) = _listing([[(100, 400, 140, 430), (700, 400, 740, 430)]])
    assert listing == {"unseen": [_outside([700.0, 362.0, 740.0, 392.0], xref, shown=True)],
                       "counts": _counts(shown=1, outside_crop=1), "unreadable_pages": []}


@pytest.mark.parametrize("media", [None, "[0 0 0 0]", "[0 0 612 0]", "7"],
                         ids=["none", "no-area", "no-height", "not-an-array"])
def test_a_crop_box_with_no_usable_media_box_is_clipped_to_us_letter(media):
    """No media box, or one with no area, reads as US Letter; the page shows
    the crop box within it. MuPDF lays the page out on all of Letter, so
    boxes are measured from Letter's top-left corner: one image sits just
    inside the crop box's top edge, 92 points below Letter's, and the last
    one left of the crop box, on Letter."""
    listing, (xref,) = _listing(
        [[(100, 400, 140, 430), (100, 30, 140, 60), (100, 694, 140, 699),
          (10, 400, 40, 430)]],
        crop="[50 100 400 700]", media=media)
    assert listing == {"unseen": [_outside([100.0, 732.0, 140.0, 762.0], xref, shown=True),
                                  _outside([10.0, 362.0, 40.0, 392.0], xref, shown=True)],
                       "counts": _counts(shown=2, outside_crop=2), "unreadable_pages": []}


def test_no_usable_media_box_and_no_crop_box_shows_us_letter():
    listing, (xref,) = _listing([[(100, 400, 140, 430), (650, 400, 690, 430)]],
                                media="[0 0 0 0]")
    assert listing == {"unseen": [_outside([650.0, 362.0, 690.0, 392.0], xref, shown=True)],
                       "counts": _counts(shown=1, outside_crop=1), "unreadable_pages": []}


@pytest.mark.parametrize("crop", ["5", "/None"])
def test_a_crop_box_that_is_not_an_array_reads_as_none(crop):
    """MuPDF and PDFium both show the media box then."""
    listing, _ = _listing([[(100, 400, 140, 430), (100, 30, 140, 60)]], crop=crop)
    assert listing == {"unseen": [], "counts": _counts(shown=2), "unreadable_pages": []}


def test_an_offset_media_box_is_measured_from_the_clipped_corner():
    listing, (xref,) = _listing([[(150, 110, 190, 140), (150, 300, 190, 330)]],
                                media="[100 100 712 892]", crop="[90 150 500 1000]")
    # shown (100, 150)-(500, 892): the first image is below it, on the paper.
    assert listing == {"unseen": [_outside([50.0, 752.0, 90.0, 782.0], xref, shown=True)],
                       "counts": _counts(shown=1, outside_crop=1), "unreadable_pages": []}


@pytest.mark.parametrize("crop", ["[700 900 900 1100]", "[612 100 700 700]",
                                  "[611.5 100 700 700]", "[50 100 400 100]"],
                         ids=["misses", "touches", "half-a-point", "no-height"])
def test_a_page_that_shows_nothing_has_every_placement_outside_crop(crop):
    """MuPDF draws nothing on any of these pages, and PDFium gives a page
    whose crop box misses its media box no area. PDFium shows the media box
    for a crop box with no height; read as nothing shown, the image is
    blanked rather than shipped unseen in the view MuPDF gives."""
    pdf, (xref,) = _boxed([[(100, 400, 140, 430), (0, 0, 612, 792)]], crop=crop)
    listing = list_unseen_images_pdf(pdf)
    assert listing["counts"] == _counts(outside_crop=2)
    assert [(u["state"], u["xref"], u["xref_shown"]) for u in listing["unseen"]] == \
        [("outside-crop", xref, False)] * 2
    assert listing["unreadable_pages"] == []


def test_a_one_point_shown_area_shows_what_it_reaches():
    listing, _ = _listing([[(0, 0, 612, 792)]], crop="[611 100 700 700]")
    assert listing["counts"] == _counts(partly_outside=1)


# ── removing and blanking what the clipped area hides ────────────────────


@pytest.mark.parametrize("rotate", [0, 90, 270])
@pytest.mark.parametrize("side", sorted(SIDES))
def test_removal_takes_an_image_the_clipped_area_hides(side, rotate):
    crop, rect, bbox = SIDES[side]
    pdf, _ = _boxed([[rect], [(100, 400, 140, 430)]], crop=crop, rotate=rotate,
                    rgbs=((196, 60, 60), (10, 10, 200)))
    sink: dict = {}
    out, _ = apply_redactions(pdf, [], unseen_sink=sink,
                              remove_images=[{"page": 0, "box": bbox}])
    assert sink == {"blanked": 0, "removed": 1, "skipped_blank": [], "skipped_remove": []}
    assert not _pixels_present(out, (196, 60, 60))
    assert _pixels_present(out, (10, 10, 200))
    assert list_unseen_images_pdf(out)["counts"] == _counts(shown=1)


def test_removal_skips_a_box_inside_the_clipped_area():
    pdf, _ = _boxed([[(100, 400, 140, 430)]], crop="[50 100 400 900]")
    sink: dict = {}
    out, _ = apply_redactions(pdf, [], unseen_sink=sink,
                              remove_images=[{"page": 0, "box": [50.0, 362.0, 90.0, 392.0]}])
    assert sink["removed"] == 0 and sink["skipped_remove"] == [
        {"page": 0, "box": [50.0, 362.0, 90.0, 392.0]}]
    assert _pixels_present(out, (196, 60, 60))


def test_blanking_the_hidden_image_leaves_every_other_image_byte_for_byte():
    pdf, (hidden, _, _) = _boxed(
        [[(100, 30, 140, 60)], [(100, 400, 140, 430)], [(200, 400, 240, 430)]],
        crop="[50 100 400 900]", rgbs=((196, 60, 60), (10, 10, 200), (20, 160, 20)))
    listing = list_unseen_images_pdf(pdf)
    assert [u["xref"] for u in listing["unseen"]] == [hidden]
    # The burn's save rewrites every stream, so the comparison is with the
    # same burn naming nothing.
    plain, _ = apply_redactions(pdf, [])
    out, _ = apply_redactions(pdf, [], blank_images=[hidden])
    before, after = _image_objects(plain), _image_objects(out)
    assert sorted(before) == sorted(after) and len(after) == 3
    changed = [x for x in after if after[x] != before[x]]
    assert len(changed) == 1
    assert _pixels_present(plain, (196, 60, 60)) and not _pixels_present(out, (196, 60, 60))
    assert _pixels_present(out, (10, 10, 200)) and _pixels_present(out, (20, 160, 20))


def _image_objects(pdf: bytes) -> dict[int, tuple]:
    """Each image object's undecoded stream and dictionary, by number."""
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        return {x: (doc.xref_stream_raw(x), doc.xref_object(x, compressed=True))
                for x in range(1, doc.xref_length())
                if doc.xref_is_stream(x)
                and doc.xref_get_key(x, "Subtype") == ("name", "/Image")}
    finally:
        doc.close()


# ── tags around no glyph ─────────────────────────────────────────────────


def _tag_page(body: bytes, extra=None, pages: int = 1) -> bytes:
    """Pages drawing a visible line; the LAST page then draws ``body``, with
    an image ``/Im`` and the font ``/helv`` in its resources."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text((60, 90), LINE, fontsize=12)
    xref = page.insert_image(pymupdf.Rect(0, 0, 10, 10), stream=_image())
    name = next(i[7] for i in page.get_images(full=True) if i[0] == xref)
    page.insert_font(fontname="helv")
    page.clean_contents()
    content = page.get_contents()[0]
    kept = b"\n".join(line for line in doc.xref_stream(content).split(b"\n")
                      if b" Do" not in line)
    if extra is not None:
        extra(doc, page)
    doc.update_stream(content, kept + b"\n" + body.replace(b"/Im ", f"/{name} ".encode()))
    data = doc.tobytes(no_new_id=True)
    doc.close()
    return data


IMG = b"q 48 0 0 48 60 300 cm /Im Do Q"


def _around(inner: bytes, tag: str = MARK) -> bytes:
    return b"/Span <</ActualText (" + tag.encode() + b")>> BDC " + inner + b" EMC"


def _glyphless(text: str = MARK) -> dict:
    return {"page": 0, "text": text, "drawn": "", "bbox": None}


def _entries(pdf: bytes) -> list[tuple[int, str, str, bool]]:
    return [(e["page"], e["text"], e["drawn"], e["bbox"] is None)
            for e in list_tag_text_pdf(pdf)["entries"]]


def test_a_tag_around_an_image_is_listed_though_extraction_returns_it():
    pdf = _tag_page(_around(IMG))
    assert MARK in pymupdf.open(stream=pdf)[0].get_text()
    assert list_tag_text_pdf(pdf) == {"entries": [_glyphless()], "unreadable_pages": []}


def test_a_tag_around_an_inline_image_is_listed():
    inline = b"q 48 0 0 48 60 300 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \x80 EI Q"
    assert list_tag_text_pdf(_tag_page(_around(inline))) == {
        "entries": [_glyphless()], "unreadable_pages": []}


def test_a_tag_around_an_image_inside_a_text_object_is_listed():
    body = b"BT /helv 12 Tf 60 150 Td " + _around(b"ET " + IMG + b" BT") + b" ET"
    assert list_tag_text_pdf(_tag_page(body)) == {
        "entries": [_glyphless()], "unreadable_pages": []}


def test_a_tag_around_an_image_whose_word_the_page_draws_is_still_listed():
    body = b"BT /helv 12 Tf 60 150 Td (" + MARK.encode() + b") Tj ET " + _around(IMG)
    assert list_tag_text_pdf(_tag_page(body)) == {
        "entries": [_glyphless()], "unreadable_pages": []}


def test_a_tag_around_a_drawing_whose_word_the_page_draws_is_listed():
    body = (b"BT /helv 12 Tf 60 150 Td (" + MARK.encode() + b") Tj ET "
            + _around(b"0 0 1 rg 60 600 100 40 re f"))
    assert list_tag_text_pdf(_tag_page(body)) == {
        "entries": [_glyphless()], "unreadable_pages": []}


@pytest.mark.parametrize("tag", ["- ", "•", "()"], ids=["hyphen", "bullet", "brackets"])
def test_a_tag_around_an_image_with_no_letter_or_digit_is_not_listed(tag):
    body = b"/Span <</ActualText (" + tag.replace("(", "\\(").replace(")", "\\)").encode(
        "latin-1", "replace") + b")>> BDC " + IMG + b" EMC"
    assert list_tag_text_pdf(_tag_page(body)) == {"entries": [], "unreadable_pages": []}


def test_a_tag_around_glyphs_and_an_image_is_listed_by_its_line_only():
    body = (b"BT /helv 12 Tf 60 150 Td " + _around(b"(visible words) Tj ET " + IMG + b" BT")
            + b" ET")
    assert _entries(_tag_page(body)) == [(0, MARK, "visible words", False)]


def test_a_tag_that_repeats_its_glyphs_beside_a_tag_around_an_image():
    body = (b"BT /helv 12 Tf 60 150 Td " + _around(b"(visible words) Tj", "visible words")
            + b" ET " + _around(IMG))
    assert list_tag_text_pdf(_tag_page(body)) == {
        "entries": [_glyphless()], "unreadable_pages": []}


def test_a_diverging_tag_and_a_tag_around_an_image_are_one_entry_each():
    body = (b"BT /helv 12 Tf 60 150 Td " + _around(b"(visible words) Tj") + b" ET "
            + _around(IMG))
    assert _entries(_tag_page(body)) == [(0, MARK, "visible words", False),
                                         (0, MARK, "", True)]


def test_two_tags_around_images_are_two_entries():
    body = _around(IMG) + b" " + _around(b"q 48 0 0 48 200 300 cm /Im Do Q", "wxotherqz")
    assert list_tag_text_pdf(_tag_page(body)) == {
        "entries": [_glyphless(), _glyphless("wxotherqz")], "unreadable_pages": []}


def _ligature_font(doc, page):
    kind, val = doc.xref_get_key(page.xref, "Resources")
    target, prefix = (int(val.split()[0]), "") if kind == "xref" else (page.xref, "Resources/")
    doc.xref_set_key(target, prefix + "Font/Lig",
                     "<</Type/Font/Subtype/Type1/BaseFont/Helvetica"
                     "/Encoding<</Type/Encoding/Differences[174/fi]>>>>")


LIGATURE = (b"BT /Lig 12 Tf 60 150 Td (de) Tj /Span <</ActualText (fi)>> BDC <AE> Tj EMC"
            b" (ne) Tj ET")
HYPHEN = b"BT /helv 12 Tf 60 200 Td (vis) Tj /Span <</ActualText ()>> BDC (-) Tj EMC ET"


def test_a_ligature_tag_and_a_hyphen_tag_are_as_they_were():
    """``de``, the ``fi`` ligature glyph tagged ``fi``, ``ne``; and a drawn
    hyphen tagged with nothing. The ligature is the line comparison's entry,
    as in 0.15.1, for a caller to judge; the hyphen is not listed."""
    assert _entries(_tag_page(LIGATURE, _ligature_font)) == [(0, "fi", "ﬁ", False)]
    assert _entries(_tag_page(HYPHEN)) == []


def test_a_ligature_and_a_hyphen_beside_a_tag_around_an_image_add_nothing():
    pdf = _tag_page(LIGATURE + b"\n" + HYPHEN + b"\n" + _around(IMG), _ligature_font)
    assert _entries(pdf) == [(0, "fi", "ﬁ", False), (0, MARK, "", True)]


def _property_list(doc, page):
    kind, val = doc.xref_get_key(page.xref, "Resources")
    target, prefix = (int(val.split()[0]), "") if kind == "xref" else (page.xref, "Resources/")
    doc.xref_set_key(target, prefix + "Properties", f"<</MC0 <</ActualText ({MARK})>>>>")


def test_a_tag_in_a_property_list_around_an_image_is_listed():
    pdf = _tag_page(b"/Span /MC0 BDC " + IMG + b" EMC", _property_list)
    assert list_tag_text_pdf(pdf) == {"entries": [_glyphless()], "unreadable_pages": []}


def _indirect_property_list(doc, page):
    kind, val = doc.xref_get_key(page.xref, "Resources")
    target, prefix = (int(val.split()[0]), "") if kind == "xref" else (page.xref, "Resources/")
    entry = doc.get_new_xref()
    doc.update_object(entry, f"<</ActualText ({MARK})>>")
    props = doc.get_new_xref()
    doc.update_object(props, f"<</MC0 {entry} 0 R>>")
    doc.xref_set_key(target, prefix + "Properties", f"{props} 0 R")


def test_a_tag_in_an_indirect_property_list_around_an_image_is_listed():
    pdf = _tag_page(b"/Span /MC0 BDC " + IMG + b" EMC", _indirect_property_list)
    assert list_tag_text_pdf(pdf) == {"entries": [_glyphless()], "unreadable_pages": []}


def test_a_hex_tag_around_an_image_is_decoded():
    utf16 = ("﻿" + MARK).encode("utf-16-be").hex().encode()
    pdf = _tag_page(b"/Span <</ActualText <" + utf16 + b">>> BDC " + IMG + b" EMC")
    assert list_tag_text_pdf(pdf) == {"entries": [_glyphless()], "unreadable_pages": []}


def test_a_property_list_tag_around_glyphs_is_listed_by_its_line_only():
    pdf = _tag_page(b"BT /helv 12 Tf 60 150 Td /Span /MC0 BDC (visible words) Tj EMC ET",
                    _property_list)
    assert _entries(pdf) == [(0, MARK, "visible words", False)]


def _forms(inner_body: bytes, outer_body: bytes = b"/Inner Do"):
    """``/Outer`` draws ``/Inner``; ``/Inner`` draws ``inner_body`` with the
    page's image and a font of its own."""
    def build(doc, page):
        kind, val = doc.xref_get_key(page.xref, "Resources")
        target, prefix = (int(val.split()[0]), "") if kind == "xref" else (page.xref, "Resources/")
        image = int(next(v for k, v in [doc.xref_get_key(target, prefix + "XObject")]
                         if k == "dict").split()[1])
        inner = doc.get_new_xref()
        doc.update_object(inner, "<</Type/XObject/Subtype/Form/BBox[0 0 612 792]"
                                 f"/Resources<</XObject<</Im {image} 0 R>>/Font<</F1"
                                 "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>")
        doc.update_stream(inner, inner_body, new=True)
        outer = doc.get_new_xref()
        doc.update_object(outer, "<</Type/XObject/Subtype/Form/BBox[0 0 612 792]"
                                 f"/Resources<</XObject<</Inner {inner} 0 R>>>>>>")
        doc.update_stream(outer, outer_body, new=True)
        doc.xref_set_key(target, prefix + "XObject",
                         f"<</Outer {outer} 0 R/Im {image} 0 R>>")
    return build


def test_a_tag_around_nested_forms_that_draw_an_image_is_listed():
    pdf = _tag_page(_around(b"/Outer Do"), _forms(b"q 48 0 0 48 60 300 cm /Im Do Q"))
    assert list_tag_text_pdf(pdf) == {"entries": [_glyphless()], "unreadable_pages": []}


def test_a_tag_around_nested_forms_that_draw_glyphs_is_not_glyphless():
    pdf = _tag_page(_around(b"/Outer Do"),
                    _forms(b"BT /F1 12 Tf 60 150 Td (visible words) Tj ET"))
    assert _entries(pdf) == [(0, MARK, "visible words", False)]


def test_a_tag_around_an_image_inside_a_nested_form_is_listed():
    inner = b"/Span <</ActualText (" + MARK.encode() + b")>> BDC q 48 0 0 48 60 300 cm /Im Do Q EMC"
    pdf = _tag_page(b"/Outer Do", _forms(inner))
    assert list_tag_text_pdf(pdf) == {"entries": [_glyphless()], "unreadable_pages": []}


def test_an_empty_string_shown_inside_the_tag_is_no_glyph():
    body = _around(b"BT /helv 12 Tf 60 150 Td () Tj [<>] TJ ET " + IMG)
    assert list_tag_text_pdf(_tag_page(body)) == {
        "entries": [_glyphless()], "unreadable_pages": []}


def test_marked_content_inside_a_tag_closes_only_itself():
    """``BMC ... EMC`` inside the tag ends there; the glyphs after it are
    still the tag's, so the tag wraps glyphs."""
    body = (b"BT /helv 12 Tf 60 150 Td "
            + _around(b"/Artifact BMC EMC /P <</MCID 0>> BDC EMC (visible words) Tj")
            + b" ET " + IMG)
    assert _entries(_tag_page(body)) == [(0, MARK, "visible words", False)]


def test_a_tag_around_a_tag_around_glyphs_wraps_them_too():
    """Both tags repeat the glyphs, so neither says anything the page does
    not draw; the outer one wraps glyphs through the inner one."""
    inner = _around(b"(visible words) Tj", "visible words")
    body = (b"BT /helv 12 Tf 60 150 Td " + _around(inner, "visible words") + b" ET "
            + IMG)
    assert list_tag_text_pdf(_tag_page(body)) == {"entries": [], "unreadable_pages": []}


def test_a_tag_the_content_never_closes_is_read_to_its_end():
    body = b"/Span <</ActualText (" + MARK.encode() + b")>> BDC " + IMG
    assert list_tag_text_pdf(_tag_page(body)) == {
        "entries": [_glyphless()], "unreadable_pages": []}


def test_a_form_without_resources_draws_with_its_callers():
    """``/Bare`` has no ``/Resources``, so the ``/Words`` it draws is the
    page's: the tag around it wraps glyphs."""
    def bare(doc, page):
        kind, val = doc.xref_get_key(page.xref, "Resources")
        target, prefix = (int(val.split()[0]), "") if kind == "xref" else (page.xref, "Resources/")
        image = doc.xref_get_key(target, prefix + "XObject")[1]
        words = doc.get_new_xref()
        doc.update_object(words, "<</Type/XObject/Subtype/Form/BBox[0 0 612 792]"
                                 "/Resources<</Font<</F1<</Type/Font/Subtype/Type1"
                                 "/BaseFont/Helvetica>>>>>>>>")
        doc.update_stream(words, b"BT /F1 12 Tf 60 150 Td (visible words) Tj ET", new=True)
        fx = doc.get_new_xref()
        doc.update_object(fx, "<</Type/XObject/Subtype/Form/BBox[0 0 612 792]>>")
        doc.update_stream(fx, b"/Words Do", new=True)
        doc.xref_set_key(target, prefix + "XObject",
                         image[:-2] + f"/Bare {fx} 0 R/Words {words} 0 R>>")
    assert _entries(_tag_page(_around(b"/Bare Do"), bare)) == [
        (0, MARK, "visible words", False)]


def test_a_form_that_draws_itself_is_read_once():
    def looped(doc, page):
        kind, val = doc.xref_get_key(page.xref, "Resources")
        target, prefix = (int(val.split()[0]), "") if kind == "xref" else (page.xref, "Resources/")
        image = doc.xref_get_key(target, prefix + "XObject")[1]
        fx = doc.get_new_xref()
        doc.update_object(fx, f"<</Type/XObject/Subtype/Form/BBox[0 0 612 792]"
                              f"/Resources<</XObject<</Loop {fx} 0 R>>>>>>")
        doc.update_stream(fx, b"/Loop Do", new=True)
        doc.xref_set_key(target, prefix + "XObject", image[:-2] + f"/Loop {fx} 0 R>>")
    pdf = _tag_page(_around(b"/Loop Do " + IMG), looped)
    assert list_tag_text_pdf(pdf) == {"entries": [_glyphless()], "unreadable_pages": []}


@pytest.mark.parametrize("inner", [b"0 0 1 rg 60 600 100 40 re f", b""],
                         ids=["around-a-drawing", "around-nothing"])
def test_a_tag_around_no_image_and_no_glyph_is_still_listed_once(inner):
    assert list_tag_text_pdf(_tag_page(_around(inner))) == {
        "entries": [_glyphless()], "unreadable_pages": []}


def test_a_page_whose_content_cannot_be_parsed_is_unreadable_not_empty():
    pdf = _tag_page(_around(IMG) + b" ) stray")
    assert MARK in pymupdf.open(stream=pdf)[0].get_text()
    assert list_tag_text_pdf(pdf) == {"entries": [], "unreadable_pages": [0]}


# ── side effects ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("rotate", [0, 90])
def test_listing_leaves_an_open_document_as_it_was(rotate):
    pdf, _ = _boxed([[(100, 30, 140, 60)]], crop="[50 100 400 900]", rotate=rotate)
    doc = pymupdf.open(stream=pdf)
    before = doc.tobytes(no_new_id=True)
    list_unseen_images(doc)
    list_tag_text(doc)
    assert doc.tobytes(no_new_id=True) == before
    tagged = pymupdf.open(stream=_tag_page(_around(IMG)))
    before = tagged.tobytes(no_new_id=True)
    list_tag_text(tagged)
    assert tagged.tobytes(no_new_id=True) == before


def test_stripping_the_image_tag_page_leaves_the_other_page_its_tags():
    doc = pymupdf.open(stream=_tag_page(_around(IMG), pages=2))
    first = doc[0]
    first.insert_font(fontname="helv")
    first.clean_contents()
    xref = first.get_contents()[0]
    doc.update_stream(xref, doc.xref_stream(xref) + b"\nBT /helv 12 Tf 60 150 Td "
                      + _around(b"(visible words) Tj", "visible words") + b" ET")
    pdf = doc.tobytes(no_new_id=True)
    assert list_tag_text_pdf(pdf)["entries"] == [
        {"page": 1, "text": MARK, "drawn": "", "bbox": None}]
    out, _ = apply_redactions(pdf, [], strip_tags_pages=[1])
    kept = pymupdf.open(stream=out)
    assert b"/ActualText" in b"".join(kept.xref_stream(x) for x in kept[0].get_contents())
    assert b"/ActualText" not in b"".join(kept.xref_stream(x) for x in kept[1].get_contents())
    assert list_tag_text_pdf(out) == {"entries": [], "unreadable_pages": []}


# ── the handle ops ───────────────────────────────────────────────────────


def test_the_handle_ops_read_the_clipped_area_and_the_image_tag():
    pdf, (xref,) = _boxed([[(100, 30, 140, 60)]], crop="[50 100 400 900]")
    handle = _OPS["open_doc"]({"pdf_b64": base64.b64encode(pdf).decode()})["handle"]
    try:
        assert _OPS["list_unseen_images_h"]({"handle": handle})["unseen"] == [
            _outside([50.0, 732.0, 90.0, 762.0], xref)]
    finally:
        _OPS["close_doc"]({"handle": handle})
    tagged = _tag_page(_around(IMG))
    handle = _OPS["open_doc"]({"pdf_b64": base64.b64encode(tagged).decode()})["handle"]
    try:
        assert _OPS["list_tag_text_h"]({"handle": handle}) == {
            "entries": [_glyphless()], "unreadable_pages": []}
    finally:
        _OPS["close_doc"]({"handle": handle})
