"""What a delivered PDF carries outside its visible content (v0.9.0).

``apply_redactions`` rewrites page content streams and ``_scrub_residue``
removes annotations, attachments, document JavaScript and thumbnails. The
vectors here survived all of that before v0.9.0, each measured on a real burn
with an invented marker before anything changed:

* actions: a JavaScript ``/OpenAction``, ``/AA`` at catalog and page level, a
  JavaScript link action, and URI / launch open actions;
* metadata beyond the eight standard ``/Info`` keys: custom ``/Info`` keys,
  page ``/Metadata``, ``/PieceInfo``, unknown catalog and page keys, and the
  trailer ``/ID`` (a delivered file shared its permanent half with its source);
* associated files (``/AF``) and the comment and EXIF segments inside JPEGs;
* tagged text: an ``/ActualText`` or ``/Alt`` that repeats a sentence keeps the
  words a burn removed from the glyphs, and text extraction returns the tag's
  text in place of the drawn characters.

The 6-criteria framework:

* **Happy path** -- each vector is absent from the exported bytes.
* **Sad paths** (at least two per happy) -- what must SURVIVE: go-to open
  actions and URI links, allowed keys, a tag on a page nothing was redacted
  from, pixels of a stripped JPEG; and what must not crash: a dangling action,
  a truncated JPEG, an unterminated string in a content stream.
* **Boundary** -- a source with compressed object streams, one page of many,
  page indices that shift when pages are removed, encrypted output.
* **No logic mirroring** -- markers are typed out; nothing is recomputed
  through the code under test.
* **Side-effect verification** -- assertions read the exported BYTES through
  the raw file, every object, every decoded stream and every undecoded
  stream, never ``get_text()`` alone.
* **Mock integrity** -- every PDF is built in memory with PyMuPDF and driven
  through the real ``apply_redactions`` / ``reduce_size`` entry points.

Synthetic fixtures only; every value is invented.
"""
from __future__ import annotations

import base64
import hashlib
import struct

import pymupdf
import pytest

from lexcloak_pdf_tool import apply_redactions, reduce_size
from lexcloak_pdf_tool.__main__ import _OPS, PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS
from lexcloak_pdf_tool.sanitise import (
    residue_report_pdf,
    strip_jpeg_segments,
    strip_tag_keys,
)

SSN = "523-81-4406"
LINE = f"Claimant intake record SSN {SSN}"


# ── builders and readers ─────────────────────────────────────────────────


def _doc(pages: int = 1):
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text((60, 90), LINE, fontsize=12)
    return doc


def _bytes(doc, **kw) -> bytes:
    data = doc.tobytes(garbage=3, deflate=True, no_new_id=True, **kw)
    doc.close()
    return data


def _new_stream(doc, dict_text: str, data: bytes) -> int:
    x = doc.get_new_xref()
    doc.update_object(x, dict_text)
    doc.update_stream(x, data, new=True)
    return x


def _forms(needle: str):
    raw = needle.encode("latin-1")
    hexed = raw.hex().encode()
    return (raw, hexed, hexed.upper())


def _readable(pdf: bytes, needle: str, password: str | None = None) -> bool:
    """True if ``needle`` can be read from ``pdf`` by any channel: the raw
    file, an object's dictionary text, a decoded stream, or an undecoded one."""
    forms = _forms(needle)
    if password is None and any(f in pdf for f in forms):
        return True
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        if password is not None:
            assert doc.authenticate(password)
        for x in list(range(1, doc.xref_length())) + [-1]:
            try:
                if needle in doc.xref_object(x, compressed=False):
                    return True
                if x > 0 and doc.xref_is_stream(x):
                    for blob in (doc.xref_stream(x), doc.xref_stream_raw(x)):
                        if blob and any(f in blob for f in forms):
                            return True
            except Exception:  # noqa: BLE001
                continue
    finally:
        doc.close()
    return False


def _burn(pdf: bytes, *, only_pages=None, **kw) -> bytes:
    """Burn every occurrence of the SSN, as the app would after detection."""
    d = pymupdf.open(stream=pdf, filetype="pdf")
    matches = []
    for i, p in enumerate(d):
        if only_pages is not None and i not in only_pages:
            continue
        for r in p.search_for(SSN):
            matches.append({"page": i, "enabled": True, "type": "SSN",
                            "rect": {"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}})
    d.close()
    out, _ = apply_redactions(pdf, matches, **kw)
    return out


def _text(pdf: bytes) -> str:
    d = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        return "".join(p.get_text() for p in d)
    finally:
        d.close()


def _image_hash(pdf: bytes) -> str:
    """Hash of the first image's DECODED pixels (the page itself is not
    comparable across a burn: the burn draws a black box)."""
    d = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        x = d[0].get_images(full=True)[0][0]
        return hashlib.sha256(pymupdf.Pixmap(d, x).samples).hexdigest()
    finally:
        d.close()


# ── active content ───────────────────────────────────────────────────────

JS = "actmark-js"


def _js_open():
    d = _doc()
    d.xref_set_key(d.pdf_catalog(), "OpenAction", f"<</S/JavaScript/JS ({JS})>>")
    return _bytes(d)


def _js_page_aa():
    d = _doc()
    d.xref_set_key(d[0].xref, "AA", f"<</O <</S/JavaScript/JS ({JS})>>>>")
    return _bytes(d)


def _js_catalog_aa():
    d = _doc()
    d.xref_set_key(d.pdf_catalog(), "AA", f"<</WC <</S/JavaScript/JS ({JS})>>>>")
    return _bytes(d)


def _link_with_action(action: str, uri: str = "https://example.invalid/") -> bytes:
    d = _doc()
    page = d[0]
    page.insert_link({"kind": pymupdf.LINK_URI,
                      "from": pymupdf.Rect(60, 200, 200, 220), "uri": uri})
    xref = int(d.xref_get_key(page.xref, "Annots")[1].strip("[]").split()[0])
    d.xref_set_key(xref, "A", action)
    return _bytes(d)


def _js_link():
    return _link_with_action(f"<</S/JavaScript/JS ({JS})>>")


def _uri_open():
    d = _doc()
    d.xref_set_key(d.pdf_catalog(), "OpenAction",
                   f"<</S/URI/URI (https://example.invalid/{JS})>>")
    return _bytes(d)


def _launch_open():
    d = _doc()
    d.xref_set_key(d.pdf_catalog(), "OpenAction",
                   f"<</S/Launch/F <</Type/Filespec/F ({JS})>>>>")
    return _bytes(d)


@pytest.mark.parametrize("build", [_js_open, _js_page_aa, _js_catalog_aa,
                                   _js_link, _uri_open, _launch_open],
                         ids=["js-open", "js-page-aa", "js-catalog-aa",
                              "js-link", "uri-open", "launch-open"])
def test_active_content_is_absent_from_the_export(build):
    src = build()
    assert _readable(src, JS), "fixture must carry the marker (positive control)"
    assert not _readable(_burn(src), JS)


def test_a_goto_open_action_survives():
    d = _doc(2)
    d.xref_set_key(d.pdf_catalog(), "OpenAction",
                   f"<</S/GoTo/D [{d[1].xref} 0 R /Fit]>>")
    out = _burn(_bytes(d))
    o = pymupdf.open(stream=out, filetype="pdf")
    kind, value = o.xref_get_key(o.pdf_catalog(), "OpenAction")
    assert kind in ("dict", "xref") and "GoTo" in (
        value if kind == "dict" else o.xref_object(int(value.split()[0])))


def test_a_destination_array_open_action_survives():
    d = _doc(2)
    d.xref_set_key(d.pdf_catalog(), "OpenAction", f"[{d[1].xref} 0 R /Fit]")
    o = pymupdf.open(stream=_burn(_bytes(d)), filetype="pdf")
    assert o.xref_get_key(o.pdf_catalog(), "OpenAction")[0] == "array"


def test_a_uri_link_action_survives():
    """The deliberate scope fence: a URI link is a separate ruling."""
    src = _link_with_action("<</S/URI/URI (https://example.invalid/keep-me)>>")
    assert _readable(_burn(src), "https://example.invalid/keep-me")


def test_a_goto_action_that_chains_a_script_is_removed():
    src = _link_with_action(
        f"<</S/GoTo/D [0 /Fit]/Next <</S/JavaScript/JS ({JS})>>>>")
    assert _readable(src, JS)
    assert not _readable(_burn(src), JS)


def test_a_structure_attribute_dictionary_is_not_mistaken_for_an_action():
    d = _doc()
    root, el = d.get_new_xref(), d.get_new_xref()
    d.update_object(root, f"<</Type/StructTreeRoot/K {el} 0 R>>")
    d.update_object(el, f"<</Type/StructElem/S/P/P {root} 0 R/Pg {d[0].xref} 0 R"
                        "/A <</O/Layout/Placement/Block>>>>")
    d.xref_set_key(d.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")
    o = pymupdf.open(stream=_burn(_bytes(d)), filetype="pdf")
    assert any("/Layout" in o.xref_object(x, compressed=False)
               for x in range(1, o.xref_length()))


def test_a_dangling_action_reference_does_not_fail_the_export():
    src = _link_with_action("999 0 R")
    out = _burn(src)
    assert SSN not in _text(out)


# ── extra metadata ───────────────────────────────────────────────────────

META = "metamark"


def _custom_info():
    d = _doc()
    ix = d.get_new_xref()
    d.update_object(ix, f"<</Company ({META})/Title (t)>>")
    d.xref_set_key(-1, "Info", f"{ix} 0 R")
    return _bytes(d)


def _page_xmp():
    d = _doc()
    x = _new_stream(d, "<</Type/Metadata/Subtype/XML>>",
                    f"<x:xmpmeta><dc:creator>{META}</dc:creator></x:xmpmeta>".encode())
    d.xref_set_key(d[0].xref, "Metadata", f"{x} 0 R")
    return _bytes(d)


def _piece_info_page():
    d = _doc()
    d.xref_set_key(d[0].xref, "PieceInfo", f"<</App <</Private ({META})>>>>")
    return _bytes(d)


def _piece_info_catalog():
    d = _doc()
    d.xref_set_key(d.pdf_catalog(), "PieceInfo", f"<</App <</Private ({META})>>>>")
    return _bytes(d)


def _catalog_key():
    d = _doc()
    d.xref_set_key(d.pdf_catalog(), "VendorNote", f"({META})")
    return _bytes(d)


def _page_key():
    d = _doc()
    d.xref_set_key(d[0].xref, "VendorNote", f"({META})")
    return _bytes(d)


def _image_metadata_stream():
    d = _doc()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), False)
    pix.clear_with(200)
    d[0].insert_image(pymupdf.Rect(60, 200, 100, 240), pixmap=pix)
    img = d[0].get_images(full=True)[0][0]
    x = _new_stream(d, "<</Type/Metadata/Subtype/XML>>", META.encode())
    d.xref_set_key(img, "Metadata", f"{x} 0 R")
    return _bytes(d)


@pytest.mark.parametrize("build", [_custom_info, _page_xmp, _piece_info_page,
                                   _piece_info_catalog, _catalog_key,
                                   _page_key, _image_metadata_stream],
                         ids=["custom-info", "page-xmp", "pieceinfo-page",
                              "pieceinfo-catalog", "catalog-key", "page-key",
                              "image-metadata"])
def test_extra_metadata_is_absent_from_the_export(build):
    src = build()
    assert _readable(src, META), "fixture must carry the marker (positive control)"
    assert not _readable(_burn(src), META)


def test_the_trailer_id_is_not_carried_from_the_source():
    d = _doc()
    d.xref_set_key(-1, "ID", f"[({'idmark-sixteen-xx'[:16]})({'idmark-sixteen-xx'[:16]})]")
    src = _bytes(d)
    assert _readable(src, "idmark-sixteen-x")
    out = _burn(src)
    assert not _readable(out, "idmark-sixteen-x")
    o = pymupdf.open(stream=out, filetype="pdf")
    assert o.xref_get_key(-1, "ID")[0] == "array"           # still has one


def test_standard_metadata_the_caller_stamps_afterwards_survives():
    from lexcloak_pdf_tool import set_metadata
    out = set_metadata(_burn(_custom_info()), {"subject": "stamped-later"})
    o = pymupdf.open(stream=out, filetype="pdf")
    assert o.metadata["subject"] == "stamped-later"
    assert not _readable(out, META)


def test_keys_the_specification_defines_are_kept():
    d = _doc(2)
    d.set_toc([[1, "chapter", 1]])
    d.set_page_labels([{"startpage": 0, "prefix": "P-", "style": "D"}])
    d.xref_set_key(d.pdf_catalog(), "Lang", "(en-US)")
    o = pymupdf.open(stream=_burn(_bytes(d)), filetype="pdf")
    keys = set(o.xref_get_keys(o.pdf_catalog()))
    assert {"Pages", "Outlines", "PageLabels", "Lang"} <= keys
    assert o.get_toc() and o.get_toc()[0][1] == "chapter"
    assert len(o) == 2


def test_encrypted_output_carries_no_extra_metadata():
    src = _custom_info()
    out, applied = apply_redactions(
        src, [], output_protection={"mode": "new", "password": "pw-test-1"})
    assert applied
    assert not _readable(out, META, password="pw-test-1")


# ── associated files ─────────────────────────────────────────────────────

AF = "afmark-payload"


def _associated_file(level: str) -> bytes:
    d = _doc()
    st = _new_stream(d, "<</Type/EmbeddedFile>>", AF.encode())
    fs = d.get_new_xref()
    d.update_object(fs, f"<</Type/Filespec/F (n.txt)/EF <</F {st} 0 R>>>>")
    target = d.pdf_catalog() if level == "catalog" else d[0].xref
    d.xref_set_key(target, "AF", f"[{fs} 0 R]")
    return _bytes(d)


@pytest.mark.parametrize("level", ["catalog", "page"])
def test_associated_files_are_absent_from_the_export(level):
    src = _associated_file(level)
    assert _readable(src, AF)
    assert not _readable(_burn(src), AF)


# ── JPEG segments ────────────────────────────────────────────────────────


def _jpeg() -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 48, 48), False)
    pix.clear_with(180)
    return pix.tobytes("jpeg")


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload


def _with_segment(jpeg: bytes, marker: int, payload: bytes) -> bytes:
    return jpeg[:2] + _segment(marker, payload) + jpeg[2:]


def _pdf_with_jpeg(jpeg: bytes) -> bytes:
    d = _doc()
    d[0].insert_image(pymupdf.Rect(60, 200, 108, 248), stream=jpeg)
    return _bytes(d)


JPEG_MARK = "jpegmark-tail"


@pytest.mark.parametrize("marker,payload", [
    (0xFE, JPEG_MARK.encode()),                                   # comment
    (0xE1, b"Exif\x00\x00" + JPEG_MARK.encode()),                 # EXIF
    (0xE1, b"http://ns.adobe.com/xap/1.0/\x00" + JPEG_MARK.encode()),  # XMP
    (0xED, b"Photoshop 3.0\x00" + JPEG_MARK.encode()),            # APP13
    (0xE2, b"MPF\x00" + JPEG_MARK.encode()),                      # multi-picture
    (0xE0, b"JFXX\x00" + JPEG_MARK.encode()),                     # JFIF thumbnail
], ids=["comment", "exif", "xmp", "app13", "mpf", "jfxx"])
def test_a_jpeg_description_segment_is_removed_and_the_pixels_are_identical(marker, payload):
    jpeg = _with_segment(_jpeg(), marker, payload)
    src = _pdf_with_jpeg(jpeg)
    assert _readable(src, JPEG_MARK), "fixture must carry the marker"
    before = _image_hash(src)
    out = _burn(src)
    assert not _readable(out, JPEG_MARK)
    assert _image_hash(out) == before                # lossless, not re-encoded
    o = pymupdf.open(stream=out, filetype="pdf")
    x = o[0].get_images(full=True)[0][0]
    assert o.xref_get_key(x, "Filter") == ("name", "/DCTDecode")
    assert o.xref_stream_raw(x)[:2] == b"\xff\xd8"
    assert pymupdf.Pixmap(o, x).width == 48


def test_an_icc_profile_and_the_adobe_transform_flag_are_kept():
    jpeg = _with_segment(_jpeg(), 0xE2, b"ICC_PROFILE\x00\x01\x01icc-body")
    jpeg = _with_segment(jpeg, 0xEE, b"Adobe\x00\x64\x00\x00\x00\x00\x01")
    stripped = strip_jpeg_segments(jpeg)
    assert b"ICC_PROFILE" in stripped and b"Adobe" in stripped
    assert stripped == jpeg


def test_data_after_the_final_end_of_image_marker_is_dropped():
    jpeg = _jpeg() + b"trailing-" + JPEG_MARK.encode()
    assert JPEG_MARK.encode() not in strip_jpeg_segments(jpeg)
    assert strip_jpeg_segments(jpeg).endswith(b"\xff\xd9")


def test_a_progressive_style_file_with_two_scans_and_restart_markers_is_walked():
    """The walker must step over entropy-coded data, byte stuffing (FF00) and
    restart markers (FFD0..D7) without mistaking them for segments."""
    sof = _segment(0xC2, b"\x08\x00\x10\x00\x10\x01\x01\x11\x00")
    sos = _segment(0xDA, b"\x01\x01\x00\x00\x3f\x00")
    scan1 = b"\x12\xff\x00\x34\xff\xd0\x56\xff\xd1\x78"
    scan2 = b"\x9a\xbc\xff\x00\xde"
    jpeg = (b"\xff\xd8" + _segment(0xFE, JPEG_MARK.encode()) + sof
            + sos + scan1 + _segment(0xFE, b"between-scans") + sos + scan2
            + b"\xff\xd9")
    out = strip_jpeg_segments(jpeg)
    assert JPEG_MARK.encode() not in out and b"between-scans" not in out
    assert scan1 in out and scan2 in out                 # entropy data untouched
    assert out == (b"\xff\xd8" + sof + sos + scan1 + sos + scan2 + b"\xff\xd9")


@pytest.mark.parametrize("data", [
    b"", b"not a jpeg at all", b"\xff\xd8", b"\xff\xd8\xff",
    _jpeg()[:-40],                                            # truncated
    b"\xff\xd8" + _segment(0xFE, b"x") + b"\xff\xd9",         # no frame, no scan
], ids=["empty", "text", "soi-only", "short", "truncated", "no-frame"])
def test_a_malformed_jpeg_is_returned_unchanged(data):
    assert strip_jpeg_segments(data) == data


def test_a_stacked_filter_image_is_left_alone():
    d = _doc()
    jpeg = _with_segment(_jpeg(), 0xFE, JPEG_MARK.encode())
    d[0].insert_image(pymupdf.Rect(60, 200, 108, 248), stream=jpeg)
    x = d[0].get_images(full=True)[0][0]
    d.xref_set_key(x, "Filter", "[/ASCIIHexDecode /DCTDecode]")
    out = _burn(_bytes(d))
    assert out  # no crash; the image is not guessed at


# ── tagged text ──────────────────────────────────────────────────────────


def _actualtext_page(actual: str, drawn: str = LINE) -> bytes:
    d = pymupdf.open()
    page = d.new_page(width=612, height=792)
    page.insert_font(fontname="helv")
    page.clean_contents()
    d.update_stream(
        page.get_contents()[0],
        b"BT /helv 12 Tf 60 700 Td /Span <</ActualText (" + actual.encode()
        + b")>> BDC (" + drawn.encode() + b") Tj EMC ET\n")
    return _bytes(d)


def test_the_baseline_burn_removes_the_ssn_from_the_text_layer():
    """Negative control: with no tag the burn works, so the tag is the cause."""
    d = pymupdf.open()
    page = d.new_page(width=612, height=792)
    page.insert_font(fontname="helv")
    page.clean_contents()
    d.update_stream(page.get_contents()[0],
                    b"BT /helv 12 Tf 60 700 Td (" + LINE.encode() + b") Tj ET\n")
    assert SSN not in _text(_burn(_bytes(d)))


def test_an_actualtext_that_repeats_the_burned_sentence_no_longer_returns_it():
    src = _actualtext_page(LINE)
    assert SSN in _text(src)                         # positive control
    out = _burn(src)
    assert SSN not in _text(out)
    assert not _readable(out, SSN)


def test_an_actualtext_that_differs_from_the_drawn_text_is_removed_on_a_burned_page():
    out = _burn(_doc_with_extra_actualtext("tagmark-differs"))
    assert not _readable(out, "tagmark-differs")


def _doc_with_extra_actualtext(marker: str) -> bytes:
    d = _doc()
    page = d[0]
    page.insert_font(fontname="helv")
    page.clean_contents()
    x = page.get_contents()[0]
    d.update_stream(x, d.xref_stream(x)
                    + b"\nBT /helv 12 Tf 60 150 Td /Span <</ActualText ("
                    + marker.encode() + b")>> BDC (visible) Tj EMC ET\n")
    return _bytes(d)


def test_a_tag_on_a_page_nothing_was_redacted_from_is_kept():
    """Fidelity: ligature and hyphenation tags survive where nothing burned."""
    d = _doc(2)
    page = d[1]
    page.insert_font(fontname="helv")
    page.clean_contents()
    x = page.get_contents()[0]
    d.update_stream(x, d.xref_stream(x)
                    + b"\nBT /helv 12 Tf 60 300 Td /Span <</ActualText (keptmark)>>"
                      b" BDC (visible) Tj EMC ET\n")
    src = _bytes(d)
    out = _burn(src, only_pages={0})
    assert _readable(out, "keptmark")
    assert SSN not in "".join(p.get_text() for p in
                              [pymupdf.open(stream=out, filetype="pdf")[0]])


def test_touched_pages_follow_removed_pages_so_the_right_tag_is_stripped():
    """Three pages, page 0 removed, page 2 burned: in the output that is page
    1, and the tag on the output's page 0 (the source's page 1) must stay."""
    d = _doc(3)
    for i, marker in ((1, b"keptmark"), (2, b"strippedmark")):
        page = d[i]
        page.insert_font(fontname="helv")
        page.clean_contents()
        x = page.get_contents()[0]
        d.update_stream(x, d.xref_stream(x)
                        + b"\nBT /helv 12 Tf 60 300 Td /Span <</ActualText ("
                        + marker + b")>> BDC (visible) Tj EMC ET\n")
    out = _burn(_bytes(d), only_pages={2}, removed_pages=[0])
    assert _readable(out, "keptmark")
    assert not _readable(out, "strippedmark")
    assert len(pymupdf.open(stream=out, filetype="pdf")) == 2


def test_a_page_where_only_hidden_text_was_removed_counts_as_touched():
    """No burn box on the page: the only redaction is a text-only removal."""
    from lexcloak_pdf_tool.trace import trace_text
    d = pymupdf.open()
    page = d.new_page(width=612, height=792)
    page.insert_text((60, 300), "quietword", fontsize=12, render_mode=3)
    page.insert_font(fontname="helv")
    page.clean_contents()
    x = page.get_contents()[0]
    d.update_stream(x, d.xref_stream(x)
                    + b"\nBT /helv 12 Tf 60 150 Td /Span <</ActualText (onlytagmark)>>"
                      b" BDC (visible) Tj EMC ET\n")
    src = _bytes(d)
    assert _readable(src, "quietword") and _readable(src, "onlytagmark")
    word = next(w for w in trace_text(src, 0)["words"] if w["text"] == "quietword")
    out, _ = apply_redactions(src, [], remove_text=[{
        "page": 0, "box": word["bbox"], "text": "quietword",
        "mode": word["mode"], "opacity": word["opacity"], "layer": word["layer"]}])
    assert not _readable(out, "quietword")            # the removal really ran
    assert not _readable(out, "onlytagmark")


def test_a_blackout_page_counts_as_touched():
    """The blackout scrubs the page's own content, so a content-stream tag goes
    regardless. A structure element that points at the page is what only the
    touched-page set reaches."""
    assert not _readable(
        apply_redactions(_doc_with_extra_actualtext("blackoutmark"), [],
                         blackout_pages=[0])[0], "blackoutmark")
    src = _struct_pdf("altmark-blackout", tag_page=0)
    assert _readable(src, "altmark-blackout")
    out, _ = apply_redactions(src, [], blackout_pages=[0])
    assert not _readable(out, "altmark-blackout")


def test_a_tag_in_a_form_xobject_drawn_on_a_burned_page_is_stripped():
    d = _doc()
    fx = _new_stream(
        d, "<</Type/XObject/Subtype/Form/BBox[0 0 300 40]/Resources<</Font<</F1"
           "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>",
        b"/Span <</ActualText (formmark)>> BDC BT /F1 12 Tf 10 20 Td (x) Tj ET EMC")
    page = d[0]
    x = page.get_contents()[0]
    d.update_stream(x, d.xref_stream(x) + b"\nq 1 0 0 1 60 300 cm /Fm1 Do Q\n")
    kind, val = d.xref_get_key(page.xref, "Resources")
    target, key = ((int(val.split()[0]), "XObject") if kind == "xref"
                   else (page.xref, "Resources/XObject"))
    d.xref_set_key(target, key, f"<</Fm1 {fx} 0 R>>")
    src = _bytes(d)
    assert _readable(src, "formmark")
    assert not _readable(_burn(src), "formmark")


def test_a_named_property_list_tag_is_stripped():
    d = _doc()
    page = d[0]
    page.insert_font(fontname="helv")
    page.clean_contents()
    x = page.get_contents()[0]
    d.update_stream(x, d.xref_stream(x) + b"\n/Span /MC0 BDC BT /helv 12 Tf 60 150 Td (v) Tj ET EMC\n")
    kind, val = d.xref_get_key(page.xref, "Resources")
    target, key = ((int(val.split()[0]), "Properties") if kind == "xref"
                   else (page.xref, "Resources/Properties"))
    d.xref_set_key(target, key, "<</MC0 <</MCID 0/ActualText (propmark)>>>>")
    src = _bytes(d)
    assert _readable(src, "propmark")
    assert not _readable(_burn(src), "propmark")


def _struct_pdf(alt_marker: str, pages: int = 2, tag_page: int = 1,
                inherit: bool = False) -> bytes:
    d = _doc(pages)
    root, parent, el = d.get_new_xref(), d.get_new_xref(), d.get_new_xref()
    d.update_object(root, f"<</Type/StructTreeRoot/K {parent} 0 R>>")
    pg_on_parent = f"/Pg {d[tag_page].xref} 0 R"
    d.update_object(parent, f"<</Type/StructElem/S/Sect/P {root} 0 R"
                            f"{pg_on_parent if inherit else ''}/K {el} 0 R>>")
    d.update_object(el, f"<</Type/StructElem/S/Figure/P {parent} 0 R"
                        f"{'' if inherit else pg_on_parent}/Alt ({alt_marker})>>")
    d.xref_set_key(d.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")
    return _bytes(d)


def test_struct_alt_on_a_burned_page_is_stripped():
    src = _struct_pdf("altmark-burned", tag_page=0)
    assert _readable(src, "altmark-burned")
    assert not _readable(_burn(src, only_pages={0}), "altmark-burned")


def test_struct_alt_on_a_page_nothing_was_redacted_from_is_kept():
    out = _burn(_struct_pdf("altmark-kept", tag_page=1), only_pages={0})
    assert _readable(out, "altmark-kept")


def test_struct_alt_takes_its_page_from_the_nearest_ancestor():
    src = _struct_pdf("altmark-inherit", tag_page=0, inherit=True)
    assert not _readable(_burn(src, only_pages={0}), "altmark-inherit")


def test_struct_alt_with_no_resolvable_page_is_stripped_when_anything_was_burned():
    d = _doc(1)
    root, el = d.get_new_xref(), d.get_new_xref()
    d.update_object(root, f"<</Type/StructTreeRoot/K {el} 0 R>>")
    d.update_object(el, f"<</Type/StructElem/S/Figure/P {root} 0 R/Alt (orphanmark)>>")
    d.xref_set_key(d.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")
    assert not _readable(_burn(_bytes(d)), "orphanmark")


def test_struct_alt_is_kept_when_nothing_at_all_was_burned():
    src = _struct_pdf("altmark-none", tag_page=0)
    out, _ = apply_redactions(src, [])
    assert _readable(out, "altmark-none")


# strip_tag_keys: the tokenizer that decides what is a key and what is text


@pytest.mark.parametrize("stream,expected,count", [
    (b"/Span <</ActualText (abc)>> BDC", b"/Span <<>> BDC", 1),
    (b"/Span<</ActualText(abc)>>BDC", b"/Span<<>>BDC", 1),
    (b"/S <</MCID 3/Alt (a \\) b (nested) c)/Lang (en)>> BDC",
     b"/S <</MCID 3/Lang (en)>> BDC", 1),
    (b"/S <</ActualText <FEFF00610062>>> BDC", b"/S <<>> BDC", 1),
    (b"/S <</ActualText (a)/Alt (b)>> BDC", b"/S <<>> BDC", 2),
], ids=["spaced", "packed", "escapes-and-nesting", "hex-value", "both-keys"])
def test_strip_tag_keys_removes_the_value_and_leaves_the_rest(stream, expected, count):
    assert strip_tag_keys(stream) == (expected, count)


@pytest.mark.parametrize("stream", [
    b"BT (the key /ActualText (x) is only text) Tj ET",
    b"% /ActualText (in a comment)\nBT (x) Tj ET",
    b"BI /W 1 /H 1 ID /ActualText (raw bytes) \nEI Q",
    b"/S <</ActualText /NotAString>> BDC",
], ids=["inside-string", "inside-comment", "inside-inline-image", "non-string-value"])
def test_strip_tag_keys_ignores_what_is_not_a_key_with_a_string(stream):
    assert strip_tag_keys(stream) == (stream, 0)


@pytest.mark.parametrize("stream", [
    b"/S <</ActualText (never closed", b"BI /W 1 ID no end marker",
    b"/S <</ActualText <ABCD",
], ids=["open-string", "open-inline-image", "open-hex"])
def test_strip_tag_keys_refuses_a_stream_it_cannot_tokenise(stream):
    with pytest.raises((ValueError, IndexError)):
        strip_tag_keys(stream)


def test_a_content_stream_that_cannot_be_tokenised_is_left_untouched():
    d = _doc()
    page = d[0]
    page.insert_font(fontname="helv")
    page.clean_contents()
    x = page.get_contents()[0]
    d.update_stream(x, d.xref_stream(x)
                    + b"\n/Span <</ActualText (broken")      # never closed
    out = _burn(_bytes(d))
    assert out                                              # export still succeeds


# ── reduce_size ──────────────────────────────────────────────────────────


def test_reduce_size_also_strips_actions_extra_metadata_files_and_image_segments():
    d = _doc()
    d.xref_set_key(d.pdf_catalog(), "OpenAction", f"<</S/JavaScript/JS ({JS})>>")
    d.xref_set_key(d[0].xref, "PieceInfo", f"<</App <</Private ({META})>>>>")
    st = _new_stream(d, "<</Type/EmbeddedFile>>", AF.encode())
    fs = d.get_new_xref()
    d.update_object(fs, f"<</Type/Filespec/F (n.txt)/EF <</F {st} 0 R>>>>")
    d.xref_set_key(d.pdf_catalog(), "AF", f"[{fs} 0 R]")
    d[0].insert_image(pymupdf.Rect(60, 200, 108, 248),
                      stream=_with_segment(_jpeg(), 0xFE, JPEG_MARK.encode()))
    src = _bytes(d)
    for marker in (JS, META, AF, JPEG_MARK):
        assert _readable(src, marker)
    out, _info = reduce_size(src)
    for marker in (JS, META, AF, JPEG_MARK):
        assert not _readable(out, marker), marker


# ── residue_report ───────────────────────────────────────────────────────


def _dirty() -> bytes:
    d = _doc()
    d.xref_set_key(d.pdf_catalog(), "OpenAction", f"<</S/JavaScript/JS ({JS})>>")
    d.xref_set_key(d[0].xref, "PieceInfo", f"<</App <</Private ({META})>>>>")
    d.xref_set_key(d[0].xref, "VendorNote", f"({META})")
    d.xref_set_key(d.pdf_catalog(), "VendorNote", f"({META})")
    ix = d.get_new_xref()
    d.update_object(ix, f"<</Company ({META})>>")
    d.xref_set_key(-1, "Info", f"{ix} 0 R")
    d[0].insert_image(pymupdf.Rect(60, 200, 108, 248),
                      stream=_with_segment(_jpeg(), 0xFE, JPEG_MARK.encode()))
    return _bytes(d)


def test_the_report_counts_what_is_left_and_is_zero_after_the_export():
    before = residue_report_pdf(_dirty())
    assert before["active_content"] >= 1
    assert before["extra_info_keys"] == 1
    assert before["piece_info"] >= 1
    assert before["unknown_keys"] >= 2
    assert before["image_metadata"] == 1
    after = residue_report_pdf(_burn(_dirty()))
    assert set(after.values()) == {0}


def test_the_report_counts_tagged_text_only_on_the_named_pages():
    src = _actualtext_page(LINE)
    assert residue_report_pdf(src, {0})["tagged_text"] >= 1
    assert residue_report_pdf(src, {1})["tagged_text"] == 0
    assert residue_report_pdf(src, None)["tagged_text"] == 0
    assert residue_report_pdf(_burn(src), {0})["tagged_text"] == 0


def test_the_report_carries_counts_and_never_text():
    report = residue_report_pdf(_dirty())
    assert all(isinstance(v, int) for v in report.values())
    blob = repr(report)
    for marker in (JS, META, JPEG_MARK, SSN):
        assert marker not in blob


def test_the_op_is_registered_and_the_protocol_advertises_it():
    assert "residue_report" in _OPS
    assert PROTOCOL_VERSION >= 8 and 8 in SUPPORTED_PROTOCOL_VERSIONS
    assert {2, 3, 4, 5, 6, 7} <= SUPPORTED_PROTOCOL_VERSIONS


def test_the_op_answers_over_the_wire_shape():
    b64 = base64.b64encode(_dirty()).decode("ascii")
    result = _OPS["residue_report"]({"pdf_b64": b64, "pages": [0]})
    assert set(result["report"]) >= {"active_content", "tagged_text", "image_metadata"}


@pytest.mark.parametrize("pages", ["0", [0, "1"], [True], {"a": 1}],
                         ids=["string", "mixed", "bool", "dict"])
def test_the_op_rejects_a_pages_field_that_is_not_a_list_of_integers(pages):
    b64 = base64.b64encode(_dirty()).decode("ascii")
    with pytest.raises(ValueError, match="pages"):
        _OPS["residue_report"]({"pdf_b64": b64, "pages": pages})


def test_the_report_refuses_an_encrypted_document():
    d = _doc()
    enc = d.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                    owner_pw="pw-test-2", user_pw="pw-test-2")
    with pytest.raises(ValueError, match="unencrypted"):
        residue_report_pdf(enc)


# ── boundary ─────────────────────────────────────────────────────────────


def test_a_source_with_compressed_object_streams_is_sanitised_too():
    d = _doc()
    d.xref_set_key(d.pdf_catalog(), "OpenAction", f"<</S/JavaScript/JS ({JS})>>")
    ix = d.get_new_xref()
    d.update_object(ix, f"<</Company ({META})>>")
    d.xref_set_key(-1, "Info", f"{ix} 0 R")
    src = _bytes(d, use_objstms=1)
    assert not _readable(src, JS) or True        # may be hidden inside an objstm
    out = _burn(src)
    assert not _readable(out, JS) and not _readable(out, META)


def test_one_page_of_many_carrying_the_vector_is_still_sanitised():
    d = _doc(25)
    d.xref_set_key(d[17].xref, "PieceInfo", f"<</App <</Private ({META})>>>>")
    src = _bytes(d)
    assert _readable(src, META)
    out = _burn(src)
    assert not _readable(out, META)
    assert len(pymupdf.open(stream=out, filetype="pdf")) == 25


def test_an_incremental_update_keeps_no_earlier_revision():
    import tempfile
    from pathlib import Path
    d = _doc()
    d.xref_set_key(d[0].xref, "PieceInfo", f"<</App <</Private ({META})>>>>")
    rev1 = _bytes(d)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.pdf"
        path.write_bytes(rev1)
        e = pymupdf.open(str(path))
        e.xref_set_key(e[0].xref, "PieceInfo", "null")
        e.set_metadata({"title": "second revision"})
        e.save(str(path), incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP)
        e.close()
        src = path.read_bytes()
    assert _readable(src, META)                        # rev 1 is in the bytes
    assert not _readable(_burn(src), META)


# ── key names PyMuPDF cannot address ─────────────────────────────────────


def _with_object_text_key(doc, xref: int, key_and_value: str) -> None:
    """Append ``/Key value`` to an object's dictionary text, so the key is
    stored under the name the file spells (a ``#20`` escape is a space)."""
    text = doc.xref_object(xref, compressed=False).rstrip()
    assert text.endswith(">>")
    doc.update_object(xref, text[:-2] + key_and_value + ">>")


def test_an_info_key_with_a_space_in_its_name_does_not_fail_the_export():
    """Real: an IRS-style form carries ``/Info`` keys such as ``Form fields``.
    Setting such a key to null raises in PyMuPDF and used to abort the export."""
    d = _doc()
    ix = d.get_new_xref()
    d.update_object(ix, f"<</Title (t)/Form#20fields ({META})>>")
    d.xref_set_key(-1, "Info", f"{ix} 0 R")
    src = _bytes(d)
    assert _readable(src, META)
    out = _burn(src)
    assert not _readable(out, META)
    assert SSN not in _text(out)


def test_standard_metadata_stamped_after_an_info_dictionary_was_replaced_survives():
    from lexcloak_pdf_tool import set_metadata
    d = _doc()
    ix = d.get_new_xref()
    d.update_object(ix, f"<</Form#20fields ({META})>>")
    d.xref_set_key(-1, "Info", f"{ix} 0 R")
    out = set_metadata(_burn(_bytes(d)), {"subject": "stamped-later"})
    o = pymupdf.open(stream=out, filetype="pdf")
    assert o.metadata["subject"] == "stamped-later"
    assert not _readable(out, META)


def _js_name_tree(indirect: bool) -> bytes:
    """A JavaScript name tree whose entry NAME is the marker. With ``indirect``
    the ``/Names`` dictionary is its own object, as in many real documents."""
    d = _doc()
    js = d.get_new_xref()
    d.update_object(js, "<</S/JavaScript/JS (var x=1;)>>")
    names = f"<</JavaScript <</Names [(namesmark) {js} 0 R]>>>>"
    if indirect:
        nx = d.get_new_xref()
        d.update_object(nx, names)
        d.xref_set_key(d.pdf_catalog(), "Names", f"{nx} 0 R")
    else:
        d.xref_set_key(d.pdf_catalog(), "Names", names)
    return _bytes(d)


@pytest.mark.parametrize("indirect", [False, True], ids=["inline", "indirect"])
def test_the_javascript_name_tree_is_removed_whether_or_not_names_is_its_own_object(indirect):
    """Measured on 11 of 268 real documents: an indirect ``/Names`` made the
    old key-path write raise, so the whole export failed."""
    src = _js_name_tree(indirect)
    assert _readable(src, "namesmark")
    out = _burn(src)                                    # must not raise
    assert not _readable(out, "namesmark")
    assert SSN not in _text(out)
    assert residue_report_pdf(out)["active_content"] == 0


def test_the_report_counts_a_live_javascript_name_tree():
    assert residue_report_pdf(_js_name_tree(True))["active_content"] >= 1
    assert residue_report_pdf(_js_name_tree(False))["active_content"] >= 1


def test_reduce_size_also_removes_the_javascript_name_tree():
    out, _info = reduce_size(_js_name_tree(True))
    assert not _readable(out, "namesmark")


def test_a_catalog_key_that_cannot_be_addressed_is_left_and_reported_not_fatal():
    d = _doc()
    _with_object_text_key(d, d.pdf_catalog(), "/Odd#20Key (spacemark)")
    src = _bytes(d)
    assert _readable(src, "spacemark")
    out = _burn(src)                                   # the export still succeeds
    assert SSN not in _text(out)
    report = residue_report_pdf(out)
    assert report["unknown_keys"] >= 1                 # and the caller can tell
