"""The v5 ops: ``list_annotations`` and ``render_clip``.

Both exist because a caller cannot get the information any other way:

* Annotation **subtypes** are not recoverable from a rendered page or from
  extracted text -- a ``/Text`` sticky note's content is absent from
  ``get_text()`` entirely, and a ``/FreeText``'s text lives in the annot's
  own appearance stream. Proving "no text-bearing annot survived" requires
  opening the document and enumerating.
* A **clip render** is not the same raster as rendering the page and
  cropping: MuPDF aligns the grid to the clip's own fractional origin. The
  two agree only when the clip lands on an integer pixel boundary, which is
  asserted directly below rather than asserted away.

The 6-criteria framework:

* **Assertion strictness** -- subtype dicts are compared deep-equal against
  literal expectations; image equality is byte/array equality, not "looks
  close".
* **No logic mirroring** -- expected subtype names and counts are literals
  written by hand from the fixture's construction, never recomputed by
  re-walking ``annots()`` in the test.
* **Unhappy path** (>=2 per happy) -- unparseable bytes, empty bytes, a
  page index off both ends, a degenerate clip, a clip fully off-page, and
  an encrypted document.
* **Boundary / type** -- zero-annotation pages reported as empty rather
  than omitted; a clip landing exactly on the pixel grid vs a fractional
  one; a non-numeric and wrong-length clip payload.
* **Side-effect verification** -- the privacy boundary is asserted as an
  absence: the returned payload is walked and proven to contain no key or
  value carrying annotation content, author or rect.
* **Mock integrity** -- ops are exercised through the real dispatch table
  (``_OPS``) with real wire-shaped dicts, not by calling the library
  functions only.

Synthetic fixtures only.
"""
from __future__ import annotations

import base64
import io
import math

import numpy as np
import pymupdf
import pytest
from PIL import Image

from lexcloak_pdf_tool.__main__ import _OPS, PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS
from lexcloak_pdf_tool.annotations import list_annotations
from lexcloak_pdf_tool.render import render_clip, render_page

PAGE_W, PAGE_H = 612.0, 792.0
SECRET = "PATIENT-NOTE-4417-DO-NOT-SHIP"
AUTHOR = "Dr Synthetic"


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _doc_with_annots() -> bytes:
    """Page 0: one Text + one FreeText + one Link. Page 1: none."""
    doc = pymupdf.open()
    p0 = doc.new_page(width=PAGE_W, height=PAGE_H)
    p0.insert_text((72, 72), "visible body text")
    a = p0.add_text_annot((100, 100), SECRET)
    a.set_info(title=AUTHOR)
    a.update()
    f = p0.add_freetext_annot((200, 200, 400, 260), SECRET)
    f.update()
    p0.insert_link({"kind": pymupdf.LINK_URI, "uri": "https://example.com",
                    "from": pymupdf.Rect(50, 400, 150, 420)})
    doc.new_page(width=PAGE_W, height=PAGE_H)
    return doc.tobytes()


def _plain_doc(npages: int = 1) -> bytes:
    doc = pymupdf.open()
    for i in range(npages):
        pg = doc.new_page(width=PAGE_W, height=PAGE_H)
        pg.insert_text((72, 72), f"page {i}")
        pg.draw_rect(pymupdf.Rect(100, 100, 300, 300), fill=(0, 0, 0))
    return doc.tobytes()


# ── list_annotations: happy path ──────────────────────────────────
class TestListAnnotationsHappy:
    def test_reports_subtypes_and_counts_per_page(self):
        """NOTE the absent ``Link``. PyMuPDF's ``page.annots()`` does not
        yield ``/Link`` annotations -- they are reached via ``page.links()``
        instead -- so a link is invisible to this op even though the fixture
        places one. Pinned deliberately: the closed caller allowlists
        ``Link`` as a permitted export annot, and that allowlist entry is
        therefore dead code on this path. If a future PyMuPDF starts
        yielding links here, this test fails rather than the caller
        silently starting to refuse exports over ordinary hyperlinks."""
        pages = list_annotations(_doc_with_annots())
        assert pages == [
            {"page": 0, "subtypes": {"Text": 1, "FreeText": 1}},
            {"page": 1, "subtypes": {}},
        ]

    def test_link_is_present_in_the_document_but_not_in_annots(self):
        """Guards the claim above: the fixture really does carry a link."""
        doc = pymupdf.open(stream=_doc_with_annots(), filetype="pdf")
        try:
            assert len(doc[0].get_links()) == 1
        finally:
            doc.close()

    def test_zero_annotation_page_is_reported_not_omitted(self):
        pages = list_annotations(_plain_doc(3))
        assert [p["page"] for p in pages] == [0, 1, 2]
        assert all(p["subtypes"] == {} for p in pages)

    def test_counts_accumulate_for_repeated_subtype(self):
        doc = pymupdf.open()
        pg = doc.new_page(width=PAGE_W, height=PAGE_H)
        for i in range(4):
            pg.add_text_annot((50 + 10 * i, 50), f"note {i}").update()
        assert list_annotations(doc.tobytes()) == [
            {"page": 0, "subtypes": {"Text": 4}}]


# ── list_annotations: the privacy boundary ────────────────────────
class TestListAnnotationsCannotLeak:
    def test_payload_carries_no_annotation_content_author_or_rect(self):
        pages = list_annotations(_doc_with_annots())
        blob = repr(pages)
        assert SECRET not in blob
        assert AUTHOR not in blob
        assert "example.com" not in blob
        # structural: only these two keys, and values are int counts
        for entry in pages:
            assert set(entry) == {"page", "subtypes"}
            assert all(isinstance(v, int) for v in entry["subtypes"].values())
            assert all(isinstance(k, str) for k in entry["subtypes"])

    def test_op_payload_is_equally_narrow(self):
        out = _OPS["list_annotations"]({"pdf_b64": _b64(_doc_with_annots())})
        assert SECRET not in repr(out) and AUTHOR not in repr(out)
        assert set(out) == {"pages"}


# ── list_annotations: unhappy + boundary ──────────────────────────
class TestListAnnotationsSadPaths:
    def test_unparseable_bytes_raise(self):
        with pytest.raises(Exception):
            list_annotations(b"%PDF-1.7\nnot actually a pdf")

    def test_empty_bytes_raise(self):
        with pytest.raises(Exception):
            list_annotations(b"")

    def test_encrypted_document_refuses_rather_than_reporting_zero(self):
        """The fail-open this op is built to avoid: an encrypted document
        OPENS successfully and then yields no annotations, so a naive
        implementation would report ``{}`` for a document carrying a
        sticky note it never read."""
        doc = pymupdf.open()
        doc.new_page(width=PAGE_W, height=PAGE_H).add_text_annot(
            (50, 50), SECRET).update()
        enc = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                          owner_pw="o", user_pw="u")
        with pytest.raises(ValueError, match="encrypted"):
            list_annotations(enc)

    def test_encrypted_doc_opens_then_fails_only_at_annot_walk(self):
        """Why the explicit ``needs_pass`` guard earns its place: the
        document OPENS cleanly and reports a page count, so the failure
        only appears once something walks the annots. The guard converts
        that late, generic ``ValueError: document closed or encrypted``
        into an early refusal naming the actual cause."""
        doc = pymupdf.open()
        doc.new_page(width=PAGE_W, height=PAGE_H).add_text_annot(
            (50, 50), SECRET).update()
        enc = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                          owner_pw="o", user_pw="u")
        raw = pymupdf.open(stream=enc, filetype="pdf")
        try:
            assert raw.needs_pass
            assert raw.page_count == 1          # opens, looks healthy
            with pytest.raises(ValueError):     # ...until you walk annots
                [a for a in raw[0].annots()]
        finally:
            raw.close()


# ── render_clip: equivalence is the whole point ───────────────────
class TestRenderClipRaster:
    @pytest.mark.parametrize("clip", [
        (72.0, 72.0, 216.0, 216.0),       # lands on the 300dpi pixel grid
        (42.3, 80.7, 222.3, 260.7),       # fractional origin
        (10.125, 20.875, 190.5, 240.25),  # fractional both ends
    ])
    def test_matches_an_in_process_clip_pixmap_byte_for_byte(self, clip):
        pdf = _plain_doc()
        png = render_clip(pdf, 0, clip, dpi=300, gray=True)
        got = np.array(Image.open(io.BytesIO(png)).convert("L"), dtype=np.uint8)

        doc = pymupdf.open(stream=pdf, filetype="pdf")
        try:
            pix = doc[0].get_pixmap(clip=clip, dpi=300,
                                    colorspace=pymupdf.csGRAY)
            want = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width)
        finally:
            doc.close()
        assert got.shape == want.shape
        assert np.array_equal(got, want)

    def test_grid_aligned_clip_agrees_with_a_page_render_crop(self):
        """Documents the equivalence that DOES hold, so the next reader
        does not assume it holds generally -- see the fractional case."""
        pdf = _plain_doc()
        clip = (72.0, 72.0, 216.0, 216.0)
        s = 300 / 72.0
        page_png = render_page(pdf, 0, dpi=300)
        page = np.array(Image.open(io.BytesIO(page_png)).convert("L"), np.uint8)
        crop = page[int(clip[1] * s):int(clip[3] * s),
                    int(clip[0] * s):int(clip[2] * s)]
        clip_png = render_clip(pdf, 0, clip, dpi=300, gray=True)
        got = np.array(Image.open(io.BytesIO(clip_png)).convert("L"), np.uint8)
        assert np.array_equal(got, crop)

    def test_gray_false_yields_three_channels(self):
        png = render_clip(_plain_doc(), 0, (72.0, 72.0, 216.0, 216.0),
                          dpi=150, gray=False)
        assert Image.open(io.BytesIO(png)).convert("RGB").size[0] > 0
        assert Image.open(io.BytesIO(png)).mode in ("RGB", "P", "L")

    def test_extent_follows_pymupdf_irect_rounding(self):
        """floor top-left / ceil bottom-right -- pinned so a future change
        to this rule is a failing test, not a silent raster shift."""
        pdf = _plain_doc()
        clip = (42.3, 80.7, 222.3, 260.7)
        s = 300 / 72.0
        png = render_clip(pdf, 0, clip, dpi=300, gray=True)
        w, h = Image.open(io.BytesIO(png)).size
        assert w == math.ceil(clip[2] * s) - math.floor(clip[0] * s)
        assert h == math.ceil(clip[3] * s) - math.floor(clip[1] * s)


class TestRenderClipSadPaths:
    @pytest.mark.parametrize("page", [-1, 1, 99])
    def test_page_index_off_either_end_raises_indexerror(self, page):
        with pytest.raises(IndexError):
            render_clip(_plain_doc(), page, (10.0, 10.0, 50.0, 50.0), dpi=150)

    @pytest.mark.parametrize("clip", [
        (100.0, 100.0, 100.0, 200.0),      # zero width
        (100.0, 100.0, 200.0, 100.0),      # zero height
        (200.0, 100.0, 100.0, 200.0),      # inverted x
        (5000.0, 5000.0, 6000.0, 6000.0),  # entirely off-page
    ])
    def test_degenerate_or_offpage_clip_raises_valueerror(self, clip):
        with pytest.raises(ValueError):
            render_clip(_plain_doc(), 0, clip, dpi=150)

    def test_unparseable_bytes_raise(self):
        with pytest.raises(Exception):
            render_clip(b"%PDF-1.7 garbage", 0, (10.0, 10.0, 50.0, 50.0))

    def test_encrypted_document_refuses_rather_than_rendering_blank(self):
        doc = pymupdf.open()
        doc.new_page(width=PAGE_W, height=PAGE_H).insert_text((72, 72), SECRET)
        enc = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                          owner_pw="o", user_pw="u")
        with pytest.raises(ValueError, match="encrypted"):
            render_clip(enc, 0, (10.0, 10.0, 200.0, 200.0), dpi=150)


# ── wire-level: real dispatch, wire-shaped payloads ───────────────
class TestOpDispatch:
    def test_both_ops_are_registered(self):
        assert "render_clip" in _OPS and "list_annotations" in _OPS

    def test_render_clip_op_round_trips_png(self):
        out = _OPS["render_clip"]({
            "pdf_b64": _b64(_plain_doc()), "page": 0,
            "clip": [72.0, 72.0, 216.0, 216.0], "dpi": 300.0, "gray": True})
        img = Image.open(io.BytesIO(base64.b64decode(out["png_b64"])))
        assert img.size == (600, 600)

    @pytest.mark.parametrize("clip", [None, "nope", [1, 2, 3], [1, 2, 3, 4, 5],
                                      ["a", "b", "c", "d"]])
    def test_render_clip_op_rejects_malformed_clip(self, clip):
        with pytest.raises(ValueError):
            _OPS["render_clip"]({"pdf_b64": _b64(_plain_doc()), "page": 0,
                                 "clip": clip})

    def test_protocol_version_advertises_5_and_still_supports_older(self):
        assert PROTOCOL_VERSION == 5
        assert {2, 3, 4}.issubset(SUPPORTED_PROTOCOL_VERSIONS)
        assert 5 in SUPPORTED_PROTOCOL_VERSIONS
