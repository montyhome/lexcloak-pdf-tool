"""Extraction-based redaction goldens — Session 659.

The S590 rotated-page goldens assert PIXELS only, and pixels are exactly the
test shape that let a scrub gap survive: a black box can land perfectly while
content under (or beyond) it stays extractable. These goldens pin the other
half of the redaction contract on every ``/Rotate`` value and content shape:

* per-match burn  -> the value is GONE from ``get_text()`` (and a co-located
  negative-control value SURVIVES, so "gone" cannot be satisfied by nuking
  the page);
* image pages     -> the burned region is DESTROYED in the embedded image
  bytes, not merely overdrawn;
* full-page blackout -> ZERO extractable chars, including glyphs hanging off
  the page box (clipped print headers whose descender slivers poke ~1-2pt
  into the page — the Session 659 live-export survivor class);
* page-edge match rects -> a rect flush against a page edge scrubs the
  clipped line beyond it (``_edge_overscan_strips``), while interior rects
  deliberately do NOT reach off-page.

All burn rects and probe points are hard-coded literals in the app-contract
as-rendered (rotation-applied) frame. Provenance: measured once against the
deterministic fixtures below via ``search_for`` + ``rotation_matrix`` on
pymupdf 1.27.2.3 AND 1.28.0 (identical to 0.01pt — base-14 Helvetica
metrics), then padded ~2pt. No transform code is mirrored here.
"""
from __future__ import annotations

import re

import fitz
import pytest

from lexcloak_pdf_tool.redact import (
    _edge_overscan_strips,
    apply_redactions,
)

SSN = "123-45-6789"
CONTROL = "CONTROL-KEEP-4455"

PORTRAIT_W, PORTRAIT_H = 612, 792


# ── fixture builders (deterministic, synthetic, PII-free) ───────────────


def _text_page_pdf(page_rotate: int, text_rotate: int = 0) -> bytes:
    """Portrait page: an SSN row (burn target), a control row (must survive),
    both drawn at ``text_rotate``; page then flagged ``/Rotate page_rotate``.

    ``text_rotate=0``  -> SSN native bbox (190.3, 256.0, 264.0, 273.9)
    ``text_rotate=90`` -> SSN native bbox (286.0, 528.0, 303.9, 601.7)
    (the real-world landscape composition: content drawn sideways, ``/Rotate``
    uprights it for display).
    """
    doc = fitz.open()
    page = doc.new_page(width=PORTRAIT_W, height=PORTRAIT_H)
    if text_rotate == 0:
        page.insert_text(fitz.Point(40, 270), f"Taxpayer SSN: {SSN}",
                         fontsize=13)
        page.insert_text(fitz.Point(40, 700), CONTROL, fontsize=13)
    elif text_rotate == 90:
        # reads bottom-to-top; SSN value lands at native (286..304, 528..602)
        page.insert_text(fitz.Point(300, 752), f"Taxpayer SSN: {SSN}",
                         fontsize=13, rotate=90)
        page.insert_text(fitz.Point(500, 752), CONTROL, fontsize=13,
                         rotate=90)
    else:  # pragma: no cover - guard against bad parametrize edits
        raise ValueError(f"unsupported text_rotate {text_rotate}")
    if page_rotate:
        page.set_rotation(page_rotate)
    out = doc.tobytes()
    doc.close()
    return out


def _image_page_pdf(page_rotate: int) -> bytes:
    """Portrait page with a 200x200 checkerboard placed at native
    (100, 100, 400, 400); page flagged ``/Rotate page_rotate``."""
    doc = fitz.open()
    page = doc.new_page(width=PORTRAIT_W, height=PORTRAIT_H)
    pm = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200))
    for y in range(200):
        for x in range(200):
            v = 255 if (x // 25 + y // 25) % 2 == 0 else 30
            pm.set_pixel(x, y, (v, v, v))
    page.insert_image(fitz.Rect(100, 100, 400, 400), pixmap=pm)
    if page_rotate:
        page.set_rotation(page_rotate)
    out = doc.tobytes()
    doc.close()
    return out


def _edge_header_pdf(page_rotate: int = 0) -> bytes:
    """The LM-Records survivor shape: a header line whose glyph boxes hang
    off the TOP page edge with descender slivers ~1.7pt in-page (baseline
    -2.5, fontsize 14 -> word boxes y in [-17.6, +1.7]), plus an in-page
    body line. Pre-fix, a full-page blackout left ``ppy ggy pyg y ( )``
    extractable on both pymupdf lines."""
    doc = fitz.open()
    page = doc.new_page(width=PORTRAIT_W, height=PORTRAIT_H)
    page.insert_text(fitz.Point(60, -2.5), "happy doggy pygmy (label)",
                     fontsize=14)
    page.insert_text(fitz.Point(60, 400), f"BODY {CONTROL}", fontsize=13)
    if page_rotate:
        page.set_rotation(page_rotate)
    out = doc.tobytes()
    doc.close()
    return out


# ── hard-coded app-contract (as-rendered) geometry ──────────────────────

# Burn rects covering the SSN row, per (text_rotate, page_rotate).
# Native boxes padded ~2pt, then expressed in the displayed frame.
BURN_RECT = {
    (0, 0): (188, 254, 266, 276),
    (0, 90): (516, 188, 538, 266),
    (0, 180): (346, 516, 424, 538),
    (0, 270): (254, 346, 276, 424),
    (90, 0): (284, 526, 306, 604),
    (90, 90): (188, 284, 266, 306),
    (90, 180): (306, 188, 328, 266),
    (90, 270): (526, 306, 604, 328),
}

# Displayed-frame probe points: center of the burned SSN region, and a point
# inside the control row (which must stay un-inked).
SSN_CENTER = {
    (0, 0): (227, 265), (0, 90): (527, 227),
    (0, 180): (385, 527), (0, 270): (265, 385),
    (90, 0): (295, 565), (90, 90): (227, 295),
    (90, 180): (317, 227), (90, 270): (565, 317),
}
CONTROL_POINT = {
    (0, 0): (100, 695), (0, 90): (97, 100),
    (0, 180): (512, 97), (0, 270): (695, 512),
    (90, 0): (509, 660), (90, 90): (132, 509),
    (90, 180): (103, 132), (90, 270): (660, 103),
}

# Burn rect over the checkerboard's native center quarter (200,200,300,300),
# expressed in the displayed frame per /Rotate.
IMAGE_BURN_RECT = {
    0: (200, 200, 300, 300),
    90: (492, 200, 592, 300),
    180: (312, 492, 412, 592),
    270: (200, 312, 300, 412),
}


# ── helpers ─────────────────────────────────────────────────────────────


def _page_text(pdf_bytes: bytes, page: int = 0) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc[page].get_text()
    finally:
        doc.close()


def _pixel_dark(pdf_bytes: bytes, page: int, x_pt: float, y_pt: float) -> bool:
    """Near-black check at an as-displayed point (get_pixmap honors /Rotate;
    72 dpi -> 1pt == 1px)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pix = doc[page].get_pixmap(matrix=fitz.Matrix(1, 1))
        r, g, b = pix.pixel(int(x_pt), int(y_pt))[:3]
        return r < 40 and g < 40 and b < 40
    finally:
        doc.close()


def _match(rect: tuple, mtype: str = "SSN") -> dict:
    x0, y0, x1, y1 = rect
    return {"id": "m1", "type": mtype, "page": 0, "enabled": True,
            "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}


# ── per-match: value gone from extraction, control survives ────────────


@pytest.mark.parametrize("page_rotate", [0, 90, 180, 270])
@pytest.mark.parametrize("text_rotate", [0, 90])
def test_burned_value_gone_from_extraction(text_rotate, page_rotate):
    """The scrub half: after a burn at the app-contract rect, the value must
    not extract — for every /Rotate and for both content shapes (upright
    native text, and sideways-native text that /Rotate uprights: the real
    landscape legal/medical composition)."""
    pdf = _text_page_pdf(page_rotate, text_rotate)
    assert SSN in _page_text(pdf)          # fixture sanity
    assert CONTROL in _page_text(pdf)

    out, _ = apply_redactions(pdf, [_match(BURN_RECT[(text_rotate,
                                                      page_rotate)])])
    redacted = _page_text(out)
    assert SSN not in redacted, (
        f"text_rotate={text_rotate} /Rotate={page_rotate}: burned value "
        f"still extractable — the scrub half missed")
    # Negative control: "gone" must not be satisfied by nuking the page.
    assert CONTROL in redacted, (
        f"text_rotate={text_rotate} /Rotate={page_rotate}: control value "
        f"was scrubbed — burn region leaked beyond the supplied rect")


@pytest.mark.parametrize("page_rotate", [0, 90, 180, 270])
@pytest.mark.parametrize("text_rotate", [0, 90])
def test_burned_region_fill_lands_and_is_localized(text_rotate, page_rotate):
    """The fill half (kept from the S590 pixel goldens): dark at the burned
    region's displayed center, NOT dark at the control row."""
    pdf = _text_page_pdf(page_rotate, text_rotate)
    out, _ = apply_redactions(pdf, [_match(BURN_RECT[(text_rotate,
                                                      page_rotate)])])
    cx, cy = SSN_CENTER[(text_rotate, page_rotate)]
    assert _pixel_dark(out, 0, cx, cy), (
        f"text_rotate={text_rotate} /Rotate={page_rotate}: fill missing at "
        f"displayed center ({cx},{cy})")
    px, py = CONTROL_POINT[(text_rotate, page_rotate)]
    assert not _pixel_dark(out, 0, px, py), (
        f"text_rotate={text_rotate} /Rotate={page_rotate}: unexpected ink "
        f"at control point ({px},{py})")


@pytest.mark.parametrize("page_rotate", [0, 90, 180, 270])
def test_burn_with_label_scrubs_value_and_extracts_label(page_rotate):
    """Labeled burns keep the same scrub contract; the overlay label itself
    becomes (harmless) extractable text on every /Rotate."""
    pdf = _text_page_pdf(page_rotate, text_rotate=0)
    out, _ = apply_redactions(pdf, [_match(BURN_RECT[(0, page_rotate)])],
                              redact_label="REDACTED")
    redacted = _page_text(out)
    assert SSN not in redacted
    assert "REDACTED" in redacted
    assert CONTROL in redacted


def test_disabled_match_is_not_burned():
    """Sad path: a disabled match must neither scrub nor ink."""
    pdf = _text_page_pdf(0, 0)
    m = _match(BURN_RECT[(0, 0)])
    m["enabled"] = False
    out, _ = apply_redactions(pdf, [m])
    assert SSN in _page_text(out)
    cx, cy = SSN_CENTER[(0, 0)]
    assert not _pixel_dark(out, 0, cx, cy)


# ── per-match redact_label (v0.6.4) ─────────────────────────────────────
#
# The document-level label stamps every box; a per-match label overrides it
# for one box, which is what lets a known-identity pseudonym ("Patient A")
# land on one person's boxes while the rest of the document keeps the
# default. Both burn in the SAME apply_redactions pass.
#
# Output bytes carry ONE source of run-to-run variation: the trailer /ID,
# a file identifier PyMuPDF regenerates on every save. Every content byte is
# stable. The additive-only contract is therefore asserted against
# _canonical_pdf() -- byte-identity modulo that identifier, which is as
# literal as "byte identical" can be made for a PDF writer.
#
# The WHOLE array is neutralized, not just part of it, because its shape is
# MuPDF-version-dependent: on 1.27 the first element is a literal string and
# only the second (hex) element varies, while on 1.29 both elements are hex
# and both vary. An earlier version of this helper neutralized only the
# trailing hex element and passed locally on 1.27 while failing CI on 1.29.

_TRAILER_ID_RE = re.compile(rb"/ID\s*\[.*?\]>>", re.S)


def _canonical_pdf(pdf_bytes: bytes) -> bytes:
    """``pdf_bytes`` with the whole random trailer /ID array neutralized."""
    return _TRAILER_ID_RE.sub(b"/ID[<FILEID>]>>", pdf_bytes)


def _labelled_match(rect: tuple, label, mid: str = "m1") -> dict:
    m = _match(rect)
    m["id"] = mid
    if label is not None:
        m["redact_label"] = label
    return m


def test_canonicalization_neutralizes_only_the_random_file_id():
    """Guard for the helper the additive-only tests lean on: two saves of the
    SAME payload differ in raw bytes (the /ID) but are canonically equal."""
    pdf = _text_page_pdf(0, 0)
    a, _ = apply_redactions(pdf, [_match(BURN_RECT[(0, 0)])],
                            redact_label="REDACTED")
    b, _ = apply_redactions(pdf, [_match(BURN_RECT[(0, 0)])],
                            redact_label="REDACTED")
    assert a != b, ("fixture assumption broken: output is now fully "
                    "deterministic, so the canonicalization is masking "
                    "nothing and these tests should compare raw bytes")
    assert _canonical_pdf(a) == _canonical_pdf(b)


def test_per_match_label_overrides_document_label_on_that_box_only():
    """The headline: a mixed payload draws the per-match label on its own box
    and the document label on the rest, in one burn pass."""
    pdf = _text_page_pdf(0, 0)
    matches = [
        _labelled_match(BURN_RECT[(0, 0)], "Patient A", mid="m1"),
        # Second box over the control row, no per-match label -> document one.
        _labelled_match((38, 688, 200, 706), None, mid="m2"),
    ]
    out, _ = apply_redactions(pdf, matches, redact_label="REDACTED")
    text = _page_text(out)
    assert "Patient A" in text, "per-match label was not drawn"
    assert "REDACTED" in text, "document label was not drawn on the unlabelled box"
    assert SSN not in text, "the labelled box stopped scrubbing"
    assert CONTROL not in text, "the unlabelled box stopped scrubbing"


def test_absent_per_match_label_falls_back_to_document_label():
    """A match with no ``redact_label`` key draws the document label."""
    pdf = _text_page_pdf(0, 0)
    out, _ = apply_redactions(pdf, [_match(BURN_RECT[(0, 0)])],
                              redact_label="REDACTED")
    assert "REDACTED" in _page_text(out)


def test_empty_per_match_label_is_identical_to_omitting_the_key():
    """``redact_label: ""`` means "no per-match label", NOT "suppress the
    document label on this box" -- the same value already means "plain black
    box" document-wide, so the inverted second meaning would be a trap."""
    pdf = _text_page_pdf(0, 0)
    absent, _ = apply_redactions(pdf, [_labelled_match(BURN_RECT[(0, 0)], None)],
                                 redact_label="REDACTED")
    empty, _ = apply_redactions(pdf, [_labelled_match(BURN_RECT[(0, 0)], "")],
                                redact_label="REDACTED")
    assert _canonical_pdf(absent) == _canonical_pdf(empty)
    assert "REDACTED" in _page_text(empty)


def test_per_match_label_on_disabled_match_draws_nothing():
    """Unhappy path: a label does not resurrect a disabled match."""
    pdf = _text_page_pdf(0, 0)
    m = _labelled_match(BURN_RECT[(0, 0)], "Patient A")
    m["enabled"] = False
    out, _ = apply_redactions(pdf, [m], redact_label="REDACTED")
    text = _page_text(out)
    assert "Patient A" not in text
    assert SSN in text, "disabled match was burned anyway"
    cx, cy = SSN_CENTER[(0, 0)]
    assert not _pixel_dark(out, 0, cx, cy)


def test_no_per_match_labels_leaves_the_document_path_untouched():
    """Additive-only: a payload carrying no per-match labels must produce the
    same output the document-level path always produced. Pinned here against
    a payload whose matches are stripped of the key entirely, so a future
    edit that (say) always writes a resolved label into each annotation would
    fail even though the visible label is unchanged."""
    pdf = _text_page_pdf(0, 0)
    plain = [{k: v for k, v in _match(BURN_RECT[(0, 0)]).items()}]
    assert "redact_label" not in plain[0]
    a, _ = apply_redactions(pdf, plain, redact_label="REDACTED")
    b, _ = apply_redactions(pdf, [_match(BURN_RECT[(0, 0)])],
                            redact_label="REDACTED")
    assert _canonical_pdf(a) == _canonical_pdf(b)


def test_per_match_label_with_no_document_label_labels_only_that_box():
    """Document label empty + one per-match label: that box is labelled, the
    other stays a plain black box."""
    pdf = _text_page_pdf(0, 0)
    matches = [
        _labelled_match(BURN_RECT[(0, 0)], "Patient A", mid="m1"),
        _labelled_match((38, 688, 200, 706), None, mid="m2"),
    ]
    out, _ = apply_redactions(pdf, matches, redact_label="")
    text = _page_text(out)
    assert "Patient A" in text
    assert SSN not in text
    assert CONTROL not in text


def test_unicode_per_match_label_is_drawn():
    """Boundary: a non-ASCII pseudonym must survive to the drawn text."""
    pdf = _text_page_pdf(0, 0)
    out, _ = apply_redactions(
        pdf, [_labelled_match(BURN_RECT[(0, 0)], "Patiënt Ä")],
        redact_label="REDACTED")
    text = _page_text(out)
    assert "Pati" in text and "nt" in text, (
        f"unicode label absent from extraction: {text!r}")
    assert SSN not in text


def test_overlong_per_match_label_still_scrubs_the_value():
    """Boundary: a label far wider than its box must not compromise the
    scrub. PyMuPDF may clip or drop the overflowing text -- what is NOT
    negotiable is that the underlying value is gone and the box is inked."""
    pdf = _text_page_pdf(0, 0)
    long_label = "Patient A " * 20
    out, _ = apply_redactions(
        pdf, [_labelled_match(BURN_RECT[(0, 0)], long_label)],
        redact_label="REDACTED")
    assert SSN not in _page_text(out)
    cx, cy = SSN_CENTER[(0, 0)]
    assert _pixel_dark(out, 0, cx, cy), "fill missing under an overlong label"


# ── image pages: pixels destroyed in the embedded image ────────────────


@pytest.mark.parametrize("page_rotate", [0, 90, 180, 270])
def test_image_region_destroyed_not_overdrawn(page_rotate):
    """Burning over an image must destroy the region INSIDE the embedded
    image bytes (blank/uniform), not merely paint over it — for every
    /Rotate. The image's un-burned corner keeps its checkerboard (the
    destruction is localized)."""
    pdf = _image_page_pdf(page_rotate)
    out, _ = apply_redactions(
        pdf, [_match(IMAGE_BURN_RECT[page_rotate], mtype="Manual Region")])

    doc = fitz.open(stream=out, filetype="pdf")
    try:
        page = doc[0]
        imgs = page.get_images(full=True)
        assert len(imgs) == 1, "embedded image count changed"
        pm = fitz.Pixmap(doc, imgs[0][0])
        if pm.colorspace and pm.colorspace.n > 3:
            pm = fitz.Pixmap(fitz.csRGB, pm)
        w, h = pm.width, pm.height

        # Center block (the burned native center quarter maps to the image
        # center for this geometry): must be uniform (destroyed).
        center_vals = {pm.pixel(x, y)[:3]
                       for y in range(int(h * 0.4), int(h * 0.6), 8)
                       for x in range(int(w * 0.4), int(w * 0.6), 8)}
        assert len(center_vals) <= 2, (
            f"/Rotate={page_rotate}: burned image region still carries "
            f"content — {sorted(center_vals)[:4]}")

        # Un-burned corner block: checkerboard survives (both tones seen).
        corner_vals = {pm.pixel(x, y)[:3]
                       for y in range(0, int(h * 0.2), 4)
                       for x in range(0, int(w * 0.2), 4)}
        assert len(corner_vals) >= 2, (
            f"/Rotate={page_rotate}: image destruction was not localized — "
            f"corner lost its checkerboard")
    finally:
        doc.close()

    # Rendered view: burned region displays dark.
    r = IMAGE_BURN_RECT[page_rotate]
    assert _pixel_dark(out, 0, (r[0] + r[2]) / 2, (r[1] + r[3]) / 2)


# ── blackout: zero extractable chars, incl. off-page sliver glyphs ──────


@pytest.mark.parametrize("page_rotate", [0, 90, 180, 270])
def test_blackout_scrubs_edge_hanging_glyphs(page_rotate):
    """The Session 659 live-export survivor class: a clipped print header
    whose descender glyphs (p/y/g) + form furniture poke ~1.7pt into the
    page survives a page-bounds blackout region on both pymupdf lines.
    The blackout invariant is solid black AND zero extractable chars —
    "most chars gone" is not redaction."""
    pdf = _edge_header_pdf(page_rotate)
    # Fixture sanity: the sliver glyphs ARE extractable pre-blackout.
    assert "ppy" in _page_text(pdf)

    out, _ = apply_redactions(pdf, [], blackout_pages=[0])
    assert _page_text(out).strip() == "", (
        f"/Rotate={page_rotate}: blackout left extractable chars: "
        f"{_page_text(out).strip()!r}")

    # Rendered: fully dark at center + all four corners (as-displayed dims
    # swap at 90/270).
    w, h = ((PORTRAIT_H, PORTRAIT_W) if page_rotate in (90, 270)
            else (PORTRAIT_W, PORTRAIT_H))
    for x, y in ((w * 0.5, h * 0.5), (15, 15), (w - 15, 15),
                 (15, h - 15), (w - 15, h - 15)):
        assert _pixel_dark(out, 0, x, y), (
            f"/Rotate={page_rotate}: blackout missed ({x:.0f},{y:.0f})")


def test_blackout_overscan_stays_on_its_page():
    """The inflated blackout region must not scrub the neighbouring page."""
    doc = fitz.open()
    p0 = doc.new_page(width=PORTRAIT_W, height=PORTRAIT_H)
    p0.insert_text(fitz.Point(60, 400), f"PAGE0 {SSN}", fontsize=13)
    p1 = doc.new_page(width=PORTRAIT_W, height=PORTRAIT_H)
    p1.insert_text(fitz.Point(60, 400), f"PAGE1 {CONTROL}", fontsize=13)
    pdf = doc.tobytes()
    doc.close()

    out, _ = apply_redactions(pdf, [], blackout_pages=[0])
    assert _page_text(out, 0).strip() == ""
    assert CONTROL in _page_text(out, 1)


# ── page-edge match rects: overscan strips ──────────────────────────────


def test_match_rect_flush_at_top_edge_scrubs_clipped_line():
    """A user rect flush against the top edge (y0=0) covering a clipped
    header's visible band must scrub the header's off-page glyphs too —
    the in-page slivers are what ``get_text`` (and copy-paste) return."""
    pdf = _edge_header_pdf(0)
    out, _ = apply_redactions(
        pdf, [_match((55, 0, 300, 12), mtype="Manual Region")])
    redacted = _page_text(out)
    assert "ppy" not in redacted and "pyg" not in redacted, (
        "clipped-header slivers survived a flush-edge rect")
    # Over-scrub guard: the body line far below the rect survives.
    assert CONTROL in redacted


def test_match_rect_interior_does_not_overscan():
    """Boundary semantics: a rect NOT flush against the edge (y0=1.5 >
    epsilon) must not reach off-page — the sliver glyphs deliberately
    survive. Interior rects keep byte-identical pre-659 behavior."""
    pdf = _edge_header_pdf(0)
    out, _ = apply_redactions(
        pdf, [_match((55, 1.5, 300, 12), mtype="Manual Region")])
    assert "ppy" in _page_text(out), (
        "interior rect unexpectedly scrubbed off-page content — overscan "
        "epsilon boundary moved")


class TestEdgeOverscanStrips:
    """Unit contract of the strip builder (pure geometry)."""

    PAGE = fitz.Rect(0, 0, 612, 792)

    def test_interior_rect_yields_no_strips(self):
        assert _edge_overscan_strips(
            fitz.Rect(50, 50, 200, 80), self.PAGE) == []

    def test_flush_top_yields_one_upward_strip(self):
        strips = _edge_overscan_strips(fitz.Rect(50, 0, 200, 30), self.PAGE)
        assert len(strips) == 1
        s = strips[0]
        assert (s.x0, s.x1) == (50, 200)
        assert s.y0 == -10000.0
        assert s.y1 == 1.0  # epsilon overlap into the page

    def test_epsilon_boundary_is_inclusive(self):
        # y0 exactly at epsilon (1.0) still counts as flush…
        assert len(_edge_overscan_strips(
            fitz.Rect(50, 1.0, 200, 30), self.PAGE)) == 1
        # …just past it does not.
        assert _edge_overscan_strips(
            fitz.Rect(50, 1.01, 200, 30), self.PAGE) == []

    def test_corner_rect_yields_two_strips(self):
        strips = _edge_overscan_strips(fitz.Rect(0, 0, 100, 100), self.PAGE)
        assert len(strips) == 2  # top + left

    def test_full_page_rect_yields_all_four(self):
        strips = _edge_overscan_strips(fitz.Rect(self.PAGE), self.PAGE)
        assert len(strips) == 4
