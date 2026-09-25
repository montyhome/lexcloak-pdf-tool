"""v0.11.0: a redaction box keeps the lines it does not reach.

MuPDF removes a glyph when a box reaches a tenth of the way into the glyph's
ascender-to-descender box, so a box over one line takes letters from the line
below without touching their ink. ``keep_uncovered_lines=True`` keeps them;
the default burn is unchanged.

Every fixture is built in memory with invented text.
"""
from __future__ import annotations

import base64

import pymupdf
import pytest

from lexcloak_pdf_tool.__main__ import _handle
from lexcloak_pdf_tool.keep_lines import plan
from lexcloak_pdf_tool.redact import apply_redactions

BANNER = "Patient: Priya Lanternfield  |  Clinic: Larkspire Family Medicine"
NAME = "Priya Lanternfield"
HEADING = "PROGRESS NOTE"
ABOVE = "The review covered the plan and the notes."
MIDDLE = "Seen by Priya Lanternfield today."
BELOW = "Follow-up visit is scheduled for the next opening."
_ACCURATE = pymupdf.TEXTFLAGS_RAWDICT | pymupdf.TEXT_ACCURATE_BBOXES
_ZOOM = 3


def _ink_box(page, text: str, pad: float = 0.2) -> pymupdf.Rect:
    """``text``'s ink on ``page``: its advance across, its glyph outlines
    down, plus ``pad`` -- the box a redaction draws over a value."""
    for block in page.get_text("rawdict", flags=_ACCURATE)["blocks"]:
        for line in block.get("lines", []):
            chars = [ch for span in line["spans"] for ch in span["chars"]]
            at = "".join(ch["c"] for ch in chars).find(text)
            if at < 0:
                continue
            ink = [ch["bbox"] for ch in chars[at:at + len(text)] if not ch["c"].isspace()]
            return pymupdf.Rect(min(b[0] for b in ink) - pad, min(b[1] for b in ink) - pad,
                                max(b[2] for b in ink) + pad, max(b[3] for b in ink) + pad)
    raise AssertionError(f"{text!r} not on the page")


def _match(rect: pymupdf.Rect, text: str = NAME, **extra) -> dict:
    return {"page": 0, "type": "Person Name", "text": text,
            "rect": {"x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1},
            **extra}


def _heading_pdf(gap: float = 11.0, extras=None) -> tuple[bytes, pymupdf.Rect]:
    """A 7 pt banner carrying ``NAME``, and a 12 pt heading ``gap`` pt below."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    if extras:
        extras(page)
    page.insert_text((45, 90), BANNER, fontsize=7)
    page.insert_text((40, 90 + gap), HEADING, fontsize=12)
    box = _ink_box(page, NAME)
    data = doc.tobytes()
    doc.close()
    return data, box


def _body_pdf(size: float = 11.0, below=None, rotate: int = 0,
              font: str = "helv") -> tuple[bytes, pymupdf.Rect]:
    """Three single-spaced lines (baseline to baseline = ``size``), ``NAME``
    on the middle one. ``below(page, point)`` draws the third line instead."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 200), ABOVE, fontsize=size, fontname=font)
    page.insert_text((72, 200 + size), MIDDLE, fontsize=size, fontname=font)
    point = (72, 200 + 2 * size)
    if below is None:
        page.insert_text(point, BELOW, fontsize=size, fontname=font)
    else:
        below(page, point)
    box = _ink_box(page, NAME)
    if rotate:
        page.set_rotation(rotate)
    data = doc.tobytes()
    doc.close()
    return data, box


def _burn(pdf: bytes, matches: list[dict], keep: bool, **kw) -> bytes:
    return apply_redactions(pdf, matches, keep_uncovered_lines=keep, **kw)[0]


def _text(pdf: bytes) -> str:
    doc = pymupdf.open(stream=pdf)
    try:
        return doc[0].get_text()
    finally:
        doc.close()


def _traced(pdf: bytes) -> str:
    """Every character the page's content carries, drawn or not."""
    doc = pymupdf.open(stream=pdf)
    try:
        return "".join(chr(c[0]) for s in doc[0].get_texttrace() for c in s["chars"])
    finally:
        doc.close()


def _gray(pdf: bytes) -> pymupdf.Pixmap:
    doc = pymupdf.open(stream=pdf)
    try:
        return doc[0].get_pixmap(matrix=pymupdf.Matrix(_ZOOM, _ZOOM),
                                 colorspace=pymupdf.csGRAY)
    finally:
        doc.close()


def _pixels(pix: pymupdf.Pixmap, rect: pymupdf.Rect) -> list[bytes]:
    x0, y0 = max(0, int(rect.x0 * _ZOOM)), max(0, int(rect.y0 * _ZOOM))
    x1, y1 = min(pix.w, int(rect.x1 * _ZOOM) + 1), min(pix.h, int(rect.y1 * _ZOOM) + 1)
    s = bytes(pix.samples)
    return [s[y * pix.stride + x0:y * pix.stride + x1] for y in range(y0, y1)]


def _heading_rect(pdf: bytes) -> pymupdf.Rect:
    doc = pymupdf.open(stream=pdf)
    try:
        return doc[0].search_for(HEADING)[0]
    finally:
        doc.close()


# ══ The defect, and the fix ══


def test_default_burn_still_takes_the_heading_below():
    """The mechanism this option exists for, on this pymupdf. If it stops
    failing, MuPDF changed its rule and the option may be unnecessary."""
    pdf, box = _heading_pdf()
    out = _text(_burn(pdf, [_match(box)], keep=False))
    assert HEADING not in out
    assert "PRO" in out   # only the letters under the box went


def test_box_does_not_touch_the_heading_ink():
    """The fixture is the defect's: the box stops above the capitals but
    reaches a tenth of the way into their glyph boxes."""
    pdf, box = _heading_pdf()
    heading = _heading_rect(pdf)
    doc = pymupdf.open(stream=pdf)
    ink_top = min(ch["bbox"][1]
                  for b in doc[0].get_text("rawdict", flags=_ACCURATE)["blocks"]
                  for ln in b.get("lines", []) for s in ln["spans"]
                  if abs(s["size"] - 12) < 0.01 for ch in s["chars"])
    doc.close()
    assert box.y1 < ink_top
    assert box.y1 - heading.y0 > 0.1 * heading.height


def test_heading_is_kept_and_the_name_removed():
    pdf, box = _heading_pdf()
    out = _burn(pdf, [_match(box)], keep=True)
    text = _text(out)
    assert HEADING in text
    assert "Priya" not in _traced(out)
    assert "Lanternfield" not in _traced(out)
    for word in ("Patient:", "Clinic:", "Larkspire", "Medicine"):
        assert word in text


#: The burn draws each box with a 1 pt border, so its black reaches half a
#: point past the rect: over the tops of the heading's round capitals here.
_BORDER = 0.5


def test_heading_renders_as_in_the_source():
    """Below the box's border, the kept heading's pixels are the source's."""
    pdf, box = _heading_pdf()
    heading = _heading_rect(pdf)
    heading.y0 = box.y1 + _BORDER + 1 / _ZOOM
    src, out = _gray(pdf), _gray(_burn(pdf, [_match(box)], keep=True))
    assert _pixels(out, heading) == _pixels(src, heading)


#: A rule wholly inside the box over ``NAME`` in ``_heading_pdf``: the burn
#: removes line art a box covers.
_UNDERLINE = ((75, 90.6), (120, 90.6))


def _furniture(page) -> None:
    """An image and two rules under the banner, one of them wholly under the
    box, for the fill/image/graphics pass."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 10), False)
    pix.set_rect(pix.irect, (200, 40, 40))
    page.insert_image(pymupdf.Rect(60, 83, 170, 93), pixmap=pix)
    page.draw_line((45, 91.5), (300, 91.5), color=(0, 0, 1), width=0.4)
    page.draw_line(*_UNDERLINE, color=(0, 0.5, 0), width=0.3)


def _drawn(pdf: bytes) -> list[tuple]:
    doc = pymupdf.open(stream=pdf)
    try:
        return sorted((tuple(round(v, 2) for v in d["rect"]), d.get("color"), d.get("fill"))
                      for d in doc[0].get_drawings())
    finally:
        doc.close()


def test_line_art_under_the_box_goes_as_in_the_default_burn():
    pdf, box = _heading_pdf(extras=_furniture)
    assert box.contains(pymupdf.Rect(_UNDERLINE[0], _UNDERLINE[1]))
    green = [d for d in _drawn(pdf) if d[1] == (0.0, 0.5, 0.0)]
    assert green                                     # the fixture draws it
    default = _drawn(_burn(pdf, [_match(box)], keep=False))
    assert not [d for d in default if d[1] == (0.0, 0.5, 0.0)]   # and the burn removes it
    assert _drawn(_burn(pdf, [_match(box)], keep=True)) == default


def test_fill_images_and_graphics_are_the_default_burns():
    """Only the kept glyphs differ from the default burn: the box, the image
    pixels and the rule under it render the same everywhere else."""
    pdf, box = _heading_pdf(extras=_furniture)
    kept = _gray(_burn(pdf, [_match(box)], keep=True))
    default = _gray(_burn(pdf, [_match(box)], keep=False))
    heading = _heading_rect(pdf)
    heading.y0 = box.y1 + _BORDER
    s_kept, s_default = bytes(kept.samples), bytes(default.samples)
    hx0, hy0 = int(heading.x0 * _ZOOM), int(heading.y0 * _ZOOM)
    hx1, hy1 = int(heading.x1 * _ZOOM) + 1, int(heading.y1 * _ZOOM) + 1
    differ = [(x, y) for y in range(kept.h) for x in range(kept.w)
              if s_kept[y * kept.stride + x] != s_default[y * default.stride + x]
              and not (hx0 <= x < hx1 and hy0 <= y < hy1)]
    assert differ == []
    # and the fill is really there, over the full box
    assert set(b"".join(_pixels(kept, pymupdf.Rect(box.x0 + 1, box.y0 + 0.5,
                                                   box.x1 - 1, box.y1 - 0.5)))) == {0}


@pytest.mark.parametrize("font", ["helv", "tiro"])
@pytest.mark.parametrize("size", [10.0, 11.0, 12.0])
def test_single_spaced_next_line_is_kept(size, font):
    pdf, box = _body_pdf(size, font=font)
    assert BELOW not in _text(_burn(pdf, [_match(box)], keep=False))
    out = _burn(pdf, [_match(box)], keep=True)
    text = _text(out)
    assert BELOW in text and ABOVE in text
    assert "Priya" not in _traced(out)
    assert "Seen by" in text and "today." in text


def test_label_is_drawn_on_the_box():
    pdf, box = _heading_pdf()
    out = _burn(pdf, [_match(box, redact_label="REDACTED")], keep=True)
    assert "REDACTED" in _text(out)
    assert HEADING in _text(out)


#: A line with no descenders, so a box can reach into its glyph boxes from
#: below without meeting its ink.
FLAT = "The notes were read and all is as noted."


def _flat_above_pdf() -> tuple[bytes, pymupdf.Rect, float]:
    """``FLAT`` then ``MIDDLE``, single-spaced at 11 pt; the box over
    ``NAME``, and ``FLAT``'s ink bottom over the box's width."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 200), FLAT, fontsize=11)
    page.insert_text((72, 211), MIDDLE, fontsize=11)
    box = _ink_box(page, NAME)
    bottom = max(ch["bbox"][3]
                 for b in page.get_text("rawdict", flags=_ACCURATE)["blocks"]
                 for ln in b.get("lines", []) for s in ln["spans"] for ch in s["chars"]
                 if ch["origin"][1] < 205 and ch["bbox"][2] > box.x0 and ch["bbox"][0] < box.x1)
    data = doc.tobytes()
    doc.close()
    return data, box, bottom


def test_line_above_is_kept_when_the_box_stops_below_its_ink():
    """A taller box (a region drawn by hand, say) that rises into the line
    above's glyph boxes but stays under its ink keeps that line."""
    pdf, box, bottom = _flat_above_pdf()
    tall = pymupdf.Rect(box.x0, bottom + 0.3, box.x1, box.y1)
    assert FLAT not in _text(_burn(pdf, [_match(tall)], keep=False))
    out = _burn(pdf, [_match(tall)], keep=True)
    assert FLAT in _text(out)
    assert "Priya" not in _traced(out)


# ══ Unhappy paths: where the old rule must still decide ══


def test_box_reaching_the_line_above_ink_still_takes_it():
    pdf, box, bottom = _flat_above_pdf()
    reaching = pymupdf.Rect(box.x0, bottom - 0.3, box.x1, box.y1)
    kept = _text(_burn(pdf, [_match(reaching)], keep=True))
    assert kept == _text(_burn(pdf, [_match(reaching)], keep=False))
    assert FLAT not in kept


def test_box_reaching_the_next_lines_ink_still_takes_it():
    """A box that meets the next line's ink by a fraction of a point removes
    what it touches, exactly as the default burn does."""
    pdf, box = _body_pdf()
    doc = pymupdf.open(stream=pdf)
    ink_top = min(ch["bbox"][1]
                  for b in doc[0].get_text("rawdict", flags=_ACCURATE)["blocks"]
                  for ln in b.get("lines", []) for s in ln["spans"]
                  for ch in s["chars"] if ch["origin"][1] > 220)
    doc.close()
    reaching = pymupdf.Rect(box.x0, box.y0, box.x1, ink_top + 0.3)
    kept = _text(_burn(pdf, [_match(reaching)], keep=True))
    assert kept == _text(_burn(pdf, [_match(reaching)], keep=False))
    assert BELOW not in kept


def test_a_glyph_beside_the_box_on_its_own_line_is_still_removed():
    """A short box that stops inside the next glyph's side bearing takes that
    glyph today; keeping lines must not change a box's own line."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 300), "Account 1111-1111-9012 was debited.", fontsize=12)
    hit = page.search_for("1111-1111-901")[0]
    two = page.search_for("2 was")[0]
    pdf = doc.tobytes()
    doc.close()
    # into the "2"'s advance by 12% of its width, short of its ink
    short = pymupdf.Rect(hit.x0, hit.y0 + 3, two.x0 + 0.12 * 6.67, hit.y1 - 3)
    kept = _text(_burn(pdf, [_match(short, "1111-1111-9012")], keep=True))
    assert kept == _text(_burn(pdf, [_match(short, "1111-1111-9012")], keep=False))
    assert "2 was" not in kept


def _invisible(page, point) -> None:
    page.insert_text(point, BELOW, fontsize=11, render_mode=3)


def _layered(page, point) -> None:
    ocg = page.parent.add_ocg("Notes", on=True)
    page.insert_text(point, BELOW, fontsize=11, oc=ocg)


def _twice(page, point) -> None:
    """Drawn twice at the same origins, as some producers fake bold."""
    page.insert_text(point, BELOW, fontsize=11)
    page.insert_text(point, BELOW, fontsize=11)


@pytest.mark.parametrize("below", [_invisible, _layered, _twice],
                         ids=["invisible", "layer", "twice"])
def test_text_that_is_not_plainly_drawn_keeps_the_old_rule(below):
    pdf, box = _body_pdf(below=below)
    assert _traced(_burn(pdf, [_match(box)], keep=True)) == \
        _traced(_burn(pdf, [_match(box)], keep=False))


def test_rotated_page_burns_as_before():
    pdf, box = _body_pdf(rotate=90)
    doc = pymupdf.open(stream=pdf)
    shown = box * doc[0].rotation_matrix   # match rects are in the shown frame
    doc.close()
    kept = _traced(_burn(pdf, [_match(shown)], keep=True))
    assert kept == _traced(_burn(pdf, [_match(shown)], keep=False))
    assert "Priya" not in kept


def _subscripted(page) -> None:
    """A 3 pt figure after the name, lowered so its glyph box lies only in
    the part of the box the heading's trim gives up."""
    hit = page.search_for(NAME)
    x = hit[0].x1 - 4 if hit else 128
    page.insert_text((x, 91.6), "2", fontsize=3)


def test_floor_repair_removes_what_the_trimmed_pass_missed():
    """The trimmed text pass cannot reach the lowered figure under the box,
    so the floor check applies the old removal: the figure goes, and with it
    the heading, which the old rule takes."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((45, 90), BANNER, fontsize=7)
    page.insert_text((40, 101), HEADING, fontsize=12)
    box = _ink_box(page, NAME)
    _subscripted(page)
    pdf = doc.tobytes()
    doc.close()
    assert "2" in _traced(pdf)
    check = pymupdf.open(stream=pdf)
    figure = [ch["bbox"] for b in check[0].get_text("rawdict", flags=_ACCURATE)["blocks"]
              for ln in b.get("lines", []) for s in ln["spans"] if s["size"] < 4
              for ch in s["chars"]][0]
    assert box.y0 <= figure[1] and figure[3] <= box.y1   # its ink is under the box
    planned = plan(check, 0, [box])
    check.close()
    assert planned is not None       # the heading would be kept...
    out = _burn(pdf, [_match(box)], keep=True)
    assert "2" not in _traced(out)   # ...but the figure under the box goes
    assert _traced(out) == _traced(_burn(pdf, [_match(box)], keep=False))


def test_a_page_with_nothing_to_keep_is_not_planned():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 300), MIDDLE, fontsize=11)
    box = _ink_box(page, NAME)
    assert plan(doc, 0, [box]) is None
    doc.close()


def test_default_is_the_historical_burn():
    """Without the flag the burn is the old one: same text, same render."""
    pdf, box = _heading_pdf(extras=_furniture)
    old = apply_redactions(pdf, [_match(box)])[0]
    new = _burn(pdf, [_match(box)], keep=False)
    assert _traced(old) == _traced(new)
    assert bytes(_gray(old).samples) == bytes(_gray(new).samples)


# ══ The wire field ══


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_wire_flag_on_both_ops():
    pdf, box = _heading_pdf()
    resp = _handle({"op": "apply_redactions", "pdf_b64": _b64(pdf),
                    "matches": [_match(box)], "keep_uncovered_lines": True})
    assert resp["ok"], resp
    assert HEADING in _text(base64.b64decode(resp["result"]["pdf_b64"]))
    handle = _handle({"op": "open_doc", "pdf_b64": _b64(pdf)})["result"]["handle"]
    resp = _handle({"op": "apply_redactions_h", "handle": handle,
                    "matches": [_match(box)], "keep_uncovered_lines": True})
    _handle({"op": "close_doc", "handle": handle})
    assert resp["ok"], resp
    assert HEADING in _text(base64.b64decode(resp["result"]["pdf_b64"]))


def test_wire_flag_absent_is_the_default_burn():
    pdf, box = _heading_pdf()
    resp = _handle({"op": "apply_redactions", "pdf_b64": _b64(pdf),
                    "matches": [_match(box)]})
    assert resp["ok"], resp
    assert HEADING not in _text(base64.b64decode(resp["result"]["pdf_b64"]))


@pytest.mark.parametrize("value", ["true", 1, None])
def test_wire_flag_must_be_a_boolean(value):
    pdf, box = _heading_pdf()
    resp = _handle({"op": "apply_redactions", "pdf_b64": _b64(pdf),
                    "matches": [_match(box)], "keep_uncovered_lines": value})
    assert not resp["ok"]
    assert "keep_uncovered_lines" in resp["error"]
