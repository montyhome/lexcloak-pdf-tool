"""A box the payload repeats is burned once (v0.11.4).

PyMuPDF's ``add_redact_annot`` rescans every annotation on the page to give
the new one a unique name, and ``apply_redactions`` rescans them to load each
one, so the burn of one page grows with the square of its box count
(pymupdf 1.28.2: about 5 s for 1,000 boxes on one page, 24 s for 2,000). A
payload that sends each of a page's boxes many times pays that square for
nothing: a box burned twice removes nothing more and draws the same fill in
the same place. ``keep_uncovered_lines`` pays it up to four times, once per
pass over the page's boxes.

``_page_boxes`` now leaves out a box identical to one already listed on its
page (same rect, label and font size). The tests pin what that may and may not
change: a payload that repeats its boxes burns to the same bytes as one that
sends each box once; a box repeated with another label, or on another page,
is still burned; and a page adds one annotation per distinct box.

Every fixture is built in memory with invented text.
"""
from __future__ import annotations

import math

import pymupdf
import pytest

from lexcloak_pdf_tool.keep_lines import plan
from lexcloak_pdf_tool.redact import apply_redactions
from test_keep_uncovered_lines import _ink_box
from test_redact_extraction_goldens import _canonical_pdf

#: Invented names with descenders, one per line of single-spaced text: a box
#: over one reaches into the glyph boxes of the line below, which is the
#: shape ``keep_uncovered_lines`` exists for.
NAMES = ("Pippa Quigley", "Gregory Jupp", "Peggy Pryor", "Jasper Gray")
SIZE = 11.0


def _body_pdf(pages: int = 1) -> tuple[bytes, list[list[pymupdf.Rect]]]:
    """Single-spaced lines, each naming one of ``NAMES``; the boxes over the
    names on each page.

    Every box has the same height, reaching as far above and below its
    baseline as the tallest name's ink, so the boxes differ only in where
    they are: the label's font size follows the box height, and a box must
    not be taken for another because the two share a label and a size.
    """
    doc = pymupdf.open()
    boxes = []
    for _ in range(pages):
        page = doc.new_page(width=612, height=792)
        baselines = [120 + i * SIZE for i in range(len(NAMES))]
        for y, name in zip(baselines, NAMES, strict=True):
            page.insert_text((72, y), f"Visit notes for {name} were filed.", fontsize=SIZE)
        ink = [_ink_box(page, name) for name in NAMES]
        # Quarter points, exact in binary, so every box's height is equal
        # to the last bit and so is the font size the burn derives from it.
        up = math.ceil(4 * max(y - r.y0 for y, r in zip(baselines, ink, strict=True))) / 4
        down = math.ceil(4 * max(r.y1 - y for y, r in zip(baselines, ink, strict=True))) / 4
        boxes.append([pymupdf.Rect(r.x0, y - up, r.x1, y + down)
                      for y, r in zip(baselines, ink, strict=True)])
    return doc.tobytes(), boxes


def _matches(boxes: list[list[pymupdf.Rect]], repeat: int = 1, **extra) -> list[dict]:
    return [{"page": pno, "type": "Person Name",
             "rect": {"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}, **extra}
            for pno, rects in enumerate(boxes) for r in rects for _ in range(repeat)]


def test_the_fixture_is_one_keep_uncovered_lines_changes():
    """Guard: the keep_uncovered_lines cases below take its own path, not
    the historical one it falls back to when nothing would be kept, and the
    boxes on a page differ only in where they are."""
    pdf, boxes = _body_pdf()
    doc = pymupdf.open(stream=pdf)
    assert plan(doc, 0, boxes[0]) is not None
    assert len({b.height for b in boxes[0]}) == 1
    assert len({(b.x0, b.y0) for b in boxes[0]}) == len(NAMES)


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("label", ["", "REDACTED"])
def test_a_repeated_box_burns_like_one(keep, label):
    pdf, boxes = _body_pdf()
    once, _ = apply_redactions(pdf, _matches(boxes), redact_label=label,
                               keep_uncovered_lines=keep)
    many, _ = apply_redactions(pdf, _matches(boxes, repeat=25), redact_label=label,
                               keep_uncovered_lines=keep)
    assert _canonical_pdf(many) == _canonical_pdf(once)
    text = pymupdf.open(stream=many)[0].get_text()
    assert not any(name in text for name in NAMES)


def test_a_page_adds_one_annotation_per_distinct_box(monkeypatch):
    added = []
    real = pymupdf.Page.add_redact_annot

    def counting(page, *a, **kw):
        added.append(page.number)
        return real(page, *a, **kw)

    monkeypatch.setattr(pymupdf.Page, "add_redact_annot", counting)
    pdf, boxes = _body_pdf()
    apply_redactions(pdf, _matches(boxes, repeat=50))
    assert len(added) == len(NAMES)


def test_the_same_box_with_another_label_is_still_burned():
    """Two matches over one value, each with its own label: two boxes, as
    before. Only a box identical in its label too is left out."""
    pdf, boxes = _body_pdf()
    first = _matches(boxes, redact_label="Client A")
    second = _matches(boxes, redact_label="Client B")
    out, _ = apply_redactions(pdf, first + second)
    text = pymupdf.open(stream=out)[0].get_text()
    assert text.count("Client A") == len(NAMES)
    assert text.count("Client B") == len(NAMES)


def test_the_same_box_on_another_page_is_burned_there():
    """Boxes are compared within a page: the same rect on two pages burns on
    both."""
    pdf, boxes = _body_pdf(pages=2)
    assert boxes[0] == boxes[1]
    out, _ = apply_redactions(pdf, _matches(boxes, repeat=3))
    doc = pymupdf.open(stream=out)
    for page in doc:
        text = page.get_text()
        assert "Visit notes for" in text
        assert not any(name in text for name in NAMES)
