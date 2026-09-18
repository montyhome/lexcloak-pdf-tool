"""``apply_redactions`` with ``remove_text`` (v7): remove text, change nothing drawn.

Every fixture is built in memory with invented text. A removal is judged three
ways: the target text is gone from the raw content (not only from
``get_text``), every other character is still there, and -- where the target
text drew nothing -- the page renders pixel-identical at 150 DPI.
"""
from __future__ import annotations

import base64

import pymupdf
import pytest

from lexcloak_pdf_tool.__main__ import _OPS, _handle
from lexcloak_pdf_tool.redact import apply_redactions
from lexcloak_pdf_tool.trace import trace_text

TARGET = "Zqvx 700-11-2233"


def _doc():
    doc = pymupdf.open()
    return doc, doc.new_page(width=612, height=792)


def _bytes(doc) -> bytes:
    data = doc.tobytes()
    doc.close()
    return data


def _word_boxes(pdf: bytes, needle_words: set[str], page: int = 0) -> list[dict]:
    """The requested words as the app sends them: box plus identity fields."""
    return [{"box": w["bbox"], "text": w["text"], "mode": w["mode"],
             "opacity": w["opacity"], "layer": w["layer"]}
            for w in trace_text(pdf, page)["words"] if w["text"] in needle_words]


def _traced_text(pdf: bytes, page: int = 0) -> str:
    """Every character the page carries, all layers, off-page included."""
    doc = pymupdf.open(stream=pdf)
    ocgs = doc.get_ocgs()
    if ocgs:
        doc.set_layer(-1, on=list(ocgs), off=[])
        doc = pymupdf.open(stream=doc.tobytes())
    return "".join(chr(c[0]) for s in doc[page].get_texttrace() for c in s["chars"])


def _stream_holds(pdf: bytes, needle: str) -> bool:
    doc = pymupdf.open(stream=pdf)
    hexed = needle.encode("latin-1").hex().encode()
    for xref in range(1, doc.xref_length()):
        if doc.xref_is_stream(xref):
            data = doc.xref_stream(xref) or b""
            if needle.encode() in data or hexed in data or hexed.upper() in data:
                return True
    return False


def _gray(pdf: bytes, page: int = 0) -> bytes:
    doc = pymupdf.open(stream=pdf)
    return doc[page].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY).samples


def _remove(pdf: bytes, boxes: list, page: int = 0, **kw):
    sink: dict = {}
    out, _ = apply_redactions(pdf, [], remove_text=[
        {"page": page, **b} for b in boxes], removal_sink=sink, **kw)
    return out, sink


def _assert_gone(pdf: bytes, needle: str = TARGET):
    assert needle.replace(" ", "") not in _traced_text(pdf).replace(" ", "")
    for part in needle.split():
        assert not _stream_holds(pdf, part), part


def _hidden(**insert_kw) -> bytes:
    doc, page = _doc()
    page.insert_text((72, 100), "Visible heading stays", fontsize=12)
    page.insert_text((72, 300), TARGET, fontsize=12, **insert_kw)
    return _bytes(doc)


VECTORS = {
    "white": dict(color=(1, 1, 1)),
    "invisible": dict(render_mode=3),
    "transparent": dict(fill_opacity=0),
}


@pytest.mark.parametrize("name", list(VECTORS))
def test_removes_text_that_draws_nothing_and_changes_no_pixel(name):
    pdf = _hidden(**VECTORS[name])
    out, sink = _remove(pdf, _word_boxes(pdf, {"Zqvx", "700-11-2233"}))
    _assert_gone(out)
    assert "Visible heading stays".replace(" ", "") in _traced_text(out).replace(" ", "")
    assert _gray(out) == _gray(pdf)
    assert sink == {"removed": [[0, 0], [0, 1]], "kept": []}


def test_removes_text_under_a_black_box_and_keeps_the_box():
    doc, page = _doc()
    page.insert_text((72, 300), TARGET, fontsize=12)
    page.draw_rect(pymupdf.Rect(66, 285, 200, 306), color=None, fill=(0, 0, 0))
    pdf = _bytes(doc)
    out, _ = _remove(pdf, _word_boxes(pdf, {"Zqvx", "700-11-2233"}))
    _assert_gone(out)
    assert _gray(out) == _gray(pdf)
    assert pymupdf.open(stream=out)[0].get_drawings(), "the box itself must stay"


def test_removes_text_under_an_image_and_keeps_the_image():
    doc, page = _doc()
    page.insert_text((72, 300), TARGET, fontsize=12)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 10), False)
    pix.clear_with(140)
    page.insert_image(pymupdf.Rect(66, 285, 246, 315), pixmap=pix,
                      keep_proportion=False)
    pdf = _bytes(doc)
    out, _ = _remove(pdf, _word_boxes(pdf, {"Zqvx", "700-11-2233"}))
    _assert_gone(out)
    assert _gray(out) == _gray(pdf)
    assert pymupdf.open(stream=out)[0].get_images()


def test_removes_text_outside_the_page():
    doc, page = _doc()
    page.insert_text((72, -20), TARGET, fontsize=12)
    page.insert_text((652, 300), TARGET, fontsize=12)
    page.insert_text((72, 300), "Visible body", fontsize=12)
    pdf = _bytes(doc)
    out, sink = _remove(pdf, _word_boxes(pdf, {"Zqvx", "700-11-2233"}))
    _assert_gone(out)
    assert sink["kept"] == []
    assert _gray(out) == _gray(pdf)


def test_removes_text_in_a_switched_off_layer_and_leaves_the_layer_off():
    doc, page = _doc()
    off = doc.add_ocg("Notes", on=False)
    page.insert_text((72, 300), TARGET, fontsize=12, oc=off)
    pdf = _bytes(doc)
    out, _ = _remove(pdf, _word_boxes(pdf, {"Zqvx", "700-11-2233"}))
    _assert_gone(out)
    check = pymupdf.open(stream=out)
    assert [v["on"] for v in check.get_ocgs().values()] == [False]
    assert _gray(out) == _gray(pdf)


def test_removes_text_clipped_out_of_view():
    doc, page = _doc()
    src = pymupdf.open()
    sp = src.new_page(width=612, height=792)
    sp.insert_text((72, 140), "Shown line", fontsize=12)
    sp.insert_text((72, 400), TARGET, fontsize=12)
    clip = pymupdf.Rect(0, 120, 612, 160)
    page.show_pdf_page(clip, src, 0, clip=clip)
    src.close()
    pdf = _bytes(doc)
    out, _ = _remove(pdf, _word_boxes(pdf, {"Zqvx", "700-11-2233"}))
    _assert_gone(out)
    assert "Shownline" in _traced_text(out).replace(" ", "")
    assert _gray(out) == _gray(pdf)


def test_punctuation_inside_the_box_goes_too():
    doc, page = _doc()
    page.insert_text((72, 100), "Visible heading stays", fontsize=12)
    page.insert_text((72, 340), "a.b,c_d 'q'", fontsize=12, color=(1, 1, 1))
    pdf = _bytes(doc)
    out, sink = _remove(pdf, _word_boxes(pdf, {"a.b,c_d", "'q'"}))
    assert sink["kept"] == []
    assert _traced_text(out).replace(" ", "") == "Visibleheadingstays"


def test_visible_words_beside_the_target_on_one_line_survive():
    doc, page = _doc()
    page.insert_text((60, 300), "Visible start", fontsize=12)
    page.insert_text((135, 300), "700-11-2233", fontsize=12, render_mode=3)
    page.insert_text((203, 300), "visible end", fontsize=12)
    pdf = _bytes(doc)
    out, sink = _remove(pdf, _word_boxes(pdf, {"700-11-2233"}))
    assert sink == {"removed": [[0, 0]], "kept": []}
    assert pymupdf.open(stream=out)[0].get_text().split() == [
        "Visible", "start", "visible", "end"]
    assert _gray(out) == _gray(pdf)


def _leaded(lead: float) -> bytes:
    doc, page = _doc()
    page.insert_text((60, 300), "Upper visible line here jgpq", fontsize=12)
    page.insert_text((60, 300 + lead), "700-11-2233 between", fontsize=12,
                     color=(1, 1, 1))
    page.insert_text((60, 300 + 2 * lead), "Lower visible Line HERE", fontsize=12)
    return _bytes(doc)


@pytest.mark.parametrize("lead", [14, 9])
def test_tight_leading_removes_the_target_and_spares_both_neighbours(lead):
    pdf = _leaded(lead)
    out, sink = _remove(pdf, _word_boxes(pdf, {"700-11-2233", "between"}))
    assert sink["kept"] == []
    text = pymupdf.open(stream=out)[0].get_text().split()
    assert text == ["Upper", "visible", "line", "here", "jgpq",
                    "Lower", "visible", "Line", "HERE"]
    _assert_gone(out, "700-11-2233 between")


def test_text_drawn_over_a_visible_line_is_kept_rather_than_harm_it():
    """No band can separate glyphs that sit on top of each other: keep, report."""
    doc, page = _doc()
    page.insert_text((60, 300), "Visible words over here", fontsize=12)
    # Same baseline, same start: every target glyph sits on a visible one.
    page.insert_text((60, 300), "Visible", fontsize=12, render_mode=3)
    pdf = _bytes(doc)
    boxes = [b for b in _word_boxes(pdf, {"Visible"}) if b["mode"] == 3]
    out, sink = _remove(pdf, boxes)
    assert sink == {"removed": [], "kept": [[0, 0]]}
    traced = _traced_text(out).replace(" ", "")
    assert traced.count("Visible") == 2
    assert "Visiblewordsoverhere" in traced
    assert _gray(out) == _gray(pdf)


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_pages(rotation):
    doc, page = _doc()
    page.insert_text((72, 100), "Visible heading stays", fontsize=12)
    page.insert_text((72, 300), TARGET, fontsize=12, render_mode=3)
    page.set_rotation(rotation)
    pdf = _bytes(doc)
    out, sink = _remove(pdf, _word_boxes(pdf, {"Zqvx", "700-11-2233"}))
    assert sink["kept"] == []
    _assert_gone(out)
    assert "Visibleheadingstays" in _traced_text(out).replace(" ", "")
    assert _gray(out) == _gray(pdf)


def test_removal_runs_before_the_burn_and_the_burn_still_applies():
    doc, page = _doc()
    page.insert_text((72, 100), "Burn this 555-01-0001", fontsize=12)
    page.insert_text((72, 300), TARGET, fontsize=12, render_mode=3)
    pdf = _bytes(doc)
    burn = [{"page": 0, "type": "SSN", "enabled": True,
             "rect": {"x0": 120, "y0": 88, "x1": 200, "y1": 104}}]
    sink: dict = {}
    out, _ = apply_redactions(pdf, burn, remove_text=[
        {"page": 0, **b} for b in _word_boxes(pdf, {"Zqvx", "700-11-2233"})],
        removal_sink=sink)
    _assert_gone(out)
    assert "555-01-0001" not in _traced_text(out)
    assert sink["kept"] == []


def test_removed_and_blacked_out_pages_are_skipped():
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=612, height=792).insert_text(
            (72, 300), TARGET, fontsize=12, render_mode=3)
    pdf = doc.tobytes()
    boxes = _word_boxes(pdf, {"Zqvx"})
    sink: dict = {}
    apply_redactions(pdf, [], removed_pages=[0], blackout_pages=[1],
                     remove_text=[{"page": p, **boxes[0]} for p in range(3)],
                     removal_sink=sink)
    assert sink == {"removed": [[2, 0]], "kept": []}


def test_a_word_that_is_not_there_is_kept_never_reported_removed():
    pdf = _hidden(render_mode=3)
    out, sink = _remove(pdf, [{"box": [400.0, 600.0, 450.0, 612.0], "text": "absent"}])
    assert sink == {"removed": [], "kept": [[0, 0]]}
    assert _gray(out) == _gray(pdf)


def test_a_word_whose_properties_differ_is_not_the_target():
    pdf = _hidden(render_mode=3)
    (item,) = _word_boxes(pdf, {"Zqvx"})
    out, sink = _remove(pdf, [{**item, "mode": 0}])
    assert sink == {"removed": [], "kept": [[0, 0]]}
    assert "Zqvx" in _traced_text(out)


def test_visible_text_in_another_layer_at_the_same_place_survives():
    """A second-language layer drawn where the visible one is: remove only it."""
    doc, page = _doc()
    off = doc.add_ocg("Other language", on=False)
    page.insert_text((72, 300), "Shown label text", fontsize=12)
    page.insert_text((72, 300), "Autre texte cache", fontsize=12, oc=off)
    pdf = _bytes(doc)
    items = _word_boxes(pdf, {"Autre", "texte", "cache"})
    out, sink = _remove(pdf, items)
    traced = _traced_text(out).replace(" ", "")
    assert "Shownlabeltext" in traced
    # Every target either went or was reported kept; the visible words never
    # went with them.
    assert len(sink["removed"]) + len(sink["kept"]) == 3
    for _, i in sink["removed"]:
        assert items[i]["text"] not in traced
    assert _gray(out) == _gray(pdf)


@pytest.mark.parametrize("bad", [
    "nope", [{"page": "x", "box": [0, 0, 1, 1]}], [{"page": 0}],
    [{"page": 0, "box": [0, 0, 1]}], [{"page": 0, "box": [5, 0, 1, 1]}],
    [{"page": 0, "box": ["a", 0, 1, 1]}],
    [{"page": 0, "box": [0, 0, 1, 1], "mode": "x"}]])
def test_malformed_remove_text_is_refused(bad):
    with pytest.raises(ValueError):
        apply_redactions(_hidden(render_mode=3), [], remove_text=bad)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_op_reports_text_removal_only_when_asked():
    pdf = _hidden(render_mode=3)
    plain = _OPS["apply_redactions"]({"pdf_b64": _b64(pdf), "matches": []})
    assert "text_removal" not in plain
    asked = _OPS["apply_redactions"]({
        "pdf_b64": _b64(pdf), "matches": [],
        "remove_text": [{"page": 0, **b}
                        for b in _word_boxes(pdf, {"Zqvx", "700-11-2233"})]})
    assert asked["text_removal"] == {"removed": [[0, 0], [0, 1]], "kept": []}
    _assert_gone(base64.b64decode(asked["pdf_b64"]))


def test_handle_op_matches_the_stateless_op():
    pdf = _hidden(render_mode=3)
    items = [{"page": 0, **b} for b in _word_boxes(pdf, {"Zqvx", "700-11-2233"})]
    handle = _handle({"op": "open_doc", "pdf_b64": _b64(pdf)})["result"]["handle"]
    via = _handle({"op": "apply_redactions_h", "handle": handle, "matches": [],
                   "remove_text": items})
    _handle({"op": "close_doc", "handle": handle})
    assert via["ok"] is True
    assert via["result"]["text_removal"] == {"removed": [[0, 0], [0, 1]], "kept": []}
    _assert_gone(base64.b64decode(via["result"]["pdf_b64"]))
