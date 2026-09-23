"""Protocol 9: text a later drawing covers only part of a word of.

A box drawn over a line of text rarely stops at a word boundary: it can end
before a word's trailing comma, or in the middle of a letter. Three pieces
let a caller deal with that without the tool deciding anything itself:

* ``trace_text`` reports each character's box, and each later fill, shading
  or image, for words a later drawing reaches into;
* ``remove_text`` entries may name some of a word's characters (``chars``);
* ``render_removed`` renders a page as it would look with some text gone, so
  a caller can compare it with the page's own render.

Every fixture is built in memory with invented text.
"""
from __future__ import annotations

import base64

import pymupdf
import pytest

from lexcloak_pdf_tool.__main__ import _OPS, _handle
from lexcloak_pdf_tool.redact import apply_redactions
from lexcloak_pdf_tool.remove_text import render_removed_doc, validate_remove_text
from lexcloak_pdf_tool.render import render_page
from lexcloak_pdf_tool.trace import trace_text

LINE = "Filed for Quillon Brask, of Eastmere, on the record."
NAME = "Quillon Brask"


def _page(rotate: int = 0, fill=(0, 0, 0), label: str | None = None,
          before: bool = False, cover: bool = True):
    """``LINE`` with a box over ``NAME`` that stops before the comma."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    if before:
        page.draw_rect(pymupdf.Rect(60, 290, 400, 306), color=None, fill=(0.9, 0.9, 0.9))
    page.insert_text((72, 300), LINE, fontsize=11)
    if cover:
        box = page.search_for(NAME)[0]
        page.draw_rect(box, color=None, fill=fill)
        if label:
            page.insert_text((box.x0 + 3, box.y1 - 3), label, fontsize=7,
                             color=(1, 1, 1))
    if rotate:
        page.set_rotation(rotate)
    data = doc.tobytes()
    doc.close()
    return data


def _words(pdf: bytes) -> dict[str, dict]:
    return {w["text"]: w for w in trace_text(pdf, 0)["words"]}


def _item(word: dict, chars=None) -> dict:
    item = {"page": 0, "box": word["bbox"], "text": word["text"],
            "mode": word["mode"], "opacity": word["opacity"],
            "layer": word["layer"]}
    if chars is not None:
        item["chars"] = chars
    return item


def _traced(pdf: bytes) -> str:
    doc = pymupdf.open(stream=pdf)
    return "".join(chr(c[0]) for s in doc[0].get_texttrace() for c in s["chars"])


def _gray(png: bytes) -> list[bytes]:
    """The render as grey rows, one ``bytes`` per pixel row."""
    pix = pymupdf.Pixmap(png)
    if pix.n > 1:
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
    rows = bytes(pix.samples)
    return [rows[y * pix.stride:y * pix.stride + pix.w] for y in range(pix.h)]


def _window(gray: list[bytes], box, dpi: float = 150, inset: int = 1) -> list[bytes]:
    s = dpi / 72
    x0, x1 = int(box[0] * s) + inset, int(box[2] * s) + 1 - inset
    return [row[x0:x1] for row in gray[int(box[1] * s) + inset:int(box[3] * s) + 1 - inset]]


# ── trace_text: character boxes and covers ────────────────────────────────


def test_a_word_a_later_box_reaches_into_carries_its_character_boxes():
    words = _words(_page())
    brask = words["Brask,"]
    assert len(brask["chars"]) == len("Brask,")
    for c in brask["chars"]:
        assert brask["bbox"][0] - 0.01 <= c[0] <= c[2] <= brask["bbox"][2] + 0.01
    # Character boxes run left to right in the order of the text.
    assert [c[0] for c in brask["chars"]] == sorted(c[0] for c in brask["chars"])


def test_words_no_later_drawing_reaches_carry_no_character_boxes():
    words = _words(_page())
    for text in ("Filed", "for", "Eastmere,", "record."):
        assert "chars" not in words[text]


def test_the_page_lists_the_cover_after_the_text_it_reaches():
    result = trace_text(_page(), 0)
    assert len(result["covers"]) == 1
    cover = result["covers"][0]
    assert cover["kind"] == "fill-path"
    assert cover["seq"] > _words(_page())["Brask,"]["span"]


def test_a_fill_drawn_before_the_text_is_not_a_cover():
    result = trace_text(_page(before=True, cover=False), 0)
    assert result["covers"] == []
    assert all("chars" not in w for w in result["words"])


def test_an_image_drawn_after_the_text_is_a_cover_of_its_own_kind():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 300), LINE, fontsize=11)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 8), False)
    pix.set_rect(pix.irect, (0, 0, 0))
    page.insert_image(page.search_for(NAME)[0], pixmap=pix, keep_proportion=False)
    result = trace_text(doc.tobytes(), 0)
    assert [c["kind"] for c in result["covers"]] == ["fill-image"]


@pytest.mark.parametrize("rotate", [90, 180, 270])
def test_character_boxes_turn_with_the_page(rotate):
    flat = _words(_page())["Brask,"]
    turned = _words(_page(rotate=rotate))["Brask,"]
    assert len(turned["chars"]) == len(flat["chars"])
    for c in turned["chars"]:
        assert turned["bbox"][0] - 0.01 <= c[0] <= c[2] <= turned["bbox"][2] + 0.01
        assert turned["bbox"][1] - 0.01 <= c[1] <= c[3] <= turned["bbox"][3] + 0.01


# ── remove_text with chars: part of a word ───────────────────────────────


def _remove(pdf: bytes, items: list):
    sink: dict = {}
    out, _ = apply_redactions(pdf, [], remove_text=items, removal_sink=sink)
    return out, sink


def test_part_of_a_word_goes_and_its_comma_stays():
    pdf = _page()
    words = _words(pdf)
    out, sink = _remove(pdf, [_item(words["Quillon"]),
                              _item(words["Brask,"], chars=[0, 1, 2, 3, 4])])
    traced = _traced(out)
    assert "Quillon" not in traced and "Brask" not in traced
    assert ", of Eastmere," in traced
    assert sink == {"removed": [[0, 0], [0, 1]], "kept": []}
    # The box hid the letters, so nothing a reader could see changed. Not
    # byte-identical: rewriting the run re-rasterises the glyphs left in it
    # by float noise (one pixel here, by a few grey levels).
    after, before = _gray(render_page(out, 0)), _gray(render_page(pdf, 0))
    assert len(after) == len(before)
    assert not any(abs(a - b) > 48 for ra, rb in zip(after, before, strict=True)
                   for a, b in zip(ra, rb, strict=True))


def test_a_whole_word_entry_behaves_as_before():
    pdf = _page()
    out, sink = _remove(pdf, [_item(_words(pdf)["Quillon"])])
    assert "Quillon" not in _traced(out)
    assert "Brask," in _traced(out)
    assert sink == {"removed": [[0, 0]], "kept": []}


def test_a_part_under_a_label_drawn_on_the_box_is_kept_where_it_cannot_be_separated():
    pdf = _page(label="HIDDEN")
    words = _words(pdf)
    out, sink = _remove(pdf, [_item(words["Quillon"], chars=list(range(7)))])
    assert sink["kept"] == [[0, 0]]
    assert "HIDDEN" in _traced(out)             # the label is never touched
    assert "Eastmere," in _traced(out)


def test_characters_past_the_end_of_the_word_mean_it_is_not_that_word():
    pdf = _page()
    out, sink = _remove(pdf, [_item(_words(pdf)["Quillon"], chars=[0, 7])])
    assert sink == {"removed": [], "kept": [[0, 0]]}
    assert "Quillon" in _traced(out)


@pytest.mark.parametrize("bad", [[], [-1], ["a"], "0"])
def test_malformed_chars_are_refused(bad):
    with pytest.raises(ValueError):
        validate_remove_text([{"page": 0, "box": [0, 0, 1, 1], "chars": bad}])


def test_chars_arrive_sorted_and_without_repeats():
    by_page = validate_remove_text([{"page": 0, "box": [0, 0, 1, 1], "chars": [3, 1, 3]}])
    assert by_page[0][0]["chars"] == [1, 3]


# ── render_removed ───────────────────────────────────────────────────────


def _render_removed(pdf: bytes, items: list, dpi: float = 150) -> dict:
    doc = pymupdf.open(stream=pdf)
    try:
        return render_removed_doc(doc, 0, validate_remove_text(items).get(0, []), dpi)
    finally:
        doc.close()


def test_removing_text_the_box_hides_changes_nothing_inside_its_box():
    pdf = _page()
    word = _words(pdf)["Brask,"]
    out = _render_removed(pdf, [_item(word, chars=[0, 1, 2, 3, 4])])
    before, after = _gray(render_page(pdf, 0)), _gray(out["png"])
    for box in word["chars"][:5]:
        assert _window(before, box) == _window(after, box)
    assert out["lost"] == [] and out["missed"] == [] and out["kept"] == []


def test_removing_a_visible_comma_changes_its_box():
    pdf = _page()
    word = _words(pdf)["Brask,"]
    out = _render_removed(pdf, [_item(word, chars=[5])])
    before, after = _gray(render_page(pdf, 0)), _gray(out["png"])
    assert _window(before, word["chars"][5]) != _window(after, word["chars"][5])


def test_a_label_lost_with_the_covered_letters_is_named():
    pdf = _page(label="HIDDEN")
    out = _render_removed(pdf, [_item(_words(pdf)["Quillon"], chars=list(range(7)))])
    assert out["lost"], "the label's characters crossed by the bands are reported"


def test_an_entry_that_names_no_word_on_the_page_is_missed():
    out = _render_removed(_page(), [{"page": 0, "box": [1, 1, 2, 2], "text": "Nowhere"}])
    assert out["missed"] == [0]


def test_nothing_removed_renders_exactly_as_the_page_does():
    pdf = _page()
    assert _render_removed(pdf, [])["png"] == render_page(pdf, 0)


@pytest.mark.parametrize("pages", [1, 3])
def test_a_switched_off_layer_stays_off_in_the_render(pages):
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=612, height=792)
    ocg = doc.add_ocg("Notes", on=False)
    page.insert_text((72, 200), "Layer text nobody sees", fontsize=11, oc=ocg)
    page.insert_text((72, 300), LINE, fontsize=11)
    pdf = doc.tobytes()
    last = pages - 1
    with pymupdf.open(stream=pdf) as opened:
        out = render_removed_doc(opened, last, [], 150)
    assert out["png"] == render_page(pdf, last)


def test_a_later_page_of_a_document_without_layers_renders_as_it_does():
    """The cheap path: only the page itself is copied."""
    doc = pymupdf.open()
    for n in range(3):
        doc.new_page(width=612, height=792).insert_text((72, 300), f"{LINE} {n}",
                                                        fontsize=11)
    pdf = doc.tobytes()
    with pymupdf.open(stream=pdf) as opened:
        word = next(w for w in trace_text(pdf, 2)["words"] if w["text"] == "Quillon")
        item = validate_remove_text([_item(word) | {"page": 2}])[2]
        nothing = render_removed_doc(opened, 2, [], 150)
        removed = render_removed_doc(opened, 2, item, 150)
    assert nothing["png"] == render_page(pdf, 2)
    assert removed["png"] != render_page(pdf, 2) and removed["missed"] == []


def test_the_render_leaves_the_document_itself_untouched():
    pdf = _page()
    doc = pymupdf.open(stream=pdf)
    word = _words(pdf)["Quillon"]
    render_removed_doc(doc, 0, validate_remove_text([_item(word)])[0], 72)
    assert "Quillon" in "".join(chr(c[0]) for s in doc[0].get_texttrace()
                                for c in s["chars"])
    doc.close()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_the_op_and_its_handle_form_agree():
    pdf = _page()
    item = _item(_words(pdf)["Brask,"], chars=[0, 1, 2, 3, 4])
    item.pop("page")
    plain = _OPS["render_removed"]({"pdf_b64": _b64(pdf), "page": 0, "dpi": 100,
                                    "remove": [item]})
    handle = _handle({"op": "open_doc", "pdf_b64": _b64(pdf)})["result"]["handle"]
    via = _handle({"op": "render_removed_h", "handle": handle, "page": 0,
                   "dpi": 100, "remove": [item]})
    _handle({"op": "close_doc", "handle": handle})
    assert via["ok"] is True
    assert via["result"] == plain


def test_the_op_refuses_a_request_with_no_remove_list():
    resp = _handle({"op": "render_removed", "pdf_b64": _b64(_page()), "page": 0})
    assert resp["ok"] is False and resp["error_type"] == "ValueError"


def test_a_page_out_of_range_is_an_index_error():
    resp = _handle({"op": "render_removed", "pdf_b64": _b64(_page()), "page": 4,
                    "remove": []})
    assert resp["ok"] is False and resp["error_type"] == "IndexError"
