"""``trace_text`` / ``trace_text_h`` (v7): every word a page carries, with how it is drawn.

Each fixture is built in memory with invented text, and each test states the
exact per-word facts the op must report. The op makes no judgement about
visibility, so the assertions are about facts read from the content stream:
render mode, opacity, optional-content state, clipping, later occluders and
the page frame.
"""
from __future__ import annotations

import base64
import json

import pymupdf
import pytest

from lexcloak_pdf_tool.__main__ import _OPS, PROTOCOL_VERSION, _handle
from lexcloak_pdf_tool.trace import trace_text


def _doc():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    return doc, page


def _bytes(doc) -> bytes:
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _words(result) -> dict[str, dict]:
    return {w["text"]: w for w in result["words"]}


def test_plain_text_is_mode_0_opaque_unclipped_uncovered():
    doc, page = _doc()
    page.insert_text((72, 100), "alpha beta", fontsize=12)
    out = trace_text(_bytes(doc), 0)
    words = _words(out)
    assert list(words) == ["alpha", "beta"]
    for w in words.values():
        assert (w["mode"], w["opacity"], w["layer"], w["layer_off"],
                w["clipped"], w["covered_by"]) == (0, 1.0, "", False, False, None)
        assert w["size"] == 12.0
    assert out["rect"] == [0.0, 0.0, 612.0, 792.0]
    assert out["rotation"] == 0
    assert out["image_cover"] == 0.0
    # One insert_text call is one text-showing run.
    assert words["alpha"]["span"] == words["beta"]["span"]


def test_words_from_separate_runs_carry_separate_span_numbers():
    doc, page = _doc()
    page.insert_text((72, 100), "first run", fontsize=12)
    page.insert_text((72, 140), "second run", fontsize=12)
    out = trace_text(_bytes(doc), 0)
    spans = [w["span"] for w in out["words"]]
    assert spans[0] == spans[1]
    assert spans[2] == spans[3]
    assert spans[1] < spans[2]


def test_render_mode_3_is_reported_and_not_clipped():
    doc, page = _doc()
    page.insert_text((72, 100), "unseen", fontsize=12, render_mode=3)
    w = _words(trace_text(_bytes(doc), 0))["unseen"]
    assert w["mode"] == 3
    # Invisible text is still in the clip-respecting extraction.
    assert w["clipped"] is False


def test_fill_opacity_zero_is_reported():
    doc, page = _doc()
    page.insert_text((72, 100), "clear", fontsize=12, fill_opacity=0)
    page.insert_text((72, 140), "solid", fontsize=12)
    words = _words(trace_text(_bytes(doc), 0))
    assert words["clear"]["opacity"] == 0.0
    assert words["solid"]["opacity"] == 1.0


def test_text_outside_the_page_is_traced_and_counts_as_clipped():
    doc, page = _doc()
    page.insert_text((72, -20), "above", fontsize=12)
    page.insert_text((652, 300), "beyond", fontsize=12)
    page.insert_text((72, 300), "inside", fontsize=12)
    words = _words(trace_text(_bytes(doc), 0))
    assert words["above"]["bbox"][3] < 0
    assert words["beyond"]["bbox"][0] > 612
    assert words["above"]["clipped"] is True
    assert words["beyond"]["clipped"] is True
    assert words["inside"]["clipped"] is False


def test_switched_off_layer_is_traced_and_the_state_is_restored():
    doc, page = _doc()
    off = doc.add_ocg("Notes", on=False)
    on = doc.add_ocg("Shown", on=True)
    page.insert_text((72, 100), "offword", fontsize=12, oc=off)
    page.insert_text((72, 140), "onword", fontsize=12, oc=on)
    data = _bytes(doc)

    words = _words(trace_text(data, 0))
    assert (words["offword"]["layer"], words["offword"]["layer_off"]) == ("Notes", True)
    assert (words["onword"]["layer"], words["onword"]["layer_off"]) == ("Shown", False)
    assert words["offword"]["clipped"] is False

    # The handle variant must leave the open document's layer state as it
    # found it, or every later render through the same handle would show the
    # switched-off layer.
    handle = _handle({"op": "open_doc", "pdf_b64": _b64(data)})["result"]["handle"]
    before = _handle({"op": "render_h", "handle": handle, "page": 0, "dpi": 72})
    traced = _handle({"op": "trace_text_h", "handle": handle, "page": 0})
    after = _handle({"op": "render_h", "handle": handle, "page": 0, "dpi": 72})
    assert traced["ok"] is True
    assert "offword" in {w["text"] for w in traced["result"]["words"]}
    assert before["result"]["png_b64"] == after["result"]["png_b64"]
    _handle({"op": "close_doc", "handle": handle})


def test_text_outside_a_clipping_path_is_clipped():
    doc, page = _doc()
    src = pymupdf.open()
    sp = src.new_page(width=612, height=792)
    sp.insert_text((72, 140), "kept", fontsize=12)
    sp.insert_text((72, 400), "cut", fontsize=12)
    clip = pymupdf.Rect(0, 120, 612, 160)
    page.show_pdf_page(clip, src, 0, clip=clip)
    src.close()
    words = _words(trace_text(_bytes(doc), 0))
    assert words["kept"]["clipped"] is False
    assert words["cut"]["clipped"] is True


def test_a_later_fill_covering_the_span_is_reported_as_path():
    doc, page = _doc()
    page.insert_text((72, 100), "under", fontsize=12)
    page.draw_rect(pymupdf.Rect(60, 85, 200, 106), color=None, fill=(0, 0, 0))
    page.draw_rect(pymupdf.Rect(60, 120, 200, 140), color=None, fill=(0, 0, 0))
    page.insert_text((72, 135), "over", fontsize=12, color=(1, 1, 1))
    words = _words(trace_text(_bytes(doc), 0))
    assert words["under"]["covered_by"] == "path"
    # Drawn after the box, so nothing covers it.
    assert words["over"]["covered_by"] is None


def test_a_later_image_covering_the_span_is_reported_as_image():
    doc, page = _doc()
    page.insert_text((72, 100), "beneath", fontsize=12)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 10), False)
    pix.clear_with(200)
    page.insert_image(pymupdf.Rect(60, 80, 240, 110), pixmap=pix,
                      keep_proportion=False)
    out = trace_text(_bytes(doc), 0)
    assert _words(out)["beneath"]["covered_by"] == "image"
    assert 0.0 < out["image_cover"] < 0.05


def test_a_partial_cover_below_the_fraction_is_not_reported():
    doc, page = _doc()
    page.insert_text((72, 100), "halfway", fontsize=12)
    page.draw_rect(pymupdf.Rect(60, 85, 95, 106), color=None, fill=(0, 0, 0))
    assert _words(trace_text(_bytes(doc), 0))["halfway"]["covered_by"] is None


def test_full_page_image_sets_image_cover_to_one():
    doc, page = _doc()
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 61, 79), False)
    pix.clear_with(255)
    page.insert_image(page.rect, pixmap=pix, keep_proportion=False)
    page.insert_text((72, 100), "layer", fontsize=12, render_mode=3)
    out = trace_text(_bytes(doc), 0)
    assert out["image_cover"] == pytest.approx(1.0)
    assert _words(out)["layer"]["covered_by"] is None


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_boxes_are_in_the_rotated_page_frame(rotation):
    doc, page = _doc()
    page.insert_text((72, 100), "turned", fontsize=12)
    upright = _words(trace_text(_bytes(doc), 0))["turned"]["bbox"]

    doc, page = _doc()
    page.insert_text((72, 100), "turned", fontsize=12)
    page.set_rotation(rotation)
    data = _bytes(doc)
    out = trace_text(data, 0)
    got = _words(out)["turned"]["bbox"]
    check = pymupdf.open(stream=data)
    expected = list(pymupdf.Rect(upright) * check[0].rotation_matrix)
    assert out["rotation"] == rotation
    assert out["rect"] == [float(v) for v in check[0].rect]
    assert got == pytest.approx(expected, abs=1e-3)


def test_empty_page_has_no_words():
    doc, _ = _doc()
    assert trace_text(_bytes(doc), 0)["words"] == []


def test_page_out_of_range_raises():
    doc, _ = _doc()
    with pytest.raises(IndexError):
        trace_text(_bytes(doc), 1)


def test_op_result_is_json_and_matches_the_handle_variant():
    doc, page = _doc()
    page.insert_text((72, 100), "same both ways", fontsize=12)
    data = _bytes(doc)
    stateless = _OPS["trace_text"]({"pdf_b64": _b64(data), "page": 0})
    handle = _handle({"op": "open_doc", "pdf_b64": _b64(data)})["result"]["handle"]
    via_handle = _handle({"op": "trace_text_h", "handle": handle, "page": 0})
    _handle({"op": "close_doc", "handle": handle})
    assert json.loads(json.dumps(stateless)) == stateless
    assert via_handle["result"] == stateless


def test_protocol_advertises_the_trace_op():
    assert PROTOCOL_VERSION >= 7
    assert {"trace_text", "trace_text_h"} <= set(_OPS)
