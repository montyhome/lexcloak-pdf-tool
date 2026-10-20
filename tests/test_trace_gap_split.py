"""``trace_text`` ends a word where the next glyph does not continue it (v0.15.1).

MuPDF's text trace keeps one span going for as long as the drawing properties
stay the same: through a ``TJ`` number that opens a gap with no space glyph,
through a ``Td`` to the next line, and through a ``Td`` backwards. Splitting
only at space characters read each of those as one word whose box spans the
gap, so a caller comparing the box with a render saw the ink of one part and
none of the other. Every fixture here is built in memory with invented text,
written as raw content so each gap is exactly the one the test names.

Helvetica at 10 pt throughout: one em is 10 pt, so a ``TJ`` number of -160
opens a 1.6 pt gap (0.16 em) and +260 pulls the next glyph back 0.26 em.
"""
from __future__ import annotations

import math

import pymupdf
import pytest

from lexcloak_pdf_tool.redact import apply_redactions
from lexcloak_pdf_tool.trace import (BACK_EM, BASELINE_EM, GAP_EM, _span_words,
                                     trace_text)


def _pdf(content: str, rotate: int = 0) -> bytes:
    """A one-page PDF whose content stream is ``content``, with Helvetica as
    ``/helv``."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((10, 10), " ", fontname="helv")   # registers /helv
    doc.update_stream(page.get_contents()[0], content.encode())
    if rotate:
        page.set_rotation(rotate)
    data = doc.tobytes()
    doc.close()
    return data


def _texts(pdf: bytes) -> list[str]:
    return [w["text"] for w in trace_text(pdf, 0)["words"]]


def _tj(kern: int, mode: int = 0) -> str:
    return f"BT /helv 10 Tf {mode} Tr 40 200 Td [(Velk){kern}(Opry)] TJ ET"


def test_the_constants_are_the_ones_the_rule_fixes():
    assert (GAP_EM, BACK_EM, BASELINE_EM) == (0.15, 0.25, 0.5)


# ── (a) a gap along the line ──────────────────────────────────────────────


@pytest.mark.parametrize("kern, words", [
    (-140, ["VelkOpry"]),           # 0.14 em: still one word
    (-160, ["Velk", "Opry"]),       # 0.16 em: a gap
    (-3000, ["Velk", "Opry"]),      # a wide gap
])
def test_a_tj_gap_ends_the_word_at_015_em(kern, words):
    assert _texts(_pdf(_tj(kern))) == words


def test_the_split_matches_mupdf_word_extraction():
    for kern in (-140, -160):
        pdf = _pdf(_tj(kern))
        doc = pymupdf.open(stream=pdf)
        mupdf = [w[4] for w in doc[0].get_text("words")]
        doc.close()
        assert _texts(pdf) == mupdf, kern


def test_split_words_get_their_own_boxes():
    joined = trace_text(_pdf(_tj(-100)), 0)["words"]
    split = trace_text(_pdf(_tj(-3000)), 0)["words"]
    assert len(joined) == 1 and len(split) == 2
    a, b = split
    assert a["bbox"][2] < b["bbox"][0]           # the gap lies between them
    assert b["bbox"][0] - a["bbox"][2] == pytest.approx(30.0, abs=0.01)
    assert a["span"] == b["span"]                # one text-showing run still


# ── (b) a jump back ───────────────────────────────────────────────────────


@pytest.mark.parametrize("kern, words", [
    (110, ["VelkOpry"]),            # a kerned pair: 0.11 em back
    (240, ["VelkOpry"]),            # 0.24 em back: still one word
    (260, ["Velk", "Opry"]),        # 0.26 em back
])
def test_a_jump_back_beyond_025_em_ends_the_word(kern, words):
    assert _texts(_pdf(_tj(kern))) == words


def test_a_td_back_along_the_line_ends_the_word():
    pdf = _pdf("BT /helv 10 Tf 200 200 Td (Velk) Tj -150 0 Td (Opry) Tj ET")
    words = trace_text(pdf, 0)["words"]
    assert [w["text"] for w in words] == ["Velk", "Opry"]
    assert words[1]["bbox"][2] < words[0]["bbox"][0]


def test_a_comma_kerned_into_the_last_letter_stays_with_its_word():
    # A kerned pair pulls a comma 0.11 em into the letter before it.
    assert _texts(_pdf("BT /helv 10 Tf 40 200 Td [(Velkor)110(,)] TJ ET")) == ["Velkor,"]


# ── (c) another baseline ──────────────────────────────────────────────────


def test_a_td_to_the_next_line_ends_the_word():
    pdf = _pdf("BT /helv 10 Tf 40 200 Td (Velk) Tj 0 -12 Td (Opry) Tj ET")
    words = trace_text(pdf, 0)["words"]
    assert [w["text"] for w in words] == ["Velk", "Opry"]
    assert words[0]["span"] == words[1]["span"]
    assert words[0]["bbox"][3] <= words[1]["bbox"][1] + 0.01   # one line each


@pytest.mark.parametrize("rise, words", [
    (3.3, ["Velk2"]),               # a superscript, a third of an em up
    (4.9, ["Velk2"]),               # just under half an em
    (5.1, ["Velk", "2"]),           # just over
])
def test_a_baseline_move_beyond_half_an_em_ends_the_word(rise, words):
    # Text rise moves the baseline without leaving the span.
    pdf = _pdf(f"BT /helv 10 Tf 40 200 Td (Velk) Tj {rise} Ts (2) Tj ET")
    assert _texts(pdf) == words


# ── direction and page frame ──────────────────────────────────────────────


@pytest.mark.parametrize("deg", [30, 45, 90, 180, 300])
def test_turned_text_splits_at_a_gap_and_nowhere_else(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    tm = f"{c:.6f} {s:.6f} {-s:.6f} {c:.6f} 200 150 Tm"
    joined = _pdf(f"BT /helv 10 Tf {tm} [(Velk)-100(Opry)] TJ ET")
    gapped = _pdf(f"BT /helv 10 Tf {tm} [(Velk)-160(Opry)] TJ ET")
    assert _texts(joined) == ["VelkOpry"]
    assert _texts(gapped) == ["Velk", "Opry"]


@pytest.mark.parametrize("rotate", [90, 180, 270])
def test_a_rotated_page_splits_like_an_upright_one(rotate):
    assert _texts(_pdf(_tj(-160), rotate)) == ["Velk", "Opry"]
    assert _texts(_pdf(_tj(-100), rotate)) == ["VelkOpry"]


def _span(wmode: int, size: float = 10.0) -> dict:
    # Three glyphs 5 pt wide; the third sits 20 pt past the second.
    chars = []
    for i, (c, x) in enumerate([("V", 0.0), ("e", 5.0), ("l", 30.0)]):
        chars.append((ord(c), i, (x, 100.0), (x, 92.0, x + 5.0, 102.0)))
    return {"chars": chars, "size": size, "dir": (1.0, 0.0), "wmode": wmode}


def test_vertical_writing_splits_only_at_spaces():
    assert [w[0] for w in _span_words(_span(0))] == ["Ve", "l"]
    assert [w[0] for w in _span_words(_span(1))] == ["Vel"]


def test_a_span_with_no_size_splits_only_at_spaces():
    assert [w[0] for w in _span_words(_span(0, size=0.0))] == ["Vel"]


def test_spaces_still_end_words_and_never_join_a_word():
    pdf = _pdf("BT /helv 10 Tf 40 200 Td [(Velk Opry)] TJ ET")
    assert _texts(pdf) == ["Velk", "Opry"]


# ── every field follows the smaller word ──────────────────────────────────


def test_clipped_is_judged_per_piece():
    # A clip path around the first piece only.
    pdf = _pdf("q 30 190 40 25 re W n "
               "BT /helv 10 Tf 40 200 Td [(Velk)-3000(Opry)] TJ ET Q")
    words = {w["text"]: w for w in trace_text(pdf, 0)["words"]}
    assert words["Velk"]["clipped"] is False
    assert words["Opry"]["clipped"] is True


def test_chars_go_only_to_the_piece_a_later_fill_reaches():
    pdf = _pdf("BT /helv 10 Tf 40 200 Td [(Velk)-3000(Opry)] TJ ET "
               "0 g 95 195 10 12 re f")
    words = {w["text"]: w for w in trace_text(pdf, 0)["words"]}
    assert "chars" not in words["Velk"]
    assert len(words["Opry"]["chars"]) == 4


def test_an_invisible_tail_past_a_gap_is_its_own_word():
    # The leak's shape: one render-mode-3 run, its first piece where ink is,
    # its tail past a gap on blank paper. Each piece now carries its own box.
    pdf = _pdf("0 g 38 195 30 14 re f "
               "BT /helv 10 Tf 3 Tr 40 200 Td [(Velk)-3000(Opry)] TJ ET")
    words = trace_text(pdf, 0)["words"]
    assert [(w["text"], w["mode"]) for w in words] == [("Velk", 3), ("Opry", 3)]
    assert words[1]["bbox"][0] > 68.0             # clear of the inked box


# ── remove_text finds the smaller words ───────────────────────────────────


def _request(pdf: bytes, text: str) -> list[dict]:
    return [{"page": 0, "box": w["bbox"], "text": w["text"], "mode": w["mode"],
             "opacity": w["opacity"], "layer": w["layer"]}
            for w in trace_text(pdf, 0)["words"] if w["text"] == text]


def test_remove_text_removes_one_piece_of_a_split_run_and_keeps_the_other():
    pdf = _pdf("BT /helv 10 Tf 40 200 Td [(Velk)-3000(Opry)] TJ ET")
    sink: dict = {}
    out, _ = apply_redactions(pdf, [], remove_text=_request(pdf, "Opry"),
                              removal_sink=sink)
    assert sink["removed"] == [[0, 0]] and sink["kept"] == []
    assert _texts(out) == ["Velk"]


def test_remove_text_finds_a_piece_after_a_line_change():
    pdf = _pdf("BT /helv 10 Tf 3 Tr 40 200 Td (Velk) Tj 0 -12 Td (Opry) Tj ET")
    sink: dict = {}
    out, _ = apply_redactions(pdf, [], remove_text=_request(pdf, "Velk"),
                              removal_sink=sink)
    assert sink["removed"] == [[0, 0]]
    assert _texts(out) == ["Opry"]


def test_a_request_naming_the_old_joined_word_is_not_found():
    # The joined box and text are no longer a word on the page, so the entry
    # is kept, never forced through.
    pdf = _pdf("BT /helv 10 Tf 40 200 Td [(Velk)-3000(Opry)] TJ ET")
    a, b = trace_text(pdf, 0)["words"]
    joined = [a["bbox"][0], a["bbox"][1], b["bbox"][2], b["bbox"][3]]
    sink: dict = {}
    out, _ = apply_redactions(pdf, [], remove_text=[
        {"page": 0, "box": joined, "text": "VelkOpry"}], removal_sink=sink)
    assert sink["kept"] == [[0, 0]] and sink["removed"] == []
    assert _texts(out) == ["Velk", "Opry"]


# ── the arithmetic, on spans built by hand ────────────────────────────────


def _pair(second_x: float, *, box_w: float = 5.0, dir_=(1.0, 0.0),
          size: float = 10.0) -> list[str]:
    """Two glyphs: the first at the origin with a ``box_w``-wide box, the
    second ``second_x`` along ``dir_`` (numbers exact in binary)."""
    ux, uy = dir_
    norm = (ux * ux + uy * uy) ** 0.5
    ox, oy = second_x * ux / norm, second_x * uy / norm
    chars = [(ord("V"), 0, (0.0, 0.0), (0.0, -8.0, box_w, 2.0)),
             (ord("e"), 1, (ox, oy), (ox, oy - 8.0, ox + 5.0, oy + 2.0))]
    span = {"chars": chars, "size": size, "dir": dir_, "wmode": 0}
    return [w[0] for w in _span_words(span)]


def test_a_gap_of_exactly_015_em_splits():
    assert _pair(6.5) == ["V", "e"]          # 5 pt advance, then 1.5 pt
    assert _pair(6.25) == ["Ve"]


def test_a_dir_that_is_not_a_unit_vector_is_normalised():
    assert _pair(6.25, dir_=(2.0, 0.0)) == ["Ve"]
    assert _pair(6.5, dir_=(2.0, 0.0)) == ["V", "e"]


def test_a_box_narrower_than_its_turn_allows_counts_as_no_advance():
    # At 30 degrees a 10 pt glyph's box is at least 5 pt wide; a 4 pt box
    # would give a negative advance, which is read as none.
    c30, s30 = 3 ** 0.5 / 2, 0.5
    assert _pair(1.0, box_w=4.0, dir_=(c30, s30)) == ["Ve"]
