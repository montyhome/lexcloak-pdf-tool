"""Tests for ``extract_pages`` — the v6 page-range split (0.7.0).

Library-level: page content fidelity, bookmark slicing + re-basing,
hierarchy clamping, range validation, source non-mutation. CLI-level:
both ops round-trip through the real subprocess, the handle variant
leaves its handle usable, and a bad range surfaces as ``ValueError``.
"""
from __future__ import annotations

import base64
import re

import pymupdf
import pytest

from lexcloak_pdf_tool import extract_pages
from lexcloak_pdf_tool.page_split import _rebase_toc, extract_pages_from_doc

from test_cli import CLISession


def _numbered_pdf(n_pages: int = 10, with_toc: bool = True) -> bytes:
    """One page per index, text ``PAGE-<i>``, optional 2-level outline."""
    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text(pymupdf.Point(72, 100), f"PAGE-{i}")
    if with_toc:
        toc = []
        for i in range(0, n_pages, 2):
            toc.append([1, f"Chapter {i}", i + 1])
            if i + 1 < n_pages:
                toc.append([2, f"Section {i}.1", i + 2])
        doc.set_toc(toc)
    out = doc.tobytes()
    doc.close()
    return out


def _page_texts(pdf_bytes: bytes) -> list[str]:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    texts = [doc[i].get_text().strip() for i in range(doc.page_count)]
    doc.close()
    return texts


class TestExtractPagesLibrary:
    def test_slice_carries_exact_page_content(self):
        part = extract_pages(_numbered_pdf(10), 4, 6)
        assert _page_texts(part) == ["PAGE-4", "PAGE-5", "PAGE-6"]

    def test_full_range_is_whole_document(self):
        src = _numbered_pdf(5)
        part = extract_pages(src, 0, 4)
        assert _page_texts(part) == [f"PAGE-{i}" for i in range(5)]

    def test_single_page_slice(self):
        part = extract_pages(_numbered_pdf(10), 9, 9)
        assert _page_texts(part) == ["PAGE-9"]

    def test_bookmarks_rebased_to_local_numbering(self):
        part = extract_pages(_numbered_pdf(10), 4, 7)
        doc = pymupdf.open(stream=part, filetype="pdf")
        toc = doc.get_toc(simple=True)
        doc.close()
        # Source pages 4-7 carry: Ch4@p5, Sec4.1@p6, Ch6@p7, Sec6.1@p8
        # (1-based source) -> local 1-based 1..4.
        assert [(t, p) for _, t, p in toc] == [
            ("Chapter 4", 1), ("Section 4.1", 2),
            ("Chapter 6", 3), ("Section 6.1", 4),
        ]

    def test_orphaned_child_level_clamped_to_legal_outline(self):
        # Slice starts at a level-2 SECTION page: its level-1 parent is
        # outside the slice, so the entry must clamp to level 1 or
        # set_toc would refuse the outline.
        part = extract_pages(_numbered_pdf(10), 1, 2)
        doc = pymupdf.open(stream=part, filetype="pdf")
        toc = doc.get_toc(simple=True)
        doc.close()
        assert toc[0][0] == 1
        assert toc[0][1] == "Section 0.1"

    def test_no_toc_source_yields_no_toc_part(self):
        part = extract_pages(_numbered_pdf(6, with_toc=False), 1, 3)
        doc = pymupdf.open(stream=part, filetype="pdf")
        assert doc.get_toc() == []
        doc.close()

    @pytest.mark.parametrize("frm,to", [(-1, 3), (3, 1), (0, 10), (10, 10)])
    def test_invalid_ranges_raise_value_error(self, frm, to):
        with pytest.raises(ValueError):
            extract_pages(_numbered_pdf(10), frm, to)

    def test_source_doc_not_mutated_or_closed(self):
        doc = pymupdf.open(stream=_numbered_pdf(8), filetype="pdf")
        before = doc.get_toc(simple=True)
        extract_pages_from_doc(doc, 2, 5)
        assert doc.page_count == 8
        assert doc.get_toc(simple=True) == before
        doc.close()

    def test_boundary_pages_render_identically(self):
        # The split's fidelity claim: a part's first/last page rasters
        # byte-identical to the same page rendered from the source.
        src_bytes = _numbered_pdf(10)
        part = extract_pages(src_bytes, 4, 6)
        src = pymupdf.open(stream=src_bytes, filetype="pdf")
        cut = pymupdf.open(stream=part, filetype="pdf")
        for src_idx, part_idx in ((4, 0), (6, 2)):
            a = src[src_idx].get_pixmap(dpi=72).tobytes("png")
            b = cut[part_idx].get_pixmap(dpi=72).tobytes("png")
            assert a == b
        src.close()
        cut.close()


class TestRebaseToc:
    def test_drops_out_of_range_and_destinationless_entries(self):
        toc = [[1, "Before", 1], [1, "In", 5], [1, "Broken", -1],
               [1, "After", 9]]
        assert _rebase_toc(toc, 3, 6) == [[1, "In", 2]]

    def test_preserves_relative_nesting_inside_slice(self):
        # Source pages 4/5/6 (1-based) under from_page=3 (0-based) land
        # on local pages 1/2/3; nesting depth is untouched.
        toc = [[1, "A", 4], [2, "A.1", 5], [3, "A.1.a", 6]]
        assert _rebase_toc(toc, 3, 6) == [
            [1, "A", 1], [2, "A.1", 2], [3, "A.1.a", 3]]


class TestExtractPagesCLI:
    def test_stateless_op_round_trip(self):
        with CLISession() as s:
            resp = s.call("extract_pages",
                          pdf_b64=base64.b64encode(
                              _numbered_pdf(10)).decode(),
                          from_page=2, to_page=4)
            assert resp["ok"] is True
            part = base64.b64decode(resp["result"]["pdf_b64"])
            assert resp["result"]["page_count"] == 3
        assert _page_texts(part) == ["PAGE-2", "PAGE-3", "PAGE-4"]

    def test_handle_op_leaves_handle_usable(self):
        with CLISession() as s:
            opened = s.call("open_doc",
                            pdf_b64=base64.b64encode(
                                _numbered_pdf(6)).decode())
            handle = opened["result"]["handle"]
            resp = s.call("extract_pages_h", handle=handle,
                          from_page=1, to_page=2)
            assert resp["ok"] is True
            after = s.call("page_count_h", handle=handle)
            assert after["ok"] is True
            assert after["result"]["count"] == 6

    def test_bad_range_reports_value_error(self):
        with CLISession() as s:
            resp = s.call("extract_pages",
                          pdf_b64=base64.b64encode(
                              _numbered_pdf(3)).decode(),
                          from_page=2, to_page=9)
            assert resp["ok"] is False
            assert resp["error_type"] == "ValueError"


# ── first_page_label (v13, 0.15.0) ──────────────────────────────────

#: A trailer ``/ID`` half is a hex string or a literal string, whichever
#: MuPDF found shorter for that save's random bytes; mask both shapes, half
#: by half (a literal may hold an unescaped ``]``).
_ID_HALF = rb"(?:<[0-9A-Fa-f]*>|\((?:\\[\s\S]|[^\\()])*\))"
_ID_RE = re.compile(rb"/ID\s*\[\s*" + _ID_HALF + rb"\s*" + _ID_HALF + rb"\s*\]")


def _mask_id(pdf_bytes: bytes) -> bytes:
    return _ID_RE.sub(b"/ID[<masked>]", pdf_bytes)


def _labels(pdf_bytes: bytes) -> tuple[list[dict], list[str]]:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc.get_page_labels(), [p.get_label() for p in doc]
    finally:
        doc.close()


def _labelled_pdf() -> bytes:
    """Six pages labelled ``A-1 A-2 A-3 B-i B-ii B-iii``."""
    doc = pymupdf.open()
    for i in range(6):
        doc.new_page(width=612, height=792).insert_text(
            pymupdf.Point(72, 100), f"PAGE-{i}")
    doc.set_page_labels([
        {"startpage": 0, "prefix": "A-", "style": "D", "firstpagenum": 1},
        {"startpage": 3, "prefix": "B-", "style": "r", "firstpagenum": 1},
    ])
    out = doc.tobytes()
    doc.close()
    return out


class TestFirstPageLabel:
    def test_labels_count_up_from_the_given_number(self):
        part = extract_pages(_numbered_pdf(10), 4, 6, first_page_label=93)
        rules, labels = _labels(part)
        assert rules == [{"startpage": 0, "prefix": "", "firstpagenum": 93,
                          "style": "D"}]
        assert labels == ["93", "94", "95"]
        assert _page_texts(part) == ["PAGE-4", "PAGE-5", "PAGE-6"]

    def test_label_one_for_a_first_part(self):
        _, labels = _labels(extract_pages(_numbered_pdf(4), 0, 1,
                                          first_page_label=1))
        assert labels == ["1", "2"]

    def test_one_page_part(self):
        _, labels = _labels(extract_pages(_numbered_pdf(10), 7, 7,
                                          first_page_label=8))
        assert labels == ["8"]

    def test_part_ending_on_the_last_page(self):
        _, labels = _labels(extract_pages(_numbered_pdf(10), 7, 9,
                                          first_page_label=8))
        assert labels == ["8", "9", "10"]

    def test_a_2000_page_source(self):
        doc = pymupdf.open()
        for _ in range(2000):
            doc.new_page(width=100, height=100)
        src = doc.tobytes()
        doc.close()
        _, labels = _labels(extract_pages(src, 1499, 1999,
                                          first_page_label=1500))
        assert len(labels) == 501
        assert labels[0] == "1500"
        assert labels[-1] == "2000"

    def test_the_labels_carry_no_prefix_string(self):
        part = extract_pages(_numbered_pdf(5), 1, 3, first_page_label=2)
        doc = pymupdf.open(stream=part, filetype="pdf")
        kind, value = doc.xref_get_key(doc.pdf_catalog(), "PageLabels")
        doc.close()
        assert (kind, value) == ("dict", "<</Nums[0<</S/D/St 2>>]>>")

    def test_source_labels_are_replaced_and_their_prefixes_gone(self):
        part = extract_pages(_labelled_pdf(), 2, 4, first_page_label=3)
        rules, labels = _labels(part)
        assert labels == ["3", "4", "5"]
        assert len(rules) == 1
        assert b"A-" not in part and b"B-" not in part

    def test_without_the_field_a_labelled_source_gives_no_labels(self):
        rules, labels = _labels(extract_pages(_labelled_pdf(), 2, 4))
        assert rules == []
        assert labels == ["", "", ""]

    def test_absent_is_the_historical_output(self):
        src = _numbered_pdf(10)
        absent = extract_pages(src, 3, 6)
        explicit_none = extract_pages(src, 3, 6, first_page_label=None)
        assert b"PageLabels" not in absent
        assert _mask_id(absent) == _mask_id(explicit_none)

    @pytest.mark.parametrize("bad", [0, -5, "93", 93.0, 1.5, True, [1]])
    def test_bad_values_raise_value_error(self, bad):
        with pytest.raises(ValueError, match="first_page_label"):
            extract_pages(_numbered_pdf(3), 0, 1, first_page_label=bad)

    def test_document_core_validates_too(self):
        doc = pymupdf.open(stream=_numbered_pdf(3), filetype="pdf")
        try:
            with pytest.raises(ValueError, match="first_page_label"):
                extract_pages_from_doc(doc, 0, 1, first_page_label=0)
        finally:
            doc.close()


class TestFirstPageLabelCLI:
    def test_stateless_op_writes_labels(self):
        with CLISession() as s:
            resp = s.call("extract_pages",
                          pdf_b64=base64.b64encode(
                              _numbered_pdf(10)).decode(),
                          from_page=2, to_page=4, first_page_label=3)
            assert resp["ok"] is True
            part = base64.b64decode(resp["result"]["pdf_b64"])
        assert _labels(part)[1] == ["3", "4", "5"]

    def test_handle_op_writes_labels_and_keeps_the_handle(self):
        with CLISession() as s:
            opened = s.call("open_doc",
                            pdf_b64=base64.b64encode(
                                _labelled_pdf()).decode())
            handle = opened["result"]["handle"]
            resp = s.call("extract_pages_h", handle=handle,
                          from_page=3, to_page=5, first_page_label=4)
            assert resp["ok"] is True
            part = base64.b64decode(resp["result"]["pdf_b64"])
            # The cached source keeps its own labels.
            again = s.call("extract_pages_h", handle=handle,
                           from_page=0, to_page=5)
            assert again["ok"] is True
        assert _labels(part)[1] == ["4", "5", "6"]

    @pytest.mark.parametrize("op", ["extract_pages", "extract_pages_h"])
    @pytest.mark.parametrize("bad", [0, -1, "93", 93.5, True])
    def test_bad_values_report_value_error(self, op, bad):
        with CLISession() as s:
            if op == "extract_pages":
                resp = s.call(op, pdf_b64=base64.b64encode(
                    _numbered_pdf(3)).decode(), from_page=0, to_page=1,
                    first_page_label=bad)
            else:
                handle = s.call("open_doc", pdf_b64=base64.b64encode(
                    _numbered_pdf(3)).decode())["result"]["handle"]
                resp = s.call(op, handle=handle, from_page=0, to_page=1,
                              first_page_label=bad)
            assert resp["ok"] is False
            assert resp["error_type"] == "ValueError"
            assert "first_page_label" in resp["error"]

    def test_null_field_is_absent(self):
        with CLISession() as s:
            resp = s.call("extract_pages",
                          pdf_b64=base64.b64encode(
                              _numbered_pdf(4)).decode(),
                          from_page=0, to_page=1, first_page_label=None)
            assert resp["ok"] is True
            part = base64.b64decode(resp["result"]["pdf_b64"])
        assert _labels(part)[0] == []
