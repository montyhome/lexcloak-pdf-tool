"""Tests for ``extract_pages`` — the v6 page-range split (0.7.0).

Library-level: page content fidelity, bookmark slicing + re-basing,
hierarchy clamping, range validation, source non-mutation. CLI-level:
both ops round-trip through the real subprocess, the handle variant
leaves its handle usable, and a bad range surfaces as ``ValueError``.
"""
from __future__ import annotations

import base64

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
