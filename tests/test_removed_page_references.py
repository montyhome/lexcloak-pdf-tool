"""A removed page leaves nothing behind (v0.13.0).

``apply_redactions`` deletes each page in ``removed_pages`` from the page
tree. Before v0.13.0 anything else that named the page object kept it, and
its content, in the saved file: a structure element's ``/Pg`` and a named
destination's target array both did, so a removed page's text shipped in a
tagged document or one with named destinations. A bookmark did not, because
deleting a page already repairs the outline.

The 6-criteria framework:

* **Happy path** -- the removed page's text is absent from the exported bytes
  for each kind of reference, on plain and encrypted output.
* **Sad paths** -- the surviving page keeps its text, its structure element
  keeps its page, a destination to a surviving page still resolves, and a
  burn that removes no page is unaffected.
* **Boundary** -- the first, a middle and the last page removed from many.
* **No logic mirroring** -- the removed text is typed out.
* **Side-effect verification** -- the exported bytes are read through the raw
  file, every object and every decoded and undecoded stream.
* **Mock integrity** -- real PDFs built in memory, through ``apply_redactions``.

Synthetic fixtures only; every value is invented.
"""
from __future__ import annotations

import pymupdf
import pytest

from lexcloak_pdf_tool import apply_redactions

GONE = "Removed page note 771-20-5813"
KEPT = "Kept page note"


def _readable(pdf: bytes, needle: str, password: str | None = None) -> bool:
    forms = (needle.encode("latin-1"), needle.encode("latin-1").hex().encode(),
             needle.encode("latin-1").hex().upper().encode())
    if password is None and any(f in pdf for f in forms):
        return True
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        if password is not None:
            assert doc.authenticate(password)
        for x in range(1, doc.xref_length()):
            try:
                if needle in doc.xref_object(x, compressed=False):
                    return True
                if doc.xref_is_stream(x):
                    for blob in (doc.xref_stream(x), doc.xref_stream_raw(x)):
                        if blob and any(f in blob for f in forms):
                            return True
            except Exception:  # noqa: BLE001
                continue
    finally:
        doc.close()
    return False


def _pages(n: int, removed: int):
    doc = pymupdf.open()
    for i in range(n):
        page = doc.new_page(width=612, height=792)
        page.insert_text((60, 90), GONE if i == removed else f"{KEPT} {i}",
                         fontsize=12)
    return doc


def _struct_ref(doc, removed: int) -> None:
    """One structure element per page, each naming its page with /Pg."""
    root = doc.get_new_xref()
    kids = []
    for i in range(len(doc)):
        el = doc.get_new_xref()
        doc.update_object(el, f"<</Type/StructElem/S/P/P {root} 0 R"
                              f"/Pg {doc[i].xref} 0 R>>")
        kids.append(f"{el} 0 R")
    doc.update_object(root, f"<</Type/StructTreeRoot/K [{' '.join(kids)}]>>")
    doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")


def _dest_ref(doc, removed: int) -> None:
    entries = " ".join(f"(page{i}) [{doc[i].xref} 0 R /Fit]" for i in range(len(doc)))
    doc.xref_set_key(doc.pdf_catalog(), "Names", f"<</Dests <</Names [{entries}]>>>>")


def _build(ref, n: int = 3, removed: int = 0) -> bytes:
    doc = _pages(n, removed)
    ref(doc, removed)
    data = doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    doc.close()
    return data


@pytest.mark.parametrize("ref", [_struct_ref, _dest_ref], ids=["struct", "dest"])
@pytest.mark.parametrize("arm", ["plain", "encrypted"])
def test_a_removed_page_referenced_elsewhere_leaves_no_text(ref, arm):
    src = _build(ref)
    assert _readable(src, GONE)
    protection = {"mode": "new", "password": "pw-test"} if arm == "encrypted" else None
    out, _ = apply_redactions(src, [], removed_pages=[0], output_protection=protection)
    password = "pw-test" if arm == "encrypted" else None
    assert not _readable(out, GONE, password)
    assert not _readable(out, "771-20-5813", password)


@pytest.mark.parametrize("removed", [0, 3, 6])
def test_first_middle_and_last_page_of_many(removed):
    src = _build(_struct_ref, n=7, removed=removed)
    out, _ = apply_redactions(src, [], removed_pages=[removed])
    assert not _readable(out, GONE)
    d = pymupdf.open(stream=out, filetype="pdf")
    try:
        assert len(d) == 6
        texts = [p.get_text().strip() for p in d]
        assert texts == [f"{KEPT} {i}" for i in range(7) if i != removed]
    finally:
        d.close()


def test_surviving_structure_elements_keep_their_pages():
    src = _build(_struct_ref, n=3, removed=1)
    out, _ = apply_redactions(src, [], removed_pages=[1])
    d = pymupdf.open(stream=out, filetype="pdf")
    try:
        page_xrefs = {d[i].xref for i in range(len(d))}
        pg = []
        for x in range(1, d.xref_length()):
            obj = d.xref_object(x, compressed=False)
            if "/StructElem" in obj:
                kind, value = d.xref_get_key(x, "Pg")
                if kind == "xref":
                    pg.append(int(value.split()[0]))
        assert len([p for p in pg if p in page_xrefs]) == 2
    finally:
        d.close()


def test_a_destination_to_a_surviving_page_still_resolves():
    src = _build(_dest_ref, n=3, removed=0)
    out, _ = apply_redactions(src, [], removed_pages=[0])
    d = pymupdf.open(stream=out, filetype="pdf")
    try:
        assert d.resolve_link("#nameddest=page2")[0] == 1
    finally:
        d.close()


def test_a_burn_that_removes_no_page_keeps_every_page():
    src = _build(_struct_ref, n=3, removed=99)
    out, _ = apply_redactions(src, [])
    d = pymupdf.open(stream=out, filetype="pdf")
    try:
        assert [p.get_text().strip() for p in d] == [f"{KEPT} {i}" for i in range(3)]
    finally:
        d.close()
