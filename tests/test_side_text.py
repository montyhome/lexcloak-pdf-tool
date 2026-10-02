"""Strings a PDF carries outside its page content (v0.13.0).

Bookmark titles, named destinations, page-label prefixes, optional-content
group names, URI link addresses and structure-element alternate text all
survived a burn and an export before v0.13.0, on the plain, encrypted and
size-reduced arms alike. ``list_side_text`` lists them with ids, and
``apply_redactions``'s ``side_text`` field rewrites the ones a caller names.

The 6-criteria framework:

* **Happy path** -- every kind is listed with its text and page, and each
  rewrite is absent from the exported bytes.
* **Sad paths** -- a string nobody named survives byte for byte; an unknown
  id, an action a kind cannot take and a colliding destination rename are
  skipped and change nothing; a malformed edit fails before anything changes;
  a malformed outline (a cycle, a title that is not a string) is counted as
  unreadable without failing; an encrypted source is refused by the lister.
* **Boundary** -- compressed object streams, an incremental revision, a link
  on one page of many, a 500-entry outline, removed pages renumbering what
  survives, the handle op, encrypted output.
* **No logic mirroring** -- every expected string is typed out.
* **Side-effect verification** -- assertions read the exported BYTES (the raw
  file, every object, every decoded and undecoded stream), and navigation is
  re-read from the exported file.
* **Mock integrity** -- every PDF is built in memory with PyMuPDF and driven
  through the real ``apply_redactions`` entry point and the CLI op table.

Synthetic fixtures only; every value is invented.
"""
from __future__ import annotations

import base64

import pymupdf
import pytest

from lexcloak_pdf_tool import apply_redactions, list_side_text_pdf
from lexcloak_pdf_tool.__main__ import _OPS, PROTOCOL_VERSION, _handle
from lexcloak_pdf_tool.side_text import validate_side_text_edits

SSN = "523-81-4406"
LINE = f"Claimant intake record SSN {SSN}"
NAME = "Harold Pemberton"


# ── builders and readers ─────────────────────────────────────────────────


def _doc(pages: int = 2):
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page(width=612, height=792)
    doc[0].insert_text((60, 90), LINE, fontsize=12)
    return doc


def _bytes(doc, **kw) -> bytes:
    data = doc.tobytes(garbage=3, deflate=True, no_new_id=True, **kw)
    doc.close()
    return data


def _readable(pdf: bytes, needle: str, password: str | None = None) -> bool:
    """True if ``needle`` can be read from ``pdf`` by any channel: the raw
    file, an object's dictionary text, a decoded stream, or an undecoded one.
    Also in UTF-16BE, the way a producer writes a non-Latin title."""
    forms = (needle.encode("latin-1"), needle.encode("latin-1").hex().encode(),
             needle.encode("latin-1").hex().upper().encode(),
             needle.encode("utf-16-be").hex().encode(),
             needle.encode("utf-16-be").hex().upper().encode())
    if password is None and any(f in pdf for f in forms):
        return True
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        if password is not None:
            assert doc.authenticate(password)
        for x in list(range(1, doc.xref_length())) + [-1]:
            try:
                if needle in doc.xref_object(x, compressed=False):
                    return True
                if x > 0 and doc.xref_is_stream(x):
                    for blob in (doc.xref_stream(x), doc.xref_stream_raw(x)):
                        if blob and any(f in blob for f in forms):
                            return True
            except Exception:  # noqa: BLE001
                continue
    finally:
        doc.close()
    return False


def _every_kind() -> bytes:
    """One of each kind, the sensitive value in each, plus benign neighbours."""
    doc = _doc(pages=2)
    p0, p1 = doc[0], doc[1]
    doc.set_toc([[1, f"Deposition of {NAME}", 1], [1, "Exhibit A", 2]])
    doc.set_page_labels([{"startpage": 0, "prefix": "PEMBERTON-", "style": "D",
                          "firstpagenum": 1}])
    doc.add_ocg("Annotations - H. Pemberton", on=True)
    p0.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(60, 200, 200, 220),
                    "uri": "mailto:harold.pemberton@example.invalid"})
    p0.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(60, 230, 200, 250),
                    "uri": "https://www.example.invalid/forms"})
    cat = doc.pdf_catalog()
    doc.xref_set_key(cat, "Names",
                     f"<</Dests <</Names [(HaroldPembertonDeposition) "
                     f"[{p1.xref} 0 R /XYZ 0 0 0] (section.2) [{p0.xref} 0 R /Fit]]>>>>")
    goto = doc.get_new_xref()
    doc.update_object(goto, "<</Type/Annot/Subtype/Link/Rect[10 10 40 40]"
                            "/A<</S/GoTo/D(HaroldPembertonDeposition)>>>>")
    doc.xref_set_key(p1.xref, "Annots", f"[{goto} 0 R]")
    root, fig, logo = doc.get_new_xref(), doc.get_new_xref(), doc.get_new_xref()
    doc.update_object(root, f"<</Type/StructTreeRoot/K [{fig} 0 R {logo} 0 R]>>")
    doc.update_object(fig, f"<</Type/StructElem/S/Figure/P {root} 0 R/Pg {p1.xref} 0 R"
                           f"/Alt (Photo ID card, SSN {SSN})>>")
    doc.update_object(logo, f"<</Type/StructElem/S/Figure/P {root} 0 R/Pg {p1.xref} 0 R"
                            "/Alt (Company logo)>>")
    doc.xref_set_key(cat, "StructTreeRoot", f"{root} 0 R")
    return _bytes(doc)


def _ssn_matches(pdf: bytes) -> list[dict]:
    """A match over every visible SSN, as the app would send after detection."""
    d = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        return [{"page": i, "enabled": True, "type": "SSN",
                 "rect": {"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}}
                for i, p in enumerate(d) for r in p.search_for(SSN)]
    finally:
        d.close()


def _by_text(listing: dict) -> dict:
    return {e["text"]: e for e in listing["entries"]}


#: What a careful caller would write for each sensitive string.
REWRITES = {
    f"Deposition of {NAME}": "Deposition of [redacted]",
    "PEMBERTON-": "[redacted]-",
    "Annotations - H. Pemberton": "Annotations - [redacted]",
    "HaroldPembertonDeposition": "destination-1",
    f"Photo ID card, SSN {SSN}": "Photo ID card, SSN [redacted]",
}
REMOVALS = ("mailto:harold.pemberton@example.invalid",)


def _edits(src: bytes) -> list[dict]:
    ids = _by_text(list_side_text_pdf(src))
    edits = [{"id": ids[old]["id"], "text": new} for old, new in REWRITES.items()]
    edits += [{"id": ids[uri]["id"], "remove": True} for uri in REMOVALS]
    return edits


# ── listing ──────────────────────────────────────────────────────────────


def test_every_kind_is_listed_with_its_text_and_page():
    listing = list_side_text_pdf(_every_kind())
    got = {(e["kind"], e["text"], e["page"]) for e in listing["entries"]}
    assert got == {
        ("outline", "Deposition of Harold Pemberton", 0),
        ("outline", "Exhibit A", 1),
        ("destination", "HaroldPembertonDeposition", 1),
        ("destination", "section.2", 0),
        ("page_label", "PEMBERTON-", 0),
        ("layer", "Annotations - H. Pemberton", None),
        ("link", "mailto:harold.pemberton@example.invalid", 0),
        ("link", "https://www.example.invalid/forms", 0),
        ("tag", "Photo ID card, SSN 523-81-4406", 1),
        ("tag", "Company logo", 1),
    }
    assert listing["unreadable"] == 0
    assert len({e["id"] for e in listing["entries"]}) == 10


def test_a_document_with_none_of_them_lists_nothing():
    assert list_side_text_pdf(_bytes(_doc())) == {"entries": [], "unreadable": 0}


def test_a_bookmark_that_opens_a_web_address_lists_its_title_and_its_link():
    doc = _doc()
    doc.set_toc([[1, "Online file", 1]])
    item = next(x for x in range(1, doc.xref_length())
                if "/Title" in doc.xref_object(x))
    doc.xref_set_key(item, "Dest", "null")
    doc.xref_set_key(item, "A", "<</S/URI/URI(https://example.invalid/case/4471)>>")
    listing = list_side_text_pdf(_bytes(doc))
    kinds = {(e["kind"], e["text"]) for e in listing["entries"]}
    assert kinds == {("outline", "Online file"),
                     ("link", "https://example.invalid/case/4471")}


def test_an_encrypted_source_is_refused_by_the_lister():
    doc = _doc()
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw",
                       owner_pw="pw")
    doc.close()
    with pytest.raises(ValueError):
        list_side_text_pdf(data)


def test_an_outline_cycle_and_a_title_that_is_not_a_string_do_not_fail():
    doc = _doc()
    doc.set_toc([[1, "First entry", 1], [1, "Second entry", 1]])
    items = [x for x in range(1, doc.xref_length()) if "/Title" in doc.xref_object(x)]
    doc.xref_set_key(items[1], "Next", f"{items[0]} 0 R")       # a cycle
    doc.xref_set_key(items[1], "Title", "42")                    # not a string
    listing = list_side_text_pdf(_bytes(doc))
    assert [e["text"] for e in listing["entries"]] == ["First entry"]
    assert listing["unreadable"] == 1


def test_a_destination_whose_key_is_not_a_string_counts_as_unreadable():
    doc = _doc()
    doc.xref_set_key(doc.pdf_catalog(), "Names",
                     f"<</Dests <</Names [17 [{doc[0].xref} 0 R /Fit]]>>>>")
    listing = list_side_text_pdf(_bytes(doc))
    assert listing == {"entries": [], "unreadable": 1}


def test_a_destination_tree_with_kids_and_the_older_dictionary_are_both_read():
    doc = _doc()
    leaf = doc.get_new_xref()
    doc.update_object(leaf, f"<</Limits[(alpha)(beta)]/Names[(alpha) "
                            f"[{doc[0].xref} 0 R /Fit] (beta) [{doc[1].xref} 0 R /Fit]]>>")
    doc.xref_set_key(doc.pdf_catalog(), "Names", f"<</Dests <</Kids [{leaf} 0 R]>>>>")
    doc.xref_set_key(doc.pdf_catalog(), "Dests",
                     f"<</Quintero [{doc[1].xref} 0 R /Fit]>>")
    listing = list_side_text_pdf(_bytes(doc))
    assert {(e["text"], e["page"]) for e in listing["entries"]} == {
        ("alpha", 0), ("beta", 1), ("Quintero", 1)}


# ── rewriting ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("arm", ["plain", "encrypted"])
def test_every_rewrite_is_absent_from_the_export(arm):
    src = _every_kind()
    protection = ({"mode": "new", "password": "pw-test"} if arm == "encrypted"
                  else None)
    out, protected = apply_redactions(src, _ssn_matches(src), side_text=_edits(src),
                                      output_protection=protection)
    assert protected
    password = "pw-test" if arm == "encrypted" else None
    for gone in ("Pemberton", "PEMBERTON", SSN, "harold.pemberton"):
        assert not _readable(out, gone, password), gone
    for new in ("Deposition of [redacted]", "Photo ID card, SSN [redacted]",
                "Annotations - [redacted]", "destination-1"):
        assert _readable(out, new, password), new


def test_strings_nobody_named_survive_byte_for_byte():
    src = _every_kind()
    out, _ = apply_redactions(src, [], side_text=_edits(src))
    after = {(e["kind"], e["text"]) for e in list_side_text_pdf(out)["entries"]}
    for kept in (("outline", "Exhibit A"), ("destination", "section.2"),
                 ("link", "https://www.example.invalid/forms"),
                 ("tag", "Company logo")):
        assert kept in after
    assert ("link", "mailto:harold.pemberton@example.invalid") not in after


def test_an_empty_edit_list_burns_exactly_like_no_field():
    src = _every_kind()
    a, _ = apply_redactions(src, [])
    b, _ = apply_redactions(src, [], side_text=[])
    assert list_side_text_pdf(a) == list_side_text_pdf(b)
    assert len(a) == len(b)


def test_a_removed_link_leaves_the_page_drawing_unchanged():
    src = _every_kind()
    out, _ = apply_redactions(src, [], side_text=_edits(src))
    d = pymupdf.open(stream=out, filetype="pdf")
    try:
        uris = [lnk.get("uri") for lnk in d[0].get_links()]
        assert uris == ["https://www.example.invalid/forms"]
        assert LINE in d[0].get_text()
    finally:
        d.close()


def test_a_renamed_destination_is_still_reached_by_the_link_that_named_it():
    src = _every_kind()
    out, _ = apply_redactions(src, [], side_text=_edits(src))
    d = pymupdf.open(stream=out, filetype="pdf")
    try:
        links = d[1].get_links()
        assert [lnk.get("nameddest") for lnk in links] == ["destination-1"]
        assert d.resolve_link("#nameddest=destination-1")[0] == 1
        assert d.resolve_link("#nameddest=section.2")[0] == 0
    finally:
        d.close()


def test_a_name_object_reference_into_the_older_dictionary_is_renamed():
    doc = _doc()
    doc.xref_set_key(doc.pdf_catalog(), "Dests",
                     f"<</QuinteroVasquez [{doc[1].xref} 0 R /Fit]>>")
    doc.set_toc([[1, "Records", 2]])
    item = next(x for x in range(1, doc.xref_length())
                if "/Title" in doc.xref_object(x))
    doc.xref_set_key(item, "Dest", "/QuinteroVasquez")
    src = _bytes(doc)
    entry = next(e for e in list_side_text_pdf(src)["entries"]
                 if e["kind"] == "destination")
    out, _ = apply_redactions(src, [], side_text=[{"id": entry["id"],
                                                   "text": "destination-1"}])
    assert not _readable(out, "QuinteroVasquez")
    d = pymupdf.open(stream=out, filetype="pdf")
    try:
        assert d.get_toc() == [[1, "Records", 2]]
    finally:
        d.close()


def test_a_rename_onto_an_existing_destination_is_skipped_as_a_whole():
    src = _every_kind()
    ids = _by_text(list_side_text_pdf(src))
    sink: dict = {}
    out, _ = apply_redactions(src, [], side_text=[
        {"id": ids["HaroldPembertonDeposition"]["id"], "text": "section.2"}],
        side_text_sink=sink)
    assert sink == {"applied": 0, "skipped": [ids["HaroldPembertonDeposition"]["id"]]}
    assert _readable(out, "HaroldPembertonDeposition")


def test_the_rebuilt_destination_tree_is_in_key_order():
    """Readers look a name up by binary search, so a rename that moves a key
    must move its entry too."""
    doc = _doc()
    doc.xref_set_key(doc.pdf_catalog(), "Names",
                     f"<</Dests <</Names [(alpha) [{doc[0].xref} 0 R /Fit] "
                     f"(Pemberton) [{doc[1].xref} 0 R /Fit] "
                     f"(mike) [{doc[0].xref} 0 R /Fit]]>>>>")
    src = _bytes(doc)
    target = _by_text(list_side_text_pdf(src))["Pemberton"]
    out, _ = apply_redactions(src, [], side_text=[{"id": target["id"],
                                                   "text": "zz-destination"}])
    keys = [e["text"] for e in list_side_text_pdf(out)["entries"]
            if e["kind"] == "destination"]
    assert keys == ["alpha", "mike", "zz-destination"]


def test_a_rename_is_refused_when_a_tree_entry_could_not_be_read():
    """Rebuilding the tree from what was read would silently drop the entry
    that was not, so the rename is skipped and the tree left as it was."""
    doc = _doc()
    doc.xref_set_key(doc.pdf_catalog(), "Names",
                     f"<</Dests <</Names [(Pemberton) [{doc[1].xref} 0 R /Fit] "
                     f"17 [{doc[0].xref} 0 R /Fit]]>>>>")
    src = _bytes(doc)
    listing = list_side_text_pdf(src)
    assert listing["unreadable"] == 1
    target = _by_text(listing)["Pemberton"]
    sink: dict = {}
    out, _ = apply_redactions(src, [], side_text=[{"id": target["id"],
                                                   "text": "destination-1"}],
                              side_text_sink=sink)
    assert sink == {"applied": 0, "skipped": [target["id"]]}
    d = pymupdf.open(stream=out, filetype="pdf")
    try:
        names = d.xref_get_key(d.pdf_catalog(), "Names/Dests/Names")[1]
    finally:
        d.close()
    assert "Pemberton" in names and "17" in names


def test_an_unknown_id_and_an_action_its_kind_cannot_take_are_skipped():
    src = _every_kind()
    ids = _by_text(list_side_text_pdf(src))
    sink: dict = {}
    apply_redactions(src, [], side_text=[
        {"id": "outline:999999", "text": "x"},
        {"id": ids["Exhibit A"]["id"], "remove": True},
        {"id": ids["mailto:harold.pemberton@example.invalid"]["id"], "text": "x"},
    ], side_text_sink=sink)
    assert sink["applied"] == 0
    assert len(sink["skipped"]) == 3


@pytest.mark.parametrize("bad", [
    "not a list",
    [{"text": "x"}],
    [{"id": 7, "text": "x"}],
    [{"id": "outline:1"}],
    [{"id": "outline:1", "text": "x", "remove": True}],
    [{"id": "outline:1", "remove": "yes"}],
    [{"id": "outline:1", "text": 5}],
])
def test_a_malformed_edit_is_refused_before_anything_changes(bad):
    with pytest.raises(ValueError):
        validate_side_text_edits(bad)
    with pytest.raises(ValueError):
        apply_redactions(_every_kind(), [], side_text=bad)


def test_a_structure_tag_can_be_removed_outright():
    src = _every_kind()
    ids = _by_text(list_side_text_pdf(src))
    out, _ = apply_redactions(src, [], side_text=[
        {"id": ids[f"Photo ID card, SSN {SSN}"]["id"], "remove": True}])
    assert not _readable(out, "Photo ID card")
    assert _readable(out, "Company logo")


# ── boundaries ───────────────────────────────────────────────────────────


def test_a_source_with_compressed_object_streams():
    doc = pymupdf.open(stream=_every_kind(), filetype="pdf")
    src = doc.tobytes(garbage=3, deflate=True, use_objstms=1)
    doc.close()
    assert b"/ObjStm" in src
    out, _ = apply_redactions(src, [], side_text=_edits(src))
    assert not _readable(out, "Pemberton")
    assert _readable(out, "Exhibit A")


def test_a_source_saved_with_an_incremental_revision(tmp_path):
    path = tmp_path / "revised.pdf"
    path.write_bytes(_every_kind())
    doc = pymupdf.open(path)
    doc.set_toc(doc.get_toc() + [[1, "Interview with Marisol Quintero", 2]])
    doc.saveIncr()
    doc.close()
    src = path.read_bytes()
    assert src.count(b"%%EOF") >= 2
    ids = _by_text(list_side_text_pdf(src))
    out, _ = apply_redactions(src, [], side_text=[
        {"id": ids["Interview with Marisol Quintero"]["id"],
         "text": "Interview with [redacted]"}])
    assert not _readable(out, "Quintero")
    assert _readable(out, "Interview with [redacted]")


def test_a_link_on_one_page_of_many_is_removed_and_the_rest_stay():
    doc = _doc(pages=12)
    for i in range(12):
        doc[i].insert_link({"kind": pymupdf.LINK_URI,
                            "from": pymupdf.Rect(60, 300, 200, 320),
                            "uri": f"https://example.invalid/page-{i}"})
    doc[7].insert_link({"kind": pymupdf.LINK_URI,
                        "from": pymupdf.Rect(60, 340, 200, 360),
                        "uri": "tel:+1-303-555-0147"})
    src = _bytes(doc)
    ids = _by_text(list_side_text_pdf(src))
    assert ids["tel:+1-303-555-0147"]["page"] == 7
    out, _ = apply_redactions(src, [], side_text=[
        {"id": ids["tel:+1-303-555-0147"]["id"], "remove": True}])
    assert not _readable(out, "555-0147")
    after = list_side_text_pdf(out)["entries"]
    assert sorted(e["text"] for e in after) == sorted(
        f"https://example.invalid/page-{i}" for i in range(12))


def test_a_500_entry_outline_lists_and_rewrites_one_entry():
    doc = _doc(pages=5)
    toc = [[1, f"Record {i:03d}", 1 + i % 5] for i in range(500)]
    toc[321] = [1, "Medical record of Marisol Quintero", 3]
    doc.set_toc(toc)
    src = _bytes(doc)
    listing = list_side_text_pdf(src)
    assert len(listing["entries"]) == 500
    target = _by_text(listing)["Medical record of Marisol Quintero"]
    out, _ = apply_redactions(src, [], side_text=[
        {"id": target["id"], "text": "Medical record of [redacted]"}])
    d = pymupdf.open(stream=out, filetype="pdf")
    try:
        titles = [t for _, t, _ in d.get_toc()]
    finally:
        d.close()
    assert len(titles) == 500
    assert titles[321] == "Medical record of [redacted]"
    assert titles[320] == "Record 320"


def test_removed_pages_do_not_stop_a_rewrite_on_a_surviving_page():
    src = _every_kind()
    edits = _edits(src)
    out, _ = apply_redactions(src, [], removed_pages=[0], side_text=edits)
    # Page 0 carried the visible SSN and is gone; the tag on page 1 that
    # repeated it was rewritten before the page removal renumbered anything.
    assert not _readable(out, "Pemberton")
    assert not _readable(out, SSN)


def test_the_cli_ops_list_and_rewrite():
    src = _every_kind()
    b64 = base64.b64encode(src).decode()
    listed = _handle({"protocol_version": PROTOCOL_VERSION, "op": "list_side_text",
                      "pdf_b64": b64})
    assert listed["ok"] and len(listed["result"]["entries"]) == 10
    resp = _handle({"protocol_version": PROTOCOL_VERSION, "op": "apply_redactions",
                    "pdf_b64": b64, "matches": [], "side_text": _edits(src)})
    assert resp["ok"]
    assert resp["result"]["side_text"] == {"applied": 6, "skipped": []}
    out = base64.b64decode(resp["result"]["pdf_b64"])
    assert not _readable(out, "Pemberton")


def test_the_cli_answers_side_text_only_when_the_request_carried_it():
    b64 = base64.b64encode(_every_kind()).decode()
    resp = _handle({"protocol_version": PROTOCOL_VERSION, "op": "apply_redactions",
                    "pdf_b64": b64, "matches": []})
    assert resp["ok"] and "side_text" not in resp["result"]


def test_the_handle_ops_list_and_rewrite():
    src = _every_kind()
    opened = _OPS["open_doc"]({"pdf_b64": base64.b64encode(src).decode()})
    handle = opened["handle"]
    try:
        listed = _OPS["list_side_text_h"]({"handle": handle})
        assert len(listed["entries"]) == 10
        result = _OPS["apply_redactions_h"]({"handle": handle, "matches": [],
                                             "side_text": _edits(src)})
        assert result["side_text"]["applied"] == 6
        assert not _readable(base64.b64decode(result["pdf_b64"]), "Pemberton")
    finally:
        _OPS["close_doc"]({"handle": handle})


def test_protocol_11_carries_the_new_op():
    assert PROTOCOL_VERSION >= 11
    assert "list_side_text" in _OPS and "list_side_text_h" in _OPS
