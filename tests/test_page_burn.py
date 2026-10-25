"""A held copy burned a page at a time gives the file one burn gives (v0.16.0).

``HeldDocument`` (``held.py``) burns each page into one held copy of the
document, in any order, and runs the whole-document steps once on a fork when
the file is wanted. These tests hold it to ``apply_redactions``: the same
payload through both must give the same file by every reading below, on an
invented document carrying every hazard ``page_burn_fixture`` lists.

"The same file" means: every page's render (72 dpi, exact), its words in order
with their boxes, every drawn image's pixel digest and box, its annotations
and its links with their targets; the outline with each entry's page, the
named destinations and where they land, the page labels, the structure tree
(every element's ``/Pg`` resolving or null, and every ``/Alt``), the metadata;
and the objects as two multisets, each object's source with every reference
blanked and its stored stream bytes. Not the bytes themselves, nor their
count: objects are numbered in the order they are made, and a reference to a
larger number takes more digits.

The 6-criteria framework:

* **Happy path** -- every page burned in a shuffled order, and in page order,
  gives the one-burn file, with the same text-removal, side-text and image
  outcomes; a held page renders and extracts as the delivered page; a page
  put back and burned again, for each of eight kinds of change, gives the
  file a fresh burn of the changed payload gives.
* **Sad paths** -- a page the assembly finds stale, or never burned, is
  burned on the fork and the file is still the one-burn file; a structure
  element two pages' burns both edit takes both pages back; a payload
  ``apply_redactions`` rejects is rejected; a bad handle, a page out of
  range, a page put back that is not burned, a page burned twice, a fork of a
  closed handle, and every op that would save a held copy in place are
  refused with a named error.
* **Boundary** -- a one-page document; every page removed but one; a page
  with more than a thousand boxes; a rotated page and a document with layers
  (both in the fixture); a stream the save writes as a CCITT bitmap; a
  burn-created stream equal to one the source holds compressed.
* **No logic mirroring** -- every comparison is against ``apply_redactions``'
  own output for the same payload, read back from the bytes; the readings
  are digests and counts, never a second implementation of the burn.
* **Side-effect verification** -- the held copy is the same, object for
  object, before and after a fork; the prepared copy is never written, after
  burns, restores and assemblies; a held copy serialises to no more than the
  source plus twice the delivered file; putting every page back leaves every
  object as the prepared copy has it.
* **Mock integrity** -- every document is built in memory with PyMuPDF and
  read through the real entry points, the CLI's ops included; no PyMuPDF call
  is mocked.

Synthetic fixtures only; every value is invented.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import random
import re
from collections import Counter

import pymupdf
import pytest

from lexcloak_pdf_tool import apply_redactions, set_metadata
from lexcloak_pdf_tool.__main__ import (
    _OPS, PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS, _DOC_CACHE, _handle,
)
from lexcloak_pdf_tool.held import HeldDocument, HeldPageError, _compress_like_save
from lexcloak_pdf_tool.unseen import list_tag_text

from page_burn_fixture import LABEL, build

# ── readings ─────────────────────────────────────────────────────────────


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _r(rect) -> tuple:
    return tuple(round(float(v), 2) for v in rect)


def page_facts(doc, i: int) -> dict:
    page = doc[i]
    links = sorted(((ln.get("kind"), _r(ln["from"]), ln.get("page"),
                     _h((ln.get("uri") or "").encode()),
                     _h(str(ln.get("nameddest") or ln.get("name") or "").encode()))
                    for ln in page.get_links()), key=repr)
    return {
        "render": _h(page.get_pixmap(dpi=72, alpha=False).samples),
        "words": _h(repr([(_r(w[:4]), w[4]) for w in page.get_text("words")]).encode()),
        "images": sorted((_r(im["bbox"]), im["digest"].hex())
                         for im in page.get_image_info(hashes=True)),
        "annots": sorted((a.type[1], _r(a.rect), _h((a.info.get("content") or "").encode()))
                         for a in page.annots()),
        "links": links,
        "rotation": page.rotation,
    }


def _struct(doc) -> dict:
    pages = {doc[i].xref for i in range(len(doc))}
    if doc.xref_get_key(doc.pdf_catalog(), "StructTreeRoot")[0] != "xref":
        return {"present": False}
    out = Counter()
    alts = []
    for x in range(1, doc.xref_length()):
        try:
            if doc.xref_get_key(x, "Type") != ("name", "/StructElem"):
                continue
        except Exception:  # noqa: BLE001
            continue
        out["elems"] += 1
        kind, value = doc.xref_get_key(x, "Pg")
        if kind == "xref":
            pg = int(value.split()[0])
            out["pg_resolves" if pg in pages else
                "pg_null" if doc.xref_object(pg).strip() == "null" else "pg_dangling"] += 1
        for key in ("Alt", "ActualText", "E"):
            v = doc.xref_get_key(x, key)
            if v[0] == "string":
                alts.append((key, _h(v[1].encode())))
    return {"present": True, **out, "alts": sorted(alts)}


def doc_facts(doc) -> dict:
    toc = [(lvl, _h(title.encode()), page) for lvl, title, page, *_ in doc.get_toc(simple=False)]
    names = sorted((_h(k.encode()), v.get("page")) for k, v in doc.resolve_names().items())
    return {"pages": len(doc), "toc": toc, "named_dests": names,
            "page_labels": [_h(repr(d).encode()) for d in doc.get_page_labels()],
            "struct": _struct(doc), "metadata": dict(doc.metadata or {})}


def facts(data: bytes) -> dict:
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        return {"doc": doc_facts(doc),
                "pages": [page_facts(doc, i) for i in range(len(doc))]}
    finally:
        doc.close()


_REF = re.compile(r"\b\d+ \d+ R\b")


def objects(data: bytes) -> Counter:
    """Every object's source with references blanked, and its stored bytes."""
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        out = Counter()
        for x in range(1, doc.xref_length()):
            raw = doc.xref_stream_raw(x) if doc.xref_is_stream(x) else b""
            out[(_REF.sub("R", doc.xref_object(x, compressed=True)), _h(raw or b""))] += 1
        return out
    finally:
        doc.close()


def assert_same_file(a: bytes, b: bytes) -> None:
    fa, fb = facts(a), facts(b)
    assert fa["doc"] == fb["doc"]
    assert len(fa["pages"]) == len(fb["pages"])
    for i, (pa, pb) in enumerate(zip(fa["pages"], fb["pages"])):
        assert pa == pb, f"page {i} differs"
    oa, ob = objects(a), objects(b)
    assert oa == ob, (f"objects differ: {sum((oa - ob).values())} only in one burn, "
                      f"{sum((ob - oa).values())} only in the held copy's")
    assert a != b       # numbering differs: the comparison above is not vacuous


def snapshot(doc) -> list:
    """Every object of ``doc``: its source and its stored stream bytes, or
    ``None`` for a number no object has."""
    out = []
    for x in range(1, doc.xref_length()):
        try:
            out.append((doc.xref_object(x, compressed=True),
                        doc.xref_stream_raw(x) if doc.xref_is_stream(x) else None))
        except RuntimeError:
            out.append(None)
    return out


# ── paths ────────────────────────────────────────────────────────────────


def one_burn(src: bytes, p: dict) -> tuple[bytes, dict]:
    p = copy.deepcopy(p)
    sinks: dict = {"removal": {}, "side": {}, "unseen": {}}
    out, _ = apply_redactions(
        src, p["matches"], redact_label=p["redact_label"],
        active_categories=p["active_categories"],
        removed_pages=p["removed_pages"], blackout_pages=p["blackout_pages"],
        remove_text=p.get("remove_text"), removal_sink=sinks["removal"],
        keep_uncovered_lines=p["keep_uncovered_lines"],
        side_text=p.get("side_text"), side_text_sink=sinks["side"],
        blank_images=p.get("blank_images"), remove_images=p.get("remove_images"),
        strip_tags_pages=p.get("strip_tags_pages"), unseen_sink=sinks["unseen"])
    return out, sinks


def open_held(src: bytes, p: dict) -> HeldDocument:
    return HeldDocument(src, blank_images=p.get("blank_images"),
                        remove_images=p.get("remove_images"))


def burn(held: HeldDocument, p: dict, pages) -> dict:
    """Burn ``pages`` into ``held`` with ``p``'s inputs for each; return the
    text-removal outcomes by page."""
    removed = set(p["removed_pages"])
    blackout = set(p["blackout_pages"]) - removed
    tags = set(p.get("strip_tags_pages") or ())
    out = {}
    for pno in pages:
        if pno in removed:
            continue
        ms = [copy.deepcopy(m) for m in p["matches"] if m["page"] == pno]
        rt = ([dict(i) for i in p["remove_text"] if i["page"] == pno]
              if p.get("remove_text") is not None else None)
        out[pno] = held.burn_page(
            pno, ms, redact_label=p["redact_label"],
            active_categories=p["active_categories"],
            keep_uncovered_lines=p["keep_uncovered_lines"],
            blackout=pno in blackout, remove_text=rt, strip_tags=pno in tags)
    return out


def assemble(held: HeldDocument, p: dict, **extra) -> tuple[bytes, dict]:
    p = copy.deepcopy(p)
    return held.assemble(
        p["matches"], redact_label=p["redact_label"],
        active_categories=p["active_categories"],
        removed_pages=p["removed_pages"], blackout_pages=p["blackout_pages"],
        remove_text=p.get("remove_text"),
        keep_uncovered_lines=p["keep_uncovered_lines"],
        side_text=p.get("side_text"), strip_tags_pages=p.get("strip_tags_pages"),
        **extra)


def shuffled(n: int, seed: int = 89) -> list[int]:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hazard():
    src, payload, hz = build(30)
    out, sinks = one_burn(src, payload)
    return {"src": src, "payload": payload, "hz": hz, "n": 30,
            "out": out, "sinks": sinks}


@pytest.fixture(scope="module")
def burned(hazard):
    """A held copy with every page burned in a shuffled order. Tests that
    change it open their own."""
    held = open_held(hazard["src"], hazard["payload"])
    burn(held, hazard["payload"], shuffled(hazard["n"]))
    yield held
    held.close()


def stale_payload(p: dict, hz: dict, n: int) -> tuple[dict, dict[str, int]]:
    """A copy of ``p`` with one change on each of eight kinds of page."""
    from page_burn_fixture import is_scanned
    q = copy.deepcopy(p)
    ms = q["matches"]
    special = set(hz.values()) | {hz["span"] + 1}
    digital = next(i for i in range(n) if not is_scanned(i, hz) and i not in special and i)
    scanned = next(i for i in range(n) if is_scanned(i, hz) and i not in special)
    logo = next(i for i in range(3, n, 10) if i not in special)
    pages = {"digital": digital, "scanned": scanned, "dense": hz["dense"],
             "rotated": hz["rotated"], "logo_box": logo, "widget": hz["widget"],
             "hidden_white": hz["hidden_white"], "alt_all_off": hz["alt_value"]}

    def first(page, pred=lambda m: True):
        return next(m for m in ms if m["page"] == page and m.get("enabled", True) and pred(m))

    box = {"x0": 300.0, "y0": 400.0, "x1": 420.0, "y1": 414.0}
    ms.append({"page": digital, "type": "Manual Region", "text": "", "enabled": True,
               "rect": dict(box)})
    first(scanned)["enabled"] = False
    r = first(hz["dense"])["rect"]
    r["x0"] += 2.0
    r["x1"] += 2.0
    ms.append({"page": hz["rotated"], "type": "Manual Region", "text": "", "enabled": True,
               "rect": {"x0": 100.0, "y0": 100.0, "x1": 220.0, "y1": 116.0}})
    first(logo, lambda m: m["type"] == "Manual Region")["enabled"] = False
    ms.append({"page": hz["widget"], "type": "Manual Region", "text": "", "enabled": True,
               "rect": dict(box)})
    ms.append({"page": hz["hidden_white"], "type": "Manual Region", "text": "",
               "enabled": True, "rect": dict(box)})
    for m in ms:
        if m["page"] == hz["alt_value"]:
            m["enabled"] = False
    return q, pages


# ── the same file ────────────────────────────────────────────────────────


def test_every_page_burned_in_a_shuffled_order_gives_the_one_burn_file(hazard, burned):
    out, outcome = assemble(burned, hazard["payload"])
    assert_same_file(hazard["out"], out)
    assert outcome["burned_at_assembly"] == []
    sinks = hazard["sinks"]
    assert outcome["removal"] == sinks["removal"]
    assert outcome["removal"]["removed"]          # the hidden words were removed
    assert outcome["side"] == sinks["side"]
    assert outcome["side"]["applied"] == 5 and outcome["side"]["skipped"] == []
    assert burned.unseen == sinks["unseen"]
    assert burned.unseen["blanked"] == 2 and burned.unseen["removed"] == 2


def test_page_order_gives_the_one_burn_file(hazard):
    held = open_held(hazard["src"], hazard["payload"])
    try:
        burn(held, hazard["payload"], range(hazard["n"]))
        out, _ = assemble(held, hazard["payload"])
    finally:
        held.close()
    assert_same_file(hazard["out"], out)


def test_a_held_page_reads_as_the_delivered_page(hazard, burned):
    delivered = pymupdf.open(stream=hazard["out"], filetype="pdf")
    removed = sorted(hazard["payload"]["removed_pages"])
    try:
        for i in range(hazard["n"]):
            if i in removed:
                continue
            j = i - sum(1 for r in removed if r < i)
            held_page, out_page = page_facts(burned.doc, i), page_facts(delivered, j)
            assert held_page["render"] == out_page["render"], f"page {i}"
            assert held_page["words"] == out_page["words"], f"page {i}"
            assert held_page["annots"] == out_page["annots"], f"page {i}"
    finally:
        delivered.close()


def test_a_held_page_reads_the_same_through_the_handle_ops(hazard):
    opened = _handle({"op": "open_held", "pdf_b64": base64.b64encode(hazard["src"]).decode(),
                      "blank_images": hazard["payload"]["blank_images"],
                      "remove_images": hazard["payload"]["remove_images"]})
    assert opened["ok"], opened
    handle = opened["result"]["handle"]
    try:
        assert opened["result"]["page_count"] == hazard["n"]
        assert opened["result"]["unseen_images"] == hazard["sinks"]["unseen"]
        page = 1
        ms = [m for m in hazard["payload"]["matches"] if m["page"] == page]
        r = _handle({"op": "burn_page_h", "handle": handle, "page": page, "matches": ms,
                     "redact_label": LABEL,
                     "active_categories": hazard["payload"]["active_categories"],
                     "keep_uncovered_lines": True})
        assert r == {"ok": True, "result": {"page": page}}
        out = _handle({"op": "open_doc",
                       "pdf_b64": base64.b64encode(hazard["out"]).decode()})["result"]["handle"]
        try:
            for op in ("render_h", "extract_text_plain_h"):
                held_read = _handle({"op": op, "handle": handle, "page": page, "dpi": 72})
                out_read = _handle({"op": op, "handle": out, "page": page, "dpi": 72})
                assert held_read["ok"] and held_read == out_read, op
            # The clean save writes the page's content again, which moves a
            # glyph's computed box by a few millionths of a point.
            traces = [_handle({"op": "trace_text_h", "handle": h, "page": page})["result"]
                      for h in (handle, out)]
            words = [[(w["text"], w["mode"], w["layer"], _r(w["bbox"])) for w in t["words"]]
                     for t in traces]
            assert words[0] == words[1] and words[0]
        finally:
            _handle({"op": "close_doc", "handle": out})
    finally:
        _handle({"op": "close_doc", "handle": handle})


# ── stale pages ──────────────────────────────────────────────────────────


def test_a_stale_page_of_each_kind_put_back_and_burned_again(hazard):
    q, pages = stale_payload(hazard["payload"], hazard["hz"], hazard["n"])
    held = open_held(hazard["src"], hazard["payload"])
    try:
        burn(held, hazard["payload"], shuffled(hazard["n"]))
        for kind, pno in pages.items():
            restored = held.restore_page(pno)
            assert restored == [pno], kind
            burn(held, q, [pno])
        out, outcome = assemble(held, q)
    finally:
        held.close()
    want, sinks = one_burn(hazard["src"], q)
    assert_same_file(want, out)
    assert outcome["burned_at_assembly"] == []
    assert outcome["removal"] == sinks["removal"]
    assert outcome["side"] == sinks["side"]


def test_the_assembly_burns_on_the_fork_what_the_held_copy_lacks(hazard):
    q, pages = stale_payload(hazard["payload"], hazard["hz"], hazard["n"])
    held = open_held(hazard["src"], hazard["payload"])
    try:
        # Half the pages burned with the old payload, and never put back.
        burn(held, hazard["payload"], range(0, hazard["n"], 2))
        held_before = snapshot(held.doc)
        out, outcome = assemble(held, q)
        assert snapshot(held.doc) == held_before
    finally:
        held.close()
    want, _ = one_burn(hazard["src"], q)
    assert_same_file(want, out)
    # Redone on the fork: the even pages the change touched (but the page it
    # switches every box off on, which is only put back) and every odd page.
    stale_even = {p for p in pages.values() if p % 2 == 0} - {hazard["hz"]["alt_value"]}
    odd = set(range(1, hazard["n"], 2))
    assert outcome["burned_at_assembly"] == sorted(stale_even | odd)


def test_an_element_two_pages_edit_takes_both_pages_back(hazard):
    a = hazard["hz"]["span"]
    held = open_held(hazard["src"], hazard["payload"])
    try:
        burn(held, hazard["payload"], shuffled(hazard["n"]))
        assert held.restore_page(a) == [a, a + 1]
        assert set(held.burned_pages) == set(range(hazard["n"])) - {
            a, a + 1, *hazard["payload"]["removed_pages"]}
        burn(held, hazard["payload"], [a + 1, a])
        out, outcome = assemble(held, hazard["payload"])
    finally:
        held.close()
    assert outcome["burned_at_assembly"] == []
    assert_same_file(hazard["out"], out)


def test_putting_every_page_back_leaves_the_prepared_copy(hazard):
    held = open_held(hazard["src"], hazard["payload"])
    try:
        prepared = snapshot(held.prepared)
        burn(held, hazard["payload"], shuffled(hazard["n"]))
        for pno in shuffled(hazard["n"], seed=3):
            if pno in held.burned_pages:
                held.restore_page(pno)
        assert held.burned_pages == []
        # Every object the source had is as prepared; what the burns made is
        # left for the save's garbage collection.
        assert snapshot(held.doc)[:len(prepared)] == prepared
        out, outcome = assemble(held, hazard["payload"])
    finally:
        held.close()
    assert outcome["burned_at_assembly"] == [
        p for p in range(hazard["n"]) if p not in hazard["payload"]["removed_pages"]]
    assert_same_file(hazard["out"], out)


def test_a_page_put_back_never_takes_what_another_page_draws(hazard):
    """MuPDF keeps one font per document for every label it writes, so the
    second page burned draws a font the first page's burn made. Putting the
    first page back must leave it."""
    held = HeldDocument(hazard["src"])
    try:
        labelled = [dict(_box(1, 300, 400, 420, 414), redact_label="Party B")]
        held.burn_page(1, copy.deepcopy(labelled))
        later = [dict(_box(5, 300, 400, 420, 414), redact_label="Party B")]
        held.burn_page(5, [dict(m, page=5) for m in copy.deepcopy(later)])
        before = page_facts(held.doc, 5)
        assert held.restore_page(1) == [1]
        assert page_facts(held.doc, 5) == before
        out, outcome = held.assemble(copy.deepcopy(labelled) + copy.deepcopy(later))
    finally:
        held.close()
    want, _ = apply_redactions(hazard["src"], copy.deepcopy(labelled) + copy.deepcopy(later))
    assert outcome["burned_at_assembly"] == [1]
    assert_same_file(want, out)


def test_a_page_burn_changes_only_what_a_restore_puts_back(hazard, burned):
    """Of the objects the prepared copy holds, a burn of every page changes
    the page dictionaries, objects one page reaches, structure elements, and
    the shared forms and content a strip of tags rewrites. Nothing else."""
    ix = burned._index
    allowed = set(ix.page_xref)
    for p in range(hazard["n"]):
        allowed |= set(ix.own[p]) | set(ix.shared[p])
    changed = {x for x in range(1, burned.prepared.xref_length())
               if burned.prepared.xref_object(x, compressed=True)
               != burned.doc.xref_object(x, compressed=True)}
    assert changed, "the burns changed nothing in place: the check is vacuous"
    assert changed <= allowed, sorted(changed - allowed)


# ── what a fork and a held copy leave behind ─────────────────────────────


def test_a_fork_leaves_the_held_copy_as_it_was(hazard, burned):
    before = snapshot(burned.doc)
    prepared = snapshot(burned.prepared)
    assemble(burned, hazard["payload"])
    assert snapshot(burned.doc) == before
    assert snapshot(burned.prepared) == prepared


def test_the_prepared_copy_is_never_written(hazard):
    q, pages = stale_payload(hazard["payload"], hazard["hz"], hazard["n"])
    held = open_held(hazard["src"], hazard["payload"])
    try:
        prepared = snapshot(held.prepared)
        burn(held, hazard["payload"], shuffled(hazard["n"]))
        for pno in pages.values():
            for back in held.restore_page(pno):
                burn(held, q, [back])
        assemble(held, q)
        assert snapshot(held.prepared) == prepared
    finally:
        held.close()


def test_a_held_copy_stays_within_the_source_plus_twice_the_file(hazard, burned):
    held_size = len(burned.doc.tobytes(no_new_id=True))
    assert held_size <= len(hazard["src"]) + 2 * len(hazard["out"])


def test_a_held_copy_stays_within_the_bound_through_restores(hazard):
    """A page burned again leaves its old burn behind for the save's garbage
    collection, so only that page's share grows."""
    from page_burn_fixture import is_scanned
    held = open_held(hazard["src"], hazard["payload"])
    try:
        burn(held, hazard["payload"], range(hazard["n"]))
        scanned = next(i for i in range(hazard["n"]) if is_scanned(i, hazard["hz"]))
        for _ in range(3):
            held.restore_page(scanned)
            burn(held, hazard["payload"], [scanned])
        held_size = len(held.doc.tobytes(no_new_id=True))
        out, _ = assemble(held, hazard["payload"])
    finally:
        held.close()
    assert held_size <= len(hazard["src"]) + 2 * len(hazard["out"])
    assert_same_file(hazard["out"], out)


# ── the stamp's metadata in the assembly's own save ──────────────────────


def test_the_stamps_metadata_rides_in_the_one_save(hazard, burned):
    stamp = {"subject": "Redacted copy", "producer": "Synthetic 1.0",
             "keywords": "redacted"}
    out, _ = assemble(burned, hazard["payload"], metadata=stamp)
    want = set_metadata(hazard["out"], stamp)
    fa, fb = facts(want), facts(out)
    assert fb["doc"]["metadata"]["subject"] == "Redacted copy"
    assert fa["doc"] == fb["doc"]
    assert fa["pages"] == fb["pages"]


# ── compression as the save does it ──────────────────────────────────────


def _image_doc(*, mask: bool, extra_icc: bool = False) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    if mask:
        x = doc.get_new_xref()
        w, h = 64, 32
        data = bytes([0xAA if (r // 4) % 2 else 0x55 for r in range(h)
                      for _ in range(w // 8)])
        doc.update_object(x, f"<< /Type /XObject /Subtype /Image /Width {w} "
                             f"/Height {h} /ImageMask true /BitsPerComponent 1 >>")
        doc.update_stream(x, data, compress=False)
        page.insert_image(pymupdf.Rect(50, 250, 306, 378), xref=x)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 40), False)
    pix.set_rect(pix.irect, (200, 30, 30))
    page.insert_image(pymupdf.Rect(50, 50, 290, 210), pixmap=pix)
    page.insert_text((60, 450), "Ada Quellbrook 555-010-4477", fontsize=12)
    data = doc.tobytes(garbage=3, deflate=True)
    if extra_icc:
        # The source already holds what a burn writes, compressed as the save
        # compresses it: a burned file of the same document.
        burned_once, _ = apply_redactions(data, [_box(0, 60, 60, 150, 120)])
        two = pymupdf.open(stream=data, filetype="pdf")
        two.insert_pdf(pymupdf.open(stream=burned_once, filetype="pdf"))
        data = two.tobytes(garbage=3, deflate=True)
    return data


def _box(page, x0, y0, x1, y1, **extra) -> dict:
    return {"page": page, "type": "Manual Region", "enabled": True,
            "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}, **extra}


@pytest.mark.parametrize("mask,extra_icc", [(True, False), (False, True)],
                         ids=["ccitt-bitmap", "matches-a-compressed-source-stream"])
def test_compressing_what_a_burn_made_leaves_the_files_objects(mask, extra_icc):
    src = _image_doc(mask=mask, extra_icc=extra_icc)
    matches = [_box(0, 60, 60, 150, 120), _box(0, 60, 260, 200, 300),
               _box(0, 55, 435, 200, 455)]
    want, _ = apply_redactions(src, copy.deepcopy(matches))
    held = HeldDocument(src)
    try:
        first = held.doc.xref_length()
        held.burn_page(0, copy.deepcopy(matches))
        filters = {held.doc.xref_get_key(x, "Filter")[1]
                   for x in range(first, held.doc.xref_length())
                   if held.doc.xref_is_stream(x)}
        if mask:
            assert "/CCITTFaxDecode" in filters
        assert "null" not in filters            # every stream it made is compressed
        out, _ = held.assemble(copy.deepcopy(matches))
    finally:
        held.close()
    assert_same_file(want, out)


def test_compressing_matches_the_save_stream_for_stream():
    """A stream compressed ahead is written by the save exactly as the save
    writes it uncompressed: two saves of the same burn, one with its streams
    compressed first, hold the same objects under the same numbers."""
    src = _image_doc(mask=True)

    def burned(pre: bool) -> bytes:
        doc = pymupdf.open(stream=src, filetype="pdf")
        first = doc.xref_length()
        page = doc[0]
        for r in ((60, 60, 150, 120), (60, 260, 200, 300)):
            page.add_redact_annot(pymupdf.Rect(*r), fill=(0, 0, 0))
        page.apply_redactions()
        del page
        if pre:
            assert _compress_like_save(doc, range(first, doc.xref_length()))
        return doc.tobytes(garbage=4, deflate=True, clean=True)

    a = pymupdf.open(stream=burned(False), filetype="pdf")
    b = pymupdf.open(stream=burned(True), filetype="pdf")
    assert a.xref_length() == b.xref_length()
    assert snapshot(a) == snapshot(b)


# ── boundaries ───────────────────────────────────────────────────────────


def _lines_doc(n: int, rotate: int = 0) -> bytes:
    doc = pymupdf.open()
    for i in range(n):
        page = doc.new_page(width=612, height=792)
        for k in range(12):
            page.insert_text((72, 90 + 24 * k), f"Line {k} of page {i}: Tobin Varre "
                             f"5501-{1000 + i * 12 + k}", fontsize=11)
        if rotate:
            page.set_rotation(rotate)
    return doc.tobytes(garbage=3, deflate=True)


def _word_matches(src: bytes, *, every: int = 1) -> list[dict]:
    doc = pymupdf.open(stream=src, filetype="pdf")
    out = []
    for i, page in enumerate(doc):
        for k, w in enumerate(page.get_text("words")):
            if k % every == 0:
                r = pymupdf.Rect(w[:4]) * page.rotation_matrix
                out.append(_box(i, r.x0, r.y0, r.x1, r.y1))
    return out


def _held_vs_one_burn(src: bytes, matches: list[dict], **kw) -> None:
    want, _ = apply_redactions(src, copy.deepcopy(matches), **copy.deepcopy(kw))
    held = HeldDocument(src)
    try:
        removed = set(kw.get("removed_pages") or ())
        n = len(held.doc)
        for pno in shuffled(n):
            if pno not in removed:
                held.burn_page(pno, [copy.deepcopy(m) for m in matches if m["page"] == pno],
                               keep_uncovered_lines=kw.get("keep_uncovered_lines", False))
        out, outcome = held.assemble(copy.deepcopy(matches), **copy.deepcopy(kw))
    finally:
        held.close()
    assert outcome["burned_at_assembly"] == []
    assert_same_file(want, out)


def test_a_one_page_document():
    src = _lines_doc(1)
    _held_vs_one_burn(src, _word_matches(src, every=3))


def test_every_page_removed_but_one():
    src = _lines_doc(4)
    _held_vs_one_burn(src, _word_matches(src, every=2), removed_pages=[0, 1, 3])


def test_every_page_removed_is_refused_as_one_burn_refuses_it():
    src = _lines_doc(2)
    with pytest.raises(ValueError, match="all pages have been removed"):
        apply_redactions(src, [], removed_pages=[0, 1])
    held = HeldDocument(src)
    try:
        with pytest.raises(ValueError, match="all pages have been removed"):
            held.assemble([], removed_pages=[0, 1])
    finally:
        held.close()


def test_a_rotated_document():
    src = _lines_doc(3, rotate=270)
    _held_vs_one_burn(src, _word_matches(src, every=2), keep_uncovered_lines=True)


def test_a_page_with_more_than_a_thousand_boxes():
    src, payload, hz = build(20, dense_boxes=1050)
    page = hz["dense"]
    matches = [m for m in payload["matches"] if m["page"] == page]
    assert len(matches) >= 1000
    doc = pymupdf.open(stream=src, filetype="pdf")
    one = pymupdf.open()
    one.insert_pdf(doc, from_page=page, to_page=page)
    src_one = one.tobytes(garbage=3, deflate=True)
    _held_vs_one_burn(src_one, [dict(m, page=0) for m in matches],
                      active_categories=["Number"])


# ── a payload apply_redactions refuses, and the held copy's own refusals ─


def test_a_payload_one_burn_refuses_is_refused(hazard, burned):
    bad = [{"page": 1, "type": "Number", "rect": {"x0": 10, "y0": 10, "x1": 5, "y1": 20}}]
    with pytest.raises(ValueError, match="invalid bounds"):
        apply_redactions(hazard["src"], copy.deepcopy(bad))
    with pytest.raises(ValueError, match="invalid bounds"):
        burned.assemble(copy.deepcopy(bad))
    fresh = HeldDocument(hazard["src"])
    try:
        with pytest.raises(ValueError, match="invalid bounds"):
            fresh.burn_page(1, copy.deepcopy(bad))
        assert fresh.burned_pages == []
    finally:
        fresh.close()
    out_of_range = [_box(99, 10, 10, 20, 20)]
    with pytest.raises(IndexError):
        apply_redactions(hazard["src"], copy.deepcopy(out_of_range))
    with pytest.raises(IndexError):
        burned.assemble(copy.deepcopy(out_of_range))
    with pytest.raises(ValueError, match="remove_text"):
        burned.assemble([], remove_text="not a list")
    with pytest.raises(ValueError, match="side_text"):
        burned.assemble([], side_text=[{"id": 3}])
    with pytest.raises(ValueError, match="metadata key"):
        burned.assemble([], metadata={"author_name": "x"})


def test_page_operations_refuse_what_the_state_does_not_allow(hazard):
    held = HeldDocument(hazard["src"])
    try:
        with pytest.raises(HeldPageError, match="not burned"):
            held.restore_page(1)
        held.burn_page(1, [])
        with pytest.raises(HeldPageError, match="already burned"):
            held.burn_page(1, [])
        with pytest.raises(IndexError):
            held.burn_page(hazard["n"], [])
        with pytest.raises(IndexError):
            held.restore_page(-1)
        with pytest.raises(ValueError, match="integer"):
            held.burn_page(True, [])
        with pytest.raises(ValueError, match="on the page being burned"):
            held.burn_page(2, [_box(3, 10, 10, 20, 20)])
        with pytest.raises(ValueError, match="on the page being burned"):
            held.burn_page(2, [], remove_text=[{"page": 3, "box": [1, 1, 2, 2]}])
        with pytest.raises(ValueError, match="boolean"):
            held.burn_page(2, [], blackout="yes")
        assert held.burned_pages == [1]
    finally:
        held.close()


def test_an_encrypted_document_is_refused():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "locked")
    src = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    with pytest.raises(ValueError, match="unencrypted"):
        HeldDocument(src)


# ── the wire ─────────────────────────────────────────────────────────────


def test_protocol_14_advertises_the_held_copy_ops():
    assert PROTOCOL_VERSION == 14
    assert 14 in SUPPORTED_PROTOCOL_VERSIONS and 13 in SUPPORTED_PROTOCOL_VERSIONS
    assert {"open_held", "burn_page_h", "restore_page_h", "list_tag_text_page_h",
            "assemble_h"} <= set(_OPS)


def test_the_held_ops_end_to_end_on_the_wire(hazard, tmp_path):
    p = hazard["payload"]
    opened = _handle({"op": "open_held", "pdf_b64": base64.b64encode(hazard["src"]).decode(),
                      "blank_images": p["blank_images"], "remove_images": p["remove_images"]})
    handle = opened["result"]["handle"]
    try:
        removed = set(p["removed_pages"])
        blackout = set(p["blackout_pages"])
        for pno in shuffled(hazard["n"]):
            if pno in removed:
                continue
            cmd = {"op": "burn_page_h", "handle": handle, "page": pno,
                   "matches": [m for m in p["matches"] if m["page"] == pno],
                   "redact_label": p["redact_label"],
                   "active_categories": p["active_categories"],
                   "keep_uncovered_lines": p["keep_uncovered_lines"],
                   "blackout": pno in blackout,
                   "strip_tags": pno in p["strip_tags_pages"]}
            if pno == hazard["hz"]["hidden_white"]:
                cmd["remove_text"] = [i for i in p["remove_text"] if i["page"] == pno]
            r = _handle(copy.deepcopy(cmd))
            assert r["ok"], r
            if "remove_text" in cmd:
                assert r["result"]["text_removal"] == hazard["sinks"]["removal"]
        out_path = str(tmp_path / "assembled.pdf")
        r = _handle(copy.deepcopy({
            "op": "assemble_h", "handle": handle, "out_path": out_path,
            "matches": p["matches"], "redact_label": p["redact_label"],
            "active_categories": p["active_categories"],
            "removed_pages": p["removed_pages"], "blackout_pages": p["blackout_pages"],
            "remove_text": p["remove_text"], "keep_uncovered_lines": True,
            "side_text": p["side_text"], "strip_tags_pages": p["strip_tags_pages"]}))
        assert r["ok"], r
        res = r["result"]
        assert res["burned_at_assembly"] == []
        assert res["text_removal"] == hazard["sinks"]["removal"]
        assert res["side_text"] == hazard["sinks"]["side"]
        assert res["unseen_images"] == hazard["sinks"]["unseen"]
        assert res["pdf_path"] == out_path
        with open(out_path, "rb") as fh:
            out = fh.read()
        assert res["pdf_size"] == len(out)
        assert_same_file(hazard["out"], out)
        pg = hazard["hz"]["span"]
        back = _handle({"op": "restore_page_h", "handle": handle, "page": pg})
        assert back == {"ok": True, "result": {"restored": [pg, pg + 1]}}
    finally:
        assert _handle({"op": "close_doc", "handle": handle})["result"] == {"closed": True}
    assert handle not in _DOC_CACHE


def test_one_pages_tags_are_the_whole_listings_for_that_page(hazard):
    doc = pymupdf.open(stream=hazard["src"], filetype="pdf")
    whole = list_tag_text(doc)
    opened = _handle({"op": "open_doc", "pdf_b64": base64.b64encode(hazard["src"]).decode()})
    handle = opened["result"]["handle"]
    try:
        tagged = {e["page"] for e in whole["entries"]}
        assert hazard["hz"]["actualtext"] in tagged
        for pno in range(hazard["n"]):
            r = _handle({"op": "list_tag_text_page_h", "handle": handle, "page": pno})
            assert r["ok"], r
            assert r["result"] == {"entries": [e for e in whole["entries"] if e["page"] == pno],
                                   "unreadable_pages": []}
        bad = _handle({"op": "list_tag_text_page_h", "handle": handle, "page": hazard["n"]})
        assert bad["ok"] is False and bad["error_type"] == "IndexError"
    finally:
        _handle({"op": "close_doc", "handle": handle})
        doc.close()


def test_one_pages_tags_on_a_held_copy_are_the_held_pages(hazard):
    page = hazard["hz"]["actualtext"]
    opened = _handle({"op": "open_held", "pdf_b64": base64.b64encode(hazard["src"]).decode()})
    handle = opened["result"]["handle"]
    try:
        before = _handle({"op": "list_tag_text_page_h", "handle": handle, "page": page})
        assert before["result"]["entries"], "the fixture's tag must be listed"
        _handle({"op": "burn_page_h", "handle": handle, "page": page, "matches": [],
                 "strip_tags": True})
        after = _handle({"op": "list_tag_text_page_h", "handle": handle, "page": page})
        assert after == {"ok": True, "result": {"entries": [], "unreadable_pages": []}}
    finally:
        _handle({"op": "close_doc", "handle": handle})


def test_the_wire_refuses_what_a_held_copy_does_not_allow(hazard):
    src_b64 = base64.b64encode(hazard["src"]).decode()
    plain = _handle({"op": "open_doc", "pdf_b64": src_b64})["result"]["handle"]
    held = _handle({"op": "open_held", "pdf_b64": src_b64})["result"]["handle"]
    try:
        for op in ("burn_page_h", "restore_page_h", "assemble_h"):
            r = _handle({"op": op, "handle": plain, "page": 0, "matches": []})
            assert r["ok"] is False and r["error_type"] == "NotAHeldCopy", (op, r)
            r = _handle({"op": op, "handle": "no-such-handle", "page": 0, "matches": []})
            assert r["ok"] is False and r["error_type"] == "HandleNotFound", (op, r)
        for op in ("apply_redactions_h", "strip_metadata_h", "set_metadata_h",
                   "insert_cover_page_h", "reduce_size_h", "encrypt_h"):
            r = _handle({"op": op, "handle": held, "matches": [], "fields": {},
                         "context": {}, "password": ""})
            assert r["ok"] is False and r["error_type"] == "NotAHeldCopy", (op, r)
        r = _handle({"op": "restore_page_h", "handle": held, "page": 0})
        assert r["ok"] is False and r["error_type"] == "HeldPageError"
        r = _handle({"op": "burn_page_h", "handle": held, "page": "0", "matches": []})
        assert r["ok"] is False and r["error_type"] == "ValueError"
        r = _handle({"op": "burn_page_h", "handle": held, "page": hazard["n"], "matches": []})
        assert r["ok"] is False and r["error_type"] == "IndexError"
        for field, value in (("blank_images", [1]), ("output_protection", {"mode": "new"})):
            r = _handle({"op": "assemble_h", "handle": held, "matches": [], field: value})
            assert r["ok"] is False and field in r["error"], r
        # The read ops a held copy serves.
        r = _handle({"op": "page_count_h", "handle": held})
        assert r == {"ok": True, "result": {"count": hazard["n"]}}
    finally:
        _handle({"op": "close_doc", "handle": plain})
        _handle({"op": "close_doc", "handle": held})
    r = _handle({"op": "assemble_h", "handle": held, "matches": []})
    assert r["ok"] is False and r["error_type"] == "HandleNotFound"
