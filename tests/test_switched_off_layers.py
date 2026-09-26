"""Switched-off optional content: which words the page does not draw.

``trace_text`` reports ``layer_off`` for a word the page leaves undrawn as the
document opens because of optional content, and ``remove_text`` reaches it.
MuPDF's ``add_ocg`` always writes a ``/Usage`` dictionary and an ``/Order``
entry, but a PDF needs neither, and a group without them is invisible to
``get_ocgs()``'s "on" and to the viewer layer list. So every fixture here is
written object by object, with only the keys each case names.

All text is invented. Every expected value is stated from how the fixture
is built: which group marks the word, and which array of the default
configuration lists that group.
"""
from __future__ import annotations

import base64
import re

import pymupdf
import pytest

from lexcloak_pdf_tool import trace as T
from lexcloak_pdf_tool.__main__ import _handle
from lexcloak_pdf_tool.redact import apply_redactions
from lexcloak_pdf_tool.remove_text import (remove_text_doc, render_removed_doc,
                                           validate_remove_text)
from lexcloak_pdf_tool.trace import LayerReadError, trace_text

HIDDEN = "Quillfeather"
SHOWN = "Ledgerline"
CONTROL = "Visible"


class Built:
    """One US-letter page written as raw objects, Helvetica as ``/F1``."""

    def __init__(self, rotate: int = 0):
        self.doc = pymupdf.open()
        self.page = self.doc.new_page(width=612, height=792)
        if rotate:
            self.page.set_rotation(rotate)
        self.font = self._obj("<</Type/Font/Subtype/Type1/BaseFont/Helvetica"
                              "/Encoding/WinAnsiEncoding>>")
        self.groups: list[int] = []
        self.props: dict[str, int] = {}
        self.forms: dict[str, int] = {}

    def _obj(self, source: str) -> int:
        xref = self.doc.get_new_xref()
        self.doc.update_object(xref, source)
        return xref

    def group(self, name: str, extra: str = "") -> int:
        xref = self._obj(f"<</Type/OCG/Name({name}){extra}>>")
        self.groups.append(xref)
        return xref

    def ocmd(self, body: str) -> int:
        return self._obj(f"<</Type/OCMD{body}>>")

    def prop(self, key: str, xref: int) -> None:
        self.props[key] = xref

    def form(self, key: str, content: str, oc: int | None = None) -> None:
        extra = f"/OC {oc} 0 R" if oc else ""
        xref = self._obj(f"<</Type/XObject/Subtype/Form/BBox[-10 -10 300 60]"
                         f"{extra}{self._resources()}>>")
        self.doc.update_stream(xref, content.encode("latin-1"))
        self.forms[key] = xref

    def config(self, ocprops: str) -> None:
        """Write ``/OCProperties`` exactly as given; ``{G0}`` etc. name groups."""
        refs = {f"G{i}": f"{x} 0 R" for i, x in enumerate(self.groups)}
        refs["ALL"] = " ".join(f"{x} 0 R" for x in self.groups)
        self.doc.xref_set_key(self.doc.pdf_catalog(), "OCProperties",
                              ocprops.format(**refs))

    def _resources(self) -> str:
        return "/Resources" + self._resource_dict()

    def _resource_dict(self) -> str:
        props = "".join(f"/{k} {x} 0 R" for k, x in self.props.items())
        forms = "".join(f"/{k} {x} 0 R" for k, x in self.forms.items())
        return (f"<</Font<</F1 {self.font} 0 R>>"
                + (f"/Properties<<{props}>>" if props else "")
                + (f"/XObject<<{forms}>>" if forms else "") + ">>")

    def finish(self, content: str) -> bytes:
        self.doc.xref_set_key(self.page.xref, "Resources", self._resource_dict())
        xref = self._obj("<<>>")
        self.doc.update_stream(xref, (content + text(60, 700, CONTROL)).encode("latin-1"))
        self.doc.xref_set_key(self.page.xref, "Contents", f"{xref} 0 R")
        data = self.doc.tobytes()
        self.doc.close()
        return data


def text(x: float, y: float, s: str, size: float = 12) -> str:
    return f"BT /F1 {size} Tf {x} {y} Td ({s}) Tj ET\n"


def marked(key: str, body: str) -> str:
    return f"/OC /{key} BDC {body}EMC\n"


def _one_group(ocprops: str, extra: str = "", name: str = "Notes") -> bytes:
    """``HIDDEN`` marked by one group, configured by ``ocprops``."""
    b = Built()
    b.prop("MC0", b.group(name, extra))
    b.config(ocprops)
    return b.finish(marked("MC0", text(60, 400, HIDDEN)))


def _words(result: dict) -> dict[str, dict]:
    return {w["text"]: w for w in result["words"]}


def _record(word: dict) -> tuple:
    return (word["text"], word["layer"], word["layer_off"], word["mode"],
            word["opacity"], word["clipped"], word["covered_by"])


def _gray_at(pdf: bytes, bbox) -> int:
    """Darkest grey value of the page's default render inside ``bbox``."""
    doc = pymupdf.open(stream=pdf)
    try:
        pix = doc[0].get_pixmap(dpi=144, colorspace=pymupdf.csGRAY,
                                clip=pymupdf.Rect(bbox))
        return min(pix.samples) if pix.samples else 255
    finally:
        doc.close()


def _in_streams(pdf: bytes, needle: str) -> bool:
    doc = pymupdf.open(stream=pdf)
    try:
        want = needle.encode("latin-1")
        return any(want in (doc.xref_stream(x) or b"")
                   for x in range(1, doc.xref_length()) if doc.xref_is_stream(x))
    finally:
        doc.close()


# ── The groups a group needs neither /Usage nor /Order to be off in ──────

USAGE = "/Usage<</CreatorInfo<</Creator(Invented)/Subtype/Artwork>>>>"

#: (id, ocprops, group extra) — the word is hidden by each.
OFF_CONFIGS = [
    ("off-neither-key", "<</OCGs[{G0}]/D<</OFF[{G0}]>>>>", ""),
    ("off-usage-only", "<</OCGs[{G0}]/D<</OFF[{G0}]>>>>", USAGE),
    ("off-order-only", "<</OCGs[{G0}]/D<</OFF[{G0}]/Order[{G0}]>>>>", ""),
    ("off-both-keys", "<</OCGs[{G0}]/D<</OFF[{G0}]/Order[{G0}]>>>>", USAGE),
    ("base-off-empty-on", "<</OCGs[{G0}]/D<</BaseState/OFF/ON[]>>>>", ""),
    ("locked", "<</OCGs[{G0}]/D<</OFF[{G0}]/Order[{G0}]/Locked[{G0}]>>>>", ""),
    ("auto-state-view",
     "<</OCGs[{G0}]/D<</ON[{G0}]/AS[<</Event/View/Category[/View]/OCGs[{G0}]>>]>>>>",
     "/Usage<</View<</ViewState/OFF>>>>"),
    ("listed-twice", "<</OCGs[{G0} {G0}]/D<</OFF[{G0} {G0}]>>>>", ""),
    ("empty-usage", "<</OCGs[{G0}]/D<</OFF[{G0}]>>>>", "/Usage<<>>"),
    ("empty-order", "<</OCGs[{G0}]/D<</OFF[{G0}]/Order[]>>>>", ""),
]


@pytest.mark.parametrize("ocprops,extra", [c[1:] for c in OFF_CONFIGS],
                         ids=[c[0] for c in OFF_CONFIGS])
def test_a_word_in_a_group_off_by_default_is_layer_off(ocprops, extra):
    words = _words(trace_text(_one_group(ocprops, extra), 0))
    assert _record(words[HIDDEN]) == (HIDDEN, "Notes", True, 0, 1.0, False, None)
    assert _record(words[CONTROL]) == (CONTROL, "", False, 0, 1.0, False, None)


#: (id, ocprops, group extra) — the word is drawn by each.
ON_CONFIGS = [
    ("on-by-default", "<</OCGs[{G0}]/D<<>>>>", ""),
    ("listed-on-under-base-off", "<</OCGs[{G0}]/D<</BaseState/OFF/ON[{G0}]>>>>", ""),
    ("view-state-on", "<</OCGs[{G0}]/D<<>>>>", "/Usage<</View<</ViewState/ON>>>>"),
    ("off-array-names-a-non-group", "<</OCGs[{G0}]/D<</OFF[<</Kind/Unrelated>>]>>>>", ""),
    # A group whose only intent is /Design under the default /View intent:
    # MuPDF draws it, so the page shows it and it is not off.
    ("intent-design-only", "<</OCGs[{G0}]/D<<>>>>", "/Intent/Design"),
]


@pytest.mark.parametrize("ocprops,extra", [c[1:] for c in ON_CONFIGS],
                         ids=[c[0] for c in ON_CONFIGS])
def test_a_word_in_a_group_on_by_default_is_not_layer_off(ocprops, extra):
    words = _words(trace_text(_one_group(ocprops, extra), 0))
    assert _record(words[HIDDEN]) == (HIDDEN, "Notes", False, 0, 1.0, False, None)


def test_an_off_outer_group_hides_a_word_in_an_on_inner_group():
    b = Built()
    outer, inner = b.group("Outer"), b.group("Inner")
    b.prop("MC0", outer)
    b.prop("MC1", inner)
    b.config("<</OCGs[{ALL}]/D<</OFF[{G0}]>>>>")
    pdf = b.finish(f"/OC /MC0 BDC /OC /MC1 BDC {text(60, 400, HIDDEN)}EMC EMC\n")
    # The innermost group names the word; the outer one hides it.
    assert _record(_words(trace_text(pdf, 0))[HIDDEN]) == \
        (HIDDEN, "Inner", True, 0, 1.0, False, None)


def test_a_form_whose_own_oc_is_off_hides_the_forms_it_draws():
    b = Built()
    off = b.group("Form layer")
    b.form("Inner", text(0, 0, HIDDEN))
    b.form("Outer", "q 1 0 0 1 20 10 cm /Inner Do Q\n", oc=off)
    b.config("<</OCGs[{G0}]/D<</OFF[{G0}]>>>>")
    pdf = b.finish("q 1 0 0 1 100 400 cm /Outer Do Q\n")
    assert _record(_words(trace_text(pdf, 0))[HIDDEN]) == \
        (HIDDEN, "Form layer", True, 0, 1.0, False, None)


def test_marked_content_around_a_form_hides_it():
    b = Built()
    b.prop("MC0", b.group("Margin"))
    b.form("Fm0", text(0, 0, HIDDEN))
    b.config("<</OCGs[{G0}]/D<</OFF[{G0}]>>>>")
    pdf = b.finish(marked("MC0", "q 1 0 0 1 100 400 cm /Fm0 Do Q "))
    assert _record(_words(trace_text(pdf, 0))[HIDDEN]) == \
        (HIDDEN, "Margin", True, 0, 1.0, False, None)


def test_a_two_deep_form_with_oc_on_both_levels():
    """Outer form's group on, inner form's group off: hidden."""
    b = Built()
    on, off = b.group("Outer on"), b.group("Inner off")
    b.form("Inner", text(0, 0, HIDDEN), oc=off)
    b.form("Outer", "q 1 0 0 1 20 10 cm /Inner Do Q\n", oc=on)
    b.config("<</OCGs[{ALL}]/D<</OFF[{G1}]>>>>")
    pdf = b.finish("q 1 0 0 1 100 400 cm /Outer Do Q\n")
    assert _record(_words(trace_text(pdf, 0))[HIDDEN]) == \
        (HIDDEN, "Inner off", True, 0, 1.0, False, None)


def test_two_groups_sharing_a_name_are_told_apart():
    """The word in the ON group named Notes is never called off because the
    other group named Notes is."""
    b = Built()
    on, off = b.group("Notes"), b.group("Notes")
    b.prop("MC0", on)
    b.prop("MC1", off)
    b.config("<</OCGs[{ALL}]/D<</OFF[{G1}]>>>>")
    pdf = b.finish(marked("MC0", text(60, 500, SHOWN)) + marked("MC1", text(60, 400, HIDDEN)))
    words = _words(trace_text(pdf, 0))
    assert _record(words[SHOWN]) == (SHOWN, "Notes", False, 0, 1.0, False, None)
    assert _record(words[HIDDEN]) == (HIDDEN, "Notes", True, 0, 1.0, False, None)


def test_the_same_word_drawn_twice_in_one_place_is_off_only_once():
    """One copy in an on group, one in an off group, both named Notes, at the
    same origin: identical characters, so the page draws exactly one of them
    and exactly one is off."""
    b = Built()
    on, off = b.group("Notes"), b.group("Notes")
    b.prop("MC0", on)
    b.prop("MC1", off)
    b.config("<</OCGs[{ALL}]/D<</OFF[{G1}]>>>>")
    pdf = b.finish(marked("MC0", text(60, 400, HIDDEN)) + marked("MC1", text(60, 400, HIDDEN)))
    copies = [w for w in trace_text(pdf, 0)["words"] if w["text"] == HIDDEN]
    assert [(w["layer"], w["layer_off"]) for w in copies] == [("Notes", False), ("Notes", True)]


def test_the_off_copy_drawn_first_under_another_name_is_the_one_off():
    """The off group's copy comes first in the content and has its own name;
    the match is by name too, so it cannot take the on copy's place."""
    b = Built()
    off, on = b.group("Draft"), b.group("Final")
    b.prop("MC0", off)
    b.prop("MC1", on)
    b.config("<</OCGs[{ALL}]/D<</OFF[{G0}]>>>>")
    pdf = b.finish(marked("MC0", text(60, 400, HIDDEN)) + marked("MC1", text(60, 400, HIDDEN)))
    copies = [w for w in trace_text(pdf, 0)["words"] if w["text"] == HIDDEN]
    assert [(w["layer"], w["layer_off"]) for w in copies] == [("Draft", True), ("Final", False)]


def test_fifty_groups_on_one_page_half_of_them_off():
    b = Built()
    body = []
    for i in range(50):
        b.prop(f"MC{i}", b.group(f"Group {i:02d}"))
        body.append(marked(f"MC{i}", text(40 + 110 * (i % 5), 600 - 28 * (i // 5),
                                          f"Word{i:02d}", size=8)))
    off = " ".join(f"{b.groups[i]} 0 R" for i in range(0, 50, 2))
    b.config("<</OCGs[{ALL}]/D<</OFF[" + off + "]>>>>")
    words = _words(trace_text(b.finish("".join(body)), 0))
    assert {w: (words[w]["layer"], words[w]["layer_off"]) for w in words if w != CONTROL} == {
        f"Word{i:02d}": (f"Group {i:02d}", i % 2 == 0) for i in range(50)}


def test_an_off_group_that_draws_no_text_reports_nothing_off():
    b = Built()
    b.prop("MC0", b.group("Stamp"))
    b.config("<</OCGs[{G0}]/D<</OFF[{G0}]>>>>")
    pdf = b.finish(marked("MC0", "q 0 g 60 380 240 40 re f Q ") + text(60, 300, SHOWN))
    words = _words(trace_text(pdf, 0))
    assert [_record(w) for w in words.values()] == [
        (SHOWN, "", False, 0, 1.0, False, None),
        (CONTROL, "", False, 0, 1.0, False, None)]


# ── Membership dictionaries: MuPDF's own evaluation, the render's ────────

#: (id, OCMD body, layer, layer_off): g1 = G0 is on, g2 = G1 is off. The
#: verdict is MuPDF's, which is how the page renders: it follows the PDF
#: specification for /AnyOn and /AllOff, reads /AllOn and /AnyOff the other
#: way round, and ignores a /VE expression (the content is drawn). The layer
#: is the last member group MuPDF opens; a /VE-only dictionary names none.
OCMDS = [
    ("any-on-g2", "/OCGs[{G1}]/P/AnyOn", "Two", True),
    ("any-on-g1-g2", "/OCGs[{G0} {G1}]/P/AnyOn", "Two", False),
    ("all-on-g1-g2", "/OCGs[{G0} {G1}]/P/AllOn", "Two", False),
    ("all-on-g1", "/OCGs[{G0}]/P/AllOn", "One", False),
    ("any-off-g1", "/OCGs[{G0}]/P/AnyOff", "One", True),
    ("any-off-g1-g2", "/OCGs[{G0} {G1}]/P/AnyOff", "Two", True),
    ("all-off-g1-g2", "/OCGs[{G0} {G1}]/P/AllOff", "Two", True),
    ("all-off-g2", "/OCGs[{G1}]/P/AllOff", "Two", False),
    ("ve-and", "/VE[/And {G0} {G1}]", "", False),
    ("ve-and-not", "/VE[/And {G0} [/Not {G1}]]", "", False),
]


def _ocmd_pdf(body: str) -> bytes:
    b = Built()
    b.group("One")
    b.group("Two")
    refs = {"G0": f"{b.groups[0]} 0 R", "G1": f"{b.groups[1]} 0 R"}
    b.prop("MC0", b.ocmd(body.format(**refs)))
    b.config("<</OCGs[{ALL}]/D<</OFF[{G1}]>>>>")
    return b.finish(marked("MC0", text(60, 400, HIDDEN)))


@pytest.mark.parametrize("body,layer,off", [o[1:] for o in OCMDS], ids=[o[0] for o in OCMDS])
def test_a_membership_dictionary_is_judged_as_the_page_renders(body, layer, off):
    pdf = _ocmd_pdf(body)
    word = _words(trace_text(pdf, 0))[HIDDEN]
    assert _record(word) == (HIDDEN, layer, off, 0, 1.0, False, None)
    # And that is what the page shows: blank paper where it is off, ink where not.
    assert (_gray_at(pdf, word["bbox"]) == 255) is off


# ── The render is the judge ──────────────────────────────────────────────


def _every_fixture() -> list[bytes]:
    out = [_one_group(c[1], c[2]) for c in OFF_CONFIGS + ON_CONFIGS]
    out += [_ocmd_pdf(o[1]) for o in OCMDS]
    return out


def test_layer_off_is_exactly_the_words_the_default_render_leaves_blank():
    """Over every fixture above: no drawn word is ever ``layer_off`` (that
    would delete visible text), and no undrawn word is missed."""
    for pdf in _every_fixture():
        for word in trace_text(pdf, 0)["words"]:
            assert (_gray_at(pdf, word["bbox"]) == 255) is word["layer_off"], word["text"]


# ── Rotation and page boxes carry over to the copy ───────────────────────


@pytest.mark.parametrize("rotate", [90, 180, 270])
def test_a_rotated_page_reports_the_same_boxes_as_without_layers(rotate):
    def build(ocprops: str | None) -> bytes:
        b = Built(rotate=rotate)
        b.prop("MC0", b.group("Notes"))
        if ocprops:
            b.config(ocprops)
        return b.finish(marked("MC0", text(60, 400, HIDDEN)))

    plain = _words(trace_text(build(None), 0))
    layered = _words(trace_text(build("<</OCGs[{G0}]/D<</OFF[{G0}]>>>>"), 0))
    assert layered[HIDDEN]["layer_off"] is True
    assert plain[HIDDEN]["layer_off"] is False
    for key in (HIDDEN, CONTROL):
        assert layered[key]["bbox"] == plain[key]["bbox"]


def test_a_page_whose_cropbox_is_offset_reports_the_same_boxes():
    def build(ocprops: str | None) -> bytes:
        b = Built()
        b.page.set_cropbox(pymupdf.Rect(40, 50, 560, 750))
        b.prop("MC0", b.group("Notes"))
        if ocprops:
            b.config(ocprops)
        return b.finish(marked("MC0", text(60, 400, HIDDEN)))

    plain = _words(trace_text(build(None), 0))
    layered = _words(trace_text(build("<</OCGs[{G0}]/D<</OFF[{G0}]>>>>"), 0))
    assert (layered[HIDDEN]["layer_off"], plain[HIDDEN]["layer_off"]) == (True, False)
    assert layered[HIDDEN]["bbox"] == plain[HIDDEN]["bbox"]


# ── Pages whose optional content cannot be read ──────────────────────────

BROKEN = [
    ("ocgs-not-an-array", "<</OCGs 7/D<<>>>>", "/OCProperties /OCGs is not an array"),
    ("no-default-config", "<</OCGs[{G0}]>>", "/OCProperties lacks a default configuration /D"),
    ("not-a-dictionary", "[{G0}]", "/OCProperties is not a dictionary"),
    ("ocgs-entry-not-a-group", "<</OCGs[5]/D<<>>>>", "an /OCGs entry is not a group dictionary"),
    ("off-not-an-array", "<</OCGs[{G0}]/D<</OFF 3>>>>", "/D /OFF is not an array"),
    ("base-state-not-a-name", "<</OCGs[{G0}]/D<</BaseState(OFF)>>>>", "/D /BaseState is not a name"),
    ("dangling-group", "<</OCGs[9999 0 R]/D<<>>>>", "an /OCGs entry is not a group dictionary"),
    ("dangling-ocgs", "<</OCGs 9999 0 R/D<<>>>>", "/OCProperties /OCGs is not an array"),
    # Listing no group excuses only a MISSING /D, never a malformed one.
    ("no-groups-d-not-a-dictionary", "<</D 3>>", "/OCProperties lacks a default configuration /D"),
    ("no-groups-off-not-an-array", "<</D<</OFF 3>>>>", "/D /OFF is not an array"),
    ("empty-ocgs-base-state-not-a-name", "<</OCGs[]/D<</BaseState 1>>>>",
     "/D /BaseState is not a name"),
]


@pytest.mark.parametrize("ocprops,message", [c[1:] for c in BROKEN], ids=[c[0] for c in BROKEN])
def test_unreadable_optional_content_fails_the_page(ocprops, message):
    pdf = _one_group(ocprops)
    with pytest.raises(LayerReadError) as caught:
        trace_text(pdf, 0)
    assert str(caught.value) == message


@pytest.mark.parametrize("ocprops,message", [c[1:] for c in BROKEN[:3]], ids=[c[0] for c in BROKEN[:3]])
def test_unreadable_optional_content_is_an_error_frame_with_no_document_text(ocprops, message):
    for op, extra in (("trace_text", {}), ("trace_text_h", None)):
        pdf = _one_group(ocprops)
        if extra is None:
            handle = _handle({"op": "open_doc", "pdf_b64": base64.b64encode(pdf).decode()})
            cmd = {"op": op, "handle": handle["result"]["handle"], "page": 0}
        else:
            cmd = {"op": op, "pdf_b64": base64.b64encode(pdf).decode(), "page": 0}
        out = _handle(cmd)
        assert out == {"ok": False, "error": message, "error_type": "LayerReadError"}


def test_a_failing_read_of_the_catalog_names_only_the_exception_type(monkeypatch):
    """A PyMuPDF call that fails mid-read (here with a message carrying page
    text) fails the page with the exception's type only."""
    pdf = _one_group("<</OCGs[{G0}]/D<</OFF[{G0}]>>>>")
    real = pymupdf.Document.xref_get_key

    def failing(self, xref, key):
        if key.startswith("OCProperties/"):
            raise pymupdf.mupdf.FzErrorFormat(f"cannot read {HIDDEN}")
        return real(self, xref, key)

    monkeypatch.setattr(pymupdf.Document, "xref_get_key", failing)
    with pytest.raises(LayerReadError) as caught:
        trace_text(pdf, 0)
    assert str(caught.value) == "optional content cannot be read (FzErrorFormat)"
    assert HIDDEN not in str(caught.value)


def test_a_copy_that_loses_default_drawn_text_fails_the_page(monkeypatch):
    """If the view with every layer drawn ever stopped matching the page, the
    comparison would call visible words hidden. It refuses instead."""
    from contextlib import contextmanager

    blank = pymupdf.open()
    blank.new_page(width=612, height=792)

    @contextmanager
    def lossy(doc, pno):
        yield blank[0]

    monkeypatch.setattr(T, "_every_layer_drawn", lossy)
    with pytest.raises(LayerReadError) as caught:
        trace_text(_one_group("<</OCGs[{G0}]/D<</OFF[{G0}]>>>>"), 0)
    assert str(caught.value) == "the page with every layer drawn lost text it draws by default"
    blank.close()


# ── Optional content that lists no group ────────────────────────────────

#: (id, ocprops) — ``/OCProperties`` naming no group. The first is the shape
#: that turned up in the wild: no ``/OCGs`` and an empty ``/D``.
NO_GROUPS = [
    ("no-ocgs-empty-d", "<</D<<>>>>"),
    ("empty-dictionary", "<<>>"),
    ("empty-ocgs-no-d", "<</OCGs[]>>"),
    ("empty-ocgs-empty-d", "<</OCGs[]/D<<>>>>"),
    ("no-ocgs-base-state-off", "<</D<</BaseState/OFF>>>>"),
    ("null-ocgs", "<</OCGs null/D<<>>>>"),
]


def _plain_page(ocprops: str | None) -> bytes:
    b = Built()
    if ocprops is not None:
        b.config(ocprops)
    return b.finish(text(60, 400, SHOWN))


@pytest.mark.parametrize("ocprops", [c[1] for c in NO_GROUPS], ids=[c[0] for c in NO_GROUPS])
def test_optional_content_that_lists_no_group_reads_as_the_page_without_it(ocprops):
    """Nothing for a default configuration to switch off, so nothing is
    refused: the trace is the one the same page gives with no
    ``/OCProperties`` at all."""
    assert trace_text(_plain_page(ocprops), 0) == trace_text(_plain_page(None), 0)


@pytest.mark.parametrize("ocprops", [c[1] for c in NO_GROUPS], ids=[c[0] for c in NO_GROUPS])
def test_optional_content_that_lists_no_group_is_still_compared(ocprops, monkeypatch):
    """Listing no group is not taken on trust: the page still goes through
    the comparison, so a view that lost default-drawn text would fail it."""
    from contextlib import contextmanager

    blank = pymupdf.open()
    blank.new_page(width=612, height=792)

    @contextmanager
    def lossy(doc, pno):
        yield blank[0]

    monkeypatch.setattr(T, "_every_layer_drawn", lossy)
    with pytest.raises(LayerReadError) as caught:
        trace_text(_plain_page(ocprops), 0)
    assert str(caught.value) == "the page with every layer drawn lost text it draws by default"
    blank.close()


#: (id, ocprops) — the word is marked by a group ``/OCGs`` does not list,
#: which the default configuration names as off or leaves to ``/BaseState
#: /OFF``. MuPDF draws such a group (so does PDFium), so the word is shown.
UNLISTED = [
    ("no-ocgs-off-names-it", "<</D<</OFF[{G0}]>>>>"),
    ("empty-ocgs-off-names-it", "<</OCGs[]/D<</OFF[{G0}]>>>>"),
    ("empty-ocgs-base-state-off", "<</OCGs[]/D<</BaseState/OFF>>>>"),
    ("empty-ocgs-no-d", "<</OCGs[]>>"),
    ("no-ocgs-empty-d", "<</D<<>>>>"),
]


@pytest.mark.parametrize("ocprops", [c[1] for c in UNLISTED], ids=[c[0] for c in UNLISTED])
def test_a_word_in_a_group_no_array_lists_is_as_drawn_as_the_render_shows(ocprops):
    pdf = _one_group(ocprops)
    words = _words(trace_text(pdf, 0))
    assert words[HIDDEN]["layer_off"] is False
    assert _gray_at(pdf, words[HIDDEN]["bbox"]) < 128      # the render draws it
    assert words[CONTROL]["layer_off"] is False


def _indirect(b: Built, source: str) -> str:
    """``source`` written as an object of its own, as a reference to it."""
    return f"{b._obj(source)} 0 R"


def test_arrays_and_the_configuration_may_be_written_as_references():
    """Any value in a PDF may be a reference to an object. An /OCGs, a /D or
    an /OFF so written is read like the same value written in place."""
    def build(indirect: str) -> bytes:
        b = Built()
        g = b.group("Notes")
        b.prop("MC0", g)
        ocgs, off = f"[{g} 0 R]", f"[{g} 0 R]"
        if indirect in ("ocgs", "all"):
            ocgs = _indirect(b, ocgs)
        if indirect in ("off", "all"):
            off = _indirect(b, off)
        d = f"<</OFF {off}>>"
        if indirect in ("d", "all"):
            d = _indirect(b, d)
        b.config(f"<</OCGs {ocgs}/D {d}>>")
        return b.finish(marked("MC0", text(60, 400, HIDDEN)))

    direct = trace_text(build(""), 0)
    assert _words(direct)[HIDDEN]["layer_off"] is True
    for indirect in ("ocgs", "d", "off", "all"):
        assert trace_text(build(indirect), 0) == direct, indirect


def test_a_base_state_written_as_a_reference_is_read_as_the_name():
    def build(base: str | None) -> bytes:
        b = Built()
        g = b.group("Notes")
        b.prop("MC0", g)
        state = "/OFF" if base is None else _indirect(b, base)
        b.config(f"<</OCGs[{g} 0 R]/D<</BaseState {state}/ON[]>>>>")
        return b.finish(marked("MC0", text(60, 400, HIDDEN)))

    direct = trace_text(build(None), 0)
    assert _words(direct)[HIDDEN]["layer_off"] is True
    assert trace_text(build("/OFF"), 0) == direct
    with pytest.raises(LayerReadError) as caught:
        trace_text(build("(OFF)"), 0)
    assert str(caught.value) == "/D /BaseState is not a name"


def test_a_reference_to_an_array_that_is_not_one_still_fails_the_page():
    b = Built()
    g = b.group("Notes")
    b.prop("MC0", g)
    b.config(f"<</OCGs[{g} 0 R]/D<</OFF {_indirect(b, '7')}>>>>")
    pdf = b.finish(marked("MC0", text(60, 400, HIDDEN)))
    with pytest.raises(LayerReadError) as caught:
        trace_text(pdf, 0)
    assert str(caught.value) == "/D /OFF is not an array"


def test_a_page_out_of_range_is_an_index_error_before_layers_are_read():
    with pytest.raises(IndexError, match="page_num 3 out of range for 1-page document"):
        trace_text(_one_group("<</OCGs 7/D<<>>>>"), 3)


def test_an_encrypted_document_is_refused_without_its_text():
    doc = pymupdf.open(stream=_one_group("<</OCGs[{G0}]/D<</OFF[{G0}]>>>>"))
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()
    out = _handle({"op": "trace_text", "pdf_b64": base64.b64encode(data).decode(), "page": 0})
    assert out["ok"] is False
    assert HIDDEN not in out["error"]


# ── Nothing is changed by reading ────────────────────────────────────────


def _state(doc) -> tuple:
    # ``no_new_id``: a save otherwise writes a fresh random second /ID.
    return (doc.tobytes(no_new_id=True), doc.xref_get_key(doc.pdf_catalog(), "OCProperties"),
            doc.get_ocgs(), doc.layer_ui_configs(), doc.is_dirty)


@pytest.mark.parametrize("ocprops", [
    "<</OCGs[{G0}]/D<</OFF[{G0}]>>>>",
    "<</OCGs[{G0}]/D<</OFF[{G0}]/Order[{G0}]>>>>",
])
def test_a_trace_changes_nothing_and_repeats_exactly(ocprops):
    pdf = _one_group(ocprops, USAGE)
    doc = pymupdf.open(stream=pdf)
    try:
        before = _state(doc)
        first = T._trace_text_doc(doc, 0)
        assert _state(doc) == before
        assert T._trace_text_doc(doc, 0) == first
        assert _state(doc) == before
    finally:
        doc.close()
    assert trace_text(pdf, 0) == first


def test_the_handle_path_changes_nothing_and_repeats_exactly():
    pdf = _one_group("<</OCGs[{G0}]/D<</OFF[{G0}]>>>>")
    opened = _handle({"op": "open_doc", "pdf_b64": base64.b64encode(pdf).decode()})
    handle = opened["result"]["handle"]
    render = {"op": "render_h", "handle": handle, "page": 0, "dpi": 72}
    before = _handle(render)
    first = _handle({"op": "trace_text_h", "handle": handle, "page": 0})
    again = _handle({"op": "trace_text_h", "handle": handle, "page": 0})
    assert first["ok"] and first == again
    assert _handle(render) == before       # the handle still renders the layer off
    _handle({"op": "close_doc", "handle": handle})


# ── remove_text reaches every word trace_text reports ────────────────────


def _items(pdf: bytes, want: str) -> list[dict]:
    return [{"page": 0, "box": w["bbox"], "text": w["text"], "mode": w["mode"],
             "opacity": w["opacity"], "layer": w["layer"]}
            for w in trace_text(pdf, 0)["words"] if w["text"] == want]


@pytest.mark.parametrize("ocprops,extra", [c[1:] for c in OFF_CONFIGS],
                         ids=[c[0] for c in OFF_CONFIGS])
def test_remove_text_removes_a_word_in_any_off_group(ocprops, extra):
    pdf = _one_group(ocprops, extra)
    items = _items(pdf, HIDDEN)
    assert len(items) == 1
    doc = pymupdf.open(stream=pdf)
    config = doc.xref_get_key(doc.pdf_catalog(), "OCProperties")
    assert remove_text_doc(doc, validate_remove_text(items)) == {"removed": [[0, 0]], "kept": []}
    assert doc.xref_get_key(doc.pdf_catalog(), "OCProperties") == config
    # Collect garbage as a delivered file is saved: the redaction writes a new
    # content stream and leaves the old one unreferenced until then.
    out = doc.tobytes(garbage=3)
    doc.close()
    assert not _in_streams(out, HIDDEN)
    assert _in_streams(out, CONTROL)


def test_remove_text_spares_the_on_word_that_shares_a_name():
    b = Built()
    on, off = b.group("Notes"), b.group("Notes")
    b.prop("MC0", on)
    b.prop("MC1", off)
    b.config("<</OCGs[{ALL}]/D<</OFF[{G1}]>>>>")
    pdf = b.finish(marked("MC0", text(60, 500, SHOWN)) + marked("MC1", text(60, 400, HIDDEN)))
    doc = pymupdf.open(stream=pdf)
    assert remove_text_doc(doc, validate_remove_text(_items(pdf, HIDDEN))) == \
        {"removed": [[0, 0]], "kept": []}
    out = doc.tobytes(garbage=3)
    doc.close()
    assert (_in_streams(out, HIDDEN), _in_streams(out, SHOWN)) == (False, True)


@pytest.mark.parametrize("ocprops,extra", [c[1:] for c in OFF_CONFIGS],
                         ids=[c[0] for c in OFF_CONFIGS])
def test_render_removed_draws_the_page_as_it_opens(ocprops, extra):
    """The render a caller compares with its own: the hidden word stays blank."""
    pdf = _one_group(ocprops, extra)
    word = _words(trace_text(pdf, 0))[HIDDEN]
    doc = pymupdf.open(stream=pdf)
    png = render_removed_doc(doc, 0, [], 144)["png"]
    doc.close()
    pix = pymupdf.Pixmap(png)
    gray = pymupdf.Pixmap(pymupdf.csGRAY, pix) if pix.n > 1 else pix
    x0, y0, x1, y1 = (int(v * 2) for v in word["bbox"])
    assert min(gray.pixel(x, y)[0] for x in range(x0, x1) for y in range(y0, y1)) == 255


def test_the_render_copy_is_the_whole_document_when_optional_content_lists_no_group():
    """Whether to copy the whole document is read from /OCProperties itself,
    so an empty /OCGs, which ``get_ocgs()`` reports as no layers at all,
    still keeps the document's configuration in the render copy."""
    from lexcloak_pdf_tool.remove_text import _copy_for_render

    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    doc.xref_set_key(doc.pdf_catalog(), "OCProperties", "<</OCGs[]/D<<>>>>")
    assert doc.get_ocgs() == {}
    copy = _copy_for_render(doc, 1)
    assert (len(copy), copy.xref_get_key(copy.pdf_catalog(), "OCProperties")[0]) == (3, "dict")
    copy.close()
    doc.xref_set_key(doc.pdf_catalog(), "OCProperties", "null")
    copy = _copy_for_render(doc, 1)
    assert len(copy) == 1
    copy.close()
    doc.close()


# ── The burn: keep-uncovered-lines never fails on layers ─────────────────


def _box(x0, y0, x1, y1) -> dict:
    return {"page": 0, "type": "Name", "text": CONTROL,
            "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}


@pytest.mark.parametrize("ocprops", [c[1] for c in BROKEN[:3]] + [OFF_CONFIGS[0][1]],
                         ids=[c[0] for c in BROKEN[:3]] + ["off-neither-key"])
@pytest.mark.parametrize("keep", [True, False], ids=["keep-lines", "old-burn"])
def test_a_burn_over_a_layered_page_never_fails(ocprops, keep):
    """A box over the control word with the hidden word on the next line
    down: the burn runs, whatever the page's optional content looks like."""
    b = Built()
    b.prop("MC0", b.group("Notes"))
    b.config(ocprops)
    pdf = b.finish(marked("MC0", text(60, 688, HIDDEN)))
    out = apply_redactions(pdf, [_box(58, 80, 110, 94)], keep_uncovered_lines=keep)[0]
    assert not _in_streams(out, CONTROL)


def test_keep_lines_treats_a_word_in_an_off_group_as_the_old_rule_does():
    """The hidden word's line sits just below the box; the old text removal
    takes its glyphs, so keep-uncovered-lines must take them too."""
    b = Built()
    b.prop("MC0", b.group("Notes"))
    b.config("<</OCGs[{G0}]/D<</OFF[{G0}]>>>>")
    pdf = b.finish(marked("MC0", text(60, 690, HIDDEN, size=11)))
    box = _box(58, 80, 110, 96)
    kept = apply_redactions(pdf, [box], keep_uncovered_lines=True)[0]
    old = apply_redactions(pdf, [box], keep_uncovered_lines=False)[0]
    assert _in_streams(kept, HIDDEN) == _in_streams(old, HIDDEN)


# ── The copy carries no /OCProperties ────────────────────────────────────


def test_the_view_with_every_layer_drawn_has_no_optional_content_properties():
    doc = pymupdf.open(stream=_one_group("<</OCGs[{G0}]/D<</OFF[{G0}]>>>>"))
    with T._every_layer_drawn(doc, 0) as page:
        assert page.parent is not doc
        assert page.parent.xref_get_key(page.parent.pdf_catalog(), "OCProperties") == ("null", "null")
        assert HIDDEN in "".join(chr(c[0]) for s in page.get_texttrace() for c in s["chars"])
    doc.close()


def test_the_view_drops_optional_content_properties_even_if_the_copy_carries_them(monkeypatch):
    """Should a PyMuPDF ``insert_pdf`` ever copy /OCProperties, the view would
    hide the switched-off layer again and its words would silently vanish.
    The view removes the key itself."""
    real = pymupdf.Document.insert_pdf

    def copying(self, src, *a, **kw):
        out = real(self, src, *a, **kw)
        # Switch the copy's own group off, as a copied /OCProperties would.
        ref = self.xref_get_key(self[0].xref, "Resources/Properties/MC0")[1]
        group = re.match(r"(\d+) 0 R", ref).group(1)
        self.xref_set_key(self.pdf_catalog(), "OCProperties",
                          f"<</OCGs[{group} 0 R]/D<</OFF[{group} 0 R]>>>>")
        return out

    monkeypatch.setattr(pymupdf.Document, "insert_pdf", copying)
    pdf = _one_group("<</OCGs[{G0}]/D<</OFF[{G0}]>>>>")
    assert _record(_words(trace_text(pdf, 0))[HIDDEN]) == \
        (HIDDEN, "Notes", True, 0, 1.0, False, None)


def test_a_document_without_optional_content_is_read_directly():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), SHOWN)
    with T._every_layer_drawn(doc, 0) as page:
        assert page.parent is doc
    doc.close()
