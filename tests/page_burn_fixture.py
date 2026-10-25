"""An invented document carrying everything a page-at-a-time burn could trip
on, and the payload a caller would send to burn it.

Every name, number, date and sentence is made up. ``build(n)`` returns
``(pdf_bytes, payload, hazards)`` for a document of ``n`` pages (at least 20):

* a logo image and a letterhead form XObject drawn by every page; the
  letterhead holds a phone number boxed on every page, and a box reaches the
  logo on every tenth page;
* scanned pages (three in five): a JPEG page image, an invisible text layer
  (``3 Tr``) drawn before the letterhead's own ``0 Tr``, and one resource
  dictionary shared by every scanned page, naming every scan;
* bookmarks, links (explicit, named and URI), named destinations, page
  labels, and a tagged structure tree with ``/Pg``, marked content and a
  ``/ParentTree``; one structure element whose marked content runs over a
  page break, so two pages' burns edit it; one bookmark, link address,
  destination name, page-label prefix and ``/Alt`` hold a value;
* a code image, a rotated page, a dense page, a form field, a sticky note
  and a FreeText, white-on-white text, an image outside the crop box, an
  inline image below it, an image only on a switched-off layer, an
  ``/ActualText`` span repeating a value, and a removed page and a
  blacked-out page that bookmarks, links, destinations and the structure tree
  all point at.

The payload is ``apply_redactions``' arguments: one match per word that
holds a value (a few switched off, a few with their own label), boxes for
what no word reaches, side-text edits and an image plan read from the
package's own listings, and the hidden words' text removal.
"""
from __future__ import annotations

import random

import pymupdf

from lexcloak_pdf_tool.side_text import list_side_text
from lexcloak_pdf_tool.trace import _trace_text_doc
from lexcloak_pdf_tool.unseen import list_unseen_images

W, H = 612.0, 792.0
NAME = "Quellbrook, Ada V"
FIRST, LAST = "Ada", "Quellbrook"
ACCOUNT = "5501228740"
PHONE = "555-010-4477"
LABEL = "REDACTED"
PARTY = "Party A"
CATEGORIES = ["Person", "Account", "Date", "Phone", "Number", "Code"]
PEOPLE = {"Ada", "Quellbrook", "Tobin", "Varre", "Mireille", "Ostrand",
          "Cass", "Dunmore", "Ilse", "Moravec"}

LINES = [
    "Opening balance carried from the prior statement period.",
    "Transfer received from Tobin Varre, reference 77120-0458.",
    "Card ending 4411 used at a grocery on 02/03/2026.",
    "Standing order 3301 paid to Mireille Ostrand on 02/05/2026.",
    "Interest credited at the variable rate for the period.",
    "Branch visit with Cass Dunmore, ticket 66012, on 02/09/2026.",
    "Cheque 000418 presented and cleared on 02/11/2026.",
    "Fee waived after review by Ilse Moravec on 02/12/2026.",
    "Direct debit 5520-3317 collected for the utility account.",
    "Correspondence sent to the address held on file.",
    "Overdraft limit of 1200 unchanged this period.",
    "Mobile alert sent to the number ending 4477.",
]

HAZARDS = ("removed", "blackout", "widget", "annots", "hidden_white", "crop_image",
           "layer_image", "rotated", "dense", "code", "actualtext", "alt_value", "span")


def hazard_pages(n: int) -> dict[str, int]:
    """Where each single-page hazard sits in a document of ``n`` pages: spread
    over it, never the first page, and never on one page twice."""
    if n < 20:
        raise ValueError("the hazard document needs at least 20 pages")
    step = (n - 3) / len(HAZARDS)
    out = {name: 2 + int(k * step) for k, name in enumerate(HAZARDS)}
    assert len(set(out.values())) == len(out)
    return out


def is_scanned(i: int, hz: dict[str, int]) -> bool:
    """Scanned pages: three in five, never a page with a hazard of its own
    (those draw their own images, which belong in their own resources), and
    never the two pages the spanning structure element joins."""
    return (i != 0 and i not in hz.values() and i - 1 != hz["span"]
            and i % 5 in (2, 3, 4))


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text(font: str, size: float, x: float, y_top: float, s: str, *, tr: int = 0,
          rgb: str | None = None, mcid: int | None = None,
          actual: str | None = None) -> bytes:
    y = H - y_top - size
    body = (f"BT {tr} Tr {rgb + ' rg ' if rgb else ''}/{font} {size:g} Tf "
            f"1 0 0 1 {x:.2f} {y:.2f} Tm ({_esc(s)}) Tj ET")
    if mcid is not None:
        props = f"/MCID {mcid}"
        if actual is not None:
            props += f" /ActualText ({_esc(actual)})"
        body = (f"/Span << {props} >> BDC {body} EMC" if actual is not None
                else f"/P << {props} >> BDC {body} EMC")
    return (body + "\n").encode("latin-1")


def _page_lines(i: int, n: int) -> tuple[list[str], list[str], str]:
    header = [f"{NAME}    Account {ACCOUNT}    Opened 06/17/2019",
              f"Statement {4100 + i}  Period 02/01/2026 to 02/28/2026"]
    rng = random.Random(i)
    body = [LINES[(i + j) % len(LINES)] for j in range(18) if rng.random() < 0.8]
    footer = f"Printed 03/02/2026 09:15   Page {i + 1} of {n}"
    return header, body, footer


def _content(i: int, n: int, *, tr: int, tagged: bool, actual_value: bool,
             mcid0: int = 0) -> tuple[bytes, int]:
    header, body, footer = _page_lines(i, n)
    out, mcid, y = [], mcid0, 70.0

    def line(font, size, x, s, **kw):
        nonlocal mcid, y
        out.append(_text(font, size, x, y, s, tr=tr,
                         mcid=mcid if tagged else None, **kw))
        mcid += 1 if tagged else 0

    for s in header:
        line("Helv", 9, 40, s)
        y += 13
    y += 10
    for j, s in enumerate(body):
        line("Cour", 9.5, 40, s,
             actual=(f"{FIRST} {LAST} {ACCOUNT}" if actual_value and j == 3 else None))
        y += 16.5
    y = 760
    line("Helv", 8, 40, footer)
    return b"".join(out), mcid


def _dense_content(boxes: int) -> bytes:
    """Lines of numbers, ``boxes`` words in all, each a value to box."""
    rng = random.Random(4242)
    out, y, left = [], 40.0, boxes
    while left > 0:
        k = min(9, left)
        out.append(_text("Cour", 5.5, 30, y,
                         " ".join(str(rng.randint(100000, 999999)) for _ in range(k))))
        left -= k
        y += 6.2
    return b"".join(out)


def _pixmap(rgb, size=(120, 40)) -> pymupdf.Pixmap:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, *size), False)
    pix.set_rect(pix.irect, rgb)
    return pix


def _png(rgb, size=(120, 40)) -> bytes:
    return _pixmap(rgb, size).tobytes("png")


def _logo_png() -> bytes:
    pix = _pixmap((24, 70, 140), (160, 54))
    pix.set_rect(pymupdf.IRect(8, 8, 46, 46), (240, 200, 60))
    pix.set_rect(pymupdf.IRect(60, 16, 150, 38), (255, 255, 255))
    return pix.tobytes("png")


def _code_png() -> bytes:
    """A square of black and white cells, standing in for a printed code."""
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 84, 84), False)
    pix.set_rect(pix.irect, (255,))
    rng = random.Random(11)
    for r in range(21):
        for c in range(21):
            if rng.random() < 0.5:
                pix.set_rect(pymupdf.IRect(c * 4, r * 4, c * 4 + 4, r * 4 + 4), (0,))
    return pix.tobytes("png")


def _scan_jpeg(content: bytes) -> bytes:
    """The page as a scanner would see it: the same text, visible, as a picture."""
    tmp = pymupdf.open()
    page = tmp.new_page(width=W, height=H)
    fx = {}
    for name, base in (("Helv", "Helvetica"), ("Cour", "Courier")):
        x = tmp.get_new_xref()
        tmp.update_object(x, f"<< /Type /Font /Subtype /Type1 /BaseFont /{base} "
                             "/Encoding /WinAnsiEncoding >>")
        fx[name] = x
    res = tmp.get_new_xref()
    tmp.update_object(res, "<< /Font << /Helv %d 0 R /Cour %d 0 R >> >>"
                      % (fx["Helv"], fx["Cour"]))
    cx = tmp.get_new_xref()
    tmp.update_object(cx, "<<>>")
    tmp.update_stream(cx, content.replace(b" 3 Tr ", b" 0 Tr "))
    tmp.xref_set_key(page.xref, "Resources", f"{res} 0 R")
    tmp.xref_set_key(page.xref, "Contents", f"{cx} 0 R")
    data = page.get_pixmap(dpi=60, colorspace=pymupdf.csGRAY).tobytes(
        "jpeg", jpg_quality=55)
    tmp.close()
    return data


def build(n: int = 30, *, dense_boxes: int = 300) -> tuple[bytes, dict, dict]:
    hz = hazard_pages(n)
    doc = pymupdf.open()
    fonts = {}
    for name, base in (("Helv", "Helvetica"), ("Cour", "Courier")):
        x = doc.get_new_xref()
        doc.update_object(x, f"<< /Type /Font /Subtype /Type1 /BaseFont /{base} "
                             "/Encoding /WinAnsiEncoding >>")
        fonts[name] = x
    # The letterhead: one form every page draws, holding a value.
    lh = doc.get_new_xref()
    doc.update_object(lh, "<< /Type /XObject /Subtype /Form /BBox [0 0 400 30] "
                          "/Resources << /Font << /Helv %d 0 R >> >> >>" % fonts["Helv"])
    doc.update_stream(lh, (
        "BT 0 Tr 0 g /Helv 8 Tf 1 0 0 1 0 18 Tm (Harrowgate Mutual  "
        "18 Linden Row, Fairhaven) Tj ET\n"
        f"BT 0 Tr 0 g /Helv 8 Tf 1 0 0 1 0 6 Tm (Help line {PHONE}  "
        "Fax 555-010-4478) Tj ET\n").encode("latin-1"))

    shared_res = doc.get_new_xref()          # filled once every scan is known
    scan_names: dict[str, int] = {}
    logo_xref = 0
    tagged_pages: dict[int, int] = {}        # page -> marked-content count
    for i in range(n):
        page = doc.new_page(width=W, height=H)
        if i == 0:
            logo_xref = page.insert_image(pymupdf.Rect(40, 18, 130, 48),
                                          stream=_logo_png())
        if i == hz["dense"]:
            content, ncid = _dense_content(dense_boxes), 0
        else:
            tagged = not is_scanned(i, hz)
            content, ncid = _content(i, n, tr=3 if is_scanned(i, hz) else 0,
                                     tagged=tagged, actual_value=i == hz["actualtext"])
            if tagged:
                tagged_pages[i] = ncid
        if i == hz["hidden_white"]:
            content += _text("Helv", 9, 300, 40, "PIN 8841 on file", rgb="1 1 1")
        pre = b""
        if is_scanned(i, hz):
            scan = page.insert_image(page.rect, stream=_scan_jpeg(content))
            scan_names[f"Scan{i}"] = scan
            pre = f"q {W:g} 0 0 {H:g} 0 0 cm /Scan{i} Do Q\n".encode()
        draw = (pre + b"q 90 0 0 30 40 744 cm /Logo Do Q\n" + content
                + b"q 1 0 0 1 140 742 cm /LH Do Q\n")
        cx = doc.get_new_xref()
        doc.update_object(cx, "<<>>")
        doc.update_stream(cx, draw)
        if is_scanned(i, hz):
            doc.xref_set_key(page.xref, "Resources", f"{shared_res} 0 R")
        else:
            res = doc.get_new_xref()
            doc.update_object(res, "<< /Font << /Helv %d 0 R /Cour %d 0 R >> "
                              "/XObject << /Logo %d 0 R /LH %d 0 R >> >>"
                              % (fonts["Helv"], fonts["Cour"], logo_xref, lh))
            doc.xref_set_key(page.xref, "Resources", f"{res} 0 R")
        doc.xref_set_key(page.xref, "Contents", f"{cx} 0 R")
        if i in tagged_pages:
            doc.xref_set_key(page.xref, "StructParents", str(i))
    scans = " ".join(f"/{k} {v} 0 R" for k, v in scan_names.items())
    doc.update_object(shared_res, "<< /Font << /Helv %d 0 R /Cour %d 0 R >> "
                      "/XObject << /Logo %d 0 R /LH %d 0 R %s >> >>"
                      % (fonts["Helv"], fonts["Cour"], logo_xref, lh, scans))
    _add_single_page_hazards(doc, hz)
    _add_structure(doc, tagged_pages, hz)
    _add_navigation(doc, n, hz)
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data, build_payload(data, hz), hz


def _add_single_page_hazards(doc, hz) -> None:
    w = doc[hz["widget"]]
    field = pymupdf.Widget()
    field.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
    field.field_name = "member_number"
    field.field_value = f"Account {ACCOUNT}"
    field.rect = pymupdf.Rect(360, 700, 560, 718)
    field.text_fontsize = 9
    w.add_widget(field)
    a = doc[hz["annots"]]
    a.add_text_annot((560, 120), f"Call {FIRST} {LAST} on {PHONE} about {ACCOUNT}")
    a.add_freetext_annot(pymupdf.Rect(360, 690, 570, 712),
                         f"{FIRST} {LAST} phoned 02/14/2026", fontsize=9)
    c = doc[hz["crop_image"]]
    c.insert_image(pymupdf.Rect(400, 712, 560, 760), stream=_png((200, 30, 30)))
    # The shared logo, placed once more below the crop box: an image the file
    # shows elsewhere, so only this placement goes (``remove_images``).
    c.insert_image(pymupdf.Rect(40, 712, 130, 742), xref=doc[0].get_images()[0][0])
    c.set_cropbox(pymupdf.Rect(0, 0, W, 700))
    inline = doc.get_new_xref()
    doc.update_object(inline, "<<>>")
    doc.update_stream(inline, b"q 50 0 0 20 200 30 cm BI /W 8 /H 8 /CS /G /BPC 8 ID "
                      + bytes(range(64, 128)) + b" EI Q\n")
    contents = doc.xref_get_key(c.xref, "Contents")
    refs = contents[1].strip("[]") if contents[0] == "array" else contents[1]
    doc.xref_set_key(c.xref, "Contents", f"[{refs} {inline} 0 R]")
    ocg = doc.add_ocg("Draft stamp", on=False)
    doc[hz["layer_image"]].insert_image(pymupdf.Rect(380, 300, 560, 360),
                                        stream=_png((30, 160, 60)), oc=ocg)
    doc[hz["code"]].insert_image(pymupdf.Rect(430, 600, 550, 720), stream=_code_png())
    doc[hz["rotated"]].set_rotation(90)


def _add_structure(doc, tagged_pages: dict[int, int], hz) -> None:
    """One structure element per tagged page, and one more whose marked
    content runs from the last line of the ``span`` page into the first line
    of the next, the shape a paragraph broken across pages has."""
    cat = doc.pdf_catalog()
    root, top, parent_tree = doc.get_new_xref(), doc.get_new_xref(), doc.get_new_xref()
    a, b = hz["span"], hz["span"] + 1
    across = doc.get_new_xref()
    elems, nums = [], []
    for i, ncid in sorted(tagged_pages.items()):
        e = doc.get_new_xref()
        ids = list(range(ncid))
        if i == a:
            ids = ids[:-1]
        elif i == b:
            ids = ids[1:]
        alt = f" /Alt (Photo of {_esc(FIRST)} {_esc(LAST)})" if i == hz["alt_value"] else ""
        doc.update_object(e, f"<< /Type /StructElem /S /P /P {top} 0 R "
                             f"/Pg {doc[i].xref} 0 R /K [{' '.join(map(str, ids))}]{alt} >>")
        elems.append(f"{e} 0 R")
        slots = [f"{e} 0 R"] * ncid
        if i == a:
            slots[-1] = f"{across} 0 R"
        elif i == b:
            slots[0] = f"{across} 0 R"
        nums.append(f"{i} [{' '.join(slots)}]")
    doc.update_object(across, (
        f"<< /Type /StructElem /S /P /P {top} 0 R /Pg {doc[a].xref} 0 R "
        f"/K [{tagged_pages[a] - 1} << /Type /MCR /Pg {doc[b].xref} 0 R /MCID 0 >>] "
        f"/E (Statement {4100 + a} for {_esc(FIRST)}) >>"))
    elems.append(f"{across} 0 R")
    doc.update_object(top, f"<< /Type /StructElem /S /Document /P {root} 0 R "
                           f"/K [{' '.join(elems)}] >>")
    doc.update_object(parent_tree, f"<< /Nums [{' '.join(nums)}] >>")
    doc.update_object(root, f"<< /Type /StructTreeRoot /K {top} 0 R "
                            f"/ParentTree {parent_tree} 0 R >>")
    doc.xref_set_key(cat, "StructTreeRoot", f"{root} 0 R")
    doc.xref_set_key(cat, "MarkInfo", "<< /Marked true >>")


def _add_navigation(doc, n, hz) -> None:
    targets = sorted({1, hz["removed"], hz["blackout"], hz["rotated"], hz["dense"],
                      n // 2, n - 1} | set(range(5, n, max(1, n // 12))))
    toc = [[1, "Contents", 1]]
    toc += [[1, f"Section {p + 1}", p + 1] for p in targets]
    toc.insert(2, [2, f"Summary for {LAST}", hz["removed"] + 1])
    doc.set_toc(toc)
    names = {f"sec-{p:04d}": p for p in targets}
    names[f"acct-{ACCOUNT}-summary"] = hz["removed"]
    pairs = " ".join(f"({k}) [{doc[v].xref} 0 R /XYZ 0 {H:g} 0]"
                     for k, v in sorted(names.items()))
    dests = doc.get_new_xref()
    doc.update_object(dests, f"<< /Names [{pairs}] >>")
    doc.xref_set_key(doc.pdf_catalog(), "Names", f"<< /Dests {dests} 0 R >>")
    first = doc[0]
    annots = []
    y = 120.0
    for k, (name, p) in enumerate(sorted(names.items())):
        if y > 700:
            break
        rect = f"[300 {H - y - 10:g} 560 {H - y:g}]"
        action = (f"/A << /S /GoTo /D ({name}) >>" if k % 2
                  else f"/Dest [{doc[p].xref} 0 R /XYZ 0 {H:g} 0]")
        x = doc.get_new_xref()
        doc.update_object(x, f"<< /Type /Annot /Subtype /Link /Rect {rect} "
                             f"/Border [0 0 0] {action} >>")
        annots.append(f"{x} 0 R")
        y += 14
    x = doc.get_new_xref()
    doc.update_object(x, "<< /Type /Annot /Subtype /Link /Rect [300 60 560 74] "
                         f"/Border [0 0 0] /A << /S /GoTo /D (acct-{ACCOUNT}-summary) >> >>")
    annots.append(f"{x} 0 R")
    uri = doc.get_new_xref()
    doc.update_object(uri, "<< /Type /Annot /Subtype /Link /Rect [300 40 560 54] "
                           "/Border [0 0 0] /A << /S /URI /URI "
                           f"(https://portal.example.org/view?acct={ACCOUNT}) >> >>")
    annots.append(f"{uri} 0 R")
    doc.xref_set_key(first.xref, "Annots", f"[{' '.join(annots)}]")
    doc.set_page_labels([
        {"startpage": 0, "prefix": "ST-", "style": "D", "firstpagenum": 1},
        {"startpage": n - 5, "prefix": f"AC{ACCOUNT}-", "style": "D", "firstpagenum": 1},
    ])


# ── the payload ─────────────────────────────────────────────────────────


def _kind(word: str, dense: bool) -> str | None:
    w = word.strip(",.;()")
    if dense:
        return "Number"
    if any(ch.isdigit() for ch in w):
        if "/" in w:
            return "Date"
        if w == ACCOUNT:
            return "Account"
        if w.count("-") == 2:
            return "Phone"
        return "Number"
    if w in PEOPLE:
        return "Person"
    return None


def _match(page: int, rect, kind: str, text: str, **extra) -> dict:
    r = pymupdf.Rect(rect)
    return {"page": page, "type": kind, "text": text, "enabled": True,
            "rect": {"x0": r.x0 - 0.6, "y0": r.y0 - 0.6,
                     "x1": r.x1 + 0.6, "y1": r.y1 + 0.6}, **extra}


def build_payload(data: bytes, hz: dict[str, int]) -> dict:
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        matches: list[dict] = []
        rng = random.Random(7)
        for i, page in enumerate(doc):
            if i in (hz["removed"], hz["blackout"]):
                continue
            for x0, y0, x1, y1, word, *_ in page.get_text("words"):
                rect = pymupdf.Rect(x0, y0, x1, y1)
                if page.rotation:
                    rect = rect * page.rotation_matrix
                kind = _kind(word, i == hz["dense"])
                if kind is None:
                    continue
                extra = {}
                if kind == "Person" and word.strip(",") in (FIRST, LAST):
                    extra["redact_label"] = PARTY
                if rng.random() < 0.02:
                    extra["enabled"] = False
                matches.append(_match(i, rect, kind, word, **extra))
        matches.append(_match(hz["code"], (430, 600, 550, 720), "Code", ""))
        matches.append(_match(hz["widget"], (360, 700, 560, 718), "Account", ACCOUNT))
        matches.append(_match(hz["annots"], (360, 690, 570, 712), "Person", LAST))
        matches.append(_match(hz["annots"], (300, 300, 420, 316), "Unlisted", "x"))
        for i in range(3, len(doc), 10):
            if i not in (hz["removed"], hz["blackout"]):
                matches.append(_match(i, (100, 40, 180, 52), "Manual Region", ""))
        payload = {
            "matches": matches, "redact_label": LABEL, "active_categories": CATEGORIES,
            "removed_pages": [hz["removed"]], "blackout_pages": [hz["blackout"]],
            "keep_uncovered_lines": True,
            "side_text": _side_edits(doc), "strip_tags_pages": [hz["actualtext"]],
            "remove_text": _hidden_words(doc, hz["hidden_white"]),
        }
        payload.update(_image_plan(doc))
        return payload
    finally:
        doc.close()


def _side_edits(doc) -> list[dict]:
    """An edit for every string outside the pages that holds a value: links
    removed, everything else rewritten with the value replaced."""
    edits = []
    for e in list_side_text(doc)["entries"]:
        values = [v for v in (ACCOUNT, FIRST, LAST) if v in e["text"]]
        if not values:
            continue
        if e["kind"] == "link":
            edits.append({"id": e["id"], "remove": True})
            continue
        text = e["text"]
        for v in values:
            text = text.replace(v, LABEL)
        edits.append({"id": e["id"], "text": text})
    return edits


def _image_plan(doc) -> dict:
    """Blank an image no placement shows; remove a placement below the crop
    box of an image the file shows elsewhere, and an inline one."""
    blank, remove = set(), []
    for u in list_unseen_images(doc)["unseen"]:
        if u["xref"] and not u["xref_shown"]:
            blank.add(u["xref"])
        elif u["state"] == "outside-crop":
            remove.append({"page": u["page"], "box": u["bbox"]})
    return {"blank_images": sorted(blank), "remove_images": remove}


def _hidden_words(doc, pno: int) -> list[dict]:
    words = _trace_text_doc(doc, pno)["words"]
    return [{"page": pno, "box": w["bbox"], "text": w["text"], "mode": w["mode"],
             "layer": w["layer"]}
            for w in words if w["text"] in ("PIN", "8841")]
