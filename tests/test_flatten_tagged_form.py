"""Flattening a tagged form leaves no widget object behind (v0.9.2).

``bake(widgets=True)`` paints each widget into its page and takes it off the
page's ``/Annots``, but a tagged form's structure tree still points at every
widget (``/K << /Type /OBJR /Obj N 0 R >>``), so the widget objects and the
parent field dictionaries behind them survived the save with their ``/V``
values. Through v0.9.1 a filled tagged form exported with every value still in
the file, though no page showed a live field.

* **Happy path** -- after ``apply_redactions`` no widget object and no field
  value is left in the delivered bytes, for a value on the widget itself and
  a value on its parent field.
* **The page is intact** -- the baked appearance is still drawn: the value
  that was not redacted is still on the page as static text.
* **Redaction still works** -- a value under a box is gone from the page and
  from the file.
* **The structure tree survives** -- it stays in the file, and its object
  reference now resolves to null.
* **Untagged forms are unchanged** -- the plain form path still flattens.

Synthetic fixture; sentinels are invented.
"""
from __future__ import annotations

import pymupdf

from lexcloak_pdf_tool import apply_redactions


OWN_VALUE = "OWNVALUE-Brisco-Hale-4471"
PARENT_VALUE = "PARENTVALUE-Ines-Carrow-2208"
OWN_RECT = (72, 100, 360, 120)
PARENT_RECT = (72, 200, 360, 220)


def _tagged_form(tagged: bool = True) -> bytes:
    """Two text fields on one page, each linked from the structure tree.

    The first carries its value on the widget. The second is a kid of a
    separate parent field dictionary that holds the value, the shape of a
    field with more than one widget.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    for name, value, rect in (("own", OWN_VALUE, OWN_RECT),
                              ("kid", PARENT_VALUE, PARENT_RECT)):
        w = pymupdf.Widget()
        w.field_name = name
        w.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
        w.field_value = value
        w.rect = pymupdf.Rect(*rect)
        w.text_fontsize = 11
        page.add_widget(w)
    own, kid = (xref for xref, _, _ in page.annot_xrefs())

    # Move the kid's name and value up to a parent field dictionary.
    parent = doc.get_new_xref()
    doc.update_object(parent, f"<< /FT /Tx /T (parent) /V ({PARENT_VALUE})"
                              f" /Kids [{kid} 0 R] >>")
    doc.xref_set_key(kid, "Parent", f"{parent} 0 R")
    doc.xref_set_key(kid, "T", "null")
    doc.xref_set_key(kid, "V", "null")
    fields = doc.xref_get_key(doc.pdf_catalog(), "AcroForm/Fields")[1]
    doc.xref_set_key(doc.pdf_catalog(), "AcroForm/Fields",
                     fields.replace(f"{kid} 0 R", f"{parent} 0 R"))

    if tagged:
        root = doc.get_new_xref()
        elems = []
        for i, xref in enumerate((own, kid)):
            elem = doc.get_new_xref()
            doc.update_object(elem, f"<< /Type /StructElem /S /Form /P {root} 0 R"
                                    f" /Pg {page.xref} 0 R"
                                    f" /K << /Type /OBJR /Obj {xref} 0 R"
                                    f" /Pg {page.xref} 0 R >> >>")
            doc.xref_set_key(xref, "StructParent", str(i))
            elems.append(elem)
        kids = " ".join(f"{e} 0 R" for e in elems)
        doc.update_object(root, f"<< /Type /StructTreeRoot /K [{kids}]"
                                f" /ParentTree << /Nums [0 {elems[0]} 0 R"
                                f" 1 {elems[1]} 0 R] >> /ParentTreeNextKey 2 >>")
        doc.xref_set_key(doc.pdf_catalog(), "StructTreeRoot", f"{root} 0 R")
        doc.xref_set_key(doc.pdf_catalog(), "MarkInfo", "<< /Marked true >>")
    out = doc.tobytes()
    doc.close()
    return out


def _objr_targets(doc) -> list[str]:
    targets = []
    for xref in range(1, doc.xref_length()):
        if doc.xref_get_key(xref, "K/Type")[1] == "/OBJR":
            targets.append(doc.xref_get_key(xref, "K/Obj")[1])
    return targets


def _widget_objects(doc) -> list[int]:
    return [x for x in range(1, doc.xref_length())
            if doc.xref_get_key(x, "Subtype")[1] == "/Widget"]


def test_fixture_is_a_tagged_form_with_both_value_shapes():
    doc = pymupdf.open(stream=_tagged_form(), filetype="pdf")
    assert doc.is_form_pdf
    assert len(_widget_objects(doc)) == 2
    assert [w.field_value for w in doc[0].widgets()] == [OWN_VALUE, PARENT_VALUE]
    assert len(_objr_targets(doc)) == 2


def test_no_widget_object_or_value_survives_a_flatten():
    out, _ = apply_redactions(_tagged_form(), [])
    assert b"/Widget" not in out
    assert b"/AcroForm" not in out
    doc = pymupdf.open(stream=out, filetype="pdf")
    assert _widget_objects(doc) == []
    for xref in range(1, doc.xref_length()):
        assert doc.xref_get_key(xref, "V")[0] == "null"
        assert doc.xref_get_key(xref, "FT")[0] == "null"
    # Both values are still drawn on the page, as static text.
    text = doc[0].get_text()
    assert OWN_VALUE in text
    assert PARENT_VALUE in text


def test_structure_tree_stays_and_points_at_nothing():
    out, _ = apply_redactions(_tagged_form(), [])
    doc = pymupdf.open(stream=out, filetype="pdf")
    assert doc.xref_get_key(doc.pdf_catalog(), "StructTreeRoot")[0] == "xref"
    # The two elements are identical once their targets are null, and the
    # save's garbage=4 merges identical objects, so count only what they hold.
    targets = _objr_targets(doc)
    assert targets and set(targets) == {"null"}


def test_a_boxed_value_is_gone_and_the_other_stays():
    x0, y0, x1, y1 = PARENT_RECT
    matches = [{"page": 0, "enabled": True, "type": "custom",
                "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}]
    out, _ = apply_redactions(_tagged_form(), matches)
    doc = pymupdf.open(stream=out, filetype="pdf")
    text = doc[0].get_text()
    assert PARENT_VALUE not in text
    assert OWN_VALUE in text
    assert _widget_objects(doc) == []
    for xref in range(1, doc.xref_length()):
        assert PARENT_VALUE not in doc.xref_object(xref)


def test_untagged_form_still_flattens():
    out, _ = apply_redactions(_tagged_form(tagged=False), [])
    doc = pymupdf.open(stream=out, filetype="pdf")
    assert _widget_objects(doc) == []
    assert not doc.is_form_pdf
    text = doc[0].get_text()
    assert OWN_VALUE in text and PARENT_VALUE in text
